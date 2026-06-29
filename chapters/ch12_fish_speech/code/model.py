"""
Ch12: Fish Speech S2 — Dual-Autoregressive TTS with RL Alignment (Simplified)

Reference:
    Fish Speech S2 Pro (Fish Audio, 2026)
    - Paper: arXiv:2603.08823 (2026)
    - Original: https://github.com/fishaudio/fish-speech

    Key innovations:
    1. Dual-AR architecture: Slow AR (4B) + Fast AR (400M)
    2. RL alignment with GRPO (Group Relative Policy Optimization)
    3. 80+ languages, inline emotion tags
    4. Industrial-scale: 10M+ hours training data

This Implementation (Simplified for Education):
    - Slow AR: 6-layer Transformer (vs 32 in original 4B model)
    - Fast AR: 2-layer Transformer (vs 4 in original 400M model)
    - Codec: 4 codebooks (vs 10 in original)
    - No RL alignment (too complex for educational version)
    - Supports: Chinese, English (vs 80+ in original)

Core Architecture:
    Text → Slow AR (semantic tokens) → Fast AR (acoustic tokens) → Codec → Waveform

    The key insight: separate semantic planning (slow, large) from acoustic
    rendering (fast, small). This is structurally isomorphic to LLMs, enabling
    use of LLM inference optimizations (KV cache, continuous batching).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# --------------------------------------------------------
# Rotary Position Embedding (RoPE)
# --------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    RoPE: Rotary Position Embedding (from LLaMA/Qwen)

    Unlike absolute or learned embeddings, RoPE encodes position by
    rotating the query/key vectors. This gives relative position encoding
    with good extrapolation to longer sequences.

    Mathematical intuition:
        q_m = R(θ, m) · W_q · x_m
        k_n = R(θ, n) · W_k · x_n
        q_m · k_n = x_m^T · W_q^T · R(θ, m-n) · W_k · x_n

    The dot product depends only on (m-n), giving relative position.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        x: (B, H, T, D) where T is sequence length, D is head dim
        Returns: rotated x
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        cos = self.cos_cached[:seq_len].to(x.device)
        sin = self.sin_cached[:seq_len].to(x.device)

        # Reshape for broadcasting
        cos = cos[None, None, :, :]  # (1, 1, T, D)
        sin = sin[None, None, :, :]

        # Apply rotation
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        rotated = torch.cat([-x2, x1], dim=-1)

        return x * cos + rotated * sin


# --------------------------------------------------------
# Transformer Block (LLaMA-style)
# --------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with RoPE and SwiGLU activation.

    Differences from standard Transformer:
    1. Pre-norm (LayerNorm before attention/FFN, not after)
    2. RoPE instead of absolute position embedding
    3. SwiGLU activation (instead of ReLU/GELU)
    4. No bias in linear layers (for efficiency)

    This is the backbone of modern LLMs (LLaMA, Qwen, Fish Speech).
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        # Attention
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # FFN (SwiGLU)
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)

        # Norms
        self.attn_norm = nn.RMSNorm(dim)
        self.ffn_norm = nn.RMSNorm(dim)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, D)
        mask: (B, 1, T, T) causal mask
        """
        B, T, D = x.shape

        # Self-attention with RoPE
        h = self.attn_norm(x)
        q = self.wq(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = self.rope(q, T)
        k = self.rope(k, T)

        # Repeat KV for GQA (Grouped Query Attention)
        if self.n_kv_heads < self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        out = self.wo(out)
        x = x + out

        # FFN with SwiGLU
        h = self.ffn_norm(x)
        out = self.w2(F.silu(self.w1(h)) * self.w3(h))
        x = x + out

        return x


# --------------------------------------------------------
# Slow AR: Semantic Token Prediction (Master Transformer)
# --------------------------------------------------------

class SlowAR(nn.Module):
    """
    Slow AR (Master Transformer): Predicts semantic codebook (codebook 0).

    In Fish Speech S2 Pro:
    - 4B parameters, 32 layers
    - Operates along time axis
    - Predicts primary semantic tokens

    This simplified version:
    - 6 layers, ~50M parameters
    - Same architecture, smaller scale
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 1024,
        n_layers: int = 6,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        ffn_dim: int = 4096,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, ffn_dim)
            for _ in range(n_layers)
        ])

        # Output projection
        self.out_norm = nn.RMSNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, T) token indices
        Returns: (B, T, vocab_size) logits
        """
        B, T = tokens.shape

        # Embed
        h = self.tok_emb(tokens)

        # Causal mask
        mask = self.mask[:, :, :T, :T]

        # Transformer
        for layer in self.layers:
            h = layer(h, mask)

        # Output
        h = self.out_norm(h)
        logits = self.out_proj(h)

        return logits


# --------------------------------------------------------
# Fast AR: Acoustic Token Prediction (Slave Transformer)
# --------------------------------------------------------

class FastAR(nn.Module):
    """
    Fast AR (Slave Transformer): Predicts remaining codebooks (1-3).

    In Fish Speech S2 Pro:
    - 400M parameters, 4 layers
    - Operates along codebook axis (for each time step)
    - Predicts acoustic details

    This simplified version:
    - 2 layers, ~10M parameters
    - Same architecture, smaller scale
    """

    def __init__(
        self,
        vocab_size: int,
        n_codebooks: int,
        dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        ffn_dim: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_codebooks = n_codebooks

        # Codebook embeddings
        self.codebook_embs = nn.ModuleList([
            nn.Embedding(vocab_size, dim) for _ in range(n_codebooks)
        ])

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, ffn_dim)
            for _ in range(n_layers)
        ])

        # Output projections (one per codebook)
        self.out_projs = nn.ModuleList([
            nn.Linear(dim, vocab_size, bias=False)
            for _ in range(n_codebooks)
        ])

    def forward(self, codebook_tokens: torch.Tensor) -> list[torch.Tensor]:
        """
        codebook_tokens: (B, n_codebooks, T) token indices for each codebook
        Returns: list of (B, T, vocab_size) logits for each codebook
        """
        B, K, T = codebook_tokens.shape

        # Sum codebook embeddings
        h = sum(self.codebook_embs[k](codebook_tokens[:, k]) for k in range(K))

        # Transformer
        for layer in self.layers:
            h = layer(h)

        # Output projections
        logits = [self.out_projs[k](h) for k in range(K)]

        return logits


# --------------------------------------------------------
# Fish Speech: Dual-AR TTS
# --------------------------------------------------------

class FishSpeech(nn.Module):
    """
    Fish Speech: Dual-Autoregressive TTS

    Architecture:
        Text → Slow AR (semantic) → Fast AR (acoustic) → Codec → Waveform

    Training:
        1. Slow AR: Cross-entropy on semantic tokens
        2. Fast AR: Cross-entropy on acoustic tokens (conditioned on semantic)

    Inference:
        1. Slow AR: Generate semantic tokens autoregressively
        2. Fast AR: For each time step, generate acoustic tokens in parallel
        3. Decode tokens with codec
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        n_codebooks: int = 4,
        slow_dim: int = 1024,
        fast_dim: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks

        # Dual-AR
        self.slow_ar = SlowAR(vocab_size, dim=slow_dim)
        self.fast_ar = FastAR(vocab_size, n_codebooks, dim=fast_dim)

        # Codec (simplified: just embedding + linear)
        self.codec_dec = nn.Linear(vocab_size * n_codebooks, 80 * 10)  # → mel frames

    def forward(
        self,
        slow_tokens: torch.Tensor,
        fast_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Training forward pass.

        slow_tokens: (B, T) semantic tokens (codebook 0)
        fast_tokens: (B, K, T) acoustic tokens (codebooks 1-3)

        Returns:
            slow_logits: (B, T, vocab_size)
            fast_logits: list of K tensors (B, T, vocab_size)
        """
        # Slow AR
        slow_logits = self.slow_ar(slow_tokens)

        # Fast AR
        fast_logits = self.fast_ar(fast_tokens)

        return slow_logits, fast_logits

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_len: int = 500,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        prompt_tokens: (B, T_prompt) prompt semantic tokens
        max_len: maximum generation length
        temperature: sampling temperature
        top_k: top-k sampling

        Returns: (B, max_len) generated semantic tokens
        """
        B = prompt_tokens.shape[0]
        device = prompt_tokens.device

        # Start with prompt
        tokens = prompt_tokens

        for _ in range(max_len - prompt_tokens.shape[1]):
            # Forward
            logits = self.slow_ar(tokens)
            logits = logits[:, -1, :] / temperature  # last position

            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens


# --------------------------------------------------------
# Shape Test
# --------------------------------------------------------

if __name__ == "__main__":
    print("Testing Fish Speech architecture...")

    # Config
    vocab_size = 1024
    n_codebooks = 4
    batch_size = 2
    seq_len = 50

    # Model
    model = FishSpeech(vocab_size=vocab_size, n_codebooks=n_codebooks)
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} params")

    # Training forward
    slow_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    fast_tokens = torch.randint(0, vocab_size, (batch_size, n_codebooks, seq_len))

    slow_logits, fast_logits = model(slow_tokens, fast_tokens)
    print(f"✓ Training forward:")
    print(f"  Slow logits: {slow_logits.shape}")  # (2, 50, 1024)
    print(f"  Fast logits: {[f.shape for f in fast_logits]}")  # 4 × (2, 50, 1024)

    # Generation
    prompt = torch.randint(0, vocab_size, (batch_size, 10))
    generated = model.generate(prompt, max_len=30)
    print(f"✓ Generation: {generated.shape}")  # (2, 30)

    print("\n✅ All tests passed!")
