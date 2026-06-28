"""
SimpleOmni — A teaching implementation of MiniMind-O (arXiv:2605.03937)
=======================================================================

MiniMind-O is a ~0.1B parameter omni model that supports:
  Input:  Text + Speech (+ Image, optional)
  Output: Text + Streaming Speech

Architecture: Thinker-Talker dual-path
  - Thinker: Transformer backbone for multimodal understanding → text output
  - Talker:  Smaller Transformer for acoustic rendering → Mimi codes → waveform

This teaching version (~50M params) simplifies:
  - Thinker: 6 layers, hidden=384 (vs 8 layers, hidden=768)
  - Talker:  2 layers, hidden=384, 4 codebooks (vs 4 layers, 8 codebooks)
  - Same bridge mechanism, MTP head, and delay pattern

Key components:
  SimpleThinker       — Causal Transformer for text + multimodal input
  SimpleTalker        — Smaller Transformer for audio code generation
  SimpleTalkerHead    — MTP head: shared base + per-codebook adapters
  SimpleTalkerEmbed   — Embedding that fuses multi-codebook input
  SimpleAudioProjector — 2-layer MLP: audio features → hidden space
  SimpleOmni          — Top-level model with forward() and stream_generate()

Based on:
  - Paper:   arXiv:2605.03937 (Gong, 2026)
  - Code:    https://github.com/jingyaogong/minimind-o
  - License: Apache 2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ===========================================================================
# Configuration
# ===========================================================================

class SimpleOmniConfig:
    """Configuration for SimpleOmni (teaching version of MiniMind-O).

    Original MiniMind-O (minimind-3o):
        hidden_size=768, num_layers=8, talker_layers=4, codebooks=8
        Trainable: ~113M params

    Teaching version (this file):
        hidden_size=384, num_layers=6, talker_layers=2, codebooks=4
        Trainable: ~31M params
    """

    def __init__(self, **kwargs):
        # Thinker (language backbone)
        self.hidden_size = kwargs.get("hidden_size", 384)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 6)
        self.num_attention_heads = kwargs.get("num_attention_heads", 6)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 2)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.intermediate_size = kwargs.get("intermediate_size", 1024)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 2048)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.dropout = kwargs.get("dropout", 0.0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)

        # Talker (acoustic renderer)
        self.talker_hidden_size = kwargs.get("talker_hidden_size", 384)
        self.num_talker_layers = kwargs.get("num_talker_layers", 2)

        # Audio codec (Mimi)
        self.num_codebooks = kwargs.get("num_codebooks", 4)  # vs 8 in original
        self.audio_vocab_size = kwargs.get("audio_vocab_size", 2082)  # 2048 codes + specials
        self.audio_pad_token = kwargs.get("audio_pad_token", 2049)
        self.audio_stop_token = kwargs.get("audio_stop_token", 2050)
        self.audio_spk_token = kwargs.get("audio_spk_token", 2051)

        # Audio encoder (SenseVoice)
        self.audio_hidden_size = kwargs.get("audio_hidden_size", 512)

        # Bridge layer: which Thinker layer feeds the Talker
        # Original: num_hidden_layers // 2 - 1 = 3 (for 8 layers)
        # Teaching: 6 // 2 - 1 = 2
        self.bridge_layer = kwargs.get("bridge_layer",
                                        self.num_hidden_layers // 2 - 1)

        # Speaker embedding (CAM++)
        self.spk_emb_size = kwargs.get("spk_emb_size", 192)

        # MTP adapter rank
        self.adapter_rank = kwargs.get("adapter_rank", 128)  # vs 256 in original

        # Special token IDs
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)


# ===========================================================================
# Building Blocks (shared by Thinker and Talker)
# ===========================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Same as LayerNorm but without mean subtraction — just scale by RMS.
    Faster and works better for Transformer hidden states.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (self.weight * x.float() * rms).type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 1e6):
    """Precompute cos/sin tables for Rotary Position Embedding (RoPE).

    RoPE encodes position by rotating Q and K vectors. The rotation angles
    decrease geometrically across dimensions, so low dims rotate fast (local
    position) and high dims rotate slow (global position).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE rotation to Q and K tensors.

    q, k: (B, T, n_heads, head_dim)
    cos, sin: (T, head_dim)  — sliced from precomputed tables
    We unsqueeze to (1, T, 1, head_dim) for broadcasting.
    """
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(2)  # (1, T, 1, head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for Grouped Query Attention (GQA).

    With GQA, we have fewer KV heads than Q heads. This function
    duplicates each KV head n_rep times to match the Q head count.
    """
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    return (x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim))


# ===========================================================================
# Attention
# ===========================================================================

class Attention(nn.Module):
    """Multi-head attention with GQA (Grouped Query Attention) and RoPE.

    GQA: fewer KV heads than Q heads → less KV-cache memory.
    RoPE: rotary position embedding → better length extrapolation.
    """

    def __init__(self, config: SimpleOmniConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False, mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)
        new_kv = (k, v) if use_cache else None

        # GQA: repeat KV heads
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = repeat_kv(k, self.n_rep).transpose(1, 2)
        v = repeat_kv(v, self.n_rep).transpose(1, 2)

        # Attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if T > 1 and past_kv is None:
            causal = torch.full((T, T), float("-inf"), device=x.device).triu(1)
            scores = scores + causal
        if mask is not None:
            scores = scores + (1.0 - mask.unsqueeze(1).unsqueeze(2)) * -1e9

        attn = F.softmax(scores.float(), dim=-1).type_as(q)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        out = self.o_proj(out)
        return out, new_kv


# ===========================================================================
# Feed-Forward Network
# ===========================================================================

class FeedForward(nn.Module):
    """SwiGLU feed-forward: gate * up → down.

    SwiGLU = SiLU(gate(x)) * up(x), then project down.
    Better than ReLU FFN at the same parameter count.
    """

    def __init__(self, config: SimpleOmniConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


# ===========================================================================
# Transformer Block
# ===========================================================================

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: RMSNorm → Attention → Residual → RMSNorm → FFN → Residual."""

    def __init__(self, config: SimpleOmniConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False, mask=None):
        residual = x
        x, kv = self.self_attn(self.attn_norm(x), cos, sin, past_kv, use_cache, mask)
        x = x + residual
        x = x + self.mlp(self.ffn_norm(x))
        return x, kv


# ===========================================================================
# Thinker (Understanding Pathway)
# ===========================================================================

class SimpleThinker(nn.Module):
    """Thinker: causal Transformer for multimodal understanding.

    Processes text tokens (and injected audio/vision features) through
    N transformer layers. The bridge_layer's output is captured as
    conditioning for the Talker.

    Key insight: the MIDDLE layer (not the last) provides the best
    conditioning signal for speech generation, because:
    - Embedding layer: too little semantic information
    - Final layer: too specialized for next-token prediction
    - Middle layer: balanced contextual + cross-modal fusion
    """

    def __init__(self, config: SimpleOmniConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, hidden_states_override=None,
                past_kvs=None, use_cache=False, mask=None):
        """
        Args:
            input_ids: (B, T) text token IDs
            hidden_states_override: (B, T, D) optional pre-computed embeddings
                                    (for injected audio/vision features)
            past_kvs: list of (k, v) tuples for KV-cache
            use_cache: whether to return new KV-cache
            mask: attention mask

        Returns:
            h_final: (B, T, D) final hidden states → text logits
            bridge: (B, T, D) middle-layer states → Talker conditioning
            presents: list of new (k, v) caches
        """
        B, T = input_ids.shape
        past_kvs = past_kvs or [None] * len(self.layers)
        start_pos = past_kvs[0][0].shape[1] if past_kvs[0] is not None else 0

        if hidden_states_override is not None:
            h = hidden_states_override
        else:
            h = self.embed_tokens(input_ids)

        cos = self.freqs_cos[start_pos:start_pos + T]
        sin = self.freqs_sin[start_pos:start_pos + T]

        bridge = h  # will be overwritten at bridge_layer
        presents = []

        for i, (layer, past_kv) in enumerate(zip(self.layers, past_kvs)):
            h, present = layer(h, cos, sin, past_kv, use_cache, mask)
            presents.append(present)
            if i == self.config.bridge_layer:
                bridge = h

        h_final = self.norm(h)
        return h_final, bridge, presents


# ===========================================================================
# MTP Head: Multi-Token Prediction for Parallel Codebook Output
# ===========================================================================

class SimpleTalkerHead(nn.Module):
    """MTP (Multi-Token Prediction) head for parallel codebook prediction.

    Instead of 8 separate output heads (one per codebook), we use:
    - A shared base linear layer (captures commonalities)
    - Per-codebook low-rank adapters (capture differences)

    logits_i = base(x) + adapter_i(x)

    This reduces parameters from 8× to ~1.5× a single head.

    The adapter rank controls how much codebook-specific capacity each
    codebook gets. rank=128 is a good balance for 4 codebooks.
    """

    def __init__(self, config: SimpleOmniConfig):
        super().__init__()
        self.base = nn.Linear(config.talker_hidden_size, config.audio_vocab_size, bias=False)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.talker_hidden_size, config.adapter_rank, bias=False),
                nn.GELU(),
                nn.Linear(config.adapter_rank, config.audio_vocab_size, bias=False),
            )
            for _ in range(config.num_codebooks)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of (B, T, audio_vocab_size) logits, one per codebook."""
        base_out = self.base(x)
        return [base_out + adapter(x) for adapter in self.adapters]


# ===========================================================================
# Talker Embedding: Fuse Multi-Codebook Input
# ===========================================================================

class SimpleTalkerEmbed(nn.Module):
    """Embedding module that fuses multi-codebook audio IDs into a single vector.

    Each codebook has its own embedding adapter. The output is the average
    of (base_embed + adapter_embed) across all codebooks.

    This allows the Talker to "see" the history of all codebook streams
    as a single sequence of hidden vectors.
    """

    def __init__(self, config: SimpleOmniConfig):
        super().__init__()
        self.num_codebooks = config.num_codebooks
        self.base = nn.Embedding(config.audio_vocab_size, config.talker_hidden_size)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(config.audio_vocab_size, config.adapter_rank),
                nn.GELU(),
                nn.Linear(config.adapter_rank, config.talker_hidden_size, bias=False),
            )
            for _ in range(config.num_codebooks)
        ])

    def forward(self, audio_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_ids: (B, num_codebooks, T) — one ID stream per codebook

        Returns:
            (B, T, talker_hidden_size) — fused embedding
        """
        base_out = self.base(audio_ids)  # (B, C, T, D)
        total = torch.zeros_like(base_out[:, 0])  # (B, T, D)
        for i in range(self.num_codebooks):
            total = total + base_out[:, i] + self.adapters[i](audio_ids[:, i])
        return total / self.num_codebooks


# ===========================================================================
# Talker (Speaking Pathway)
# ===========================================================================

class SimpleTalker(nn.Module):
    """Talker: converts Thinker's semantic states into audio codebook codes.

    Architecture:
    1. Receive bridge states from Thinker → project to talker_hidden
    2. Receive audio code history → embed via TalkerEmbed
    3. Combine: hidden = embed_proj(bridge) * text_scale + codec_proj(audio_emb) * audio_scale
    4. Process through N Transformer blocks
    5. Output via MTP head: one set of logits per codebook

    The text_scale and audio_scale are learnable parameters that control
    the balance between semantic conditioning and autoregressive audio history.
    """

    def __init__(self, config: SimpleOmniConfig):
        super().__init__()
        self.config = config

        # Transformer blocks (same architecture as Thinker but smaller)
        talker_config = SimpleOmniConfig(
            hidden_size=config.talker_hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
        )
        self.layers = nn.ModuleList([
            TransformerBlock(talker_config) for _ in range(config.num_talker_layers)
        ])
        self.norm = RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps)

        # Input projections
        self.embed_tokens = SimpleTalkerEmbed(config)
        self.embed_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.talker_hidden_size),
            RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps),
        )
        self.codec_proj = nn.Sequential(
            nn.Linear(config.talker_hidden_size, config.talker_hidden_size),
            nn.GELU(),
            nn.Linear(config.talker_hidden_size, config.talker_hidden_size),
            RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps),
        )

        # Learnable scaling factors
        self.text_scale = nn.Parameter(torch.tensor(3.0))
        self.audio_scale = nn.Parameter(torch.tensor(1.0))

        # Speaker embedding projection
        self.spk_proj = nn.Linear(config.spk_emb_size, config.talker_hidden_size, bias=False)

        # Output head
        self.lm_head = SimpleTalkerHead(config)

        # RoPE
        head_dim = config.talker_hidden_size // config.num_attention_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(
            head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)


# ===========================================================================
# Audio Projector: SenseVoice features → Thinker hidden space
# ===========================================================================

class SimpleAudioProjector(nn.Module):
    """2-layer MLP that projects audio encoder features into the Thinker's
    hidden space.

    SenseVoice outputs 512-d features; Thinker expects hidden_size-d.
    The MLP learns a nonlinear mapping with LayerNorm for stability.
    """

    def __init__(self, config: SimpleOmniConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.audio_hidden_size),
            nn.Linear(config.audio_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ===========================================================================
# SimpleOmni: Top-Level Model
# ===========================================================================

class SimpleOmni(nn.Module):
    """End-to-end omni model: text + speech → text + streaming speech.

    Forward pass:
    1. Thinker processes text (with injected audio features at placeholder positions)
    2. Bridge captures middle-layer states
    3. Talker combines bridge states + audio code history
    4. Talker outputs logits for each codebook

    Generation:
    1. Thinker generates text tokens autoregressively
    2. Talker generates audio codes with delay pattern
    3. Mimi decoder converts codes to 24 kHz waveform incrementally

    Args:
        config: SimpleOmniConfig
        audio_encoder: frozen SenseVoice model (or None for text-only)
    """

    def __init__(self, config: SimpleOmniConfig = None, audio_encoder=None):
        super().__init__()
        self.config = config or SimpleOmniConfig()

        # Core components
        self.thinker = SimpleThinker(self.config)
        self.talker = SimpleTalker(self.config)
        self.audio_proj = SimpleAudioProjector(self.config)

        # Text output head
        self.text_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.text_head.weight = self.thinker.embed_tokens.weight

        # Frozen audio encoder (set externally)
        self.audio_encoder = audio_encoder

    def forward(self, input_ids, audio_ids=None, audio_features=None,
                spk_emb=None, past_kvs=None, use_cache=False, mask=None):
        """
        Forward pass for training.

        Args:
            input_ids: (B, T) text token IDs
                      or (B, 1+num_codebooks, T) if audio_ids are packed
            audio_ids: (B, num_codebooks, T) audio codebook IDs (if not packed)
            audio_features: (B, T_audio, audio_hidden) from SenseVoice (optional)
            spk_emb: (B, spk_emb_size) speaker embedding (optional)
            past_kvs: KV-cache from previous forward
            use_cache: whether to return KV-cache
            mask: attention mask

        Returns:
            dict with 'text_logits', 'audio_logits', 'past_kvs'
        """
        # Unpack if needed
        if input_ids.dim() == 3:
            audio_ids = input_ids[:, :self.config.num_codebooks]
            input_ids = input_ids[:, self.config.num_codebooks]

        B, T = input_ids.shape

        # Split KV-caches between Thinker and Talker
        n_thinker = len(self.thinker.layers)
        if past_kvs is None:
            past_kvs = [None] * (n_thinker + len(self.talker.layers))
        thinker_kvs = past_kvs[:n_thinker]
        talker_kvs = past_kvs[n_thinker:]

        # ---- THINKER ----
        # Inject audio features at placeholder positions (if provided)
        h_override = None
        if audio_features is not None:
            text_emb = self.thinker.embed_tokens(input_ids)
            # TODO: inject audio_features at audio placeholder positions
            # For now, pass text_emb as-is
            h_override = text_emb

        h_final, bridge, thinker_presents = self.thinker(
            input_ids, h_override, thinker_kvs, use_cache, mask
        )

        # ---- TALKER ----
        # Embed audio codes
        if audio_ids is not None:
            talker_emb = self.talker.embed_tokens(audio_ids)
        else:
            talker_emb = torch.zeros(B, T, self.config.talker_hidden_size,
                                     device=input_ids.device)

        # Inject speaker embedding at spk_token positions
        if spk_emb is not None:
            spk_hidden = self.talker.spk_proj(spk_emb)  # (B, talker_hidden)
            # TODO: inject at spk_token positions in talker_emb

        # Combine: text conditioning + audio history
        text_cond = self.talker.embed_proj(bridge) * self.talker.text_scale
        audio_cond = self.talker.codec_proj(talker_emb) * self.talker.audio_scale
        talker_input = text_cond + audio_cond

        # Process through Talker transformer
        cos = self.talker.freqs_cos[:T]
        sin = self.talker.freqs_sin[:T]

        h = talker_input
        talker_presents = []
        for layer, past_kv in zip(self.talker.layers, talker_kvs):
            h, present = layer(h, cos, sin, past_kv, use_cache, mask)
            talker_presents.append(present)

        h_talker = self.talker.norm(h)

        # Output heads
        text_logits = self.text_head(h_final)           # (B, T, vocab_size)
        audio_logits = self.talker.lm_head(h_talker)    # list of (B, T, audio_vocab_size)

        return {
            "text_logits": text_logits,
            "audio_logits": audio_logits,
            "past_kvs": thinker_presents + talker_presents,
        }

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.75,
                 top_p=0.9, stream=False, **kwargs):
        """Generate text and audio codes autoregressively.

        For streaming generation, yields (text_tokens, audio_frame) tuples.
        An audio_frame is available once all codebooks have produced codes
        for that time step (after the delay pattern settles).

        Args:
            input_ids: (1, T) prompt token IDs
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature for text
            top_p: nucleus sampling threshold
            stream: if True, yields incrementally

        Returns:
            If stream=False: (generated_ids, audio_codes)
            If stream=True: generator of (ids, audio_frame)
        """
        # TODO: implement full streaming generation with delay pattern
        # See stream_generate() in the original model_omni.py for reference
        pass

    @torch.inference_mode()
    def stream_generate(self, input_ids, max_new_tokens=256, temperature=0.75,
                        top_p=0.9, spk_emb=None, ref_codes=None, **kwargs):
        """Streaming generation with delay pattern.

        The key insight for streaming:
        - Text is generated one step ahead of audio
        - Audio codebooks are staggered: CB-0 starts at step 0, CB-1 at step 1, ...
        - After num_codebooks steps, a complete audio frame is available
        - Mimi decoder can incrementally decode frames → streaming playback

        Yields:
            (text_ids_or_None, audio_frame_or_None)
            - text_ids: current generated text tokens (None when text is done)
            - audio_frame: list of num_codebooks codes (None until delay settles)
        """
        # TODO: implement
        # Reference: model_omni.py stream_generate() in the original repo
        pass


# ===========================================================================
# Loss Computation
# ===========================================================================

def compute_omni_loss(text_logits, audio_logits, text_labels, audio_labels,
                      num_codebooks=4, stop_weight=10.0):
    """Compute combined text + audio loss for training.

    Text loss: standard cross-entropy on Thinker output.
    Audio loss: per-codebook cross-entropy on Talker output, with
                stop tokens weighted 10× higher.

    Why weight stop tokens?
    - Stop tokens are rare (1 per utterance) but critical
    - Without weighting, the model doesn't learn when to stop speaking
    - 10× weight was found empirically in the original paper

    Args:
        text_logits: (B, T, vocab_size)
        audio_logits: list of num_codebooks × (B, T, audio_vocab_size)
        text_labels: (B, T) with -100 for positions to ignore
        audio_labels: (B, num_codebooks, T) with -100 for positions to ignore
        num_codebooks: number of audio codebook layers
        stop_weight: weight multiplier for stop tokens (default 10×)

    Returns:
        (total_loss, text_loss, audio_loss)
    """
    # Text loss
    text_loss = F.cross_entropy(
        text_logits.view(-1, text_logits.size(-1)),
        text_labels.view(-1),
        ignore_index=-100,
    )

    # Audio loss (per-codebook)
    audio_loss = torch.tensor(0.0, device=text_logits.device)
    for i in range(num_codebooks):
        logits_i = audio_logits[i].view(-1, audio_logits[i].size(-1))
        labels_i = audio_labels[:, i, :].reshape(-1)
        per_token_loss = F.cross_entropy(logits_i, labels_i,
                                         ignore_index=-100, reduction='none')
        valid_mask = (labels_i != -100).float()
        stop_mask = (labels_i == 2050).float()  # audio_stop_token
        weighted_loss = per_token_loss * valid_mask * (1 + stop_mask * (stop_weight - 1))
        n_valid = valid_mask.sum().clamp(min=1)
        audio_loss = audio_loss + weighted_loss.sum() / n_valid
    audio_loss = audio_loss / num_codebooks

    return text_loss + audio_loss, text_loss, audio_loss


# ===========================================================================
# Utility: Parameter counting
# ===========================================================================

def count_parameters(model: nn.Module, verbose: bool = True) -> dict:
    """Count trainable and total parameters by module."""
    counts = {}
    for name, module in model.named_children():
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in module.parameters())
        counts[name] = {"trainable": trainable, "total": total}
        if verbose:
            print(f"  {name:20s}: {trainable/1e6:8.2f}M trainable / {total/1e6:8.2f}M total")

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    counts["_total"] = {"trainable": total_trainable, "total": total_params}
    if verbose:
        print(f"  {'TOTAL':20s}: {total_trainable/1e6:8.2f}M trainable / {total_params/1e6:8.2f}M total")
    return counts


# ===========================================================================
# Quick sanity check
# ===========================================================================

if __name__ == "__main__":
    config = SimpleOmniConfig()
    model = SimpleOmni(config)

    print("SimpleOmni parameter count:")
    count_parameters(model)

    # Quick forward pass test
    B, T = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    audio_ids = torch.randint(0, config.audio_vocab_size, (B, config.num_codebooks, T))
    spk_emb = torch.randn(B, config.spk_emb_size)

    out = model(input_ids, audio_ids=audio_ids, spk_emb=spk_emb)
    print(f"\nText logits shape:  {out['text_logits'].shape}")
    print(f"Audio logits:       {len(out['audio_logits'])} codebooks × {out['audio_logits'][0].shape}")
    print(f"Past KVs:           {len(out['past_kvs'])} layers")
    print("\nForward pass OK!")
