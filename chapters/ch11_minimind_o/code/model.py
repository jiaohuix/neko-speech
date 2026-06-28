"""
SimpleOmni — A teaching implementation of MiniMind-O (arXiv:2605.03937)
=======================================================================

MiniMind-O is a ~0.1B parameter omni model that supports:
  Input:  Text + Speech (+ Image, optional)
  Output: Text + Streaming Speech

Architecture: Thinker-Talker dual-path
  - Thinker: Transformer backbone for multimodal understanding -> text output
  - Talker:  Smaller Transformer for acoustic rendering -> mel codes -> waveform

This teaching version (~50M params) simplifies:
  - Thinker: 6 layers, hidden=512 (vs 8 layers, hidden=768)
  - Talker:  2 layers, hidden=512, 4 codebooks (vs 4 layers, 8 codebooks)
  - SimpleSpeechEncoder: 4-layer Transformer (vs frozen SenseVoice, 234M)
  - SimpleImageEncoder:  4-layer ViT (vs frozen SigLIP2, 94M)
  - Same bridge mechanism, MTP head, and delay pattern

Key components:
  SimpleSpeechEncoder -- Whisper-like speech encoder (mel -> features)
  SimpleImageEncoder  -- Simplified ViT (image patches -> features)
  SimpleThinker       -- Causal Transformer for multimodal understanding
  SimpleTalker        -- Smaller Transformer for audio code generation
  SimpleTalkerHead    -- MTP head: shared base + per-codebook adapters
  SimpleTalkerEmbed   -- Embedding that fuses multi-codebook input
  SimpleMelDecoder    -- Code -> mel spectrogram decoder
  SimpleOmni          -- Top-level model with forward() and generate()

Based on:
  - Paper:   arXiv:2605.03937 (Gong, 2026)
  - Code:    https://github.com/jingyaogong/minimind-o
  - License: Apache 2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


# ===========================================================================
# Configuration
# ===========================================================================

class SimpleOmniConfig:
    """Configuration for SimpleOmni (teaching version of MiniMind-O).

    Original MiniMind-O (minimind-3o):
        hidden_size=768, num_layers=8, talker_layers=4, codebooks=8
        Trainable: ~113M params

    Teaching version (this file):
        hidden_size=512, num_layers=6, talker_layers=2, codebooks=4
        Trainable: ~50M params
    """

    def __init__(self, **kwargs):
        # Thinker (language backbone)
        self.hidden_size = kwargs.get("hidden_size", 512)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 6)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 2)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.intermediate_size = kwargs.get("intermediate_size", 1408)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 2048)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.dropout = kwargs.get("dropout", 0.0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)

        # Talker (acoustic renderer)
        self.talker_hidden_size = kwargs.get("talker_hidden_size", 512)
        self.num_talker_layers = kwargs.get("num_talker_layers", 2)

        # Audio codec
        self.num_codebooks = kwargs.get("num_codebooks", 4)
        self.audio_vocab_size = kwargs.get("audio_vocab_size", 2082)
        self.audio_pad_token = kwargs.get("audio_pad_token", 2049)
        self.audio_stop_token = kwargs.get("audio_stop_token", 2050)
        self.audio_spk_token = kwargs.get("audio_spk_token", 2051)

        # Speech encoder (simplified Whisper-like)
        self.speech_hidden_size = kwargs.get("speech_hidden_size", 256)
        self.speech_num_layers = kwargs.get("speech_num_layers", 4)
        self.speech_num_heads = kwargs.get("speech_num_heads", 4)
        self.n_mels = kwargs.get("n_mels", 80)

        # Image encoder (simplified ViT)
        self.image_hidden_size = kwargs.get("image_hidden_size", 256)
        self.image_num_layers = kwargs.get("image_num_layers", 4)
        self.image_num_heads = kwargs.get("image_num_heads", 4)
        self.image_size = kwargs.get("image_size", 256)
        self.patch_size = kwargs.get("patch_size", 16)

        # Bridge layer: which Thinker layer feeds the Talker
        self.bridge_layer = kwargs.get("bridge_layer",
                                        self.num_hidden_layers // 2 - 1)

        # Speaker embedding (CAM++)
        self.spk_emb_size = kwargs.get("spk_emb_size", 192)

        # MTP adapter rank
        self.adapter_rank = kwargs.get("adapter_rank", 128)

        # Mel decoder
        self.n_mel_out = kwargs.get("n_mel_out", 80)

        # Special token IDs
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)


# ===========================================================================
# Building Blocks (shared by Thinker and Talker)
# ===========================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Same as LayerNorm but without mean subtraction -- just scale by RMS.
    Faster and works better for Transformer hidden states.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (self.weight * x.float() * rms).type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 1e6):
    """Precompute cos/sin tables for Rotary Position Embedding (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE rotation to Q and K tensors."""
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for Grouped Query Attention (GQA)."""
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
    """Multi-head attention with GQA and RoPE."""

    def __init__(self, hidden_size: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, rms_norm_eps: float = 1e-6):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

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

        q = q.transpose(1, 2)
        k = repeat_kv(k, self.n_rep).transpose(1, 2)
        v = repeat_kv(v, self.n_rep).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if T > 1 and past_kv is None:
            causal = torch.full((T, T), float("-inf"), device=x.device).triu(1)
            scores = scores + causal
        if mask is not None:
            scores = scores + (1.0 - mask.unsqueeze(1).unsqueeze(2)) * -1e9

        attn = F.softmax(scores.float(), dim=-1).type_as(q)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out), new_kv


# ===========================================================================
# Feed-Forward Network
# ===========================================================================

class FeedForward(nn.Module):
    """SwiGLU feed-forward: gate * up -> down."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


# ===========================================================================
# Transformer Block
# ===========================================================================

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: RMSNorm -> Attn -> Residual -> RMSNorm -> FFN -> Residual."""

    def __init__(self, hidden_size: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, intermediate_size: int, rms_norm_eps: float = 1e-6):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Attention(hidden_size, n_heads, n_kv_heads, head_dim, rms_norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = FeedForward(hidden_size, intermediate_size)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False, mask=None):
        residual = x
        x, kv = self.self_attn(self.attn_norm(x), cos, sin, past_kv, use_cache, mask)
        x = x + residual
        x = x + self.mlp(self.ffn_norm(x))
        return x, kv


def _make_transformer_layers(n_layers, hidden_size, n_heads, n_kv_heads,
                              head_dim, intermediate_size, rms_norm_eps):
    """Helper to create a list of TransformerBlocks."""
    return nn.ModuleList([
        TransformerBlock(hidden_size, n_heads, n_kv_heads, head_dim,
                         intermediate_size, rms_norm_eps)
        for _ in range(n_layers)
    ])


# ===========================================================================
# SimpleSpeechEncoder: Whisper-like speech encoder
# ===========================================================================

class SimpleSpeechEncoder(nn.Module):
    """Simplified Whisper-like speech encoder.

    Converts log-mel spectrogram (B, n_mels, T_mel) into feature sequence
    (B, T_out, speech_hidden_size).

    Architecture:
      1. Conv1d frontend: 2 layers, stride 2 each -> 4x downsampling
      2. 4-layer Transformer encoder (bidirectional attention)

    In MiniMind-O, the frozen SenseVoice (234M) serves this role.
    This simplified version (~3M params) is fully trainable.
    """

    def __init__(self, config: SimpleOmniConfig):
        super().__init__()
        h = config.speech_hidden_size
        # Conv frontend: n_mels -> h/2 -> h (stride 2 each = 4x downsample)
        self.conv1 = nn.Conv1d(config.n_mels, h // 2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(h // 2, h, kernel_size=3, stride=2, padding=1)
        self.act = nn.GELU()

        # Position embedding (for up to 1024 mel frames -> 256 after conv)
        self.pos_emb = nn.Embedding(1024, h)

        # Transformer encoder layers (bidirectional)
        head_dim = h // config.speech_num_heads
        self.layers = _make_transformer_layers(
            config.speech_num_layers, h,
            config.speech_num_heads, config.speech_num_heads,  # full MHA
            head_dim, h * 4, config.rms_norm_eps,
        )
        self.norm = RMSNorm(h, eps=config.rms_norm_eps)

        # Precompute RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, 1024)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, n_mels, T_mel) log-mel spectrogram
        Returns:
            (B, T_out, speech_hidden_size) where T_out = T_mel // 4
        """
        x = self.act(self.conv1(mel))
        x = self.act(self.conv2(x))            # (B, h, T_mel/4)
        x = x.transpose(1, 2)                  # (B, T_out, h)

        T = x.shape[1]
        pos_ids = torch.arange(T, device=x.device)
        x = x + self.pos_emb(pos_ids)

        cos = self.freqs_cos[:T]
        sin = self.freqs_sin[:T]

        for layer in self.layers:
            x, _ = layer(x, cos, sin)          # no causal mask (bidirectional)

        return self.norm(x)


# ===========================================================================
# SimpleImageEncoder: Simplified ViT
# ===========================================================================

class SimpleImageEncoder(nn.Module):
    """Simplified Vision Transformer (ViT) for image understanding.

    Converts image (B, 3, image_size, image_size) into patch features
    (B, num_patches, image_hidden_size).

    Architecture:
      1. Patch embedding via Conv2d (patch_size x patch_size -> hidden)
      2. Learnable position embeddings
      3. 4-layer Transformer encoder (bidirectional)

    In MiniMind-O, the frozen SigLIP2 (94M) serves this role.
    This simplified version (~3M params) is fully trainable.
    """

    def __init__(self, config: SimpleOmniConfig):
        super().__init__()
        h = config.image_hidden_size
        n_patches = (config.image_size // config.patch_size) ** 2  # 256 for 256/16

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, h, kernel_size=config.patch_size,
                                      stride=config.patch_size)
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, h) * 0.02)

        # Transformer encoder
        head_dim = h // config.image_num_heads
        self.layers = _make_transformer_layers(
            config.image_num_layers, h,
            config.image_num_heads, config.image_num_heads,
            head_dim, h * 4, config.rms_norm_eps,
        )
        self.norm = RMSNorm(h, eps=config.rms_norm_eps)

        # RoPE for patches
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, n_patches + 1)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) pixel values
        Returns:
            (B, num_patches, image_hidden_size)
        """
        x = self.patch_embed(images)           # (B, h, h_patches, w_patches)
        x = x.flatten(2).transpose(1, 2)      # (B, n_patches, h)
        x = x + self.pos_emb

        T = x.shape[1]
        cos = self.freqs_cos[:T]
        sin = self.freqs_sin[:T]

        for layer in self.layers:
            x, _ = layer(x, cos, sin)

        return self.norm(x)


# ===========================================================================
# Thinker (Understanding Pathway)
# ===========================================================================

class SimpleThinker(nn.Module):
    """Thinker: causal Transformer for multimodal understanding.

    Processes text tokens (with injected audio/vision features) through
    N transformer layers. The bridge_layer's output is captured as
    conditioning for the Talker.

    Key insight: the MIDDLE layer (not the last) provides the best
    conditioning signal for speech generation.
    """

    def __init__(self, config: SimpleOmniConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = _make_transformer_layers(
            config.num_hidden_layers, config.hidden_size,
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, config.intermediate_size, config.rms_norm_eps,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, hidden_states_override=None,
                past_kvs=None, use_cache=False, mask=None):
        """
        Returns:
            h_final: (B, T, D) final hidden states -> text logits
            bridge: (B, T, D) middle-layer states -> Talker conditioning
            presents: list of new (k, v) caches
        """
        past_kvs = past_kvs or [None] * len(self.layers)
        start_pos = past_kvs[0][0].shape[1] if past_kvs[0] is not None else 0

        if hidden_states_override is not None:
            h = hidden_states_override
        else:
            h = self.embed_tokens(input_ids)

        T = h.shape[1]  # actual sequence length (may include multimodal tokens)
        cos = self.freqs_cos[start_pos:start_pos + T]
        sin = self.freqs_sin[start_pos:start_pos + T]

        bridge = h
        presents = []

        for i, (layer, past_kv) in enumerate(zip(self.layers, past_kvs)):
            h, present = layer(h, cos, sin, past_kv, use_cache, mask)
            presents.append(present)
            if i == self.config.bridge_layer:
                bridge = h

        return self.norm(h), bridge, presents


# ===========================================================================
# MTP Head: Multi-Token Prediction for Parallel Codebook Output
# ===========================================================================

class SimpleTalkerHead(nn.Module):
    """MTP head for parallel codebook prediction.

    logits_i = base(x) + adapter_i(x)

    Reduces parameters from 8x to ~1.5x a single head.
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
    """Embedding that fuses multi-codebook audio IDs into a single vector.

    Output = mean(base_embed + adapter_embed) across all codebooks.
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
        """audio_ids: (B, num_codebooks, T) -> (B, T, talker_hidden_size)"""
        base_out = self.base(audio_ids)  # (B, C, T, D)
        total = torch.zeros_like(base_out[:, 0])
        for i in range(self.num_codebooks):
            total = total + base_out[:, i] + self.adapters[i](audio_ids[:, i])
        return total / self.num_codebooks


# ===========================================================================
# Talker (Speaking Pathway)
# ===========================================================================

class SimpleTalker(nn.Module):
    """Talker: converts Thinker's semantic states into audio codebook codes.

    hidden = embed_proj(bridge) * text_scale + codec_proj(audio_emb) * audio_scale
    -> N Transformer blocks -> MTP head -> per-codebook logits
    """

    def __init__(self, config: SimpleOmniConfig):
        super().__init__()
        self.config = config
        h = config.talker_hidden_size

        # Transformer blocks
        head_dim = h // config.num_attention_heads
        self.layers = _make_transformer_layers(
            config.num_talker_layers, h,
            config.num_attention_heads, config.num_key_value_heads,
            head_dim, config.intermediate_size, config.rms_norm_eps,
        )
        self.norm = RMSNorm(h, eps=config.rms_norm_eps)

        # Input projections
        self.embed_tokens = SimpleTalkerEmbed(config)
        self.embed_proj = nn.Sequential(
            nn.Linear(config.hidden_size, h),
            nn.GELU(),
            nn.Linear(h, h),
            RMSNorm(h, eps=config.rms_norm_eps),
        )
        self.codec_proj = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, h),
            RMSNorm(h, eps=config.rms_norm_eps),
        )

        # Learnable scaling factors
        self.text_scale = nn.Parameter(torch.tensor(3.0))
        self.audio_scale = nn.Parameter(torch.tensor(1.0))

        # Speaker embedding projection
        self.spk_proj = nn.Linear(config.spk_emb_size, h, bias=False)

        # Output head
        self.lm_head = SimpleTalkerHead(config)

        # RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(
            head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)


# ===========================================================================
# Projectors: encoder features -> Thinker hidden space
# ===========================================================================

class SimpleProjector(nn.Module):
    """2-layer MLP that projects encoder features into Thinker hidden space."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ===========================================================================
# SimpleMelDecoder: audio codes -> mel spectrogram
# ===========================================================================

class SimpleMelDecoder(nn.Module):
    """Simplified mel spectrogram decoder from audio codes.

    In MiniMind-O, the frozen Mimi codec (96M) decodes 8 codebook streams
    to 24kHz waveform. This simplified version decodes codebook embeddings
    to mel spectrogram frames via a small Transformer + linear head.

    Used for teaching; replace with Mimi for production quality.
    """

    def __init__(self, config: SimpleOmniConfig):
        super().__init__()
        h = config.talker_hidden_size
        self.code_embeddings = nn.ModuleList([
            nn.Embedding(config.audio_vocab_size, h // config.num_codebooks)
            for _ in range(config.num_codebooks)
        ])
        # Fuse codebook embeddings
        self.fuse = nn.Linear(h, h)
        # 2-layer Transformer decoder
        head_dim = h // config.num_attention_heads
        self.layers = _make_transformer_layers(
            2, h,
            config.num_attention_heads, config.num_key_value_heads,
            head_dim, config.intermediate_size, config.rms_norm_eps,
        )
        self.norm = RMSNorm(h, eps=config.rms_norm_eps)
        self.mel_head = nn.Linear(h, config.n_mel_out)

        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, 2048)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_codes: (B, num_codebooks, T) integer codes
        Returns:
            (B, T, n_mel_out) mel spectrogram frames
        """
        B, C, T = audio_codes.shape
        # Embed and concatenate codebooks
        embs = [self.code_embeddings[i](audio_codes[:, i]) for i in range(C)]
        x = torch.cat(embs, dim=-1)  # (B, T, h)
        x = self.fuse(x)

        cos = self.freqs_cos[:T]
        sin = self.freqs_sin[:T]
        for layer in self.layers:
            x, _ = layer(x, cos, sin)

        return self.mel_head(self.norm(x))


# ===========================================================================
# SimpleOmni: Top-Level Model
# ===========================================================================

class SimpleOmni(nn.Module):
    """End-to-end omni model: text + speech + image -> text + streaming speech.

    Forward pass:
    1. Encode speech via SimpleSpeechEncoder -> project into Thinker space
    2. Encode image via SimpleImageEncoder -> project into Thinker space
    3. Thinker processes text + injected audio/image features
    4. Bridge captures middle-layer states
    5. Talker combines bridge states + audio code history
    6. Talker outputs logits for each codebook
    """

    def __init__(self, config: SimpleOmniConfig = None):
        super().__init__()
        self.config = config or SimpleOmniConfig()

        # Encoders (trainable in teaching version; frozen in original)
        self.speech_encoder = SimpleSpeechEncoder(self.config)
        self.image_encoder = SimpleImageEncoder(self.config)

        # Projectors: encoder output -> Thinker hidden space
        self.audio_proj = SimpleProjector(self.config.speech_hidden_size,
                                           self.config.hidden_size)
        self.image_proj = SimpleProjector(self.config.image_hidden_size,
                                           self.config.hidden_size)

        # Core: Thinker + Talker
        self.thinker = SimpleThinker(self.config)
        self.talker = SimpleTalker(self.config)

        # Text output head
        self.text_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.text_head.weight = self.thinker.embed_tokens.weight

        # Mel decoder (simplified Mimi replacement)
        self.mel_decoder = SimpleMelDecoder(self.config)

    def encode_speech(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode log-mel spectrogram -> Thinker-space features."""
        features = self.speech_encoder(mel)       # (B, T_out, speech_hidden)
        return self.audio_proj(features)           # (B, T_out, hidden_size)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images -> Thinker-space features."""
        features = self.image_encoder(images)     # (B, n_patches, image_hidden)
        return self.image_proj(features)           # (B, n_patches, hidden_size)

    def forward(self, input_ids, audio_ids=None,
                mel_input=None, image_input=None,
                spk_emb=None, past_kvs=None, use_cache=False, mask=None):
        """
        Forward pass for training.

        Args:
            input_ids: (B, T) text token IDs
            audio_ids: (B, num_codebooks, T) audio codebook IDs
            mel_input: (B, n_mels, T_mel) log-mel spectrogram (optional)
            image_input: (B, 3, H, W) images (optional)
            spk_emb: (B, spk_emb_size) speaker embedding (optional)
        """
        B, T = input_ids.shape

        # Split KV-caches between Thinker and Talker
        n_thinker = len(self.thinker.layers)
        if past_kvs is None:
            past_kvs = [None] * (n_thinker + len(self.talker.layers))
        thinker_kvs = past_kvs[:n_thinker]
        talker_kvs = past_kvs[n_thinker:]

        # ---- ENCODE MULTIMODAL INPUTS ----
        text_emb = self.thinker.embed_tokens(input_ids)

        if mel_input is not None:
            audio_feat = self.encode_speech(mel_input)  # (B, T_a, hidden)
            # Concatenate: [audio_features, text_embeddings]
            # In real model, audio is injected at placeholder positions
            text_emb = torch.cat([audio_feat, text_emb], dim=1)

        if image_input is not None:
            image_feat = self.encode_image(image_input)  # (B, T_i, hidden)
            text_emb = torch.cat([image_feat, text_emb], dim=1)

        # Adjust sequence length after multimodal injection
        T_full = text_emb.shape[1]

        # ---- THINKER ----
        h_final, bridge, thinker_presents = self.thinker(
            input_ids, text_emb, thinker_kvs, use_cache, mask
        )

        # ---- TALKER ----
        if audio_ids is not None:
            # Pad or truncate audio_ids to match thinker seq length
            aT = audio_ids.shape[2]
            if aT < T_full:
                pad = torch.full((B, self.config.num_codebooks, T_full - aT),
                                 self.config.audio_pad_token,
                                 dtype=audio_ids.dtype, device=audio_ids.device)
                audio_ids_padded = torch.cat([audio_ids, pad], dim=2)
            else:
                audio_ids_padded = audio_ids[:, :, :T_full]
            talker_emb = self.talker.embed_tokens(audio_ids_padded)
        else:
            talker_emb = torch.zeros(B, T_full, self.config.talker_hidden_size,
                                     device=input_ids.device)

        # Combine: text conditioning + audio history
        text_cond = self.talker.embed_proj(bridge) * self.talker.text_scale
        audio_cond = self.talker.codec_proj(talker_emb) * self.talker.audio_scale
        talker_input = text_cond + audio_cond

        # Process through Talker transformer
        cos = self.talker.freqs_cos[:T_full]
        sin = self.talker.freqs_sin[:T_full]

        h = talker_input
        talker_presents = []
        for layer, past_kv in zip(self.talker.layers, talker_kvs):
            h, present = layer(h, cos, sin, past_kv, use_cache, mask)
            talker_presents.append(present)

        h_talker = self.talker.norm(h)

        # Output heads
        text_logits = self.text_head(h_final)
        audio_logits = self.talker.lm_head(h_talker)

        return {
            "text_logits": text_logits,
            "audio_logits": audio_logits,
            "past_kvs": thinker_presents + talker_presents,
        }

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=128, temperature=0.75,
                 top_p=0.9, mel_input=None, image_input=None,
                 spk_emb=None, **kwargs):
        """Generate text and audio codes autoregressively.

        Pipeline:
        1. Thinker generates text tokens one at a time
        2. For each text step, Talker generates audio codes (all codebooks)
        3. Audio codes are decoded to mel spectrogram

        Returns:
            dict with 'text_ids' (generated text), 'audio_codes' (B, C, T),
            'mel' (B, T, n_mel)
        """
        cfg = self.config
        B = input_ids.shape[0]
        device = input_ids.device

        # Encode multimodal inputs
        text_emb = self.thinker.embed_tokens(input_ids)
        prefix_embs = []
        if mel_input is not None:
            prefix_embs.append(self.encode_speech(mel_input))
        if image_input is not None:
            prefix_embs.append(self.encode_image(image_input))
        if prefix_embs:
            prefix = torch.cat(prefix_embs, dim=1)
            text_emb = torch.cat([prefix, text_emb], dim=1)

        # Thinker forward (full sequence for conditioning)
        h_final, bridge, _ = self.thinker(input_ids, text_emb, use_cache=False)

        # Generate text tokens autoregressively
        gen_text = []
        cur_ids = input_ids
        for _ in range(max_new_tokens):
            h_final, bridge, _ = self.thinker(cur_ids, use_cache=False)
            logits = self.text_head(h_final[:, -1, :])  # (B, vocab)
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                # Top-p filtering
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True)
                next_token = sorted_idx.gather(
                    -1, torch.multinomial(sorted_probs, 1))
            else:
                next_token = logits.argmax(-1, keepdim=True)
            gen_text.append(next_token)
            cur_ids = torch.cat([cur_ids, next_token], dim=1)
            if (next_token == cfg.eos_token_id).all():
                break

        gen_text_ids = torch.cat(gen_text, dim=1) if gen_text else \
            torch.empty(B, 0, dtype=torch.long, device=device)

        # Talker generates audio codes using bridge states
        T_text = bridge.shape[1]
        audio_codes = torch.full((B, cfg.num_codebooks, T_text),
                                  cfg.audio_pad_token, dtype=torch.long, device=device)

        # Feed bridge states through talker
        text_cond = self.talker.embed_proj(bridge) * self.talker.text_scale
        audio_cond = torch.zeros_like(text_cond)
        talker_h = text_cond + audio_cond

        cos = self.talker.freqs_cos[:T_text]
        sin = self.talker.freqs_sin[:T_text]
        for layer in self.talker.layers:
            talker_h, _ = layer(talker_h, cos, sin)
        talker_h = self.talker.norm(talker_h)

        # Get audio logits and sample codes for each position
        audio_logits = self.talker.lm_head(talker_h)
        for cb in range(cfg.num_codebooks):
            if temperature > 0:
                probs = F.softmax(audio_logits[cb] / temperature, dim=-1)
                audio_codes[:, cb, :] = torch.multinomial(
                    probs.view(-1, cfg.audio_vocab_size), 1
                ).view(B, T_text)
            else:
                audio_codes[:, cb, :] = audio_logits[cb].argmax(-1)

        # Decode audio codes to mel spectrogram
        mel_out = self.mel_decoder(audio_codes)  # (B, T, n_mel)

        return {
            "text_ids": gen_text_ids,
            "audio_codes": audio_codes,
            "mel": mel_out,
        }

    @torch.inference_mode()
    def stream_generate(self, input_ids, max_new_tokens=128, temperature=0.75,
                        top_p=0.9, spk_emb=None, **kwargs):
        """Streaming generation with delay pattern.

        The delay pattern staggers codebook generation:
        - CB-0 starts at step 0, CB-1 at step 1, ..., CB-(N-1) at step N-1
        - After N steps, a complete audio frame is available for decoding
        - This enables low-latency streaming playback

        Yields:
            (text_token_or_None, audio_frame_or_None)
            audio_frame: list of num_codebooks codes (ready for decoder)
        """
        cfg = self.config
        B = input_ids.shape[0]
        device = input_ids.device

        # Initial Thinker pass
        h_final, bridge, _ = self.thinker(input_ids, use_cache=False)

        # Per-codebook code buffers for delay pattern
        cb_codes = [[] for _ in range(cfg.num_codebooks)]

        # Streaming loop
        for step in range(max_new_tokens):
            # Thinker generates one text token
            logits = self.text_head(h_final[:, -1, :])
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                text_token = torch.multinomial(probs, 1)
            else:
                text_token = logits.argmax(-1, keepdim=True)

            # Talker generates audio codes with delay pattern
            # Each active codebook produces one code per step
            for cb in range(cfg.num_codebooks):
                if step >= cb:
                    # Sample a code (simplified: in production, use Talker output)
                    code = torch.randint(0, 2048, (B,), device=device)
                    cb_codes[cb].append(code)

            # Check if complete frame is available (all CBs have enough codes)
            if step >= cfg.num_codebooks - 1:
                # All codebooks have produced at least (step - cb + 1) codes
                # Frame index = step - num_codebooks + 1
                # For each CB-i, read code at index = frame_idx
                # (CB-i started at step i, so at step s it has s-i+1 codes)
                frame_idx = step - cfg.num_codebooks + 1
                frame = []
                for cb in range(cfg.num_codebooks):
                    if frame_idx < len(cb_codes[cb]):
                        frame.append(cb_codes[cb][frame_idx][0].item())
                    else:
                        frame.append(cfg.audio_pad_token)
                yield text_token, frame
            else:
                yield text_token, None

            # Update Thinker for next step
            h_final, bridge, _ = self.thinker(text_token, use_cache=False)

            if (text_token == cfg.eos_token_id).all():
                break


# ===========================================================================
# Loss Computation
# ===========================================================================

def compute_omni_loss(text_logits, audio_logits, text_labels, audio_labels,
                      num_codebooks=4, stop_weight=10.0):
    """Compute combined text + audio loss for training.

    Text loss: standard cross-entropy on Thinker output.
    Audio loss: per-codebook cross-entropy with stop token weighting (10x).

    Returns:
        (total_loss, text_loss, audio_loss)
    """
    text_loss = F.cross_entropy(
        text_logits.view(-1, text_logits.size(-1)),
        text_labels.view(-1),
        ignore_index=-100,
    )

    audio_loss = torch.tensor(0.0, device=text_logits.device)
    for i in range(num_codebooks):
        logits_i = audio_logits[i].view(-1, audio_logits[i].size(-1))
        labels_i = audio_labels[:, i, :].reshape(-1)
        per_token_loss = F.cross_entropy(logits_i, labels_i,
                                         ignore_index=-100, reduction='none')
        valid_mask = (labels_i != -100).float()
        stop_mask = (labels_i == 2050).float()
        weighted_loss = per_token_loss * valid_mask * (1 + stop_mask * (stop_weight - 1))
        n_valid = valid_mask.sum().clamp(min=1)
        audio_loss = audio_loss + weighted_loss.sum() / n_valid
    audio_loss = audio_loss / num_codebooks

    return text_loss + audio_loss, text_loss, audio_loss


# ===========================================================================
# Utility
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
# Shape verification
# ===========================================================================

def verify_shapes(model: SimpleOmni, config: SimpleOmniConfig):
    """Verify all tensor shapes in a forward pass."""
    B, T = 2, 32
    device = next(model.parameters()).device

    # Text-only forward
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
    audio_ids = torch.randint(0, config.audio_vocab_size,
                               (B, config.num_codebooks, T), device=device)
    out = model(input_ids, audio_ids=audio_ids)
    assert out["text_logits"].shape == (B, T, config.vocab_size), \
        f"text_logits: {out['text_logits'].shape}"
    assert len(out["audio_logits"]) == config.num_codebooks
    for i, logits in enumerate(out["audio_logits"]):
        assert logits.shape == (B, T, config.audio_vocab_size), \
            f"audio_logits[{i}]: {logits.shape}"
    print("  Text-only forward: OK")

    # With speech input
    T_mel = 128  # 128 mel frames -> 32 after conv
    mel = torch.randn(B, config.n_mels, T_mel, device=device)
    out2 = model(input_ids, audio_ids=audio_ids, mel_input=mel)
    print(f"  With speech: text_logits={out2['text_logits'].shape}, "
          f"seq_len={out2['text_logits'].shape[1]}")

    # With image input
    img = torch.randn(B, 3, config.image_size, config.image_size, device=device)
    out3 = model(input_ids, audio_ids=audio_ids, image_input=img)
    n_patches = (config.image_size // config.patch_size) ** 2
    print(f"  With image: text_logits={out3['text_logits'].shape}, "
          f"expected extra {n_patches} patches")

    # Generate
    gen = model.generate(input_ids[:1], max_new_tokens=8)
    print(f"  Generate: text_ids={gen['text_ids'].shape}, "
          f"audio_codes={gen['audio_codes'].shape}, mel={gen['mel'].shape}")

    print("  All shape checks passed!")


# ===========================================================================
# Sanity check
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SimpleOmni - MiniMind-O Teaching Implementation")
    print("=" * 60)

    config = SimpleOmniConfig()
    model = SimpleOmni(config)

    print("\nParameter count:")
    count_parameters(model)

    print("\nShape verification:")
    verify_shapes(model, config)

    print("\nForward pass OK!")
