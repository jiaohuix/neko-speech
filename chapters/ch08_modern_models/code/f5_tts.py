"""
Ch08.1: F5-TTS — Flow Matching with Transformer (Simplified)

Reference:
    F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
    Chen et al., 2024 (ACL 2025)

Core Idea:
    Traditional diffusion models learn to *denoise* step-by-step.
    Flow Matching learns a *velocity field* v_t(x) that transports
    Gaussian noise (t=0) directly to data (t=1) along a straight line.

    Training:
        x0 ~ N(0, I)           # noise
        x1 = real mel           # data
        t  ~ U(0, 1)
        x_t = (1 - t) * x0 + t * x1       # linear interpolation (optimal transport)
        v_target = x1 - x0                 # constant velocity field
        loss = ||v_theta(x_t, t, cond) - v_target||^2

    Inference (Euler ODE integration):
        x_{t+dt} = x_t + dt * v_theta(x_t, t, cond)

Why Flow Matching?
    - Simpler than diffusion (no noise schedule, no Markov chain)
    - Straight-line trajectories → fewer inference steps (10-20 vs 50-1000)
    - Training is just regression on a straight-line path

This Implementation:
    - Transformer (not DiT) to keep complexity low
    - Text + reference mel as conditioning
    - Predicts velocity field v_t(x)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------
# Sinusoidal Time Embedding (same idea as diffusion models)
# --------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Embed continuous time t ∈ [0, 1] into a vector.
    Same formula as Transformer positional encoding,
    but applied to a scalar time value.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) float
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device) / half
        )
        args = t[:, None] * freqs[None, :] * 1000.0
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb  # (B, dim)


# --------------------------------------------------------
# Transformer Block (Pre-LN, standard)
# --------------------------------------------------------

class TransformerBlock(nn.Module):
    """Standard Pre-LN Transformer block."""

    def __init__(self, dim, heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        # Self-attention with residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        # FFN with residual
        x = x + self.ff(self.norm2(x))
        return x


# --------------------------------------------------------
# F5-TTS Velocity Network (Transformer-based)
# --------------------------------------------------------

class F5VelocityNet(nn.Module):
    """
    Predicts the velocity field v_t(x_t) for Flow Matching.

    Input Construction:
        We concatenate along the time axis:
            [ref_mel | x_t (noisy target)]
        and feed them through a shared Transformer.

        Text conditioning: cross-attention to text embeddings.
        Time conditioning: add sinusoidal time embedding to all tokens.

    Args:
        mel_dim:        mel bins (e.g., 80)
        text_dim:       text encoder output dim
        dim:            model hidden dim
        n_layers:       number of Transformer blocks
        max_mel_len:    maximum mel frames (for positional encoding)
    """

    def __init__(
        self,
        mel_dim=80,
        text_dim=512,
        dim=512,
        heads=8,
        n_layers=6,
        max_mel_len=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.mel_dim = mel_dim
        self.dim = dim

        # Project mel to model dim
        self.mel_proj = nn.Linear(mel_dim, dim)

        # Project text to model dim (for cross-attention keys/values)
        self.text_proj = nn.Linear(text_dim, dim)

        # Time embedding → added to mel tokens
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        # Positional encoding for mel tokens (ref + target concatenated)
        self.pos_emb = nn.Embedding(max_mel_len, dim)

        # Cross-attention to text: each mel token attends to text
        self.cross_attn = nn.ModuleList([
            nn.ModuleDict({
                "norm_q": nn.LayerNorm(dim),
                "norm_kv": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(
                    dim, heads, dropout=dropout, batch_first=True
                ),
            })
            for _ in range(n_layers)
        ])

        # Self-attention Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, ff_mult=4, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output: project back to mel velocity
        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mel_dim),
        )

    def forward(self, x_t, t, text_emb, text_mask, ref_mel=None, ref_mask=None):
        """
        Args:
            x_t:       (B, T_tgt, mel_dim)  noisy mel at time t
            t:         (B,)                 flow time ∈ [0, 1]
            text_emb:  (B, T_text, text_dim)  text encoder outputs
            text_mask: (B, T_text)  True=padding
            ref_mel:   (B, T_ref, mel_dim)  reference mel (optional)
            ref_mask:  (B, T_ref)  True=padding

        Returns:
            velocity:  (B, T_tgt, mel_dim) predicted velocity for target only
        """
        B, T_tgt, _ = x_t.shape

        # --- 1. Prepare mel input: concat [ref_mel, x_t] ---
        if ref_mel is not None:
            T_ref = ref_mel.size(1)
            mel_in = torch.cat([ref_mel, x_t], dim=1)  # (B, T_ref+T_tgt, mel_dim)
            T_total = T_ref + T_tgt
            mask_mel = torch.cat([ref_mask, torch.zeros(B, T_tgt, dtype=torch.bool, device=x_t.device)], dim=1)
        else:
            mel_in = x_t
            T_total = T_tgt
            T_ref = 0
            mask_mel = torch.zeros(B, T_tgt, dtype=torch.bool, device=x_t.device)

        # --- 2. Project mel + add time + position ---
        h = self.mel_proj(mel_in)                # (B, T_total, dim)
        h = h + self.time_embed(t)[:, None, :]   # broadcast time to all tokens
        positions = torch.arange(T_total, device=h.device)
        h = h + self.pos_emb(positions)[None, :, :]

        # --- 3. Text key/value (shared across layers) ---
        kv = self.text_proj(text_emb)            # (B, T_text, dim)

        # --- 4. Transformer blocks with cross-attention to text ---
        # Build attention mask: True where padding (PyTorch convention)
        attn_mask = mask_mel  # (B, T_total)
        key_padding_mask_text = text_mask  # (B, T_text)

        for block, cross in zip(self.blocks, self.cross_attn):
            # Cross-attention: mel queries attend to text
            q = cross["norm_q"](h)
            kv_n = cross["norm_kv"](kv)
            ca, _ = cross["attn"](
                q, kv_n, kv_n,
                key_padding_mask=key_padding_mask_text,
                need_weights=False,
            )
            h = h + ca
            # Self-attention
            h = block(h, attn_mask=None)  # we use key_padding via masking below

        # --- 5. Extract only the target portion (discard ref) ---
        h_tgt = h[:, T_ref:, :]  # (B, T_tgt, dim)
        velocity = self.out_proj(h_tgt)  # (B, T_tgt, mel_dim)
        return velocity


# --------------------------------------------------------
# F5-TTS: Flow Matching Training + Inference
# --------------------------------------------------------

class F5TTS(nn.Module):
    """
    F5-TTS: Flow Matching based TTS.

    Training:
        Given (text, ref_mel, target_mel):
            1. Sample t ~ U(0,1), x0 ~ N(0, I)
            2. x_t = (1-t) * x0 + t * target_mel
            3. v_pred = VelocityNet(x_t, t, text, ref_mel)
            4. loss = MSE(v_pred, target_mel - x0)

    Inference:
        Start from x_0 ~ N(0, I), integrate ODE with Euler steps:
            for t in 0 → 1:
                x_{t+dt} = x_t + dt * v(x_t, t, text, ref_mel)
    """

    def __init__(
        self,
        mel_dim=80,
        text_dim=512,
        dim=512,
        heads=8,
        n_layers=6,
        n_inference_steps=20,
    ):
        super().__init__()
        self.mel_dim = mel_dim
        self.n_inference_steps = n_inference_steps
        self.velocity_net = F5VelocityNet(
            mel_dim=mel_dim,
            text_dim=text_dim,
            dim=dim,
            heads=heads,
            n_layers=n_layers,
        )

    def flow_matching_loss(
        self,
        target_mel, target_mask,
        text_emb, text_mask,
        ref_mel=None, ref_mask=None,
    ):
        """
        Compute Flow Matching loss.

        Args:
            target_mel:  (B, T, mel_dim) ground truth mel
            target_mask: (B, T) True=padding
            text_emb:    (B, T_text, text_dim)
            text_mask:   (B, T_text) True=padding
            ref_mel:     (B, T_ref, mel_dim) reference audio
            ref_mask:    (B, T_ref) True=padding

        Returns:
            loss: scalar MSE between predicted and target velocity
        """
        B, T, D = target_mel.shape
        device = target_mel.device

        # 1. Sample noise, time
        x0 = torch.randn_like(target_mel)         # (B, T, D)
        t = torch.rand(B, device=device)          # (B,)

        # 2. Linear interpolation: x_t = (1-t)*x0 + t*x1
        t_expand = t[:, None, None]               # (B, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * target_mel

        # 3. Target velocity: v = x1 - x0 (constant along straight line)
        v_target = target_mel - x0

        # 4. Predict velocity
        v_pred = self.velocity_net(
            x_t, t, text_emb, text_mask,
            ref_mel=ref_mel, ref_mask=ref_mask,
        )

        # 5. MSE loss (mask out padding frames)
        diff = (v_pred - v_target) ** 2
        valid = (~target_mask).float().unsqueeze(-1)  # (B, T, 1)
        loss = (diff * valid).sum() / valid.sum().clamp(min=1) / D
        return loss

    @torch.no_grad()
    def sample(
        self,
        n_frames,
        text_emb, text_mask,
        ref_mel=None, ref_mask=None,
        n_steps=None,
    ):
        """
        Generate mel spectrogram via Euler ODE integration.

        Args:
            n_frames:   number of mel frames to generate
            text_emb:   (1, T_text, text_dim)
            text_mask:  (1, T_text)
            ref_mel:    (1, T_ref, mel_dim)
            ref_mask:   (1, T_ref)
            n_steps:    number of Euler steps (default: self.n_inference_steps)

        Returns:
            mel: (1, n_frames, mel_dim)
        """
        n_steps = n_steps or self.n_inference_steps
        device = text_emb.device
        dt = 1.0 / n_steps

        # Start from Gaussian noise at t=0
        x = torch.randn(1, n_frames, self.mel_dim, device=device)

        # Euler integration: t goes from 0 to 1
        for i in range(n_steps):
            t = torch.tensor([i / n_steps], device=device)
            v = self.velocity_net(
                x, t, text_emb, text_mask,
                ref_mel=ref_mel, ref_mask=ref_mask,
            )
            x = x + dt * v

        return x


# --------------------------------------------------------
# Simple Text Encoder (shared across models)
# --------------------------------------------------------

class SimpleTextEncoder(nn.Module):
    """
    Minimal text encoder: Embedding + Conv + Transformer.
    Outputs (B, T_text, text_dim) for conditioning.
    """

    def __init__(self, vocab_size=256, embed_dim=256, text_dim=512, n_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, text_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(text_dim, heads=4, dropout=0.1)
            for _ in range(n_layers)
        ])

    def forward(self, text_ids, text_mask=None):
        x = self.proj(self.embedding(text_ids))  # (B, T, text_dim)
        for layer in self.layers:
            x = layer(x)
        return x


# --------------------------------------------------------
# Quick Sanity Check
# --------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cpu"

    # Hyperparams (tiny for demo)
    mel_dim, text_dim, dim = 80, 256, 256
    B, T_text, T_ref, T_tgt = 2, 20, 50, 100

    # Model
    text_enc = SimpleTextEncoder(text_dim=text_dim).to(device)
    model = F5TTS(
        mel_dim=mel_dim, text_dim=text_dim, dim=dim,
        heads=4, n_layers=2, n_inference_steps=10,
    ).to(device)

    # Dummy inputs
    text_ids = torch.randint(0, 100, (B, T_text), device=device)
    text_mask = torch.zeros(B, T_text, dtype=torch.bool, device=device)
    ref_mel = torch.randn(B, T_ref, mel_dim, device=device)
    ref_mask = torch.zeros(B, T_ref, dtype=torch.bool, device=device)
    tgt_mel = torch.randn(B, T_tgt, mel_dim, device=device)
    tgt_mask = torch.zeros(B, T_tgt, dtype=torch.bool, device=device)

    # Forward: training
    text_emb = text_enc(text_ids)
    loss = model.flow_matching_loss(
        tgt_mel, tgt_mask, text_emb, text_mask,
        ref_mel=ref_mel, ref_mask=ref_mask,
    )
    print(f"[Train] Flow Matching Loss: {loss.item():.4f}")

    # Forward: inference
    model.eval()
    gen = model.sample(
        n_frames=T_tgt,
        text_emb=text_emb[:1], text_mask=text_mask[:1],
        ref_mel=ref_mel[:1], ref_mask=ref_mask[:1],
    )
    print(f"[Infer] Generated mel shape: {tuple(gen.shape)}")
    print("[OK] F5-TTS sanity check passed.")
