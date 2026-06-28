"""
SimpleVoxCPM — A tokenizer-free TTS model (teaching implementation)
==================================================================

Based on the VoxCPM architecture (arXiv:2509.24650).

Key idea: instead of predicting discrete audio tokens (like VALL-E),
VoxCPM generates speech in a continuous latent space using conditional
flow matching. The AudioVAE compresses audio to continuous latents;
the TSLM plans semantics; the RALM refines acoustics; and a small
diffusion model (LocDiT + CFM) generates each latent patch.

Components:
  SimpleAudioVAE  : Causal CNN encoder/decoder (waveform <-> continuous latent)
  SimpleLocEnc    : Local encoder — compress one latent patch to a hidden vector
  SimpleFSQ       : Finite scalar quantization — semantic bottleneck
  SimpleTSLM      : Text-Semantic Language Model — autoregressive semantic planning
  SimpleRALM      : Residual Acoustic Language Model — autoregressive acoustic refinement
  SimpleDiT       : Local Diffusion Transformer — denoise one latent patch
  SimpleCFM       : Conditional Flow Matching — training objective + Euler sampler
  SimpleVoxCPM    : End-to-end pipeline wrapper

~46M parameters total (scaled down from the original 0.5B).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CausalConv1dBlock(nn.Module):
    """Causal Conv1d + GroupNorm + SiLU + residual.

    Causal padding: left-pad by (kernel_size - 1) so the output at time t
    depends only on inputs at times <= t.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.pad_size = (kernel_size - 1)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=0)
        self.norm = nn.GroupNorm(1, out_ch)
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1, stride=stride) if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = F.pad(x, (self.pad_size, 0))          # causal (left) padding
        out = self.conv(out)
        out = self.norm(out)
        out = F.silu(out)
        return out + self.residual(x)


# ---------------------------------------------------------------------------
# AudioVAE
# ---------------------------------------------------------------------------

class SimpleAudioVAE(nn.Module):
    """Causal CNN AudioVAE.

    Encoder: 4 causal conv layers, strides [2, 5, 8, 8] → total stride 640.
             16 kHz waveform → 25 Hz continuous latent (dim=32).
    Decoder: mirror architecture with transposed convolutions.
    """

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: list = None,
        latent_dim: int = 32,
        decoder_dim: int = 256,
        decoder_rates: list = None,
    ):
        super().__init__()
        self.encoder_rates = encoder_rates or [2, 5, 8, 8]   # product = 640
        self.decoder_rates = decoder_rates or [8, 8, 5, 2]   # product = 640
        self.chunk_size = math.prod(self.encoder_rates)        # 640 samples per latent frame
        self.latent_dim = latent_dim

        # --- Encoder ---
        enc_layers = []
        in_ch = 1
        out_ch = encoder_dim
        for i, stride in enumerate(self.encoder_rates):
            enc_layers.append(CausalConv1dBlock(in_ch, out_ch, kernel_size=7, stride=stride))
            in_ch = out_ch
            out_ch = min(out_ch * 2, encoder_dim * 8)
        self.encoder = nn.Sequential(*enc_layers)
        self.enc_to_latent = nn.Conv1d(in_ch, latent_dim, kernel_size=1)

        # --- Decoder ---
        self.latent_to_dec = nn.Conv1d(latent_dim, decoder_dim, kernel_size=1)
        dec_layers = []
        in_ch = decoder_dim
        out_ch = decoder_dim
        for i, stride in enumerate(self.decoder_rates):
            out_ch = max(decoder_dim // (2 ** (i + 1)), 16)
            dec_layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_ch, out_ch, kernel_size=stride * 2, stride=stride,
                                       padding=stride // 2),
                    nn.GroupNorm(1, out_ch),
                    nn.SiLU(),
                )
            )
            in_ch = out_ch
        self.decoder = nn.Sequential(*dec_layers)
        self.dec_to_out = nn.Conv1d(in_ch, 1, kernel_size=7, padding=3)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, T_samples) -> (B, latent_dim, T_latent)"""
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)               # (B, 1, T)
        h = self.encoder(waveform)
        return self.enc_to_latent(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, latent_dim, T_latent) -> (B, T_samples)"""
        h = self.latent_to_dec(z)
        h = self.decoder(h)
        out = self.dec_to_out(h)
        return out.squeeze(1)                               # (B, T_samples)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        z = self.encode(waveform)
        return self.decode(z)


# ---------------------------------------------------------------------------
# LocEnc  (Local Encoder)
# ---------------------------------------------------------------------------

class SimpleLocEnc(nn.Module):
    """Compress a single latent patch into one hidden vector.

    Input:  (B, patch_size, latent_dim) — one patch of continuous latents
    Output: (B, hidden_dim)              — pooled hidden representation

    Uses a CLS token + bidirectional self-attention + CLS pooling
    (following the original VoxCPM LocEnc design).
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 512,
                 num_heads: int = 4, num_layers: int = 2, ffn_dim: int = 1024):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        B, P, D = patch.shape
        x = self.in_proj(patch)                              # (B, P, hidden)
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, hidden)
        x = torch.cat([cls, x], dim=1)                       # (B, P+1, hidden)
        x = self.encoder(x)                                  # bidirectional
        return x[:, 0, :]                                    # CLS pool → (B, hidden)


# ---------------------------------------------------------------------------
# FSQ  (Finite Scalar Quantization)
# ---------------------------------------------------------------------------

class SimpleFSQ(nn.Module):
    """Finite Scalar Quantization — differentiable semantic bottleneck.

    Maps hidden → low-dim → tanh → round * scale → straight-through → hidden.

    During training, the straight-through estimator passes gradients through
    the non-differentiable rounding operation.  At inference the rounding is
    applied directly.

    The scale controls the number of quantization levels:
      scale=9 → 19 distinct values in [-1, 1] (step = 1/9).
    """

    def __init__(self, in_dim: int = 512, latent_dim: int = 128, scale: int = 9):
        super().__init__()
        self.scale = scale
        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)                                  # project down
        h = torch.tanh(h)                                    # bound to [-1, 1]
        if self.training:
            q = torch.round(h * self.scale) / self.scale
            h = h + (q - h).detach()                         # straight-through estimator
        else:
            h = torch.round(h * self.scale) / self.scale
        return self.out_proj(h)                              # project back up


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    """Pre-LN transformer block used by both TSLM and RALM."""

    def __init__(self, hidden: int, num_heads: int, ffn: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class SimpleTransformer(nn.Module):
    """GPT-style causal transformer (shared backbone for TSLM and RALM)."""

    def __init__(self, input_dim: int, hidden: int = 512, num_heads: int = 8,
                 ffn: int = 2048, num_layers: int = 8, max_len: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.Linear(input_dim, hidden) if input_dim != hidden else nn.Identity()
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.blocks = nn.ModuleList([
            _TransformerBlock(hidden, num_heads, ffn, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Full-sequence forward (teacher forcing)."""
        B, T, _ = x.shape
        x = self.input_proj(x)
        if positions is None:
            positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(positions)
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.norm(x)

    def forward_step(self, x_step: torch.Tensor, position: int) -> torch.Tensor:
        """Single-token forward for autoregressive generation (with KV-cache would go here).

        x_step:   (B, input_dim)  one token's embedding
        position: scalar int      position index

        Returns:  (B, hidden)     output at this position
        """
        B = x_step.shape[0]
        pos = torch.full((B,), position, device=x_step.device, dtype=torch.long)
        x = self.input_proj(x_step) + self.pos_emb(pos)       # (B, hidden)
        x = x.unsqueeze(1)                                     # (B, 1, hidden)
        for block in self.blocks:
            x = block(x, causal_mask=None)                     # (B, 1, hidden)
        x = self.norm(x)
        return x.squeeze(1)                                    # (B, hidden)


# ---------------------------------------------------------------------------
# DiT blocks  (for LocDiT / CFM estimator)
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    """Adaptive Layer Normalization (scale + shift from conditioning).

    Accepts cond of shape (B, cond_dim) and x of shape (B, seq, hidden).
    The shift/scale are unsqueezed to broadcast along the sequence axis.
    """

    def __init__(self, hidden: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, hidden * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.proj(cond).chunk(2, dim=-1)       # each (B, hidden)
        shift = shift.unsqueeze(1)                            # (B, 1, hidden)
        scale = scale.unsqueeze(1)                            # (B, 1, hidden)
        return self.norm(x) * (1.0 + scale) + shift


class DiTBlock(nn.Module):
    """DiT block with AdaLN conditioning on (timestep + context)."""

    def __init__(self, hidden: int, num_heads: int, ffn: int, cond_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.adaln1 = AdaLN(hidden, cond_dim)
        self.attn = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True,
        )
        self.adaln2 = AdaLN(hidden, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn, hidden), nn.Dropout(dropout),
        )
        self.adaln_out = AdaLN(hidden, cond_dim)
        self.ada_scale = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, hidden))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.adaln1(x, cond)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.adaln2(x, cond)
        x = x + self.ffn(h)
        h = self.adaln_out(x, cond)
        # ada_scale(cond) has shape (B, hidden); unsqueeze to (B, 1, hidden) for broadcast
        return x + self.ada_scale(cond).unsqueeze(1) * h


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal time embedding (same as in diffusion/transformer literature)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb if dim % 2 == 0 else F.pad(emb, (0, 1))


class SimpleDiT(nn.Module):
    """Local Diffusion Transformer — the CFM velocity estimator.

    Given a noisy latent patch, timestep, and conditioning vector from
    TSLM+RALM, predicts the velocity field for flow matching.
    """

    def __init__(self, num_layers: int = 4, hidden: int = 256, ffn: int = 1024,
                 num_heads: int = 4, in_channels: int = 32, cond_dim: int = 512):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.time_proj = nn.Linear(hidden, hidden)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden, num_heads, ffn, hidden) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden, in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, P, in_channels)  noisy latent patch
        t:    (B,)                 timestep scalars in [0, 1]
        cond: (B, cond_dim)        conditioning from TSLM+RALM
        """
        h = self.in_proj(x)
        t_emb = self.time_proj(_sinusoidal_embedding(t, h.shape[-1]))
        c = self.cond_proj(cond) + t_emb                     # combined conditioning
        for block in self.blocks:
            h = block(h, c)
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# CFM  (Conditional Flow Matching)
# ---------------------------------------------------------------------------

class SimpleCFM(nn.Module):
    """Conditional Flow Matching — training objective + Euler ODE sampler.

    Training:
        Sample t ~ Uniform(0,1), noise z ~ N(0, I).
        Interpolate: y_t = (1 - t) * x1 + t * z
        Target velocity: v = z - x1   (optimal transport path)
        Loss = MSE(DiT(y_t, t, mu), v)

    Inference:
        Start from z ~ N(0, I), take Euler steps:
            x_{t-dt} = x_t - dt * DiT(x_t, t, mu)
        for n_steps steps from t=1 to t=0.
    """

    def __init__(self, estimator: SimpleDiT, n_steps: int = 10, sigma_min: float = 1e-5):
        super().__init__()
        self.estimator = estimator
        self.n_steps = n_steps
        self.sigma_min = sigma_min

    def compute_loss(
        self,
        x1: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Flow matching training loss for one patch.

        x1:   (B, P, D) target latent patch
        cond: (B, cond_dim) conditioning from TSLM+RALM
        """
        B = x1.shape[0]
        z = torch.randn_like(x1)                             # noise
        t = torch.rand(B, device=x1.device)                  # uniform time in [0, 1]
        t = t * (1.0 - self.sigma_min) + self.sigma_min      # avoid t=0

        # Interpolate along optimal transport path
        y = (1.0 - t.view(-1, 1, 1)) * x1 + t.view(-1, 1, 1) * z

        # Target velocity (derivative of interpolation path)
        v_target = z - x1

        # Predicted velocity
        v_pred = self.estimator(y, t, cond)

        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        shape: Tuple[int, ...],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Euler ODE solver — integrate from t=1 (noise) to t=0 (data).

        cond:  (B, cond_dim) conditioning vector
        shape: (B, P, D)     output latent patch shape
        """
        x = torch.randn(shape, device=cond.device) * temperature
        dt = 1.0 / self.n_steps
        for i in range(self.n_steps):
            t_val = 1.0 - i * dt
            t = torch.full((shape[0],), t_val, device=cond.device)
            v = self.estimator(x, t, cond)
            x = x - dt * v                                   # Euler step
        return x


# ---------------------------------------------------------------------------
# SimpleVoxCPM  (full pipeline wrapper)
# ---------------------------------------------------------------------------

class SimpleVoxCPM(nn.Module):
    """End-to-end tokenizer-free TTS model.

    Pipeline (training):
        waveform → AudioVAE.encode → latents
        latents → LocEnc → audio_emb
        text_emb + audio_emb (interleaved) → TSLM → semantic hidden
        semantic hidden → FSQ → quantized
        quantized → RALM → acoustic hidden
        semantic + acoustic → conditioning μ
        (noisy patch, t, μ) → DiT → velocity prediction
        loss = MSE(velocity_pred, z - x1)

    Pipeline (inference):
        text_emb → TSLM.forward_step (with KV-cache conceptually)
        → FSQ → RALM.forward_step → conditioning μ
        → CFM.sample(μ) → latent patch
        → AudioVAE.decode → waveform chunk
    """

    def __init__(
        self,
        vocab_size: int = 256,
        encoder_dim: int = 64,
        latent_dim: int = 32,
        decoder_dim: int = 256,
        patch_size: int = 1,
        loc_enc_hidden: int = 512,
        loc_enc_layers: int = 2,
        tslm_hidden: int = 512,
        tslm_layers: int = 8,
        tslm_heads: int = 8,
        tslm_ffn: int = 2048,
        fsq_latent: int = 128,
        fsq_scale: int = 9,
        ralm_hidden: int = 512,
        ralm_layers: int = 4,
        ralm_heads: int = 8,
        ralm_ffn: int = 2048,
        dit_hidden: int = 256,
        dit_layers: int = 4,
        dit_heads: int = 4,
        dit_ffn: int = 1024,
        cfm_steps: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.tslm_hidden = tslm_hidden

        # --- Audio VAE ---
        self.audio_vae = SimpleAudioVAE(
            encoder_dim=encoder_dim, latent_dim=latent_dim, decoder_dim=decoder_dim,
        )

        # --- Text embedding ---
        self.text_emb = nn.Embedding(vocab_size, tslm_hidden)

        # --- LocEnc ---
        self.loc_enc = SimpleLocEnc(
            latent_dim=latent_dim, hidden_dim=loc_enc_hidden,
            num_layers=loc_enc_layers, ffn_dim=loc_enc_hidden * 4,
        )
        self.enc_to_tslm = nn.Linear(loc_enc_hidden, tslm_hidden)

        # --- TSLM ---
        self.tslm = SimpleTransformer(
            input_dim=tslm_hidden, hidden=tslm_hidden, num_heads=tslm_heads,
            ffn=tslm_ffn, num_layers=tslm_layers,
        )

        # --- FSQ ---
        self.fsq = SimpleFSQ(in_dim=tslm_hidden, latent_dim=fsq_latent, scale=fsq_scale)

        # --- RALM ---
        self.ralm = SimpleTransformer(
            input_dim=tslm_hidden, hidden=ralm_hidden, num_heads=ralm_heads,
            ffn=ralm_ffn, num_layers=ralm_layers,
        )

        # --- Projections to DiT conditioning ---
        self.lm_to_dit = nn.Linear(tslm_hidden, dit_hidden)
        self.res_to_dit = nn.Linear(ralm_hidden, dit_hidden)

        # --- LocDiT + CFM ---
        self.dit = SimpleDiT(
            num_layers=dit_layers, hidden=dit_hidden, ffn=dit_ffn,
            num_heads=dit_heads, in_channels=latent_dim * patch_size,
            cond_dim=dit_hidden,
        )
        self.cfm = SimpleCFM(self.dit, n_steps=cfm_steps)

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------
    def forward(
        self,
        text_tokens: torch.Tensor,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """Full training forward pass. Returns flow-matching loss.

        text_tokens: (B, L_text)  integer token IDs
        waveform:    (B, T_samples)  16 kHz audio
        """
        B = waveform.shape[0]

        # 1. Encode audio → continuous latents
        with torch.no_grad():                                # VAE typically frozen during LM training
            z = self.audio_vae.encode(waveform)              # (B, D, T_latent)
        z = z.transpose(1, 2)                                # (B, T_latent, D)
        T_lat = z.shape[1]

        # 2. Optional: patchify (group consecutive frames)
        usable = T_lat - (T_lat % self.patch_size)
        z = z[:, :usable, :]
        T_patches = usable // self.patch_size
        z_patches = z.reshape(B, T_patches, self.patch_size * self.latent_dim)  # (B, N, P*D)

        # 3. Encode each patch via LocEnc
        patch_emb = self.loc_enc(
            z.reshape(B * T_patches, self.patch_size, self.latent_dim)
        )                                                    # (B*N, hidden)
        patch_emb = patch_emb.reshape(B, T_patches, -1)      # (B, N, loc_enc_hidden)
        audio_emb = self.enc_to_tslm(patch_emb)              # (B, N, tslm_hidden)

        # 4. Build combined text+audio sequence
        L = text_tokens.shape[1]
        text_h = self.text_emb(text_tokens)                  # (B, L, tslm_hidden)

        # Sequence layout:  [text_0 ... text_{L-1}  BOS  audio_0 ... audio_{N-2}]
        # Targets:          [audio_0 ... audio_{N-1}]
        bos = torch.zeros(B, 1, self.tslm_hidden, device=text_h.device)
        audio_input = audio_emb[:, :-1, :]                   # shift right by 1
        combined = torch.cat([text_h, bos, audio_input], dim=1)  # (B, L+N, H)

        # 5. TSLM forward (causal)
        tslm_out = self.tslm(combined)                       # (B, L+N, H)

        # Extract positions corresponding to audio (after text+BOS)
        tslm_audio = tslm_out[:, L:, :]                      # (B, N, H)

        # 6. FSQ — semantic bottleneck
        fsq_out = self.fsq(tslm_audio)                       # (B, N, H)

        # 7. RALM forward (causal, conditioned on FSQ output)
        ralm_out = self.ralm(fsq_out)                        # (B, N, H)

        # 8. Fusion: additive (v1 style)
        dit_cond = self.lm_to_dit(tslm_audio) + self.res_to_dit(ralm_out)  # (B, N, dit_hidden)

        # 9. CFM loss: predict each patch from its conditioning
        # Condition at position i predicts patch at position i
        # (in a real system there'd be an offset; here we pair them directly for simplicity)
        n_predict = min(dit_cond.shape[1], z_patches.shape[1])
        cond = dit_cond[:, :n_predict, :]                    # (B, M, dit_hidden)
        targets = z_patches[:, :n_predict, :]                # (B, M, P*D)

        # Flatten batch and sequence for CFM loss
        cond_flat = cond.reshape(B * n_predict, -1)          # (B*M, dit_hidden)
        tgt_flat = targets.reshape(B * n_predict, self.patch_size, self.latent_dim)

        loss = self.cfm.compute_loss(tgt_flat, cond_flat)
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        text_tokens: torch.Tensor,
        n_steps: int = 50,
        temperature: float = 1.0,
        prompt_waveform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive inference: text → waveform.

        text_tokens:    (1, L_text)
        n_steps:        number of latent patches to generate
        temperature:    noise scale for CFM initial noise
        prompt_waveform: optional reference audio for voice cloning (not implemented here)

        Returns: (1, T_samples) waveform at 16 kHz
        """
        self.eval()
        B = text_tokens.shape[0]
        assert B == 1, "batched inference not supported"

        # 1. Encode text
        text_h = self.text_emb(text_tokens)                  # (1, L, H)

        # We'll collect generated latent patches
        generated_patches = []
        prev_audio_emb = torch.zeros(1, 1, self.tslm_hidden, device=text_h.device)  # BOS

        # Cache: in a real system we'd use KV-cache; here we recompute for clarity
        audio_history = prev_audio_emb

        for step in range(n_steps):
            # 2. Build sequence and run TSLM (only last position matters)
            combined = torch.cat([text_h, audio_history], dim=1)
            tslm_out = self.tslm(combined)                   # (1, L+step+1, H)
            tslm_step = tslm_out[:, -1, :]                   # (1, H)

            # 3. FSQ
            fsq_step = self.fsq(tslm_step)                   # (1, H)

            # 4. RALM (simplified: process only current FSQ output)
            ralm_step = self.ralm.forward_step(fsq_step, position=step)  # (1, H)

            # 5. Fusion → conditioning for DiT
            dit_cond = self.lm_to_dit(tslm_step) + self.res_to_dit(ralm_step)  # (1, dit_hidden)

            # 6. CFM sample — generate one latent patch via Euler ODE
            patch = self.cfm.sample(
                dit_cond,
                shape=(1, self.patch_size, self.latent_dim),
                temperature=temperature,
            )                                                 # (1, P, D)
            generated_patches.append(patch)

            # 7. Update audio history for next AR step
            # LocEnc compresses the patch to a hidden vector; enc_to_tslm projects it
            new_audio_emb = self.enc_to_tslm(self.loc_enc(patch))  # (1, H)
            new_audio_emb = new_audio_emb.unsqueeze(1)        # (1, 1, H)
            audio_history = torch.cat([audio_history, new_audio_emb], dim=1)

        # 8. Concatenate all patches → latent sequence
        z = torch.cat(generated_patches, dim=1)              # (1, n_steps*P, D)
        z = z.reshape(1, n_steps * self.patch_size, self.latent_dim)
        z = z.transpose(1, 2)                                # (1, D, T)

        # 9. Decode to waveform
        waveform = self.audio_vae.decode(z)                  # (1, T_samples)
        return waveform

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def count_params(self) -> dict:
        """Return parameter count per component."""
        counts = {}
        for name, module in [
            ("audio_vae", self.audio_vae),
            ("text_emb", self.text_emb),
            ("loc_enc", self.loc_enc),
            ("tslm", self.tslm),
            ("fsq", self.fsq),
            ("ralm", self.ralm),
            ("dit", self.dit),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(counts.values())
        return counts


# ---------------------------------------------------------------------------
# Shape verification
# ---------------------------------------------------------------------------

def verify_shapes():
    """Run shape checks on all components (executed when running this file directly)."""
    import torch

    device = "cpu"
    B, T_samples = 2, 16000                                  # 1 second at 16 kHz

    print("=" * 60)
    print("Shape Verification — SimpleVoxCPM")
    print("=" * 60)

    # AudioVAE
    vae = SimpleAudioVAE().to(device)
    wav = torch.randn(B, T_samples, device=device)
    z = vae.encode(wav)
    wav_recon = vae.decode(z)
    print(f"[AudioVAE]  input:    {tuple(wav.shape)}")
    print(f"            latent:   {tuple(z.shape)}  (latent_dim=32, rate=25Hz)")
    print(f"            recon:    {tuple(wav_recon.shape)}")
    assert z.shape == (B, 32, T_samples // 640)
    print("            ✓ AudioVAE shapes OK")

    # LocEnc
    loc_enc = SimpleLocEnc().to(device)
    patch = torch.randn(B, 1, 32, device=device)             # 1 patch of size 1
    enc_out = loc_enc(patch)
    print(f"[LocEnc]    input:    {tuple(patch.shape)}")
    print(f"            output:   {tuple(enc_out.shape)}")
    assert enc_out.shape == (B, 512)
    print("            ✓ LocEnc shapes OK")

    # FSQ
    fsq = SimpleFSQ().to(device)
    h = torch.randn(B, 512, device=device)
    q = fsq(h)
    print(f"[FSQ]       input:    {tuple(h.shape)}")
    print(f"            output:   {tuple(q.shape)}")
    assert q.shape == h.shape
    print("            ✓ FSQ shapes OK")

    # TSLM
    tslm = SimpleTransformer(input_dim=512, hidden=512, num_heads=8,
                             ffn=2048, num_layers=8).to(device)
    seq = torch.randn(B, 10, 512, device=device)
    out = tslm(seq)
    print(f"[TSLM]      input:    {tuple(seq.shape)}")
    print(f"            output:   {tuple(out.shape)}")
    assert out.shape == seq.shape
    print("            ✓ TSLM shapes OK")

    # RALM
    ralm = SimpleTransformer(input_dim=512, hidden=512, num_heads=8,
                             ffn=2048, num_layers=4).to(device)
    out = ralm(seq)
    print(f"[RALM]      input:    {tuple(seq.shape)}")
    print(f"            output:   {tuple(out.shape)}")
    assert out.shape == seq.shape
    print("            ✓ RALM shapes OK")

    # DiT
    dit = SimpleDiT(num_layers=4, hidden=256, in_channels=32, cond_dim=512).to(device)
    noisy = torch.randn(B, 1, 32, device=device)
    t = torch.rand(B, device=device)
    cond = torch.randn(B, 512, device=device)
    v = dit(noisy, t, cond)
    print(f"[DiT]       input:    {tuple(noisy.shape)}, cond: {tuple(cond.shape)}")
    print(f"            output:   {tuple(v.shape)}")
    assert v.shape == noisy.shape
    print("            ✓ DiT shapes OK")

    # Full model
    model = SimpleVoxCPM().to(device)
    text = torch.randint(0, 256, (B, 8), device=device)
    # Need enough audio for at least 1 patch: 640 samples minimum
    wav_short = torch.randn(B, 640 * 4, device=device)       # 4 latent frames
    loss = model(text, wav_short)
    print(f"[VoxCPM]    text:     {tuple(text.shape)}")
    print(f"            audio:    {tuple(wav_short.shape)}")
    print(f"            loss:     {loss.item():.4f}")
    print("            ✓ Full forward OK")

    # Param counts
    counts = model.count_params()
    print()
    print("Parameter counts:")
    for k, v in counts.items():
        print(f"  {k:12s}: {v:>12,}  ({v/1e6:.1f}M)")

    print()
    print("All shape checks passed! ✓")


if __name__ == "__main__":
    verify_shapes()
