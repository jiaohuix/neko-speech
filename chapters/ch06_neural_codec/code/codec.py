"""
Ch06: EnCodec Mini — 波形级神经音频编解码器

EnCodec 风格的音频压缩模型，直接操作波形（非 Mel 频谱）。
与 ch07 的 NeuralCodec（Mel 域）不同，本模型在波形域工作。

Architecture:
    Waveform (B, 1, T_wav)
        ↓ Encoder (4× Conv1d stride, 320× temporal downsampling)
    Latent (B, latent_dim, T_latent)
        ↓ Residual Vector Quantizer (8 codebooks × 1024 entries)
    Quantized Latent (B, latent_dim, T_latent) + Codes (B, K, T_latent)
        ↓ Decoder (4× ConvTranspose1d stride, 320× upsampling)
    Reconstructed Waveform (B, 1, T_wav)

Reference:
    Defossez et al., 2022. High Fidelity Neural Audio Compression.
    van den Oord et al., 2017. Neural Discrete Representation Learning (VQ-VAE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ResBlock1d(nn.Module):
    """1D Residual block: Conv → ELU → Conv + skip."""

    def __init__(self, dim, kernel_size=3, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(dim, dim, kernel_size, dilation=dilation, padding=pad),
            nn.ELU(),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, x):
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Residual Vector Quantizer
# ---------------------------------------------------------------------------

class ResidualVectorQuantizer(nn.Module):
    """
    Residual VQ (RVQ): multi-level vector quantization.

    Level 0 quantizes the original signal.
    Level k quantizes the residual from levels 0..k-1.

    This gives a coarse-to-fine representation:
        Level 0: rough shape (prosody, rhythm)
        Level 1: finer detail (pitch contour)
        Level 2+: spectral detail (timbre)

    Args:
        dim:        latent dimension
        num_codes:  entries per codebook
        num_codebooks: number of RVQ levels
    """

    def __init__(self, dim=128, num_codes=1024, num_codebooks=8):
        super().__init__()
        self.dim = dim
        self.num_codes = num_codes
        self.num_codebooks = num_codebooks

        self.quantizers = nn.ModuleList([
            nn.Embedding(num_codes, dim) for _ in range(num_codebooks)
        ])

        # Initialize codebooks
        for emb in self.quantizers:
            nn.init.uniform_(emb.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z):
        """
        Args:
            z: (B, dim, T) — continuous latent

        Returns:
            z_q: (B, dim, T) — quantized output (sum of all levels)
            codes: (B, num_codebooks, T) — token indices
            vq_loss: scalar — total VQ loss
        """
        residual = z
        all_codes = []
        all_quantized = []
        total_vq_loss = 0.0

        for i, emb in enumerate(self.quantizers):
            cb = emb.weight  # (num_codes, dim)
            B, D, T = residual.shape

            # Flatten: (B*T, dim)
            r_flat = residual.permute(0, 2, 1).reshape(-1, D)

            # Nearest neighbor lookup
            dist = (
                r_flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * r_flat @ cb.t()
                + cb.pow(2).sum(dim=1, keepdim=True).t()
            )
            indices = dist.argmin(dim=1)
            quantized = cb[indices].reshape(B, T, D).permute(0, 2, 1)

            # VQ loss for this level
            commitment = F.mse_loss(quantized.detach(), residual)
            codebook = F.mse_loss(quantized, residual.detach())
            total_vq_loss = total_vq_loss + codebook + 0.25 * commitment

            # Straight-through estimator
            quantized_st = residual + (quantized - residual).detach()

            all_codes.append(indices.reshape(B, T))
            all_quantized.append(quantized_st)

            # Update residual
            residual = residual - quantized_st

        # Sum all quantized levels
        z_q = torch.stack(all_quantized, dim=0).sum(dim=0)
        codes = torch.stack(all_codes, dim=1)  # (B, K, T)

        return z_q, codes, total_vq_loss / self.num_codebooks

    def decode_tokens(self, codes):
        """
        Look up codebook embeddings and sum.

        Args:
            codes: (B, K, T)

        Returns:
            z_q: (B, dim, T)
        """
        z_q = 0
        for i, emb in enumerate(self.quantizers):
            idx = codes[:, i]  # (B, T)
            q = emb(idx)       # (B, T, dim)
            z_q = z_q + q.permute(0, 2, 1)
        return z_q


# ---------------------------------------------------------------------------
# EnCodec Mini
# ---------------------------------------------------------------------------

class EnCodecMini(nn.Module):
    """
    Simplified EnCodec: waveform-level neural audio codec.

    Operates directly on waveforms (not mel spectrograms).
    320× temporal compression at 24kHz → 75 Hz frame rate.

    Args:
        base_dim:      base channel count (doubled at each downsample)
        latent_dim:    bottleneck dimension (for VQ)
        num_codebooks: number of RVQ levels
        num_codes:     entries per codebook
    """

    def __init__(
        self,
        base_dim=32,
        latent_dim=128,
        num_codebooks=8,
        num_codes=1024,
    ):
        super().__init__()
        self.base_dim = base_dim
        self.latent_dim = latent_dim
        self._num_codebooks = num_codebooks
        self._num_codes = num_codes

        # Downsampling ratios: 4 × 4 × 4 × 5 = 320
        strides = [4, 4, 4, 5]
        self.hop_length = 1
        for s in strides:
            self.hop_length *= s  # 320

        # --- Encoder ---
        enc_layers = [
            nn.Conv1d(1, base_dim, 7, padding=3),
            nn.ELU(),
        ]
        ch = base_dim
        for s in strides:
            enc_layers.extend([
                nn.Conv1d(ch, ch * 2, s * 2, stride=s, padding=s // 2),
                nn.ELU(),
                ResBlock1d(ch * 2, kernel_size=3, dilation=1),
                ResBlock1d(ch * 2, kernel_size=3, dilation=3),
            ])
            ch *= 2
        enc_layers.append(nn.Conv1d(ch, latent_dim, 1))
        self.encoder = nn.Sequential(*enc_layers)

        # --- Decoder ---
        dec_layers = [nn.Conv1d(latent_dim, ch, 1)]
        for s in reversed(strides):
            dec_layers.extend([
                ResBlock1d(ch, kernel_size=3, dilation=1),
                ResBlock1d(ch, kernel_size=3, dilation=3),
                nn.ConvTranspose1d(ch, ch // 2, s * 2, stride=s, padding=s // 2),
                nn.ELU(),
            ])
            ch //= 2
        dec_layers.append(nn.Conv1d(ch, 1, 7, padding=3))
        dec_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec_layers)

        # --- Quantizer ---
        self.quantizer = ResidualVectorQuantizer(
            dim=latent_dim, num_codes=num_codes, num_codebooks=num_codebooks,
        )

    def forward(self, wav):
        """
        Args:
            wav: (B, 1, T) — input waveform

        Returns:
            dict with:
                wav_hat: (B, 1, T) — reconstructed waveform
                vq_loss: scalar — VQ commitment + codebook loss
                tokens: (B, K, T_latent) — discrete codes
        """
        z = self.encoder(wav)             # (B, latent_dim, T_latent)
        z_q, codes, vq_loss = self.quantizer(z)
        wav_hat = self.decoder(z_q)        # (B, 1, T_wav')

        # Match input length
        T_in = wav.shape[-1]
        if wav_hat.shape[-1] > T_in:
            wav_hat = wav_hat[..., :T_in]
        elif wav_hat.shape[-1] < T_in:
            wav_hat = F.pad(wav_hat, (0, T_in - wav_hat.shape[-1]))

        return {"wav_hat": wav_hat, "vq_loss": vq_loss, "tokens": codes}

    def encode(self, wav):
        """Waveform → discrete tokens."""
        z = self.encoder(wav)
        _, codes, _ = self.quantizer(z)
        return codes

    def decode(self, codes):
        """Discrete tokens → waveform."""
        z_q = self.quantizer.decode_tokens(codes)
        return self.decoder(z_q)

    def n_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

def _stft_loss(wav, wav_hat, n_fft, hop_length, win_length):
    """Single-resolution STFT L1 loss (magnitude)."""
    window = torch.hann_window(win_length, device=wav.device)

    min_len = min(wav.shape[-1], wav_hat.shape[-1])
    w = wav[..., :min_len].squeeze(1)
    wh = wav_hat[..., :min_len].squeeze(1)

    spec_r = torch.stft(w, n_fft, hop_length, win_length, window,
                        return_complex=True).abs()
    spec_g = torch.stft(wh, n_fft, hop_length, win_length, window,
                        return_complex=True).abs()

    return F.l1_loss(spec_g, spec_r)


def codec_loss(wav, wav_hat, vq_loss):
    """
    Multi-Resolution STFT + L1 + VQ loss.

    Args:
        wav:     (B, 1, T) — original waveform
        wav_hat: (B, 1, T) — reconstructed waveform
        vq_loss: scalar — from the quantizer

    Returns:
        total_loss: scalar
        loss_dict: dict with l1, spec, vq, total
    """
    # Waveform L1
    l1 = F.l1_loss(wav_hat, wav)

    # Multi-Resolution STFT loss
    resolutions = [
        (512, 128, 512),
        (1024, 256, 1024),
        (2048, 512, 2048),
    ]
    spec_loss = 0.0
    for n_fft, hop, win in resolutions:
        spec_loss = spec_loss + _stft_loss(wav, wav_hat, n_fft, hop, win)
    spec_loss = spec_loss / len(resolutions)

    total = l1 + spec_loss + vq_loss

    return total, {
        "l1": l1.item(),
        "spec": spec_loss.item(),
        "vq": vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss,
        "total": total.item(),
    }


# ---------------------------------------------------------------------------
# Shape Verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Ch06 EnCodec Mini — Shape Verification")
    print("=" * 60)

    model = EnCodecMini(
        base_dim=32, latent_dim=128,
        num_codebooks=8, num_codes=1024,
    )

    B = 2
    sr = 24000
    duration = 2.0
    T = int(sr * duration)

    wav = torch.randn(B, 1, T)
    out = model(wav)

    print(f"Input:        {wav.shape}")
    print(f"Output:       {out['wav_hat'].shape}")
    print(f"Tokens:       {out['tokens'].shape}")
    print(f"VQ loss:      {out['vq_loss']:.4f}")
    print(f"Hop length:   {model.hop_length}")
    print(f"Frame rate:   {sr / model.hop_length:.1f} Hz")
    print(f"Parameters:   {model.n_params():,}")

    # Test codec_loss
    total, details = codec_loss(wav, out["wav_hat"], out["vq_loss"])
    print(f"\nLoss breakdown:")
    for k, v in details.items():
        print(f"  {k}: {v:.4f}")

    # Test encode / decode roundtrip
    codes = model.encode(wav)
    wav_recon = model.decode(codes)
    print(f"\nEncode → Decode:")
    print(f"  Codes:   {codes.shape}")
    print(f"  Recon:   {wav_recon.shape}")

    print("\nAll shapes OK!")
