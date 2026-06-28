"""
Ch07: Neural Audio Codec — Simplified EnCodec for VALL-E

This codec converts mel spectrograms into discrete multi-level token sequences.
It is a simplified version of EnCodec (Defossez et al., 2022) designed for
educational purposes.

Architecture:
    Mel Spectrogram (B, 80, T_mel)
        ↓ Encoder (3× Conv1d stride-2, 8× temporal downsampling)
    Latent (B, dim, T_latent)
        ↓ Multi-level Vector Quantizer (4 codebooks × 256 entries)
    Quantized Latent (B, dim, T_latent)  +  Codes (B, num_levels, T_latent)
        ↓ Decoder (3× ConvTranspose1d stride-2, 8× temporal upsampling)
    Reconstructed Mel (B, 80, T_mel)

The key idea: each audio frame is represented by a STACK of discrete tokens.
Level 0 captures coarse structure (rhythm, pitch contour).
Levels 1-3 capture progressively finer acoustic detail.

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

class ResBlock(nn.Module):
    """Residual block: two Conv1d layers with a skip connection."""

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=pad),
            nn.ELU(),
            nn.Conv1d(dim, dim, kernel_size, padding=pad),
        )

    def forward(self, x):
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Vector Quantizer (Multi-level, independent)
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """
    Multi-level Vector Quantizer.

    For each frame, independently quantizes into `num_levels` codebooks.
    This is simpler than Residual VQ (used in real EnCodec) but captures
    the same core idea: multiple levels of discrete representation.

    Each codebook has `codebook_size` entries of dimension `dim`.
    """

    def __init__(self, codebook_size=256, dim=128, num_levels=4):
        super().__init__()
        self.codebook_size = codebook_size
        self.num_levels = num_levels

        # One codebook per level: (num_levels, codebook_size, dim)
        self.embeddings = nn.Embedding(num_levels * codebook_size, dim)
        nn.init.uniform_(self.embeddings.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def _get_codebook(self, level):
        """Get the codebook for a specific level."""
        offset = level * self.codebook_size
        return self.embeddings.weight[offset:offset + self.codebook_size]

    def quantize_level(self, x, level):
        """
        Quantize x using the codebook for the given level.

        Args:
            x: (B, dim, T) — continuous latent
            level: int — which codebook to use

        Returns:
            indices: (B, T) — token indices in [0, codebook_size)
            quantized: (B, dim, T) — quantized vectors
            loss: scalar — VQ loss for this level
        """
        cb = self._get_codebook(level)  # (codebook_size, dim)
        B, D, T = x.shape

        # x_flat: (B*T, dim), cb: (codebook_size, dim)
        x_flat = x.permute(0, 2, 1).reshape(-1, D)

        # Compute distances: ||x - e||^2 = ||x||^2 - 2*x.e + ||e||^2
        dist = (
            x_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * x_flat @ cb.t()
            + cb.pow(2).sum(dim=1, keepdim=True).t()
        )  # (B*T, codebook_size)

        indices = dist.argmin(dim=1)  # (B*T,)
        quantized = cb[indices]       # (B*T, dim)

        # VQ losses
        commitment_loss = F.mse_loss(quantized.detach(), x_flat)
        codebook_loss = F.mse_loss(quantized, x_flat.detach())
        loss = codebook_loss + 0.25 * commitment_loss

        # Straight-through estimator: pass gradients from quantized to x
        quantized = x_flat + (quantized - x_flat).detach()

        # Reshape back
        indices = indices.reshape(B, T)
        quantized = quantized.reshape(B, T, D).permute(0, 2, 1)

        return indices, quantized, loss

    def forward(self, z):
        """
        Quantize z using all levels.

        Args:
            z: (B, dim, T) — encoder output

        Returns:
            z_q: (B, dim, T) — averaged quantized output
            codes: (B, num_levels, T) — token indices for each level
            vq_loss: scalar — total VQ loss
        """
        all_codes = []
        all_quantized = []
        total_loss = 0

        for level in range(self.num_levels):
            indices, quantized, loss = self.quantize_level(z, level)
            all_codes.append(indices)
            all_quantized.append(quantized)
            total_loss = total_loss + loss

        # Average the quantized outputs from all levels
        z_q = torch.stack(all_quantized, dim=0).mean(dim=0)
        codes = torch.stack(all_codes, dim=1)  # (B, num_levels, T)

        return z_q, codes, total_loss / self.num_levels

    def decode_tokens(self, codes):
        """
        Look up codebook embeddings for given token indices.

        Args:
            codes: (B, num_levels, T)

        Returns:
            z_q: (B, dim, T) — averaged quantized output
        """
        all_quantized = []
        for level in range(self.num_levels):
            cb = self._get_codebook(level)
            idx = codes[:, level]  # (B, T)
            q = cb[idx]            # (B, T, dim)
            all_quantized.append(q.permute(0, 2, 1))  # (B, dim, T)

        return torch.stack(all_quantized, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Neural Codec (full encoder-quantizer-decoder pipeline)
# ---------------------------------------------------------------------------

class NeuralCodec(nn.Module):
    """
    Simplified neural audio codec.

    Converts mel spectrograms ↔ multi-level discrete tokens.

    Temporal compression: 8× (three stride-2 conv layers)
    So 1 second of audio (100 mel frames @ 16kHz/256hop) → ~13 codec frames.

    Args:
        mel_bins:    Number of mel frequency bins (default 80)
        hidden_dim:  Encoder/decoder hidden dimension
        codebook_size: Number of entries per codebook level
        num_levels:  Number of VQ codebook levels
    """

    def __init__(
        self,
        mel_bins=80,
        hidden_dim=128,
        codebook_size=256,
        num_levels=4,
    ):
        super().__init__()
        self.mel_bins = mel_bins
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.num_levels = num_levels

        # Encoder: mel → latent (8× temporal downsampling)
        self.encoder = nn.Sequential(
            nn.Conv1d(mel_bins, hidden_dim, 7, padding=3),
            nn.ELU(),
            # Downsample block 1: stride 2
            nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ELU(),
            ResBlock(hidden_dim),
            # Downsample block 2: stride 2
            nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ELU(),
            ResBlock(hidden_dim),
            # Downsample block 3: stride 2
            nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ELU(),
            ResBlock(hidden_dim),
        )

        # Decoder: latent → mel (8× temporal upsampling)
        self.decoder = nn.Sequential(
            ResBlock(hidden_dim),
            # Upsample block 1: stride 2
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ELU(),
            ResBlock(hidden_dim),
            # Upsample block 2: stride 2
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ELU(),
            ResBlock(hidden_dim),
            # Upsample block 3: stride 2
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(hidden_dim, mel_bins, 7, padding=3),
        )

        # Vector Quantizer
        self.quantizer = VectorQuantizer(codebook_size, hidden_dim, num_levels)

    def forward(self, mel):
        """
        Encode → Quantize → Decode.

        Args:
            mel: (B, mel_bins, T_mel) — input mel spectrogram

        Returns:
            mel_hat: (B, mel_bins, T_mel) — reconstructed mel
            codes: (B, num_levels, T_latent) — discrete tokens
            vq_loss: scalar — VQ commitment/codebook loss
        """
        z = self.encoder(mel)              # (B, hidden_dim, T_latent)
        z_q, codes, vq_loss = self.quantizer(z)
        mel_hat = self.decoder(z_q)         # (B, mel_bins, T_mel)
        return mel_hat, codes, vq_loss

    def encode(self, mel):
        """
        Mel → discrete multi-level tokens.

        Args:
            mel: (B, mel_bins, T_mel)

        Returns:
            codes: (B, num_levels, T_latent)
        """
        z = self.encoder(mel)
        _, codes, _ = self.quantizer(z)
        return codes

    def decode(self, codes):
        """
        Discrete tokens → mel spectrogram.

        Args:
            codes: (B, num_levels, T_latent)

        Returns:
            mel_hat: (B, mel_bins, T_mel)
        """
        z_q = self.quantizer.decode_tokens(codes)
        return self.decoder(z_q)

    def get_embeddings_level(self, level):
        """Get codebook embeddings for a specific level: (codebook_size, dim)."""
        return self.quantizer._get_codebook(level)


# ---------------------------------------------------------------------------
# Shape Verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing NeuralCodec shapes...")

    codec = NeuralCodec(
        mel_bins=80, hidden_dim=128,
        codebook_size=256, num_levels=4,
    )

    B, T_mel = 2, 160  # ~1.6 seconds of audio at 16kHz/256hop
    mel = torch.randn(B, 80, T_mel)

    # Forward: encode → quantize → decode
    mel_hat, codes, vq_loss = codec(mel)
    print(f"Input mel:    {mel.shape}")        # (2, 80, 160)
    print(f"Output mel:   {mel_hat.shape}")     # (2, 80, 160)
    print(f"Codes:        {codes.shape}")       # (2, 4, 20)
    print(f"VQ loss:      {vq_loss.item():.4f}")
    print(f"Code range:   [{codes.min()}, {codes.max()}]")

    # Encode only
    tokens = codec.encode(mel)
    print(f"Tokens:       {tokens.shape}")      # (2, 4, 20)

    # Decode only
    mel_from_tokens = codec.decode(tokens)
    print(f"From tokens:  {mel_from_tokens.shape}")  # (2, 80, 160)

    total_params = sum(p.numel() for p in codec.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("All shapes OK!")
