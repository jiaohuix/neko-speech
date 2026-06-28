"""
Ch04: FastSpeech2 — Non-Autoregressive TTS

Reference:
    FastSpeech 2: Fast and High-Quality End-to-End Text to Speech
    Ren et al., 2021  (https://arxiv.org/abs/2006.04558)

Architecture:
    Text -> Embedding -> FFT Encoder -> (+ pitch/energy)
                         |
                  Length Regulator  (expand by predicted durations)
                         |
                  FFT Decoder -> Linear -> Mel Spectrogram

Key idea vs Tacotron2 (Ch02):
    - No autoregressive decoding  => parallel generation
    - Explicit duration modeling   => no attention alignment headaches
    - 10-100x faster inference
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------
# Sinusoidal Positional Encoding
# --------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):
        """x: (B, T, d) or (T, B, d) -- adds PE to last dim."""
        if x.dim() == 3 and x.shape[0] != 1 and x.shape[1] != 1:
            # (B, T, d) or (T, B, d): add pe[:, :x.size(1)]
            return x + self.pe[:, : x.size(1)]
        return x + self.pe[:, : x.size(0)]


# --------------------------------------------------------
# Feed-Forward Transformer (FFT) Block
# --------------------------------------------------------

class FFTBlock(nn.Module):
    """One FFT block: LayerNorm -> Self-Attention -> Residual -> LN -> FFN -> Residual."""

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """x: (B, T, d), mask: (B, T) bool True=pad. Returns (B, T, d)."""
        h = self.norm1(x)
        h_t = h.transpose(0, 1)  # (T, B, d) for nn.MultiheadAttention
        a, _ = self.attn(h_t, h_t, h_t, key_padding_mask=mask)
        x = x + self.dropout(a.transpose(0, 1))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class FFTStack(nn.Module):
    """Stack of N FFT blocks with positional encoding."""

    def __init__(self, d_model=256, nhead=2, d_ff=1024, n_layers=4, dropout=0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            FFTBlock(d_model, nhead, d_ff, dropout) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """x: (B, T, d), mask: (B, T) True=pad."""
        x = self.pos_enc(x).transpose(0, 1)  # (T, B, d)
        for block in self.blocks:
            x = block(x.transpose(0, 1), mask).transpose(0, 1)
        return self.dropout(x.transpose(0, 1))  # (B, T, d)


# --------------------------------------------------------
# Variance Predictor (shared by Duration / Pitch / Energy)
# --------------------------------------------------------

class VariancePredictor(nn.Module):
    """
    2-layer Conv1d predictor: (B, T, d) -> (B, T) scalar per token.
    Used for duration, pitch, and energy prediction.
    """

    def __init__(self, d_model=256, d_hidden=256, kernel_size=3, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(d_model, d_hidden, kernel_size, padding=pad)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.conv2 = nn.Conv1d(d_hidden, d_hidden, kernel_size, padding=pad)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_hidden, 1)

    def forward(self, x):
        """(B, T, d) -> (B, T)"""
        # Conv1d operates on (B, C, T); LayerNorm on last dim (T->C after transpose)
        h = self.conv1(x.transpose(1, 2))       # (B, d_h, T)
        h = self.norm1(h.transpose(1, 2))        # (B, T, d_h)
        h = F.relu(h)
        h = self.dropout(h).transpose(1, 2)      # (B, d_h, T)
        h = self.conv2(h)                        # (B, d_h, T)
        h = self.norm2(h.transpose(1, 2))        # (B, T, d_h)
        h = F.relu(h)
        h = self.dropout(h)
        return self.proj(h).squeeze(-1)           # (B, T)


# --------------------------------------------------------
# Length Regulator
# --------------------------------------------------------

class LengthRegulator(nn.Module):
    """Expand encoder output by predicted durations (repeat_interleave)."""

    def forward(self, x, durations, mask=None):
        """x: (B, T_text, d), durations: (B, T_text) int, mask: (B, T_text) bool."""
        B = x.shape[0]
        out = []
        for i in range(B):
            if mask is not None:
                valid = ~mask[i]
                xi, di = x[i, valid], durations[i, valid]
            else:
                xi, di = x[i], durations[i]
            di = di.clamp(min=0).long()
            out.append(xi.repeat_interleave(di, dim=0))
        return nn.utils.rnn.pad_sequence(out, batch_first=True)


# --------------------------------------------------------
# FastSpeech2 (Full Model)
# --------------------------------------------------------

class FastSpeech2(nn.Module):
    """
    FastSpeech2: non-autoregressive TTS with variance adapters.

    Input:  text tokens  (B, T_text)
    Output: mel spectrogram (B, n_mels, T_mel)

    Variance adapters predict:
        - Duration: frames per phoneme (for Length Regulator)
        - Pitch:    spectral centroid proxy (for prosody control)
        - Energy:   frame loudness (for dynamics control)
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        n_mels: int = 80,
        nhead: int = 2,
        d_ff: int = 1024,
        enc_layers: int = 4,
        dec_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_mels = n_mels

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.encoder = FFTStack(d_model, nhead, d_ff, enc_layers, dropout)

        # Variance adapters
        self.dur_predictor = VariancePredictor(d_model)
        self.pitch_predictor = VariancePredictor(d_model)
        self.energy_predictor = VariancePredictor(d_model)
        self.pitch_embed = nn.Linear(1, d_model)
        self.energy_embed = nn.Linear(1, d_model)

        self.decoder = FFTStack(d_model, nhead, d_ff, dec_layers, dropout)
        self.mel_proj = nn.Linear(d_model, n_mels)

    # ----- helpers -----

    @staticmethod
    def _pad_mask(lengths, max_len):
        """(B,) -> (B, max_len) bool, True = pad."""
        return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)

    @staticmethod
    def _pitch_energy_from_mel(mel, durations, text_mask):
        """Derive pitch (spectral centroid) and energy (mean magnitude) from mel."""
        B, T_text = durations.shape
        pitch = torch.zeros(B, T_text, device=mel.device)
        energy = torch.zeros(B, T_text, device=mel.device)
        freqs = torch.arange(mel.shape[-1], device=mel.device, dtype=mel.dtype)

        for i in range(B):
            pos = 0
            for j in range(T_text):
                if text_mask is not None and text_mask[i, j]:
                    break
                d = int(durations[i, j].item())
                if d > 0 and pos + d <= mel.shape[1]:
                    seg = mel[i, pos: pos + d, :]  # (d, n_mels)
                    mag = seg.exp().mean(dim=0)     # (n_mels,)
                    total = mag.sum().clamp(min=1e-6)
                    pitch[i, j] = (mag * freqs).sum() / total
                    energy[i, j] = seg.mean()
                    pos += d
                else:
                    pos += max(d, 0)
        return pitch, energy

    # ----- forward (training) -----

    def forward(self, text, text_lens, mel, mel_lens,
                durations=None, pitch=None, energy=None):
        """
        Training forward pass.
        Returns: mel_pred (B, n_mels, T_mel), dur_pred, pitch_pred, energy_pred (B, T_text)
        """
        B = text.shape[0]
        t_mask = self._pad_mask(text_lens, text.shape[1])

        # --- Encoder ---
        x = self.embedding(text) * math.sqrt(self.d_model)
        x = self.encoder(x, mask=t_mask)

        # --- Default durations: uniform distribution ---
        if durations is None:
            durations = torch.ones_like(text, dtype=torch.long)
            for i in range(B):
                tl, ml = int(text_lens[i]), int(mel_lens[i])
                if tl > 0:
                    base = ml // tl
                    rem = ml - base * tl
                    durations[i, :tl] = base
                    if rem > 0:
                        durations[i, :rem] += 1

        # --- Derive pitch/energy from mel if not provided ---
        if pitch is None or energy is None:
            p, e = self._pitch_energy_from_mel(mel, durations, t_mask)
            if pitch is None:
                pitch = p
            if energy is None:
                energy = e

        # --- Variance predictors ---
        dur_pred = self.dur_predictor(x)
        pitch_pred = self.pitch_predictor(x)
        energy_pred = self.energy_predictor(x)

        # --- Add pitch & energy embeddings ---
        x = x + self.pitch_embed(pitch.unsqueeze(-1))
        x = x + self.energy_embed(energy.unsqueeze(-1))

        # --- Length Regulator ---
        x = LengthRegulator()(x, durations, mask=t_mask)

        # --- Decoder ---
        m_mask = self._pad_mask(mel_lens, x.shape[1])
        x = self.decoder(x, mask=m_mask)

        # --- Mel projection -> (B, n_mels, T_mel) ---
        mel_pred = self.mel_proj(x).transpose(1, 2)

        return mel_pred, dur_pred, pitch_pred, energy_pred

    # ----- inference -----

    @torch.no_grad()
    def inference(self, text, text_lens=None,
                  pitch_scale=1.0, energy_scale=1.0):
        """Parallel (non-autoregressive) inference. Returns mel (B, n_mels, T_mel)."""
        self.eval()
        B = text.shape[0]
        if text_lens is None:
            text_lens = torch.full((B,), text.shape[1], device=text.device)
        t_mask = self._pad_mask(text_lens, text.shape[1])

        # Encoder
        x = self.embedding(text) * math.sqrt(self.d_model)
        x = self.encoder(x, mask=t_mask)

        # Predict duration (log -> exp -> round)
        log_dur = self.dur_predictor(x)
        durations = torch.clamp(torch.exp(log_dur) - 1, min=0).long()
        if t_mask is not None:
            durations = durations.masked_fill(t_mask, 0)
        # Ensure at least 1 frame per real token
        real = ~t_mask if t_mask is not None else torch.ones_like(text, dtype=torch.bool)
        durations = torch.where(real & (durations == 0),
                                torch.ones_like(durations), durations)

        # Predict & add pitch/energy
        pitch = self.pitch_predictor(x) * pitch_scale
        energy = self.energy_predictor(x) * energy_scale
        x = x + self.pitch_embed(pitch.unsqueeze(-1))
        x = x + self.energy_embed(energy.unsqueeze(-1))

        # Length Regulator
        x = LengthRegulator()(x, durations, mask=t_mask)
        if x.shape[1] == 0:
            x = torch.zeros(B, 1, self.d_model, device=x.device)

        # Decoder
        x = self.decoder(x)
        mel = self.mel_proj(x).transpose(1, 2)  # (B, n_mels, T)
        return mel


# --------------------------------------------------------
# Shape Verification
# --------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("FastSpeech2 -- Shape Verification")
    print("=" * 55)
    B, T_text, T_mel, n_mels = 2, 15, 80, 80
    model = FastSpeech2(vocab_size=100, d_model=256, n_mels=n_mels,
                        nhead=2, d_ff=1024, enc_layers=4, dec_layers=4)
    text = torch.randint(1, 100, (B, T_text))
    text_lens = torch.tensor([15, 10])
    mel = torch.randn(B, T_mel, n_mels)
    mel_lens = torch.tensor([80, 60])

    mel_pred, dur_pred, pitch_pred, energy_pred = model(text, text_lens, mel, mel_lens)
    print(f"  text:         {text.shape}")
    print(f"  mel target:   {mel.shape}")
    print(f"  mel_pred:     {mel_pred.shape}")
    assert mel_pred.shape == (B, n_mels, T_mel)
    print(f"  dur_pred:     {dur_pred.shape}")
    print(f"  pitch_pred:   {pitch_pred.shape}")
    print(f"  energy_pred:  {energy_pred.shape}")

    mel_inf = model.inference(text[:1], text_lens[:1])
    print(f"  mel_inf:      {mel_inf.shape}")
    assert mel_inf.shape[0] == 1 and mel_inf.shape[1] == n_mels

    n = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters: {n:,}")
    print("All shapes OK!")
