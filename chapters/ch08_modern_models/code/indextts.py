"""
Ch08.3: IndexTTS — Controllable TTS with Pinyin/Tone Control (Simplified)

Reference:
    IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot
    Text-to-Speech System (Xiaomi, 2025)

Core Idea — Controllability:
    Modern TTS isn't just about quality. Industrial systems need:
    1. Pinyin control:  explicitly specify pronunciation (解决多音字问题)
    2. Tone control:   override the default tone of a syllable
    3. Duration control: stretch or compress specific syllables
    4. Emotion control:  modulate prosody without retraining

    IndexTTS achieves this by *disentangling* linguistic content from
    acoustic features. The pinyin+tone sequence is an explicit, editable
    intermediate representation between text and audio.

    Text → G2P (grapheme-to-phoneme) → Pinyin + Tones → Acoustic Model → Mel → Vocoder
                                         ↑
                                    user can override here

This Implementation:
    - PinyinEmbedding: embed (initial, final, tone) tuples
    - Duration Predictor: predict frames per syllable
    - Length Regulator: expand syllable-level to frame-level
    - Flow-based acoustic decoder: syllable features → mel
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------
# Pinyin Embedding: (initial, final, tone) → vector
# --------------------------------------------------------

class PinyinEmbedding(nn.Module):
    """
    Chinese pinyin has three components:
        - Initial (声母): b, p, m, f, ...  (21 standard initials + null)
        - Final   (韵母): a, o, e, i, u, ... (35+ finals)
        - Tone    (声调): 1-5 (5 = neutral tone / 轻声)

    Example: "ma3" (马, horse) → initial="m", final="a", tone=3

    We embed each component separately and sum them.
    This allows the model to learn that "ma1" (妈) and "ma3" (马)
    share phonetic content but differ in tone.
    """

    def __init__(self, n_initials=25, n_finals=40, n_tones=6, embed_dim=256):
        super().__init__()
        self.initial_emb = nn.Embedding(n_initials, embed_dim)
        self.final_emb = nn.Embedding(n_finals, embed_dim)
        self.tone_emb = nn.Embedding(n_tones, embed_dim)
        self.proj = nn.Linear(embed_dim * 3, embed_dim)

    def forward(self, initials, finals, tones):
        """
        Args:
            initials: (B, N_syl)  int tensor, initial ids
            finals:   (B, N_syl)  int tensor, final ids
            tones:    (B, N_syl)  int tensor, tone ids (0-5)

        Returns:
            pinyin_emb: (B, N_syl, embed_dim)
        """
        e_i = self.initial_emb(initials)   # (B, N, D)
        e_f = self.final_emb(finals)       # (B, N, D)
        e_t = self.tone_emb(tones)         # (B, N, D)
        combined = torch.cat([e_i, e_f, e_t], dim=-1)  # (B, N, 3D)
        return self.proj(combined)         # (B, N, D)


# --------------------------------------------------------
# Duration Predictor: syllable → number of mel frames
# --------------------------------------------------------

class DurationPredictor(nn.Module):
    """
    Predict how many mel frames each syllable should occupy.

    This is crucial for natural rhythm. Without it, every syllable
    would get the same duration → robotic speech.

    Architecture: Conv stack → linear → softplus (duration must be > 0).

    During training: use ground-truth durations (from forced alignment).
    During inference: use predicted durations.
    """

    def __init__(self, input_dim=256, hidden_dim=256, n_layers=2):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Conv1d(d, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            d = hidden_dim
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        """
        Args:
            x:    (B, N_syl, input_dim) syllable features
            mask: (B, N_syl) True=padding

        Returns:
            durations: (B, N_syl) predicted frames per syllable (float, >0)
        """
        h = x.transpose(1, 2)        # (B, D, N)
        h = self.net(h)              # (B, hidden, N)
        h = h.transpose(1, 2)        # (B, N, hidden)
        d = self.proj(h).squeeze(-1) # (B, N)
        d = F.softplus(d)            # ensure positive

        if mask is not None:
            d = d.masked_fill(mask, 0.0)
        return d


# --------------------------------------------------------
# Length Regulator: expand syllable-level → frame-level
# --------------------------------------------------------

def length_regulate(x, durations):
    """
    Expand syllable features to frame-level by repeating each syllable
    according to its duration.

    Example:
        x = [A, B, C]         (3 syllables)
        durations = [2, 3, 1] → [A, A, B, B, B, C]  (6 frames)

    Args:
        x:         (B, N_syl, D)
        durations: (B, N_syl) int or float (rounded)

    Returns:
        expanded:  (B, T_total, D)
        total_lens: (B,) total frames per sample
    """
    B, N, D = x.shape
    dur_int = durations.round().long()
    total_lens = dur_int.sum(dim=1)  # (B,)
    T_max = total_lens.max().item()

    expanded = torch.zeros(B, T_max, D, device=x.device, dtype=x.dtype)
    for b in range(B):
        pos = 0
        for n in range(N):
            d = dur_int[b, n].item()
            if d > 0 and pos + d <= T_max:
                expanded[b, pos:pos + d, :] = x[b, n, :].unsqueeze(0).expand(d, -1)
                pos += d
    return expanded, total_lens


# --------------------------------------------------------
# IndexTTS: Full Model
# --------------------------------------------------------

class IndexTTS(nn.Module):
    """
    IndexTTS simplified: controllable TTS with pinyin input.

    Pipeline:
        Pinyin (initials, finals, tones)
            ↓ PinyinEmbedding
        Syllable features (B, N_syl, D)
            ↓ DurationPredictor → durations
            ↓ LengthRegulator (expand to frames)
        Frame features (B, T, D)
            ↓ Transformer decoder
        Mel (B, T, mel_dim)

    Control Points:
        - Change tones tensor → different pronunciation
        - Scale durations → faster/slower speech
        - Replace specific syllables → fix multi-pronunciation chars
    """

    def __init__(
        self,
        mel_dim=80,
        pinyin_dim=256,
        dim=256,
        heads=4,
        n_layers=4,
    ):
        super().__init__()
        self.mel_dim = mel_dim

        # Pinyin embedding
        self.pinyin_emb = PinyinEmbedding(embed_dim=pinyin_dim)

        # Duration predictor
        self.dur_pred = DurationPredictor(input_dim=pinyin_dim, hidden_dim=dim)

        # Syllable → frame feature transformer
        self.pre_net = nn.Sequential(
            nn.Linear(pinyin_dim, dim),
            nn.ReLU(),
        )

        # Frame-level decoder (bidirectional, since all frames are known after LR)
        from f5_tts import TransformerBlock  # reuse
        self.decoder = nn.ModuleList([
            TransformerBlock(dim, heads=heads, dropout=0.1)
            for _ in range(n_layers)
        ])

        self.out_proj = nn.Linear(dim, mel_dim)

    def forward(
        self,
        initials, finals, tones,
        syl_mask,
        durations=None,
        dur_scale=1.0,
    ):
        """
        Args:
            initials:   (B, N_syl) initial ids
            finals:     (B, N_syl) final ids
            tones:      (B, N_syl) tone ids
            syl_mask:   (B, N_syl) True=padding
            durations:  (B, N_syl) ground truth durations (training)
            dur_scale:  float, scale factor for inference speed control

        Returns:
            mel:         (B, T, mel_dim) predicted mel
            dur_pred:    (B, N_syl) predicted durations
            frame_lens:  (B,) total frames per sample
        """
        # 1. Embed pinyin
        syl_emb = self.pinyin_emb(initials, finals, tones)  # (B, N, D)

        # 2. Predict durations (always, for loss)
        dur_pred = self.dur_pred(syl_emb, mask=syl_mask)

        # 3. Use GT durations in training, predicted in inference
        dur_used = durations if durations is not None else dur_pred * dur_scale

        # 4. Length regulate: syllable → frame
        h = self.pre_net(syl_emb)  # (B, N, dim)
        h_frames, frame_lens = length_regulate(h, dur_used)  # (B, T, dim)

        # 5. Transformer decoder
        for layer in self.decoder:
            h_frames = layer(h_frames)

        # 6. Project to mel
        mel = self.out_proj(h_frames)  # (B, T, mel_dim)
        return mel, dur_pred, frame_lens

    def loss(
        self,
        initials, finals, tones,
        syl_mask,
        gt_durations,
        gt_mel,
    ):
        """
        Training loss = Mel MSE + Duration MSE.
        """
        mel_pred, dur_pred, frame_lens = self.forward(
            initials, finals, tones, syl_mask,
            durations=gt_durations,
        )

        # Mel loss (truncate to same length)
        T = min(mel_pred.size(1), gt_mel.size(1))
        mel_loss = F.mse_loss(mel_pred[:, :T, :], gt_mel[:, :T, :])

        # Duration loss (log domain, more stable)
        dur_loss = F.mse_loss(
            torch.log(dur_pred + 1e-6),
            torch.log(gt_durations.float() + 1e-6),
        )

        return mel_loss + 0.5 * dur_loss, mel_loss, dur_loss


# --------------------------------------------------------
# Quick Sanity Check
# --------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    mel_dim, pinyin_dim, dim = 80, 128, 128

    B, N_syl = 2, 8  # 8 syllables per sentence (e.g., "ni3 hao3 shi4 jie4 ...")

    model = IndexTTS(
        mel_dim=mel_dim, pinyin_dim=pinyin_dim, dim=dim,
        heads=4, n_layers=2,
    )

    # Dummy pinyin: "ni3 hao3 ma5" = (n, i, 3), (h, ao, 3), (m, a, 5)
    initials = torch.randint(0, 21, (B, N_syl))
    finals = torch.randint(0, 35, (B, N_syl))
    tones = torch.randint(1, 5, (B, N_syl))  # tones 1-4
    syl_mask = torch.zeros(B, N_syl, dtype=torch.bool)

    # Ground truth durations: 10-30 frames per syllable
    gt_dur = torch.randint(10, 30, (B, N_syl)).float()
    T_total = int(gt_dur.sum(dim=1).max().item())
    gt_mel = torch.randn(B, T_total, mel_dim)

    # Training
    total_loss, mel_loss, dur_loss = model.loss(
        initials, finals, tones, syl_mask, gt_dur, gt_mel
    )
    print(f"[Train] Total: {total_loss.item():.4f}  "
          f"Mel: {mel_loss.item():.4f}  Dur: {dur_loss.item():.4f}")

    # Inference: change tone of first syllable (e.g., ma1 → ma3)
    model.eval()
    with torch.no_grad():
        # Default tones
        mel_default, _, _ = model(initials, finals, tones, syl_mask)
        # Override tone of syllable 0 to tone 1 (妈)
        tones_modified = tones.clone()
        tones_modified[:, 0] = 1
        mel_modified, _, _ = model(initials, finals, tones_modified, syl_mask)

    print(f"[Infer] Default mel shape: {tuple(mel_default.shape)}")
    print(f"[Infer] Modified mel shape: {tuple(mel_modified.shape)}")
    diff = (mel_default - mel_modified).abs().mean().item()
    print(f"[Control] Tone change causes mel diff: {diff:.4f}")
    print("[OK] IndexTTS sanity check passed.")
