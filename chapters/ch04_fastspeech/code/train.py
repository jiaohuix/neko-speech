"""
Ch04: FastSpeech2 Training Script

Usage:
    python train.py --data-dir ../../data/processed --epochs 50 --batch-size 8

Requires:
    - data/processed/train.list   (wav_path|speaker|language|text)
    - data/processed/wavs/*.wav
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import FastSpeech2


# --------------------------------------------------------
# Text Tokenizer (same as Ch02)
# --------------------------------------------------------

class CharTokenizer:
    """Character-level tokenizer. 0 = <pad>."""

    def __init__(self, chars=None):
        if chars is None:
            chars = (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
                "，。！？、；：""''（）【】《》"
                " "
            )
        self.chars = chars
        self.vocab = {c: i + 1 for i, c in enumerate(chars)}
        self.pad_id = 0
        self.vocab_size = len(self.vocab) + 1

    @classmethod
    def from_texts(cls, texts):
        chars = "".join(sorted(set(c for t in texts for c in t)))
        return cls(chars)

    def encode(self, text):
        return [self.vocab.get(c, self.pad_id) for c in text]


# --------------------------------------------------------
# Audio Processing
# --------------------------------------------------------

def compute_mel(wave, sr=16000, n_fft=1024, hop_length=256, n_mels=80):
    """Log-mel spectrogram via NumPy (matching Ch01/Ch02)."""
    window = np.hamming(n_fft)
    wave = np.pad(wave, (n_fft // 2, n_fft // 2), mode="constant")
    n_frames = 1 + (len(wave) - n_fft) // hop_length

    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        frame = wave[i * hop_length: i * hop_length + n_fft]
        if len(frame) == n_fft:
            stft[:, i] = np.fft.rfft(frame * window, n=n_fft)

    magnitude = np.abs(stft)

    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700.0)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595.0) - 1)

    fft_freqs = np.linspace(0, sr // 2, n_fft // 2 + 1)
    mel_pts = np.linspace(hz_to_mel(0), hz_to_mel(sr // 2), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        l, c, r = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (fft_freqs - l) / (c - l)
        dn = (r - fft_freqs) / (r - c)
        fb[i] = np.maximum(0, np.minimum(up, dn))

    return np.log(fb @ magnitude + 1e-10).T  # (T, n_mels)


# --------------------------------------------------------
# Duration Estimation
# --------------------------------------------------------

def estimate_uniform_durations(text_len, mel_len):
    """
    Distribute mel frames evenly across text tokens.

    This is the simplest duration estimation. Production FastSpeech2
    uses alignments from a trained teacher model (e.g. Tacotron2) or
    Montreal Forced Aligner (MFA) for more accurate durations.
    """
    if text_len == 0:
        return []
    base = mel_len // text_len
    rem = mel_len - base * text_len
    durs = [base] * text_len
    for i in range(rem):
        durs[i] += 1
    return durs


# --------------------------------------------------------
# Dataset
# --------------------------------------------------------

class FastSpeechDataset(Dataset):
    """
    FastSpeech2 dataset.

    For each sample, provides:
      - text:     tokenized text
      - mel:      log-mel spectrogram  (T_mel, 80)
      - durations: estimated frames per token (uniform)
      - pitch:    spectral centroid per token (from mel)
      - energy:   mean magnitude per token (from mel)
    """

    def __init__(self, data_dir, max_duration_sec=15, target_sr=16000):
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr
        self.max_duration_sec = max_duration_sec

        manifest = self.data_dir / "train.list"
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")

        raw = []
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 4:
                    wav_path = self.data_dir / parts[0]
                    if wav_path.exists():
                        raw.append({"wav": wav_path, "text": parts[3]})

        # Filter by duration
        self.samples = []
        for s in raw:
            try:
                if sf.info(s["wav"]).duration <= max_duration_sec:
                    self.samples.append(s)
            except Exception:
                continue

        # Tokenizer
        self.tokenizer = CharTokenizer.from_texts([s["text"] for s in self.samples])
        print(f"[dataset] {len(self.samples)} samples, vocab={self.tokenizer.vocab_size}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load & resample
        wave, sr = sf.read(str(s["wav"]))
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        if sr != self.target_sr:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.target_sr)

        # Mel
        mel = compute_mel(wave, sr=self.target_sr)  # (T_mel, 80)

        # Tokenize
        text_ids = self.tokenizer.encode(s["text"])
        text_len = len(text_ids)
        mel_len = len(mel)

        # Uniform durations
        durs = estimate_uniform_durations(text_len, mel_len)

        # Pitch & energy from mel (phoneme-level averages)
        mel_t = torch.FloatTensor(mel)
        pitch = torch.zeros(text_len)
        energy = torch.zeros(text_len)
        freqs = torch.arange(mel_t.shape[-1], dtype=mel_t.dtype)
        pos = 0
        for j in range(text_len):
            d = durs[j] if j < len(durs) else 0
            if d > 0 and pos + d <= mel_len:
                seg = mel_t[pos: pos + d]           # (d, 80)
                mag = seg.exp().mean(dim=0)          # (80,)
                total = mag.sum().clamp(min=1e-6)
                pitch[j] = (mag * freqs).sum() / total  # spectral centroid
                energy[j] = seg.mean()                # loudness proxy
                pos += d
            else:
                pos += max(d, 0)

        return {
            "text": torch.LongTensor(text_ids),
            "mel": mel_t,
            "durations": torch.LongTensor(durs),
            "pitch": pitch,
            "energy": energy,
        }


def collate_fn(batch):
    """Pad variable-length sequences to batch max."""
    B = len(batch)
    max_t = max(b["text"].shape[0] for b in batch)
    max_m = max(b["mel"].shape[0] for b in batch)

    text = torch.zeros(B, max_t, dtype=torch.long)
    mel = torch.zeros(B, max_m, 80)
    durs = torch.zeros(B, max_t, dtype=torch.long)
    pitch = torch.zeros(B, max_t)
    energy = torch.zeros(B, max_t)
    text_lens = torch.zeros(B, dtype=torch.long)
    mel_lens = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        tl = b["text"].shape[0]
        ml = b["mel"].shape[0]
        text[i, :tl] = b["text"]
        mel[i, :ml] = b["mel"]
        durs[i, :tl] = b["durations"]
        pitch[i, :tl] = b["pitch"]
        energy[i, :tl] = b["energy"]
        text_lens[i] = tl
        mel_lens[i] = ml

    return dict(text=text, mel=mel, durations=durs,
                pitch=pitch, energy=energy,
                text_lens=text_lens, mel_lens=mel_lens)


# --------------------------------------------------------
# Loss
# --------------------------------------------------------

class FastSpeech2Loss(nn.Module):
    """
    FastSpeech2 multi-task loss:
        L = L_mel + L_dur + L_pitch + L_energy

    Duration loss is in log-space (durations are highly skewed).
    """

    def forward(self, mel_pred, dur_pred, pitch_pred, energy_pred,
                mel_tgt, dur_tgt, pitch_tgt, energy_tgt,
                text_lens, mel_lens):
        B, T_text = dur_pred.shape
        t_mask = torch.arange(T_text, device=dur_pred.device).unsqueeze(0) < text_lens.unsqueeze(1)
        T_mel = mel_pred.shape[-1]
        m_mask = torch.arange(T_mel, device=mel_pred.device).unsqueeze(0) < mel_lens.unsqueeze(1)

        # Mel: pred is (B, 80, T), target is (B, T, 80) -- transpose target
        mel_loss = nn.functional.mse_loss(
            mel_pred, mel_tgt.transpose(1, 2), reduction="none"
        ).mean(dim=1)  # (B, T_mel)
        mel_loss = (mel_loss * m_mask).sum() / m_mask.sum().clamp(min=1)

        # Duration (log-space)
        log_dur_tgt = torch.log(dur_tgt.float() + 1)
        dur_loss = nn.functional.mse_loss(dur_pred, log_dur_tgt, reduction="none")
        dur_loss = (dur_loss * t_mask).sum() / t_mask.sum().clamp(min=1)

        # Pitch & energy
        pitch_loss = nn.functional.mse_loss(pitch_pred, pitch_tgt, reduction="none")
        pitch_loss = (pitch_loss * t_mask).sum() / t_mask.sum().clamp(min=1)
        energy_loss = nn.functional.mse_loss(energy_pred, energy_tgt, reduction="none")
        energy_loss = (energy_loss * t_mask).sum() / t_mask.sum().clamp(min=1)

        total = mel_loss + dur_loss + pitch_loss + energy_loss
        return total, {
            "mel": mel_loss.item(),
            "dur": dur_loss.item(),
            "pitch": pitch_loss.item(),
            "energy": energy_loss.item(),
        }


# --------------------------------------------------------
# Training
# --------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="  train"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        mel_pred, dur_pred, pitch_pred, energy_pred = model(
            batch["text"], batch["text_lens"],
            batch["mel"], batch["mel_lens"],
            batch["durations"], batch["pitch"], batch["energy"],
        )
        loss, _ = criterion(
            mel_pred, dur_pred, pitch_pred, energy_pred,
            batch["mel"], batch["durations"],
            batch["pitch"], batch["energy"],
            batch["text_lens"], batch["mel_lens"],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train FastSpeech2")
    parser.add_argument("--data-dir", type=str, default="../../../data/processed")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, default="../checkpoints")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Dataset
    dataset = FastSpeechDataset(args.data_dir, max_duration_sec=15, target_sr=16000)
    if len(dataset) == 0:
        print("[error] No data. Run data/download_neko_1k.py first.")
        return
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate_fn, num_workers=0)

    # Model
    tokenizer = dataset.tokenizer
    model = FastSpeech2(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} parameters")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = FastSpeech2Loss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Save tokenizer
    with open(save_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump({"chars": tokenizer.chars}, f, ensure_ascii=False)

    # Loss log
    log_path = save_dir / "loss_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,loss,mel,dur,pitch,energy\n")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()

        # Detailed loss for last batch
        model.eval()
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in next(iter(loader)).items()}
            mp, dp, pp, ep = model(
                batch["text"], batch["text_lens"],
                batch["mel"], batch["mel_lens"],
                batch["durations"], batch["pitch"], batch["energy"],
            )
            _, details = criterion(
                mp, dp, pp, ep,
                batch["mel"], batch["durations"],
                batch["pitch"], batch["energy"],
                batch["text_lens"], batch["mel_lens"],
            )
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch:3d}/{args.epochs}] "
              f"loss={avg_loss:.4f}  "
              f"mel={details['mel']:.4f} dur={details['dur']:.4f} "
              f"pit={details['pitch']:.4f} eng={details['energy']:.4f}  "
              f"lr={lr_now:.1e}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},"
                    f"{details['mel']:.6f},{details['dur']:.6f},"
                    f"{details['pitch']:.6f},{details['energy']:.6f}\n")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "tokenizer_chars": tokenizer.chars,
            }, save_dir / f"fs2_epoch_{epoch}.pt")
            print(f"  -> saved checkpoint epoch {epoch}")

    # Final
    torch.save({
        "model_state_dict": model.state_dict(),
        "tokenizer_chars": tokenizer.chars,
    }, save_dir / "fs2_final.pt")
    print(f"[done] Final model: {save_dir / 'fs2_final.pt'}")


if __name__ == "__main__":
    main()
