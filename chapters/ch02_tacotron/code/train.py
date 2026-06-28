"""
Ch02: Tacotron2 Training Script (Minimal)

Usage:
    python train.py \
        --data-dir ../../data/processed \
        --epochs 100 \
        --batch-size 8

Requires:
    - data/processed/train.list      (manifest)
    - data/processed/wavs/*.wav      (audio files)
"""

import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import Tacotron2


# --------------------------------------------------------
# Text Processing (Minimal)
# --------------------------------------------------------

class CharTokenizer:
    """Simple character-level tokenizer built from data."""

    def __init__(self, chars=None):
        if chars is None:
            # Default base chars
            chars = (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
                "，。！？、；：""''（）【】《》"
                " "
            )
        self.chars = chars
        self.vocab = {c: i + 1 for i, c in enumerate(self.chars)}  # 0 = pad
        self.pad_id = 0
        self.vocab_size = len(self.vocab) + 1

    @classmethod
    def from_texts(cls, texts):
        """Build tokenizer from a list of texts."""
        unique_chars = set()
        for t in texts:
            unique_chars.update(t)
        # Sort for determinism
        chars = "".join(sorted(unique_chars))
        return cls(chars)

    def encode(self, text):
        return [self.vocab.get(c, self.pad_id) for c in text]


# --------------------------------------------------------
# Audio Processing
# --------------------------------------------------------

def compute_mel(wave, sr=16000, n_fft=1024, hop_length=256, n_mels=80):
    """NumPy-based mel spectrogram (matching Ch01)."""
    # STFT
    window = np.hamming(n_fft)
    pad_len = n_fft // 2
    wave = np.pad(wave, (pad_len, pad_len), mode="constant")
    n_frames = 1 + (len(wave) - n_fft) // hop_length

    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        frame = wave[i * hop_length:i * hop_length + n_fft]
        if len(frame) == n_fft:
            stft_matrix[:, i] = np.fft.rfft(frame * window, n=n_fft)

    magnitude = np.abs(stft_matrix)

    # Mel filter bank
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700.0)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595.0) - 1)

    fft_freqs = np.linspace(0, sr // 2, n_fft // 2 + 1)
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sr // 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    mel_filter = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        up = (fft_freqs - left) / (center - left)
        down = (right - fft_freqs) / (right - center)
        mel_filter[i] = np.maximum(0, np.minimum(up, down))

    mel_spec = mel_filter @ magnitude
    log_mel = np.log(mel_spec + 1e-10)
    return log_mel.T  # (T, n_mels)


# --------------------------------------------------------
# Dataset
# --------------------------------------------------------

class NekoDataset(Dataset):
    def __init__(self, data_dir, max_text_len=100, max_mel_len=500, max_duration_sec=15, target_sr=16000):
        self.data_dir = Path(data_dir)
        self.max_text_len = max_text_len
        self.max_mel_len = max_mel_len
        self.max_duration_sec = max_duration_sec
        self.target_sr = target_sr

        # Load manifest
        manifest_path = self.data_dir / "train.list"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        raw_samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 4:
                    wav_rel, speaker, lang, text = parts[0], parts[1], parts[2], parts[3]
                    wav_path = self.data_dir / wav_rel
                    if wav_path.exists():
                        raw_samples.append({
                            "wav_path": wav_path,
                            "text": text,
                        })

        # Filter by duration (quick check via file size estimate or sf.info)
        self.samples = []
        for s in raw_samples:
            try:
                info = sf.info(s["wav_path"])
                if info.duration <= self.max_duration_sec:
                    self.samples.append(s)
            except Exception:
                continue

        # Build tokenizer from dataset texts
        all_texts = [s["text"] for s in self.samples]
        self.tokenizer = CharTokenizer.from_texts(all_texts)

        print(f"[dataset] Loaded {len(self.samples)} samples (filtered from {len(raw_samples)})")
        print(f"[dataset] Vocab size: {self.tokenizer.vocab_size}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        wave, sr = sf.read(sample["wav_path"])
        if wave.ndim > 1:
            wave = wave.mean(axis=1)

        # Resample to target_sr if needed
        if sr != self.target_sr:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.target_sr)

        # Compute mel at target sample rate
        mel = compute_mel(wave, sr=self.target_sr)  # (T, 80)

        # Truncate long mels
        if len(mel) > self.max_mel_len:
            mel = mel[:self.max_mel_len]

        # Tokenize text
        text_ids = self.tokenizer.encode(sample["text"])

        return {
            "text": torch.LongTensor(text_ids),
            "mel": torch.FloatTensor(mel),
        }


def collate_fn(batch):
    """Pad sequences to same length within batch."""
    texts = [b["text"] for b in batch]
    mels = [b["mel"] for b in batch]

    # Pad text
    text_lens = [len(t) for t in texts]
    max_text_len = max(text_lens)
    text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    for i, t in enumerate(texts):
        text_padded[i, :len(t)] = t

    # Pad mel
    mel_lens = [len(m) for m in mels]
    max_mel_len = max(mel_lens)
    mel_padded = torch.zeros(len(batch), max_mel_len, 80)
    for i, m in enumerate(mels):
        mel_padded[i, :len(m)] = m

    return {
        "text": text_padded,
        "mel": mel_padded,
        "text_lens": torch.LongTensor(text_lens),
        "mel_lens": torch.LongTensor(mel_lens),
    }


# --------------------------------------------------------
# Loss
# --------------------------------------------------------

class TacotronLoss(nn.Module):
    """
    Tacotron2 Loss:
        1. MSE on mel (before + after PostNet)
        2. BCE on stop token
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, mel_before, mel_after, stop_logits, mel_targets, mel_lens):
        # Mask padding: (B, T_mel)
        B, T_mel, _ = mel_targets.shape
        mask = torch.zeros(B, T_mel, device=mel_targets.device)
        for i, l in enumerate(mel_lens):
            mask[i, :l] = 1.0

        mel_before = mel_before * mask.unsqueeze(-1)
        mel_after = mel_after * mask.unsqueeze(-1)
        mel_targets = mel_targets * mask.unsqueeze(-1)

        # Mel losses
        loss_mel_before = self.mse(mel_before, mel_targets)
        loss_mel_after = self.mse(mel_after, mel_targets)

        # Stop token loss
        stop_targets = torch.zeros_like(stop_logits)
        for i, l in enumerate(mel_lens):
            if l < stop_logits.shape[1]:
                stop_targets[i, l - 1] = 1.0  # last frame = stop

        loss_stop = self.bce(stop_logits, stop_targets)

        return loss_mel_before + loss_mel_after + loss_stop


# --------------------------------------------------------
# Training
# --------------------------------------------------------

def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0

    use_amp = scaler is not None
    for batch in tqdm(dataloader, desc="Training"):
        text = batch["text"].to(device)
        mel = batch["mel"].to(device)
        mel_lens = batch["mel_lens"]

        optimizer.zero_grad()

        if use_amp:
            with autocast("cuda"):
                mel_before, mel_after, stop_logits = model(text, mel)
                loss = criterion(mel_before, mel_after, stop_logits, mel, mel_lens)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            mel_before, mel_after, stop_logits = model(text, mel)
            loss = criterion(mel_before, mel_after, stop_logits, mel, mel_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train Tacotron2")
    parser.add_argument("--data-dir", type=str, default="../../../data/processed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, default="../checkpoints")
    parser.add_argument("--max-duration", type=float, default=25,
                        help="Max audio duration in seconds")
    parser.add_argument("--max-mel-len", type=int, default=500,
                        help="Max mel frames per sample")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using {device}")

    # Disable cudnn to avoid FP16 conv compatibility issues on some GPUs
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    # Dataset
    dataset = NekoDataset(args.data_dir, max_duration_sec=args.max_duration,
                          max_mel_len=args.max_mel_len, target_sr=16000)
    if len(dataset) == 0:
        print("[error] No data found. Run data/prepare.py first.")
        return

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )

    # Model (use dataset's tokenizer vocab)
    tokenizer = dataset.tokenizer
    model = Tacotron2(
        vocab_size=tokenizer.vocab_size,
        mel_dim=80,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = TacotronLoss()

    # AMP
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    if scaler:
        print("[amp] Mixed precision enabled (FP16)")

    # Training loop
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Save tokenizer config
    tokenizer_path = save_dir / "tokenizer_config.json"
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump({"chars": tokenizer.chars}, f, ensure_ascii=False)
    print(f"[save] Tokenizer config: {tokenizer_path}")

    # Loss log
    loss_log_path = save_dir / "loss_log.csv"
    with open(loss_log_path, "w") as f:
        f.write("epoch,loss\n")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device, scaler)
        print(f"[epoch {epoch}/{args.epochs}] loss: {avg_loss:.4f}")

        with open(loss_log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f}\n")

        if epoch % 5 == 0:
            ckpt_path = save_dir / f"tacotron_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "tokenizer_chars": tokenizer.chars,
            }, ckpt_path)
            print(f"[save] Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = save_dir / "tacotron_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "tokenizer_chars": tokenizer.chars,
    }, final_path)
    print(f"[done] Final model saved: {final_path}")


if __name__ == "__main__":
    main()
