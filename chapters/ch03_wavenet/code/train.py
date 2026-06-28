"""
Ch03: WaveNet Training Script

Usage:
    python train.py \
        --data-dir ../../../data/processed \
        --epochs 20 \
        --batch-size 4

Requires:
    - data/processed/train.list      (manifest)
    - data/processed/wavs/*.wav      (audio files)

Train WaveNet vocoder: learns to generate waveform conditioned on Mel spectrogram.
Loss: cross-entropy over mu-law quantized waveform classes (256 classes).
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import WaveNet, mu_law_encode


# --------------------------------------------------------
# Audio / Mel helpers
# --------------------------------------------------------

def load_audio(path, sr=16000):
    """Load audio, convert to mono, resample to target sr."""
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def compute_mel(wave, sr=16000, n_fft=1024, hop_length=256, n_mels=80):
    """Compute log-mel spectrogram via librosa.

    Returns:
        mel: (n_mels, T) numpy array
    """
    mel = librosa.feature.melspectrogram(
        y=wave, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )
    return np.log(mel + 1e-5)


# --------------------------------------------------------
# Dataset
# --------------------------------------------------------

class WaveNetDataset(Dataset):
    """
    WaveNet 训练数据集。

    每条数据：
        - mel:    (n_mels, segment_length // hop_length)  — 条件
        - wav:    (segment_length,)                       — 目标 (mu-law 编码)

    从完整音频中随机裁剪 segment_length 长度的片段。
    """

    def __init__(
        self,
        data_dir,
        segment_length=8192,
        sample_rate=16000,
        max_duration_sec=15,
        hop_length=256,
        n_mels=80,
    ):
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Load manifest
        manifest = self.data_dir / "train.list"
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")

        self.wav_paths = []
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 4:
                    wav_path = self.data_dir / parts[0]
                    if wav_path.exists():
                        try:
                            info = sf.info(str(wav_path))
                            if info.duration <= max_duration_sec:
                                self.wav_paths.append(wav_path)
                        except Exception:
                            continue

        print(f"[dataset] Loaded {len(self.wav_paths)} audio files")

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav = load_audio(str(self.wav_paths[idx]), sr=self.sample_rate)

        # Random crop
        if len(wav) > self.segment_length:
            start = np.random.randint(0, len(wav) - self.segment_length)
            wav = wav[start : start + self.segment_length]
        elif len(wav) < self.segment_length:
            wav = np.pad(wav, (0, self.segment_length - len(wav)))

        # Compute mel spectrogram
        mel = compute_mel(wav, sr=self.sample_rate, hop_length=self.hop_length,
                          n_mels=self.n_mels)  # (n_mels, T_mel)

        # mu-law encode waveform
        wav_tensor = torch.from_numpy(wav).float()
        wav_mu = mu_law_encode(wav_tensor)  # (T_wav,)

        return {
            "mel": torch.from_numpy(mel).float(),     # (n_mels, T_mel)
            "wav": wav_mu.long(),                       # (T_wav,)
        }


def collate_fn(batch):
    """Pad to same length within batch."""
    max_mel_t = max(b["mel"].shape[1] for b in batch)
    max_wav_t = max(b["wav"].shape[0] for b in batch)
    n_mels = batch[0]["mel"].shape[0]
    B = len(batch)

    mels = torch.zeros(B, n_mels, max_mel_t)
    wavs = torch.full((B, max_wav_t), 128, dtype=torch.long)  # 128 = mu-law silence

    for i, b in enumerate(batch):
        mt = b["mel"].shape[1]
        wt = b["wav"].shape[0]
        mels[i, :, :mt] = b["mel"]
        wavs[i, :wt] = b["wav"]

    return {"mel": mels, "wav": wavs}


# --------------------------------------------------------
# Training
# --------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None

    for batch in tqdm(loader, desc="  train"):
        mel = batch["mel"].to(device)
        wav = batch["wav"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast("cuda"):
                logits = model(mel, wav)               # (B, 256, T)
                loss = criterion(logits, wav)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(mel, wav)
            loss = criterion(logits, wav)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train WaveNet vocoder")
    parser.add_argument("--data-dir", type=str, default="../../../data/processed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, default="../checkpoints")
    parser.add_argument("--segment-length", type=int, default=8192,
                        help="Training segment length in samples (default 8192 = 0.5s)")
    parser.add_argument("--res-channels", type=int, default=64)
    parser.add_argument("--skip-channels", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=10)
    parser.add_argument("--n-cycles", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Dataset
    dataset = WaveNetDataset(
        args.data_dir,
        segment_length=args.segment_length,
        sample_rate=16000,
    )
    if len(dataset) == 0:
        print("[error] No data found. Run data/download_neko_1k.py first.")
        return

    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )

    # Model
    model = WaveNet(
        n_mels=80,
        res_channels=args.res_channels,
        skip_channels=args.skip_channels,
        n_blocks=args.n_blocks,
        n_cycles=args.n_cycles,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # AMP
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    if scaler:
        print("[amp] Mixed precision enabled")

    # Train
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    loss_log_path = save_dir / "wavenet_loss_log.csv"
    with open(loss_log_path, "w") as f:
        f.write("epoch,loss\n")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, loader, optimizer, criterion, device, scaler)
        print(f"[epoch {epoch}/{args.epochs}] loss: {avg_loss:.4f}")

        with open(loss_log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f}\n")

        if epoch % 5 == 0:
            ckpt_path = save_dir / f"wavenet_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "res_channels": args.res_channels,
                    "skip_channels": args.skip_channels,
                    "n_blocks": args.n_blocks,
                    "n_cycles": args.n_cycles,
                },
            }, ckpt_path)
            print(f"[save] {ckpt_path}")

    # Final
    final_path = save_dir / "wavenet_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "res_channels": args.res_channels,
            "skip_channels": args.skip_channels,
            "n_blocks": args.n_blocks,
            "n_cycles": args.n_cycles,
        },
    }, final_path)
    print(f"[done] Final model saved: {final_path}")


if __name__ == "__main__":
    main()
