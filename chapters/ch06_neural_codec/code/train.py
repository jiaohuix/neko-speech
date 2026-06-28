"""
Ch06: EnCodec Mini Training Script

Usage:
    # 用真实数据训练 (推荐)
    python train.py \
        --data-dir ../../../data/processed \
        --epochs 50 \
        --batch-size 4 \
        --lr 1e-4 \
        --ckpt-dir ../checkpoints

    # 快速测试 (合成数据)
    python train.py --synthetic --epochs 5

训练目标:
    让 codec 学会 "压缩→重建" 音频。
    L = L1(wav, wav_hat) + MR-STFT(wav, wav_hat) + VQ loss

训练完成后:
    - checkpoint 保存到 checkpoints/
    - loss 曲线保存到 outputs/
    - 用 test_codec.py 测试重建质量
"""

import argparse
import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from codec import EnCodecMini, codec_loss


# ============================================================
# Dataset
# ============================================================

class AudioDataset(Dataset):
    """
    简单音频数据集.

    从 train.list 读取音频文件，裁剪到固定长度。
    格式: wavs/000001.wav|speaker|lang|text
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 24000,
        clip_seconds: float = 3.0,
        hop_length: int = 320,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.clip_len = int(sample_rate * clip_seconds)
        # 确保 clip_len 是 hop_length 的整数倍
        self.clip_len = (self.clip_len // hop_length) * hop_length

        # 读取 manifest
        manifest_path = self.data_dir / "train.list"
        self.audio_files = []
        with open(manifest_path) as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 1:
                    audio_path = self.data_dir / parts[0]
                    if audio_path.exists():
                        self.audio_files.append(str(audio_path))

        print(f"Loaded {len(self.audio_files)} audio files from {data_dir}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        wav, sr = sf.read(audio_path, dtype="float32")

        # 确保单声道
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        # 随机裁剪
        if len(wav) > self.clip_len:
            start = random.randint(0, len(wav) - self.clip_len)
            wav = wav[start : start + self.clip_len]
        elif len(wav) < self.clip_len:
            # 循环补齐
            repeats = math.ceil(self.clip_len / len(wav))
            wav = np.tile(wav, repeats)[: self.clip_len]

        # 归一化
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak * 0.95

        return torch.FloatTensor(wav).unsqueeze(0)  # [1, T]


class SyntheticDataset(Dataset):
    """
    合成音频数据集 (用于快速验证).

    生成不同频率的正弦波 + 谐波，测试 codec 能否学到基本压缩。
    不能代表真实训练效果，但可以验证代码正确性。
    """

    def __init__(
        self,
        n_samples: int = 200,
        sample_rate: int = 24000,
        clip_seconds: float = 2.0,
        hop_length: int = 320,
    ):
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.clip_len = int(sample_rate * clip_seconds)
        self.clip_len = (self.clip_len // hop_length) * hop_length

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t = np.arange(self.clip_len) / self.sample_rate

        # 随机基频 + 谐波
        f0 = random.uniform(100, 500)
        wav = np.zeros_like(t)
        for harmonic in range(1, 6):
            amp = random.uniform(0.1, 0.5) / harmonic
            wav += amp * np.sin(2 * np.pi * f0 * harmonic * t)

        # 加一点噪声
        wav += np.random.randn(len(wav)) * 0.01

        # 归一化
        wav = wav / np.abs(wav).max() * 0.9

        return torch.FloatTensor(wav).unsqueeze(0)


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Model ----
    model = EnCodecMini(
        base_dim=args.base_dim,
        latent_dim=args.latent_dim,
        num_codebooks=args.num_codebooks,
        num_codes=args.num_codes,
    ).to(device)
    print(f"Model: {model.n_params():,} parameters")
    print(f"Hop length: {model.hop_length}")

    # ---- Data ----
    if args.synthetic:
        dataset = SyntheticDataset(
            n_samples=200,
            sample_rate=args.sample_rate,
            clip_seconds=args.clip_seconds,
            hop_length=model.hop_length,
        )
    else:
        dataset = AudioDataset(
            data_dir=args.data_dir,
            sample_rate=args.sample_rate,
            clip_seconds=args.clip_seconds,
            hop_length=model.hop_length,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Optimizer ----
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # ---- Checkpoint dir ----
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSV log ----
    log_path = output_dir / "loss_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "step", "total", "l1", "spec", "vq"])

    # ---- Training loop ----
    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {"total": 0, "l1": 0, "spec": 0, "vq": 0}
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, wav in enumerate(pbar):
            wav = wav.to(device)

            # Forward
            out = model(wav)
            total_loss, loss_dict = codec_loss(
                wav, out["wav_hat"], out["vq_loss"],
            )

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Log
            global_step += 1
            n_batches += 1
            for k in epoch_losses:
                epoch_losses[k] += loss_dict.get(k, total_loss.item() if k == "total" else 0)

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "l1": f"{loss_dict['l1']:.4f}",
                "spec": f"{loss_dict['spec']:.4f}",
                "vq": f"{loss_dict['vq']:.4f}",
            })

            # Log to CSV every N steps
            if global_step % 50 == 0:
                log_writer.writerow([
                    epoch, global_step,
                    f"{total_loss.item():.4f}",
                    f"{loss_dict['l1']:.4f}",
                    f"{loss_dict['spec']:.4f}",
                    f"{loss_dict['vq']:.4f}",
                ])
                log_file.flush()

        # Epoch summary
        scheduler.step()
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        print(f"Epoch {epoch}: "
              f"total={epoch_losses['total']:.4f} "
              f"l1={epoch_losses['l1']:.4f} "
              f"spec={epoch_losses['spec']:.4f} "
              f"vq={epoch_losses['vq']:.4f}")

        # Save checkpoint
        if epoch_losses["total"] < best_loss:
            best_loss = epoch_losses["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": {
                    "base_dim": args.base_dim,
                    "latent_dim": args.latent_dim,
                    "num_codebooks": args.num_codebooks,
                    "num_codes": args.num_codes,
                    "sample_rate": args.sample_rate,
                },
            }, ckpt_dir / "codec_best.pt")

        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": {
                    "base_dim": args.base_dim,
                    "latent_dim": args.latent_dim,
                    "num_codebooks": args.num_codebooks,
                    "num_codes": args.num_codes,
                    "sample_rate": args.sample_rate,
                },
            }, ckpt_dir / f"codec_epoch_{epoch}.pt")

        # Codebook usage stats
        _print_codebook_stats(model, loader, device)

    log_file.close()

    # Save final
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "config": {
            "base_dim": args.base_dim,
            "latent_dim": args.latent_dim,
            "num_codebooks": args.num_codebooks,
            "num_codes": args.num_codes,
            "sample_rate": args.sample_rate,
        },
    }, ckpt_dir / "codec_final.pt")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}")

    # Plot loss curve
    _plot_loss_curve(log_path, output_dir / "loss_curve.png")


def _print_codebook_stats(model, loader, device):
    """打印码本使用率统计."""
    model.eval()
    all_tokens = []
    with torch.no_grad():
        for wav in loader:
            wav = wav.to(device)
            out = model(wav)
            all_tokens.append(out["tokens"])
            if len(all_tokens) >= 5:  # 只看前 5 个 batch
                break

    tokens = torch.cat(all_tokens, dim=0)  # [N, K, T]
    num_codes = model.quantizer.quantizers[0].num_codes
    print("  Codebook usage:")
    for k in range(tokens.shape[1]):
        unique = tokens[:, k, :].unique().numel()
        print(f"    Layer {k}: {unique}/{num_codes} codes used "
              f"({unique / num_codes * 100:.1f}%)")
    model.train()


def _plot_loss_curve(log_path: Path, output_path: Path):
    """绘制训练损失曲线."""
    try:
        import matplotlib.pyplot as plt

        epochs, totals, l1s, specs, vqs = [], [], [], [], []
        with open(log_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                epochs.append(int(row[0]))
                totals.append(float(row[2]))
                l1s.append(float(row[3]))
                specs.append(float(row[4]))
                vqs.append(float(row[5]))

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(epochs, totals)
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Step")

        axes[0, 1].plot(epochs, l1s)
        axes[0, 1].set_title("L1 Loss (waveform)")
        axes[0, 1].set_xlabel("Step")

        axes[1, 0].plot(epochs, specs)
        axes[1, 0].set_title("MR-STFT Loss (spectral)")
        axes[1, 0].set_xlabel("Step")

        axes[1, 1].plot(epochs, vqs)
        axes[1, 1].set_title("VQ Loss (codebook)")
        axes[1, 1].set_xlabel("Step")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Loss curve saved to {output_path}")
    except ImportError:
        print("matplotlib not available, skipping loss curve plot")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train EnCodec Mini")

    # Data
    parser.add_argument("--data-dir", type=str,
                        default="../../../data/processed",
                        help="Path to data directory with train.list")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (for quick testing)")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=4)

    # Model
    parser.add_argument("--base-dim", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--num-codes", type=int, default=1024)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=10)

    # Output
    parser.add_argument("--ckpt-dir", type=str, default="../checkpoints")
    parser.add_argument("--output-dir", type=str, default="../outputs")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
