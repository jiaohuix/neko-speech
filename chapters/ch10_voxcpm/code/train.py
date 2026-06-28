"""
SimpleVoxCPM Training Script
============================

End-to-end training with conditional flow matching loss.

The AudioVAE is frozen (random init, not trained here — in a real pipeline
it would be pretrained separately). The TSLM, RALM, LocEnc, FSQ, and LocDiT
are trained jointly to minimize the flow matching MSE loss.

Usage:
    python train.py --epochs 10 --batch-size 4 --lr 1e-4 --audio-len 3200
"""

import argparse
import math
import time
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Allow running from the code/ directory or parent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SimpleVoxCPM


# ---------------------------------------------------------------------------
# Simulated dataset
# ---------------------------------------------------------------------------

class SimulatedTTSDataset(Dataset):
    """Generates random (text, audio) pairs for demonstration.

    In a real pipeline you'd replace this with a dataset that loads
    actual text transcriptions and 16 kHz waveforms.
    """

    def __init__(self, n_samples: int = 200, vocab_size: int = 256,
                 text_len: int = 16, audio_len: int = 3200):
        self.n_samples = n_samples
        self.vocab_size = vocab_size
        self.text_len = text_len
        self.audio_len = audio_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        text = torch.randint(0, self.vocab_size, (self.text_len,))
        # Simulate a waveform: a sum of sinusoids with random frequencies
        t = torch.linspace(0, self.audio_len / 16000, self.audio_len)
        freqs = torch.rand(3) * 400 + 100                   # 100-500 Hz
        audio = sum(torch.sin(2 * math.pi * f * t) for f in freqs)
        audio = audio / (audio.abs().max() + 1e-8) * 0.5     # normalize
        return text, audio


# ---------------------------------------------------------------------------
# Cosine annealing with warmup
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup + cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for batch_idx, (text, audio) in enumerate(loader):
        text = text.to(device)
        audio = audio.to(device)

        optimizer.zero_grad()
        loss = model(text, audio)
        loss.backward()
        # Gradient clipping (optional, helps with flow matching stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            avg_loss = total_loss / n_batches
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            print(f"  [Epoch {epoch}] Batch {batch_idx+1}/{len(loader)}  "
                  f"loss={avg_loss:.4f}  lr={lr:.2e}  ({elapsed:.1f}s)")

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train SimpleVoxCPM")
    parser.add_argument("--epochs", type=int, default=3,
                        help="number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="batch size (default: 2, fits in 12GB VRAM)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="peak learning rate (default: 1e-4)")
    parser.add_argument("--audio-len", type=int, default=3200,
                        help="audio length in samples (default: 3200 = 0.2s)")
    parser.add_argument("--text-len", type=int, default=16,
                        help="text token sequence length (default: 16)")
    parser.add_argument("--n-train", type=int, default=64,
                        help="number of training samples (default: 64)")
    parser.add_argument("--n-val", type=int, default=8,
                        help="number of validation samples (default: 8)")
    parser.add_argument("--warmup-steps", type=int, default=50,
                        help="warmup steps for cosine schedule (default: 50)")
    parser.add_argument("--save-path", type=str, default="checkpoints/voxcpm.pt",
                        help="checkpoint save path")
    parser.add_argument("--device", type=str, default=None,
                        help="device (default: auto-detect)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Audio length: {args.audio_len} samples ({args.audio_len/16000:.3f}s)")

    # --- Dataset ---
    train_ds = SimulatedTTSDataset(
        n_samples=args.n_train, text_len=args.text_len, audio_len=args.audio_len,
    )
    val_ds = SimulatedTTSDataset(
        n_samples=args.n_val, text_len=args.text_len, audio_len=args.audio_len,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # --- Model ---
    model = SimpleVoxCPM(
        vocab_size=256,
        encoder_dim=64,
        latent_dim=32,
        decoder_dim=256,
        patch_size=1,
        loc_enc_hidden=512,
        loc_enc_layers=2,
        tslm_hidden=512,
        tslm_layers=8,
        tslm_heads=8,
        tslm_ffn=2048,
        fsq_latent=128,
        fsq_scale=9,
        ralm_hidden=512,
        ralm_layers=4,
        ralm_heads=8,
        ralm_ffn=2048,
        dit_hidden=256,
        dit_layers=4,
        dit_heads=4,
        dit_ffn=1024,
        cfm_steps=10,
    ).to(device)

    counts = model.count_params()
    print(f"Total parameters: {counts['total']:,} ({counts['total']/1e6:.1f}M)")
    for k, v in counts.items():
        if k != "total":
            print(f"  {k:12s}: {v/1e6:.1f}M")

    # Freeze AudioVAE (in a real pipeline it's pretrained separately)
    for p in model.audio_vae.parameters():
        p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,} ({trainable/1e6:.1f}M)")

    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-2,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    # --- Training loop ---
    best_val_loss = float("inf")
    print(f"\n{'='*60}")
    print(f"Training SimpleVoxCPM for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for text, audio in val_loader:
                text, audio = text.to(device), audio.to(device)
                loss = model(text, audio)
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(n_val, 1)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"time={elapsed:.1f}s\n")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "vocab_size": 256, "latent_dim": 32,
                    "tslm_hidden": 512, "tslm_layers": 8,
                    "audio_len": args.audio_len,
                },
            }, args.save_path)
            print(f"  Saved checkpoint to {args.save_path} (val_loss={val_loss:.4f})")

    print(f"\n{'='*60}")
    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
