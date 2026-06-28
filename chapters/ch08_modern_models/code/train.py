"""
Ch08: Training Script — Train F5-TTS on Dummy Data

This script demonstrates the training loop for F5-TTS.
Uses synthetic data for quick validation; replace with real
LJSpeech / LibriTTS data for actual training.

Usage:
    python train.py --model f5_tts --epochs 50
    python train.py --model cosyvoice --epochs 50
    python train.py --model indextts --epochs 50

For real training, you would:
    1. Precompute mel spectrograms (see ch01 code)
    2. Use a proper DataLoader with real audio
    3. Add a learning rate scheduler (cosine annealing)
    4. Log to TensorBoard / W&B
    5. Save checkpoints every N epochs
"""

import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from f5_tts import F5TTS, SimpleTextEncoder
from cosyvoice import CosyVoice
from indextts import IndexTTS


# --------------------------------------------------------
# Synthetic Dataset (for testing the training loop)
# --------------------------------------------------------

class SyntheticTTSDataset(Dataset):
    """
    Generates random mel + text pairs.
    Replace this with real data for actual training.
    """

    def __init__(self, n_samples=200, mel_dim=80, T_mel=100, T_text=20, T_ref=40):
        self.n_samples = n_samples
        self.mel_dim = mel_dim
        self.T_mel = T_mel
        self.T_text = T_text
        self.T_ref = T_ref

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "text_ids": torch.randint(0, 100, (self.T_text,)),
            "text_mask": torch.zeros(self.T_text, dtype=torch.bool),
            "ref_mel": torch.randn(self.T_ref, self.mel_dim),
            "ref_mask": torch.zeros(self.T_ref, dtype=torch.bool),
            "tgt_mel": torch.randn(self.T_mel, self.mel_dim),
            "tgt_mask": torch.zeros(self.T_mel, dtype=torch.bool),
            # For IndexTTS: pinyin
            "initials": torch.randint(0, 21, (10,)),
            "finals": torch.randint(0, 35, (10,)),
            "tones": torch.randint(1, 5, (10,)),
            "syl_mask": torch.zeros(10, dtype=torch.bool),
            "gt_dur": torch.randint(8, 15, (10,)).float(),
        }


# --------------------------------------------------------
# Training Loops
# --------------------------------------------------------

def train_f5_tts(args):
    """Train F5-TTS with Flow Matching loss."""
    device = args.device
    dataset = SyntheticTTSDataset(n_samples=args.n_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    text_enc = SimpleTextEncoder(text_dim=args.text_dim).to(device)
    model = F5TTS(
        mel_dim=args.mel_dim, text_dim=args.text_dim,
        dim=args.dim, heads=args.heads, n_layers=args.n_layers,
    ).to(device)

    params = list(text_enc.parameters()) + list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    print(f"Training F5-TTS | params: {sum(p.numel() for p in params):,}")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            text_emb = text_enc(batch["text_ids"])
            loss = model.flow_matching_loss(
                batch["tgt_mel"], batch["tgt_mask"],
                text_emb, batch["text_mask"],
                ref_mel=batch["ref_mel"], ref_mask=batch["ref_mask"],
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | Loss: {avg:.4f}")


def train_cosyvoice(args):
    """Train CosyVoice with AR next-frame prediction loss."""
    device = args.device
    dataset = SyntheticTTSDataset(n_samples=args.n_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    text_enc = SimpleTextEncoder(text_dim=args.text_dim).to(device)
    model = CosyVoice(
        mel_dim=args.mel_dim, text_dim=args.text_dim,
        spk_dim=args.spk_dim, dim=args.dim,
        heads=args.heads, n_layers=args.n_layers,
    ).to(device)

    params = list(text_enc.parameters()) + list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    print(f"Training CosyVoice | params: {sum(p.numel() for p in params):,}")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            text_emb = text_enc(batch["text_ids"])
            loss = model.ar_loss(
                batch["tgt_mel"], batch["tgt_mask"],
                text_emb, batch["text_mask"],
                batch["ref_mel"], batch["ref_mask"],
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | Loss: {avg:.4f}")


def train_indextts(args):
    """Train IndexTTS with mel + duration loss."""
    device = args.device
    dataset = SyntheticTTSDataset(n_samples=args.n_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = IndexTTS(
        mel_dim=args.mel_dim, pinyin_dim=args.pinyin_dim,
        dim=args.dim, heads=args.heads, n_layers=args.n_layers,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"Training IndexTTS | params: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            loss, _, _ = model.loss(
                batch["initials"], batch["finals"], batch["tones"],
                batch["syl_mask"], batch["gt_dur"],
                gt_mel=batch["tgt_mel"],  # will be truncated
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | Loss: {avg:.4f}")


# --------------------------------------------------------
# Main
# --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["f5_tts", "cosyvoice", "indextts"],
                        default="f5_tts")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--mel-dim", type=int, default=80)
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--spk-dim", type=int, default=128)
    parser.add_argument("--pinyin-dim", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"=== Ch08 Training: {args.model} ===")
    if args.model == "f5_tts":
        train_f5_tts(args)
    elif args.model == "cosyvoice":
        train_cosyvoice(args)
    elif args.model == "indextts":
        train_indextts(args)
    print("Done.")
