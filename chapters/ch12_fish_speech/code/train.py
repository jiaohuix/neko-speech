"""
Ch12: Fish Speech Training Script

Usage:
    python train.py --data-dir ../../data/processed --epochs 20 --batch-size 4

Training Dual-AR TTS:
    1. Slow AR: Predict semantic tokens (codebook 0)
    2. Fast AR: Predict acoustic tokens (codebooks 1-3)

Loss:
    L = L_slow + L_fast
    L_slow: Cross-entropy on semantic tokens
    L_fast: Sum of cross-entropy on acoustic tokens
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import FishSpeech


# --------------------------------------------------------
# Synthetic Dataset (for demonstration)
# --------------------------------------------------------

class SyntheticCodecDataset(Dataset):
    """
    Synthetic dataset for Dual-AR training.

    In real Fish Speech:
    - Codec: 10 codebooks, ~21 Hz frame rate
    - Semantic tokens: codebook 0 (high-level linguistic content)
    - Acoustic tokens: codebooks 1-9 (detailed acoustic information)

    This simplified version:
    - 4 codebooks, 10 Hz frame rate
    - Generates random token sequences for training
    """

    def __init__(self, n_samples: int = 100, seq_len: int = 100, vocab_size: int = 1024, n_codebooks: int = 4):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks

        # Generate synthetic data
        self.data = []
        for _ in range(n_samples):
            # Slow tokens: semantic (codebook 0)
            slow_tokens = torch.randint(0, vocab_size, (seq_len,))

            # Fast tokens: acoustic (codebooks 1-3)
            fast_tokens = torch.randint(0, vocab_size, (n_codebooks - 1, seq_len))

            self.data.append((slow_tokens, fast_tokens))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


# --------------------------------------------------------
# Loss Function
# --------------------------------------------------------

class DualARLoss(nn.Module):
    """
    Dual-AR loss: cross-entropy on both slow and fast predictions.

    L = L_slow + λ · L_fast

    where:
        L_slow = CE(slow_logits, slow_tokens)
        L_fast = Σ CE(fast_logits[k], fast_tokens[k])
    """

    def __init__(self, lambda_fast: float = 1.0):
        super().__init__()
        self.lambda_fast = lambda_fast
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        slow_logits: torch.Tensor,
        fast_logits: list[torch.Tensor],
        slow_tokens: torch.Tensor,
        fast_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        slow_logits: (B, T, vocab_size)
        fast_logits: list of K tensors (B, T, vocab_size)
        slow_tokens: (B, T)
        fast_tokens: (B, K, T)
        """
        B, T, V = slow_logits.shape

        # Slow AR loss
        slow_logits_flat = slow_logits.view(-1, V)
        slow_tokens_flat = slow_tokens.view(-1)
        loss_slow = self.ce(slow_logits_flat, slow_tokens_flat)

        # Fast AR loss
        loss_fast = 0.0
        K = len(fast_logits)
        for k in range(K):
            fast_logits_flat = fast_logits[k].view(-1, V)
            fast_tokens_flat = fast_tokens[:, k].view(-1)
            loss_fast += self.ce(fast_logits_flat, fast_tokens_flat)
        loss_fast /= K

        # Total
        loss = loss_slow + self.lambda_fast * loss_fast

        return loss, {
            "loss_slow": loss_slow.item(),
            "loss_fast": loss_fast.item(),
        }


# --------------------------------------------------------
# Training Loop
# --------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    total_slow = 0.0
    total_fast = 0.0

    for slow_tokens, fast_tokens in tqdm(loader, desc="  train"):
        slow_tokens = slow_tokens.to(device)
        fast_tokens = fast_tokens.to(device)

        # Prepend BOS token (0)
        B = slow_tokens.shape[0]
        bos = torch.zeros(B, 1, dtype=slow_tokens.dtype, device=device)
        slow_input = torch.cat([bos, slow_tokens[:, :-1]], dim=1)

        # Forward
        slow_logits, fast_logits = model(slow_input, fast_tokens)

        # Loss
        loss, losses = criterion(slow_logits, fast_logits, slow_tokens, fast_tokens)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_slow += losses["loss_slow"]
        total_fast += losses["loss_fast"]

    n = len(loader)
    return total_loss / n, total_slow / n, total_fast / n


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Fish Speech")
    parser.add_argument("--data-dir", type=str, default="../../data/processed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default="../checkpoints")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--n-codebooks", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Dataset (synthetic for now)
    dataset = SyntheticCodecDataset(
        n_samples=100,
        seq_len=100,
        vocab_size=args.vocab_size,
        n_codebooks=args.n_codebooks,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = FishSpeech(
        vocab_size=args.vocab_size,
        n_codebooks=args.n_codebooks,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} parameters")
    print(f"[model] Slow AR: {sum(p.numel() for p in model.slow_ar.parameters()):,}")
    print(f"[model] Fast AR: {sum(p.numel() for p in model.fast_ar.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Loss
    criterion = DualARLoss()

    # Save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Training
    print(f"\n[training] {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        avg_loss, avg_slow, avg_fast = train_epoch(model, loader, optimizer, criterion, device)
        print(f"[epoch {epoch}/{args.epochs}] loss={avg_loss:.4f} slow={avg_slow:.4f} fast={avg_fast:.4f}")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = save_dir / f"fish_speech_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(args),
            }, ckpt_path)
            print(f"[save] {ckpt_path}")

    # Final checkpoint
    final_path = save_dir / "fish_speech_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(args),
    }, final_path)
    print(f"[done] Final model saved: {final_path}")


if __name__ == "__main__":
    main()
