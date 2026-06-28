"""
SimpleOmni Training Script
==========================

Two-stage training for the SimpleOmni teaching model:

Stage 1: T2A (Text-to-Audio)
  - Align text with speech output
  - Thinker learns semantic conditions
  - Talker learns to generate Mimi codes
  - ~1-2 hours on RTX 3060

Stage 2: A2A (Audio-to-Audio)
  - Add speech input pathway
  - First warm up audio projector only
  - Then fine-tune full model at lower LR
  - ~1-2 hours on RTX 3060

Based on the MiniMind-O training pipeline (trainer/train_sft_omni.py).

Usage:
    # Stage 1: T2A
    python train.py --stage t2a --data_path ../data/sft_t2a_mini.parquet

    # Stage 2: A2A (projector warmup)
    python train.py --stage a2a_proj --data_path ../data/sft_a2a_mini.parquet --from_weight t2a

    # Stage 2: A2A (full fine-tune)
    python train.py --stage a2a --data_path ../data/sft_a2a_mini.parquet --from_weight t2a
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from contextlib import nullcontext

from model import SimpleOmni, SimpleOmniConfig, compute_omni_loss, count_parameters


# ===========================================================================
# Training Utilities
# ===========================================================================

def get_cosine_lr(step: int, total_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    """Cosine learning rate schedule with warmup.

    The original MiniMind-O uses cosine decay. We add a short linear warmup
    (first 10% of steps) for stability.
    """
    warmup_steps = total_steps // 10
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_step(model, batch, optimizer, scaler, config, accumulation_step=1):
    """Single training step with mixed precision.

    Returns dict of loss values.
    """
    input_ids = batch["input_ids"]       # (B, T)
    text_labels = batch["text_labels"]   # (B, T)
    audio_ids = batch["audio_ids"]       # (B, num_codebooks, T)
    audio_labels = batch["audio_labels"] # (B, num_codebooks, T)
    audio_features = batch.get("audio_features")  # (B, T_audio, 512) or None
    spk_emb = batch.get("spk_emb")       # (B, 192) or None

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = model(
            input_ids,
            audio_ids=audio_ids,
            audio_features=audio_features,
            spk_emb=spk_emb,
        )
        total_loss, text_loss, audio_loss = compute_omni_loss(
            out["text_logits"],
            out["audio_logits"],
            text_labels,
            audio_labels,
            num_codebooks=config.num_codebooks,
        )
        loss = total_loss / accumulation_step

    scaler.scale(loss).backward()
    return {
        "total": total_loss.item(),
        "text": text_loss.item(),
        "audio": audio_loss.item(),
    }


# ===========================================================================
# Main Training Loop
# ===========================================================================

def train(args):
    """Main training function."""
    import math

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Config ----
    config = SimpleOmniConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_talker_layers=args.num_talker_layers,
        num_codebooks=args.num_codebooks,
    )

    # ---- Model ----
    model = SimpleOmni(config)
    if args.from_weight and os.path.exists(args.from_weight):
        state_dict = torch.load(args.from_weight, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {args.from_weight}")
    model.to(device)
    count_parameters(model)

    # ---- Freeze for projector-only training ----
    if args.stage == "a2a_proj":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.audio_proj.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Projector-only training: {trainable/1e6:.2f}M trainable params")

    # ---- Optimizer ----
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    # ---- Data ----
    # TODO: implement OmniDataset for parquet loading
    # For now, use dummy data for skeleton verification
    print(f"Loading data from {args.data_path}")
    # train_loader = DataLoader(dataset, batch_size=args.batch_size, ...)

    # ---- Training loop ----
    total_steps = args.epochs * args.steps_per_epoch
    global_step = 0
    model.train()

    print(f"\nTraining: {args.stage}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.learning_rate}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  Total steps: {total_steps}")
    print()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # TODO: iterate over actual data loader
        for step in range(args.steps_per_epoch):
            global_step += 1

            # Cosine LR
            lr = get_cosine_lr(global_step, total_steps, args.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # TODO: load actual batch
            batch = {
                "input_ids": torch.randint(0, config.vocab_size, (args.batch_size, args.max_seq_len), device=device),
                "text_labels": torch.full((args.batch_size, args.max_seq_len), -100, device=device),
                "audio_ids": torch.full((args.batch_size, config.num_codebooks, args.max_seq_len), config.audio_pad_token, device=device),
                "audio_labels": torch.full((args.batch_size, config.num_codebooks, args.max_seq_len), -100, device=device),
            }

            losses = train_step(model, batch, optimizer, scaler, config, args.accumulation_steps)

            if global_step % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}, Step {step}/{args.steps_per_epoch}, "
                      f"loss={losses['total']:.4f}, text={losses['text']:.4f}, "
                      f"audio={losses['audio']:.4f}, lr={lr:.6f}")

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} done in {epoch_time/60:.1f} min")

        # Save checkpoint
        save_path = os.path.join(args.save_dir, f"{args.stage}_epoch{epoch+1}.pth")
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimpleOmni Training")

    # Stage
    parser.add_argument("--stage", type=str, default="t2a",
                        choices=["t2a", "a2a_proj", "a2a"],
                        help="Training stage")

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=256)

    # Model
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--num_talker_layers", type=int, default=2)
    parser.add_argument("--num_codebooks", type=int, default=4)
    parser.add_argument("--from_weight", type=str, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--steps_per_epoch", type=int, default=1000)

    # Logging & saving
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    args = parser.parse_args()

    # Adjust LR for fine-tuning stages
    if args.stage == "a2a":
        args.learning_rate = min(args.learning_rate, 2e-5)
        args.max_seq_len = max(args.max_seq_len, 384)

    train(args)
