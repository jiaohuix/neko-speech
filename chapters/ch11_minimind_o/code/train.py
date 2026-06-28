"""
SimpleOmni Training Script
==========================

Two-stage training for the SimpleOmni teaching model:

Stage 1: T2A (Text-to-Audio)
  - Align text with speech output
  - Thinker learns semantic conditions
  - Talker learns to generate audio codes
  - ~1-2 hours on RTX 3060

Stage 2: A2A (Audio-to-Audio)
  - Add speech input pathway
  - First warm up audio projector only
  - Then fine-tune full model at lower LR
  - ~1-2 hours on RTX 3060

Usage:
    # Stage 1: T2A (with simulated data for quick test)
    python train.py --stage t2a --data_path dummy --steps_per_epoch 100

    # Stage 2: A2A
    python train.py --stage a2a --data_path dummy --from_weight checkpoints/t2a_epoch1.pth
"""

import os
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext

from model import SimpleOmni, SimpleOmniConfig, compute_omni_loss, count_parameters


# ===========================================================================
# Simulated Multimodal Dataset
# ===========================================================================

class DummyOmniDataset(Dataset):
    """Simulated multimodal dataset for training verification.

    Generates random text tokens, audio codebook IDs, and optional mel
    spectrograms. Labels have realistic structure:
    - text_labels: non-(-100) only at "response" positions
    - audio_labels: non-(-100) only at audio generation positions

    In real training, replace with actual T2A/A2A data (parquet format).
    """

    def __init__(self, config: SimpleOmniConfig, num_samples: int = 1000,
                 seq_len: int = 64, include_mel: bool = False):
        self.config = config
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.include_mel = include_mel

        # Pre-generate data for speed
        self.data = []
        for _ in range(num_samples):
            sample = self._make_sample()
            self.data.append(sample)

    def _make_sample(self):
        cfg = self.config
        T = self.seq_len
        n_cb = cfg.num_codebooks

        # Text: random tokens with structure [prompt | response | eos | pad]
        prompt_len = T // 3
        response_len = T // 3
        pad_len = T - prompt_len - response_len

        input_ids = torch.randint(3, cfg.vocab_size, (T,))
        input_ids[0] = cfg.bos_token_id

        # Labels: -100 for prompt and padding, actual tokens for response
        text_labels = torch.full((T,), -100, dtype=torch.long)
        resp_start = prompt_len
        resp_end = prompt_len + response_len
        text_labels[resp_start:resp_end] = input_ids[resp_start:resp_end]
        text_labels[resp_end] = cfg.eos_token_id

        # Audio codes: pad before response, actual codes during response,
        # stop token at end, pad after
        audio_ids = torch.full((n_cb, T), cfg.audio_pad_token, dtype=torch.long)
        audio_labels = torch.full((n_cb, T), -100, dtype=torch.long)

        for cb in range(n_cb):
            # Stagger start by codebook index (delay pattern)
            cb_start = resp_start + cb
            cb_end = min(resp_end, T - 1)
            if cb_start < cb_end:
                audio_ids[cb, cb_start:cb_end] = torch.randint(
                    0, 2048, (cb_end - cb_start,))
                audio_labels[cb, cb_start:cb_end] = audio_ids[cb, cb_start:cb_end]
                # Stop token at end
                if cb_end < T:
                    audio_ids[cb, cb_end] = cfg.audio_stop_token
                    audio_labels[cb, cb_end] = cfg.audio_stop_token

        sample = {
            "input_ids": input_ids,
            "text_labels": text_labels,
            "audio_ids": audio_ids,
            "audio_labels": audio_labels,
        }

        if self.include_mel:
            # Simulated mel spectrogram (random noise)
            T_mel = T * 4  # before conv downsampling
            sample["mel_input"] = torch.randn(cfg.n_mels, T_mel)

        return sample

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


# ===========================================================================
# Training Utilities
# ===========================================================================

def get_cosine_lr(step: int, total_steps: int, max_lr: float,
                  min_lr: float = 0.0) -> float:
    """Cosine learning rate schedule with linear warmup (first 10%)."""
    warmup_steps = max(total_steps // 10, 1)
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_step(model, batch, optimizer, scaler, config, accumulation_step=1):
    """Single training step with mixed precision.

    Returns dict of loss values.
    """
    input_ids = batch["input_ids"]
    text_labels = batch["text_labels"]
    audio_ids = batch["audio_ids"]
    audio_labels = batch["audio_labels"]
    mel_input = batch.get("mel_input")

    # If mel_input is provided, pad labels for the mel prefix
    # (mel features are prepended to the text sequence)
    if mel_input is not None:
        with torch.no_grad():
            mel_feat_len = mel_input.shape[-1] // 4  # conv 4x downsample
        # Pad text_labels with -100 at the front for mel prefix
        pad = torch.full((text_labels.shape[0], mel_feat_len), -100,
                         dtype=text_labels.dtype, device=text_labels.device)
        text_labels = torch.cat([pad, text_labels], dim=1)
        # Pad audio_labels similarly (audio logits cover full sequence)
        audio_pad = torch.full(
            (audio_labels.shape[0], config.num_codebooks, mel_feat_len), -100,
            dtype=audio_labels.dtype, device=audio_labels.device)
        audio_labels = torch.cat([audio_pad, audio_labels], dim=2)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                  enabled=torch.cuda.is_available()):
        out = model(
            input_ids,
            audio_ids=audio_ids,
            mel_input=mel_input,
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
    scaler = torch.cuda.amp.GradScaler(
        enabled=(args.dtype == "float16" and torch.cuda.is_available())
    )

    # ---- Data ----
    include_mel = (args.stage in ("a2a", "a2a_proj"))
    dataset = DummyOmniDataset(
        config, num_samples=args.steps_per_epoch * args.batch_size,
        seq_len=args.max_seq_len, include_mel=include_mel,
    )
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    print(f"Dataset: {len(dataset)} samples, seq_len={args.max_seq_len}")

    # ---- Training loop ----
    total_steps = args.epochs * len(train_loader)
    global_step = 0
    model.train()

    print(f"\nTraining: {args.stage}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.learning_rate}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  Steps/epoch: {len(train_loader)}")
    print(f"  Total steps: {total_steps}")
    print()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_losses = {"total": 0.0, "text": 0.0, "audio": 0.0}
        n_batches = 0

        for step, batch in enumerate(train_loader):
            global_step += 1
            n_batches += 1

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Cosine LR
            lr = get_cosine_lr(global_step, total_steps, args.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            losses = train_step(model, batch, optimizer, scaler, config,
                                args.accumulation_steps)

            if global_step % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for k in epoch_losses:
                epoch_losses[k] += losses[k]

            if step % args.log_interval == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}, "
                      f"Step {step}/{len(train_loader)}, "
                      f"loss={losses['total']:.4f}, "
                      f"text={losses['text']:.4f}, "
                      f"audio={losses['audio']:.4f}, "
                      f"lr={lr:.6f}")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        print(f"\nEpoch {epoch+1} summary:")
        print(f"  Avg loss: total={avg_loss['total']:.4f}, "
              f"text={avg_loss['text']:.4f}, audio={avg_loss['audio']:.4f}")
        print(f"  Time: {epoch_time/60:.1f} min")

        # Save checkpoint
        save_path = os.path.join(args.save_dir, f"{args.stage}_epoch{epoch+1}.pth")
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  Saved: {save_path}\n")


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
    parser.add_argument("--data_path", type=str, default="dummy")
    parser.add_argument("--max_seq_len", type=int, default=64)

    # Model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--num_talker_layers", type=int, default=2)
    parser.add_argument("--num_codebooks", type=int, default=4)
    parser.add_argument("--from_weight", type=str, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--steps_per_epoch", type=int, default=100)

    # Logging & saving
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    args = parser.parse_args()

    # Adjust LR for fine-tuning stages
    if args.stage == "a2a":
        args.learning_rate = min(args.learning_rate, 2e-5)
        args.max_seq_len = max(args.max_seq_len, 96)

    train(args)
