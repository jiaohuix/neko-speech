"""
Ch09: GPT-SoVITS -- Two-Stage Training Script

Training pipeline:
    Stage 1: AR model (text -> semantic tokens)
        - CrossEntropy loss
        - AdamW optimizer
        - Simulated data (random phonemes + HuBERT-like features)

    Stage 2: SoVITS vocoder (semantic tokens -> waveform)
        - GAN training (generator + discriminator)
        - Losses: mel loss, KL loss, generator loss, feature loss
        - AdamW optimizer for both G and D

Usage:
    # Stage 1: AR model
    python train.py --stage 1 --epochs 10 --batch-size 4 --lr 1e-4

    # Stage 2: SoVITS vocoder
    python train.py --stage 2 --epochs 10 --batch-size 4 --lr 1e-4
"""

import argparse
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path for neko imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from model import (
    SimpleAR, SimpleRVQ, GPTSoVITS,
    kl_loss, mel_loss,
)


# ===================================================================
# Simulated Datasets
# ===================================================================

class SimulatedARDataset(Dataset):
    """
    Simulated dataset for AR model training.

    Generates random phoneme IDs and semantic token IDs that loosely
    mimic the real data distribution:
      - Phoneme IDs in [0, phoneme_vocab_size)
      - Semantic IDs in [0, 1024) with EOS=1024 at the end
      - Text length ~ 10-30 tokens
      - Audio length ~ 30-100 tokens (at 50 Hz, 0.6-2s)
    """

    def __init__(self, num_samples=500, min_text=8, max_text=20, min_audio=30, max_audio=80):
        self.num_samples = num_samples
        self.min_text = min_text
        self.max_text = max_text
        self.min_audio = min_audio
        self.max_audio = max_audio

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Use idx as seed for reproducibility within epoch
        rng = torch.Generator().manual_seed(idx)
        t_len = torch.randint(self.min_text, self.max_text + 1, (1,), generator=rng).item()
        a_len = torch.randint(self.min_audio, self.max_audio + 1, (1,), generator=rng).item()

        phoneme_ids = torch.randint(1, 512, (t_len,), generator=rng)
        # Semantic tokens + EOS at the end
        semantic_ids = torch.randint(0, 1024, (a_len,), generator=rng)
        semantic_ids[-1] = 1024  # EOS

        return {
            'phoneme_ids': phoneme_ids,
            'semantic_ids': semantic_ids,
            'text_len': t_len,
            'audio_len': a_len,
        }


class SimulatedSoVITSDataset(Dataset):
    """
    Simulated dataset for SoVITS vocoder training.

    Generates:
      - Random phoneme IDs
      - Random linear spectrogram (mimicking n_fft=2048 -> 1025 channels)
      - Random reference mel (mimicking 128 mels)
      - Random waveform segment
    """

    def __init__(self, num_samples=500, spec_channels=1025, n_mels=128,
                 min_text=8, max_text=20, min_spec=30, max_spec=60):
        self.num_samples = num_samples
        self.spec_channels = spec_channels
        self.n_mels = n_mels
        self.min_text = min_text
        self.max_text = max_text
        self.min_spec = min_spec
        self.max_spec = max_spec
        # Hop product from Generator upsample rates: 10*8*2*2*2 = 640
        self.hop_product = 640

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = torch.Generator().manual_seed(idx + 10000)
        t_len = torch.randint(self.min_text, self.max_text + 1, (1,), generator=rng).item()
        s_len = torch.randint(self.min_spec, self.max_spec + 1, (1,), generator=rng).item()

        phoneme_ids = torch.randint(1, 512, (t_len,), generator=rng)
        # Spectrogram: small random values to simulate log-magnitude
        spec = torch.randn(self.spec_channels, s_len, generator=rng) * 0.5
        # Reference mel (separate length, typically shorter)
        ref_len = max(20, s_len // 2)
        ref_mel = torch.randn(self.n_mels, ref_len, generator=rng) * 0.5
        # Real waveform: the target for discriminator
        wav_len = s_len * self.hop_product
        waveform = torch.randn(1, wav_len, generator=rng) * 0.1

        return {
            'phoneme_ids': phoneme_ids,
            'spec': spec,
            'ref_mel': ref_mel,
            'waveform': waveform,
            'text_len': t_len,
            'spec_len': s_len,
        }


# ===================================================================
# Collate functions
# ===================================================================

def ar_collate_fn(batch):
    """Pad variable-length sequences for AR model."""
    max_text = max(b['text_len'] for b in batch)
    max_audio = max(b['audio_len'] for b in batch)
    B = len(batch)

    phoneme_ids = torch.zeros(B, max_text, dtype=torch.long)
    semantic_ids = torch.zeros(B, max_audio, dtype=torch.long)

    for i, b in enumerate(batch):
        phoneme_ids[i, :b['text_len']] = b['phoneme_ids']
        semantic_ids[i, :b['audio_len']] = b['semantic_ids']

    return {
        'phoneme_ids': phoneme_ids,
        'semantic_ids': semantic_ids,
    }


def sovits_collate_fn(batch):
    """Pad variable-length sequences for SoVITS model."""
    max_text = max(b['text_len'] for b in batch)
    max_spec = max(b['spec_len'] for b in batch)
    max_ref = max(b['ref_mel'].shape[1] for b in batch)
    max_wav = max(b['waveform'].shape[1] for b in batch)
    B = len(batch)

    spec_channels = batch[0]['spec'].shape[0]
    n_mels = batch[0]['ref_mel'].shape[0]

    phoneme_ids = torch.zeros(B, max_text, dtype=torch.long)
    specs = torch.zeros(B, spec_channels, max_spec)
    ref_mels = torch.zeros(B, n_mels, max_ref)
    waveforms = torch.zeros(B, 1, max_wav)

    for i, b in enumerate(batch):
        tl = b['text_len']
        sl = b['spec_len']
        rl = b['ref_mel'].shape[1]
        wl = b['waveform'].shape[1]

        phoneme_ids[i, :tl] = b['phoneme_ids']
        specs[i, :, :sl] = b['spec']
        ref_mels[i, :, :rl] = b['ref_mel']
        waveforms[i, :, :wl] = b['waveform']

    return {
        'phoneme_ids': phoneme_ids,
        'spec': specs,
        'ref_mel': ref_mels,
        'waveform': waveforms,
    }


# ===================================================================
# Stage 1: AR Model Training
# ===================================================================

def train_stage1(args):
    """Train the autoregressive model (text -> semantic tokens)."""
    print("=" * 60)
    print("Stage 1: Training AR Model (GPT-style Transformer)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Model
    model = SimpleAR(
        dim=384, n_heads=8, n_layers=8,
        phoneme_vocab_size=512, vocab_size=1025,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Data
    dataset = SimulatedARDataset(num_samples=args.num_samples)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=ar_collate_fn, num_workers=0,
    )
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # Optimizer (AdamW, simplified from ScaledAdam)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )

    # Warmup + cosine decay schedule
    total_steps = len(dataloader) * args.epochs
    warmup_steps = min(200, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # Training loop
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch in dataloader:
            phoneme_ids = batch['phoneme_ids'].to(device)
            semantic_ids = batch['semantic_ids'].to(device)

            # Teacher forcing: input = semantic_ids[:-1], target = semantic_ids[1:]
            # But our model takes the full semantic_ids and predicts from each position.
            # We use semantic_ids as input and shift the target by 1.
            input_ids = semantic_ids[:, :-1]   # all but last
            target_ids = semantic_ids[:, 1:]   # all but first

            logits = model(phoneme_ids, input_ids)  # (B, T-1, vocab)

            # CrossEntropy expects (B, C, T) or (B*T, C)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

        elapsed = time.time() - epoch_start
        avg_loss = epoch_loss / len(dataloader)
        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}/{args.epochs} | "
              f"loss: {avg_loss:.4f} | lr: {lr:.6f} | "
              f"time: {elapsed:.1f}s")

    # Save checkpoint
    save_path = os.path.join(args.save_dir, 'ar_model.pt')
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': args.epochs,
        'config': {
            'dim': 384, 'n_heads': 8, 'n_layers': 8,
            'phoneme_vocab_size': 512, 'vocab_size': 1025,
        },
    }, save_path)
    print(f"\nSaved AR model to {save_path}")


# ===================================================================
# Stage 2: SoVITS Vocoder Training
# ===================================================================

def train_stage2(args):
    """Train the SoVITS vocoder (semantic tokens -> waveform) with GAN training."""
    print("=" * 60)
    print("Stage 2: Training SoVITS Vocoder (VITS-based)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Model
    model = GPTSoVITS().to(device)
    gen_params = sum(p.numel() for n, p in model.named_parameters()
                     if p.requires_grad and not n.startswith('discriminator'))
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Generator parameters:  {gen_params:,} ({gen_params/1e6:.1f}M)")
    print(f"Discriminator params:  {disc_params:,} ({disc_params/1e6:.1f}M)")
    print(f"Total trainable:       {total_params:,} ({total_params/1e6:.1f}M)")

    # Data
    dataset = SimulatedSoVITSDataset(num_samples=args.num_samples)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=sovits_collate_fn, num_workers=0,
    )
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # Optimizers
    # Collect generator params (everything except discriminator and frozen quantizer)
    gen_param_list = [p for n, p in model.named_parameters()
                      if p.requires_grad and not n.startswith('discriminator')]
    disc_param_list = list(model.discriminator.parameters())

    opt_gen = torch.optim.AdamW(gen_param_list, lr=args.lr, betas=(0.8, 0.99), eps=1e-9)
    opt_disc = torch.optim.AdamW(disc_param_list, lr=args.lr, betas=(0.8, 0.99), eps=1e-9)

    # Training loop
    c_mel = 45.0  # mel loss weight
    c_kl = 1.0    # KL loss weight

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        epoch_mel_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_start = time.time()

        for batch in dataloader:
            phoneme_ids = batch['phoneme_ids'].to(device)
            spec = batch['spec'].to(device)
            ref_mel = batch['ref_mel'].to(device)
            waveform = batch['waveform'].to(device)

            # --- Generator forward ---
            y_hat, m_p, logs_p, m_q, logs_q = model.forward_stage2(
                phoneme_ids, spec, ref_mel
            )

            # Trim to same length (generator output may differ from waveform)
            min_len = min(y_hat.size(2), waveform.size(2))
            y_hat_trim = y_hat[:, :, :min_len]
            waveform_trim = waveform[:, :, :min_len]

            # --- Discriminator step ---
            d_loss, g_loss, feat_loss = model.discriminator(waveform_trim, y_hat_trim)

            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()

            # --- Generator step ---
            # Recompute gen/feat loss (now with gradient through generator)
            _, g_loss, feat_loss = model.discriminator(waveform_trim, y_hat_trim)

            # Mel loss
            loss_mel = mel_loss(waveform_trim, y_hat_trim) * c_mel

            # KL loss
            loss_kl = kl_loss(m_p, logs_p, m_q, logs_q) * c_kl

            # Total generator loss
            loss_gen_total = g_loss + feat_loss + loss_mel + loss_kl

            opt_gen.zero_grad()
            loss_gen_total.backward()
            opt_gen.step()

            epoch_gen_loss += g_loss.item()
            epoch_disc_loss += d_loss.item()
            epoch_mel_loss += loss_mel.item()
            epoch_kl_loss += loss_kl.item()

        elapsed = time.time() - epoch_start
        n_batches = len(dataloader)
        print(f"  Epoch {epoch}/{args.epochs} | "
              f"gen: {epoch_gen_loss/n_batches:.4f} | "
              f"disc: {epoch_disc_loss/n_batches:.4f} | "
              f"mel: {epoch_mel_loss/n_batches:.4f} | "
              f"kl: {epoch_kl_loss/n_batches:.4f} | "
              f"time: {elapsed:.1f}s")

    # Save checkpoint
    save_path = os.path.join(args.save_dir, 'sovits_model.pt')
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': args.epochs,
    }, save_path)
    print(f"\nSaved SoVITS model to {save_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description='GPT-SoVITS Training')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2],
                        help='Training stage: 1 (AR model) or 2 (SoVITS vocoder)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='Number of simulated training samples (default: 500)')
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints'),
                        help='Checkpoint save directory')
    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Stage:      {args.stage}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Samples:    {args.num_samples}")
    print(f"  Save dir:   {args.save_dir}")
    print()

    if args.stage == 1:
        train_stage1(args)
    else:
        train_stage2(args)


if __name__ == '__main__':
    main()
