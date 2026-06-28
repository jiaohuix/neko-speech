"""
Ch07: VALL-E Training Pipeline

Two-phase training:
    Phase 1: Train the Neural Codec (mel → discrete tokens → mel)
    Phase 2: Train VALL-E (text + audio tokens → next audio token)

Data format:
    Expects audio files + text transcriptions in the format:
        wav_path|speaker|language|text

Usage:
    # Phase 1: Train codec
    python train.py --phase codec --data-dir ../../data/processed --epochs 30

    # Phase 2: Train VALL-E (requires trained codec)
    python train.py --phase valle \
        --data-dir ../../data/processed \
        --codec-checkpoint ../checkpoints/codec_final.pt \
        --epochs 30

    # Quick shape test (no data needed)
    python train.py --phase test
"""

import argparse
import os
import sys
import json
import csv
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from codec import NeuralCodec
from valle import VALLE


# ---------------------------------------------------------------------------
# Audio Processing
# ---------------------------------------------------------------------------

def load_audio(path, target_sr=16000):
    """Load audio, convert to mono, resample if needed."""
    data, sr = sf.read(path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        ratio = target_sr / sr
        new_len = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, new_len)
        data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
    return data, target_sr


def compute_mel(waveform, sr=16000, n_fft=1024, hop_size=256, n_mels=80):
    """Compute log-mel spectrogram."""
    if isinstance(waveform, np.ndarray):
        waveform = torch.FloatTensor(waveform)
    window = torch.hann_window(n_fft)
    stft = torch.stft(waveform, n_fft, hop_size, window=window, return_complex=True)
    magnitude = stft.abs()

    # Mel filterbank
    fmax = sr / 2
    mel_min = 0
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, fmax, n_freqs)
    filters = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        low, center, high = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        for j, f in enumerate(freqs):
            if low <= f <= center:
                filters[i, j] = (f - low) / (center - low + 1e-10)
            elif center < f <= high:
                filters[i, j] = (high - f) / (high - center + 1e-10)

    mel_basis = torch.FloatTensor(filters)
    mel = mel_basis @ magnitude.pow(2)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """Character-level tokenizer."""

    def __init__(self):
        self.char2idx = {'<pad>': 0, '<unk>': 1}
        self.next_idx = 2

    def add_chars(self, texts):
        for text in texts:
            for c in text:
                if c not in self.char2idx:
                    self.char2idx[c] = self.next_idx
                    self.next_idx += 1

    def encode(self, text):
        return [self.char2idx.get(c, 1) for c in text]

    @property
    def vocab_size(self):
        return self.next_idx

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'char2idx': self.char2idx}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            config = json.load(f)
        tok = cls()
        tok.char2idx = config['char2idx']
        tok.next_idx = max(tok.char2idx.values()) + 1
        return tok


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioTextDataset(Dataset):
    """
    Loads audio + text pairs from the data directory.
    Expects a train.list file with format: wav_path|speaker|language|text
    """

    def __init__(self, data_dir, tokenizer=None, max_duration=15.0, sr=16000):
        self.sr = sr
        self.max_duration = max_duration
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.samples = []

        list_file = os.path.join(data_dir, 'train.list')
        if not os.path.exists(list_file):
            print(f"Warning: {list_file} not found. Using wavs/ directory only.")
            wavs = glob.glob(os.path.join(data_dir, 'wavs', '*.wav'))
            for w in sorted(wavs):
                self.samples.append({
                    'wav_path': w,
                    'text': os.path.basename(w).replace('.wav', ''),
                })
        else:
            with open(list_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('|')
                    if len(parts) >= 4:
                        wav_path, speaker, lang, text = parts[0], parts[1], parts[2], parts[3]
                    elif len(parts) >= 2:
                        wav_path, text = parts[0], parts[-1]
                    else:
                        continue

                    # Make wav_path relative to data_dir if not absolute
                    if not os.path.isabs(wav_path):
                        wav_path = os.path.join(data_dir, wav_path)

                    self.samples.append({
                        'wav_path': wav_path,
                        'text': text,
                    })

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            audio, _ = load_audio(sample['wav_path'], self.sr)
        except Exception as e:
            print(f"Error loading {sample['wav_path']}: {e}")
            audio = np.zeros(self.sr, dtype=np.float32)

        # Trim to max duration
        max_samples = int(self.max_duration * self.sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        mel = compute_mel(audio, self.sr)

        text_ids = self.tokenizer.encode(sample['text'])
        text_ids = torch.LongTensor(text_ids)

        return {
            'mel': mel,              # (n_mels, T_mel)
            'text_ids': text_ids,    # (T_text,)
            'text': sample['text'],
            'wav_path': sample['wav_path'],
        }


def collate_fn(batch):
    """Collate batch with padding."""
    # Pad mels to same length
    max_mel_len = max(b['mel'].shape[1] for b in batch)
    n_mels = batch[0]['mel'].shape[0]

    # Pad texts to same length
    max_text_len = max(b['text_ids'].shape[0] for b in batch)

    mels = torch.zeros(len(batch), n_mels, max_mel_len)
    texts = torch.zeros(len(batch), max_text_len, dtype=torch.long)

    for i, b in enumerate(batch):
        T_mel = b['mel'].shape[1]
        mels[i, :, :T_mel] = b['mel']

        T_text = b['text_ids'].shape[0]
        texts[i, :T_text] = b['text_ids']

    return {'mel': mels, 'text_ids': texts}


# ---------------------------------------------------------------------------
# Phase 1: Train Codec
# ---------------------------------------------------------------------------

def train_codec(args):
    """Train the neural codec to reconstruct mel spectrograms."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training codec on {device}")

    codec = NeuralCodec(
        mel_bins=80,
        hidden_dim=args.codec_dim,
        codebook_size=args.codebook_size,
        num_levels=args.num_levels,
    ).to(device)

    optimizer = torch.optim.Adam(codec.parameters(), lr=args.lr)

    # Dataset
    tokenizer = SimpleTokenizer()
    dataset = AudioTextDataset(args.data_dir, tokenizer)
    if len(dataset) == 0:
        print("Error: No data found. Run data/download_neko_1k.py first.")
        sys.exit(1)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    # Build tokenizer from data
    all_texts = [s['text'] for s in dataset.samples]
    tokenizer.add_chars(all_texts)

    ckpt_dir = os.path.join(args.data_dir, '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    loss_log = []
    for epoch in range(args.epochs):
        total_loss = 0
        total_recon = 0
        total_vq = 0
        n_batches = 0

        codec.train()
        pbar = tqdm(dataloader, desc=f"Codec Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            mel = batch['mel'].to(device)

            mel_hat, codes, vq_loss = codec(mel)

            # Reconstruction loss
            recon_loss = F.l1_loss(mel_hat, mel)

            # Total loss
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'vq': f"{vq_loss.item():.4f}",
            })

        avg_loss = total_loss / max(n_batches, 1)
        avg_recon = total_recon / max(n_batches, 1)
        avg_vq = total_vq / max(n_batches, 1)
        loss_log.append({'epoch': epoch+1, 'loss': avg_loss, 'recon': avg_recon, 'vq': avg_vq})

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}  recon={avg_recon:.4f}  vq={avg_vq:.4f}")

        # Save checkpoints
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f'codec_epoch_{epoch+1}.pt')
            torch.save(codec.state_dict(), ckpt_path)

    # Save final
    final_path = os.path.join(ckpt_dir, 'codec_final.pt')
    torch.save(codec.state_dict(), final_path)
    tokenizer.save(os.path.join(ckpt_dir, 'tokenizer_config.json'))

    # Save loss log
    log_path = os.path.join(ckpt_dir, 'codec_loss_log.json')
    with open(log_path, 'w') as f:
        json.dump(loss_log, f, indent=2)

    print(f"\nCodec training complete!")
    print(f"  Final checkpoint: {final_path}")
    print(f"  Tokenizer config: {os.path.join(ckpt_dir, 'tokenizer_config.json')}")
    print(f"  Loss log: {log_path}")


# ---------------------------------------------------------------------------
# Phase 2: Train VALL-E
# ---------------------------------------------------------------------------

def train_valle(args):
    """Train VALL-E (AR + NAR) using codec tokens."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training VALL-E on {device}")

    # Load tokenizer
    ckpt_dir = os.path.join(args.data_dir, '..', 'checkpoints')
    tok_path = os.path.join(ckpt_dir, 'tokenizer_config.json')
    if os.path.exists(tok_path):
        tokenizer = SimpleTokenizer.load(tok_path)
    else:
        tokenizer = SimpleTokenizer()

    # Load codec (frozen)
    codec = NeuralCodec(
        mel_bins=80,
        hidden_dim=args.codec_dim,
        codebook_size=args.codebook_size,
        num_levels=args.num_levels,
    ).to(device)
    codec.eval()

    if args.codec_checkpoint:
        codec.load_state_dict(torch.load(args.codec_checkpoint, map_location=device))
        print(f"Loaded codec from: {args.codec_checkpoint}")
    else:
        # Try default path
        default_codec = os.path.join(ckpt_dir, 'codec_final.pt')
        if os.path.exists(default_codec):
            codec.load_state_dict(torch.load(default_codec, map_location=device))
            print(f"Loaded codec from: {default_codec}")
        else:
            print("Warning: No codec checkpoint found. Using random codec.")

    for p in codec.parameters():
        p.requires_grad = False

    # Build VALL-E
    valle = VALLE(
        vocab_size=tokenizer.vocab_size,
        audio_codebook_size=args.codebook_size,
        dim=args.valle_dim,
        num_heads=args.num_heads,
        ar_layers=args.ar_layers,
        nar_layers=args.nar_layers,
        num_levels=args.num_levels,
    ).to(device)

    ar_optimizer = torch.optim.Adam(valle.ar_model.parameters(), lr=args.lr)
    nar_optimizer = torch.optim.Adam(
        list(valle.nar_model.parameters()) + list(valle.text_encoder.parameters()),
        lr=args.lr,
    )

    # Dataset
    dataset = AudioTextDataset(args.data_dir, tokenizer)
    if len(dataset) == 0:
        print("Error: No data found.")
        sys.exit(1)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    # Ensure tokenizer covers all characters
    all_texts = [s['text'] for s in dataset.samples]
    tokenizer.add_chars(all_texts)

    os.makedirs(ckpt_dir, exist_ok=True)
    loss_log = []

    for epoch in range(args.epochs):
        total_ar_loss = 0
        total_nar_loss = 0
        n_batches = 0

        valle.train()
        pbar = tqdm(dataloader, desc=f"VALL-E Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            mel = batch['mel'].to(device)
            text_ids = batch['text_ids'].to(device)

            # Encode mel to codec tokens (frozen)
            with torch.no_grad():
                codes = codec.encode(mel)  # (B, num_levels, T_latent)

            # Skip if too short
            if codes.shape[2] < 3:
                continue

            # ---- AR Training (level 0) ----
            ar_logits = valle.forward_ar(text_ids, codes)  # (B, T_audio, codebook_size)
            ar_targets = codes[:, 0, :]                      # (B, T_audio)

            # Shift: predict token t from tokens 0..t-1
            ar_loss = F.cross_entropy(
                ar_logits[:, :-1, :].reshape(-1, args.codebook_size),
                ar_targets[:, 1:].reshape(-1),
                ignore_index=0,
            )

            ar_optimizer.zero_grad()
            ar_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(valle.ar_model.parameters(), 1.0)
            ar_optimizer.step()

            # ---- NAR Training (levels 1-3) ----
            nar_loss_total = 0
            for level in range(1, args.num_levels):
                nar_logits = valle.forward_nar(text_ids, codes, target_level=level)
                nar_targets = codes[:, level, :]

                level_loss = F.cross_entropy(
                    nar_logits.reshape(-1, args.codebook_size),
                    nar_targets.reshape(-1),
                    ignore_index=0,
                )
                nar_loss_total = nar_loss_total + level_loss
            nar_loss_total = nar_loss_total / (args.num_levels - 1)

            nar_optimizer.zero_grad()
            nar_loss_total.backward()
            torch.nn.utils.clip_grad_norm_(valle.nar_model.parameters(), 1.0)
            nar_optimizer.step()

            total_ar_loss += ar_loss.item()
            total_nar_loss += nar_loss_total.item()
            n_batches += 1

            pbar.set_postfix({
                'ar': f"{ar_loss.item():.4f}",
                'nar': f"{nar_loss_total.item():.4f}",
            })

        avg_ar = total_ar_loss / max(n_batches, 1)
        avg_nar = total_nar_loss / max(n_batches, 1)
        loss_log.append({'epoch': epoch+1, 'ar_loss': avg_ar, 'nar_loss': avg_nar})

        print(f"Epoch {epoch+1}: AR={avg_ar:.4f}  NAR={avg_nar:.4f}")

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f'valle_epoch_{epoch+1}.pt')
            torch.save(valle.state_dict(), ckpt_path)

    # Save final
    final_path = os.path.join(ckpt_dir, 'valle_final.pt')
    torch.save(valle.state_dict(), final_path)
    tokenizer.save(os.path.join(ckpt_dir, 'tokenizer_config.json'))

    log_path = os.path.join(ckpt_dir, 'valle_loss_log.json')
    with open(log_path, 'w') as f:
        json.dump(loss_log, f, indent=2)

    print(f"\nVALL-E training complete!")
    print(f"  Final checkpoint: {final_path}")
    print(f"  Loss log: {log_path}")


# ---------------------------------------------------------------------------
# Shape Test (no data needed)
# ---------------------------------------------------------------------------

def test_shapes():
    """Verify all model shapes without requiring data."""
    print("=" * 60)
    print("  VALL-E Shape Verification Test")
    print("=" * 60)

    device = 'cpu'

    # Test codec
    print("\n--- Neural Codec ---")
    codec = NeuralCodec(
        mel_bins=80, hidden_dim=128,
        codebook_size=256, num_levels=4,
    )
    mel = torch.randn(2, 80, 160)
    mel_hat, codes, vq_loss = codec(mel)
    print(f"  Input:  {mel.shape}")
    print(f"  Output: {mel_hat.shape}")
    print(f"  Codes:  {codes.shape}")
    print(f"  VQ loss: {vq_loss.item():.4f}")

    # Test VALL-E
    print("\n--- VALL-E ---")
    valle = VALLE(
        vocab_size=256,
        audio_codebook_size=256,
        dim=256,
        num_heads=4,
        ar_layers=4,
        nar_layers=4,
        num_levels=4,
    )

    B, T_text, T_audio = 2, 15, 50
    text_ids = torch.randint(0, 256, (B, T_text))
    audio_codes = torch.randint(0, 256, (B, 4, T_audio))

    # AR
    ar_logits = valle.forward_ar(text_ids, audio_codes)
    ar_loss = F.cross_entropy(
        ar_logits[:, :-1, :].reshape(-1, 256),
        audio_codes[:, 0, 1:].reshape(-1),
    )
    print(f"  AR logits: {ar_logits.shape}  loss: {ar_loss.item():.4f}")

    # NAR
    nar_logits = valle.forward_nar(text_ids, audio_codes, target_level=1)
    nar_loss = F.cross_entropy(
        nar_logits.reshape(-1, 256),
        audio_codes[:, 1, :].reshape(-1),
    )
    print(f"  NAR logits: {nar_logits.shape}  loss: {nar_loss.item():.4f}")

    # Generation
    prompt = audio_codes[:, :, :20]
    generated = valle.generate(text_ids, prompt, max_new_tokens=30)
    print(f"  Generated: {generated.shape}")

    # Parameter counts
    codec_params = sum(p.numel() for p in codec.parameters())
    valle_params = sum(p.numel() for p in valle.parameters())
    ar_params = sum(p.numel() for p in valle.ar_model.parameters())
    nar_params = sum(p.numel() for p in valle.nar_model.parameters())

    print(f"\n--- Parameters ---")
    print(f"  Codec:     {codec_params:>10,}")
    print(f"  VALL-E:    {valle_params:>10,}")
    print(f"    AR:      {ar_params:>10,}")
    print(f"    NAR:     {nar_params:>10,}")
    print(f"\nAll shape tests passed!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VALL-E Training")
    parser.add_argument('--phase', type=str, required=True,
                        choices=['codec', 'valle', 'test'],
                        help='Training phase: codec, valle, or test')
    parser.add_argument('--data-dir', type=str, default='../../data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--codec-checkpoint', type=str, default=None,
                        help='Path to pre-trained codec (for VALL-E phase)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)

    # Codec args
    parser.add_argument('--codec-dim', type=int, default=128,
                        help='Codec hidden dimension')
    parser.add_argument('--codebook-size', type=int, default=256,
                        help='Number of entries per codebook level')
    parser.add_argument('--num-levels', type=int, default=4,
                        help='Number of VQ codebook levels')

    # VALL-E args
    parser.add_argument('--valle-dim', type=int, default=256,
                        help='VALL-E hidden dimension')
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--ar-layers', type=int, default=6,
                        help='Number of AR Transformer layers')
    parser.add_argument('--nar-layers', type=int, default=6,
                        help='Number of NAR Transformer layers')

    args = parser.parse_args()

    if args.phase == 'test':
        test_shapes()
    elif args.phase == 'codec':
        train_codec(args)
    elif args.phase == 'valle':
        train_valle(args)


if __name__ == "__main__":
    main()
