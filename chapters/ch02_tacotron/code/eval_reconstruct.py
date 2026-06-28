"""
Reconstruct audio from training set using teacher forcing.

This tests whether the model has actually learned to map text -> mel,
by feeding ground-truth mel frames as decoder input.

Usage:
    python eval_reconstruct.py \
        --checkpoint ../checkpoints/tacotron_epoch_10.pt \
        --data-dir ../../../data/processed \
        --sample-idx 0 \
        --output ../outputs/recon_sample0.wav
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from inference import mel_to_waveform
from model import Tacotron2
from train import NekoDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="../../../data/processed")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--output", type=str, default="../outputs/recon_sample.wav")
    parser.add_argument("--save-gt", type=str, default=None,
                        help="Also save ground-truth audio for comparison")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load checkpoint first to get tokenizer
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "tokenizer_chars" in ckpt:
        from train import CharTokenizer
        tokenizer = CharTokenizer(ckpt["tokenizer_chars"])
        model_state = ckpt.get("model_state_dict", ckpt)
    else:
        model_state = ckpt
        tokenizer = None

    # Load dataset
    dataset = NekoDataset(args.data_dir, max_duration_sec=25, target_sr=16000)
    sample = dataset[args.sample_idx]

    # Re-encode text with checkpoint's tokenizer (vocab may differ from current dataset)
    raw_text = dataset.samples[args.sample_idx]["text"]
    if tokenizer is None:
        tokenizer = dataset.tokenizer
    text_ids = tokenizer.encode(raw_text)
    text = torch.LongTensor([text_ids]).to(device)
    mel_gt = sample["mel"].unsqueeze(0).to(device)  # (1, T_mel, 80)

    print(f"[sample] idx={args.sample_idx}")
    print(f"[sample] text tokens: {text.shape}")
    print(f"[sample] mel shape: {mel_gt.shape}")

    # Load model
    model = Tacotron2(vocab_size=tokenizer.vocab_size, mel_dim=80).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # Forward with teacher forcing
    with torch.no_grad():
        mel_before, mel_after, stop_logits = model(text, mel_gt)

    # Compute reconstruction error
    mse_before = torch.nn.functional.mse_loss(mel_before, mel_gt).item()
    mse_after = torch.nn.functional.mse_loss(mel_after, mel_gt).item()
    print(f"[recon] MSE before PostNet: {mse_before:.4f}")
    print(f"[recon] MSE after  PostNet: {mse_after:.4f}")

    # Convert predicted mel -> audio
    mel_pred = mel_after[0].cpu().numpy()  # (T, 80)
    print(f"[recon] Predicted mel range: [{mel_pred.min():.2f}, {mel_pred.max():.2f}]")

    waveform_pred = mel_to_waveform(mel_pred, sr=16000)
    sf.write(args.output, waveform_pred, 16000)
    print(f"[save] Reconstructed audio: {args.output}")

    # Optionally save ground truth for comparison
    if args.save_gt:
        mel_gt_np = mel_gt[0].cpu().numpy()
        waveform_gt = mel_to_waveform(mel_gt_np, sr=16000)
        sf.write(args.save_gt, waveform_gt, 16000)
        print(f"[save] Ground-truth audio: {args.save_gt}")


if __name__ == "__main__":
    main()
