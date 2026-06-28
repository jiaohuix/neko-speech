"""
Ch03: WaveNet Inference

Usage:
    # Mel → Waveform (ground truth mel, compare with Griffin-Lim)
    python inference.py \
        --checkpoint ../checkpoints/wavenet_final.pt \
        --source ../../../data/processed/wavs/000001.wav \
        --output-dir ../outputs

    # Short generation (faster for testing)
    python inference.py \
        --checkpoint ../checkpoints/wavenet_final.pt \
        --source ../../../data/processed/wavs/000001.wav \
        --max-samples 8000

    # With Tacotron2 predicted mel
    python inference.py \
        --checkpoint ../checkpoints/wavenet_final.pt \
        --tacotron-checkpoint ../../ch02_tacotron/checkpoints/tacotron_final.pt \
        --text "你好" \
        --output-dir ../outputs

Flow:
    1. Compute (or load) Mel spectrogram
    2. WaveNet: mel → waveform (autoregressive)
    3. Griffin-Lim: mel → waveform (baseline)
    4. Save both for comparison
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa

from model import WaveNet, mu_law_encode, mu_law_decode


# --------------------------------------------------------
# Audio / Mel helpers
# --------------------------------------------------------

def compute_mel(wave, sr=16000, n_fft=1024, hop_length=256, n_mels=80):
    """Log-mel spectrogram, matching training pipeline."""
    mel = librosa.feature.melspectrogram(
        y=wave, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )
    return np.log(mel + 1e-5)


def griffin_lim_reconstruct(mel_log, sr=16000, n_fft=1024, hop_length=256,
                            n_mels=80, n_iter=60):
    """Griffin-Lim baseline for comparison."""
    mel = np.exp(mel_log)  # (n_mels, T)

    # Build mel filter bank and pseudo-inverse
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_basis_inv = np.linalg.pinv(mel_basis)
    magnitude = np.maximum(0, mel_basis_inv @ mel)  # (n_freqs, T)

    waveform = librosa.griffinlim(
        magnitude, n_iter=n_iter, hop_length=hop_length, win_length=n_fft,
    )
    return waveform


# --------------------------------------------------------
# Tacotron2 mel generation (optional)
# --------------------------------------------------------

def get_tacotron2_mel(checkpoint_path, text, device):
    """Load Tacotron2 and generate mel from text."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ch02_tacotron" / "code"))
    from model import Tacotron2
    from train import CharTokenizer

    ckpt = torch.load(checkpoint_path, map_location=device)
    tokenizer = CharTokenizer(ckpt.get("tokenizer_chars", ""))
    tacotron = Tacotron2(vocab_size=tokenizer.vocab_size, mel_dim=80).to(device)
    tacotron.load_state_dict(ckpt.get("model_state_dict", ckpt))
    tacotron.eval()

    text_ids = tokenizer.encode(text)
    text_tensor = torch.LongTensor([text_ids]).to(device)

    with torch.no_grad():
        _, mel_after = tacotron.inference(text_tensor, max_len=500)

    mel = mel_after[0].cpu().numpy()  # (T, 80)
    mel = mel.T  # (80, T) — match librosa convention
    return mel


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WaveNet Inference: Mel -> Waveform")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="WaveNet checkpoint path")
    parser.add_argument("--source", type=str, default=None,
                        help="Source wav for ground-truth mel (default: first wav in dataset)")
    parser.add_argument("--output-dir", type=str, default="../outputs")
    parser.add_argument("--max-samples", type=int, default=16000,
                        help="Max waveform samples to generate (16000 = 1s @ 16kHz)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (<1 = more deterministic)")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--hop-length", type=int, default=256)
    # Optional: Tacotron2 integration
    parser.add_argument("--tacotron-checkpoint", type=str, default=None,
                        help="Tacotron2 checkpoint for predicted mel")
    parser.add_argument("--text", type=str, default=None,
                        help="Text for Tacotron2 mel generation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load WaveNet --
    print("[load] Loading WaveNet...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {})
    model = WaveNet(
        n_mels=args.n_mels,
        res_channels=config.get("res_channels", 64),
        skip_channels=config.get("skip_channels", 128),
        n_blocks=config.get("n_blocks", 10),
        n_cycles=config.get("n_cycles", 3),
        hop_length=args.hop_length,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] WaveNet parameters: {total_params:,}")

    # -- Get mel spectrogram --
    if args.tacotron_checkpoint and args.text:
        # From Tacotron2 prediction
        print(f"[mel] Generating mel from Tacotron2: '{args.text}'")
        mel = get_tacotron2_mel(args.tacotron_checkpoint, args.text, device)
        source_name = "tacotron2"
    else:
        # From ground truth wav
        if args.source:
            wav_path = args.source
        else:
            # Default: first wav in dataset
            default_path = Path(args.checkpoint).parent.parent.parent.parent / "data" / "processed" / "wavs" / "000001.wav"
            if default_path.exists():
                wav_path = str(default_path)
            else:
                print("[error] No source wav specified. Use --source or provide a valid path.")
                return

        print(f"[load] Loading audio: {wav_path}")
        wave, _ = librosa.load(wav_path, sr=args.sr, mono=True)
        mel = compute_mel(wave, sr=args.sr, n_mels=args.n_mels,
                          hop_length=args.hop_length)  # (n_mels, T)
        source_name = Path(wav_path).stem

    # Trim mel to match max_samples
    max_mel_frames = args.max_samples // args.hop_length + 2
    if mel.shape[1] > max_mel_frames:
        mel = mel[:, :max_mel_frames]

    n_samples = min(args.max_samples, (mel.shape[1] - 1) * args.hop_length)

    # -- Griffin-Lim baseline --
    print("[griffin-lim] Reconstructing baseline...")
    wav_gl = griffin_lim_reconstruct(mel, sr=args.sr, n_mels=args.n_mels,
                                     hop_length=args.hop_length)
    wav_gl = wav_gl[:n_samples]
    if len(wav_gl) < n_samples:
        wav_gl = np.pad(wav_gl, (0, n_samples - len(wav_gl)))
    wav_gl = wav_gl / (np.max(np.abs(wav_gl)) + 1e-8) * 0.9

    gl_path = output_dir / f"griffinlim_{source_name}.wav"
    sf.write(str(gl_path), wav_gl, args.sr)
    print(f"[save] Griffin-Lim: {gl_path}")

    # -- WaveNet generation --
    print(f"[wavenet] Generating {n_samples} samples (this is slow)...")
    mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).to(device)  # (1, 80, T)

    import time
    t0 = time.time()
    wav_mu = model.generate(mel_tensor, n_samples=n_samples,
                            temperature=args.temperature)
    elapsed = time.time() - t0
    print(f"[wavenet] Generated in {elapsed:.1f}s "
          f"({n_samples / elapsed:.0f} samples/s, "
          f"RTF={elapsed / (n_samples / args.sr):.1f}x)")

    wav_wavenet = mu_law_decode(wav_mu).cpu().numpy()
    wav_wavenet = wav_wavenet / (np.max(np.abs(wav_wavenet)) + 1e-8) * 0.9

    wn_path = output_dir / f"wavenet_{source_name}.wav"
    sf.write(str(wn_path), wav_wavenet, args.sr)
    print(f"[save] WaveNet: {wn_path}")

    # -- Save ground truth if available --
    if not args.tacotron_checkpoint:
        gt = wave[:n_samples]
        gt = gt / (np.max(np.abs(gt)) + 1e-8) * 0.9
        gt_path = output_dir / f"ground_truth_{source_name}.wav"
        sf.write(str(gt_path), gt, args.sr)
        print(f"[save] Ground truth: {gt_path}")

    # -- Quality comparison --
    min_len = min(len(wav_gl), len(wav_wavenet))
    if not args.tacotron_checkpoint:
        min_len = min(min_len, len(gt))
        gt_trim = gt[:min_len]

    gl_trim = wav_gl[:min_len]
    wn_trim = wav_wavenet[:min_len]

    print(f"\n{'=' * 60}")
    print("Quality Metrics (vs ground truth)")
    print(f"{'=' * 60}")
    if not args.tacotron_checkpoint:
        snr_gl = 10 * np.log10(
            np.sum(gt_trim**2) / (np.sum((gt_trim - gl_trim)**2) + 1e-8)
        )
        snr_wn = 10 * np.log10(
            np.sum(gt_trim**2) / (np.sum((gt_trim - wn_trim)**2) + 1e-8)
        )
        print(f"  Griffin-Lim  SNR: {snr_gl:.2f} dB")
        print(f"  WaveNet      SNR: {snr_wn:.2f} dB")
    print(f"  WaveNet generation: {elapsed:.1f}s for {n_samples/args.sr:.2f}s audio")
    print(f"  RTF: {elapsed / (n_samples / args.sr):.1f}x real-time")
    print(f"\n  Note: WaveNet SNR may be lower than Griffin-Lim on short")
    print(f"  training, but perceptual quality (naturalness) is the real metric.")
    print(f"  Listen to the output files to judge!")

    print(f"\n[done] Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
