"""
Ch02: Tacotron2 Inference

Usage:
    python inference.py \
        --checkpoint checkpoints/tacotron_final.pt \
        --text "你好，我是猫娘。" \
        --output neko_output.wav

Then use Ch01's Griffin-Lim or a proper vocoder to convert mel → waveform.
"""

import argparse

import numpy as np
import soundfile as sf
import torch

from model import Tacotron2
from train import CharTokenizer


from scipy import signal


def griffin_lim(magnitude, n_fft=1024, hop_length=256, n_iter=60):
    """Griffin-Lim reconstruction using scipy for robust STFT/ISTFT."""
    # Start with random phase
    angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    stft_recon = magnitude * angles

    for _ in range(n_iter):
        # ISTFT
        _, output = signal.istft(stft_recon, fs=16000, nperseg=n_fft, noverlap=n_fft - hop_length, window="hamming")

        # STFT
        _, _, stft_new = signal.stft(output, fs=16000, nperseg=n_fft, noverlap=n_fft - hop_length, window="hamming")

        # Match shape
        if stft_new.shape != magnitude.shape:
            min_frames = min(stft_new.shape[1], magnitude.shape[1])
            stft_new = stft_new[:, :min_frames]
            magnitude = magnitude[:, :min_frames]

        angles = stft_new / (np.abs(stft_new) + 1e-10)
        stft_recon = magnitude * angles

    # Final ISTFT
    _, output = signal.istft(stft_recon, fs=16000, nperseg=n_fft, noverlap=n_fft - hop_length, window="hamming")
    return output


def mel_to_waveform(mel, sr=16000, n_fft=1024, hop_length=256, n_mels=80):
    """
    Convert log-mel spectrogram back to waveform via Griffin-Lim.

    This is a crude approximation — a proper vocoder (WaveNet, HiFi-GAN)
    would sound much better.
    """
    # 1. exp(log-mel) → mel
    mel = np.exp(mel)  # (T, n_mels)

    # 2. Mel → linear magnitude (approximate pseudo-inverse)
    # Build mel filter bank
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700.0)

    def mel_to_hz(mel_v):
        return 700 * (10 ** (mel_v / 2595.0) - 1)

    fft_freqs = np.linspace(0, sr // 2, n_fft // 2 + 1)
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sr // 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    mel_filter = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        up = (fft_freqs - left) / (center - left)
        down = (right - fft_freqs) / (right - center)
        mel_filter[i] = np.maximum(0, np.minimum(up, down))

    # Pseudo-inverse (very rough approximation)
    mel_filter_inv = np.linalg.pinv(mel_filter)  # (n_mels, n_freqs) → (n_freqs, n_mels)
    magnitude = np.dot(mel_filter_inv, mel.T)  # (n_freqs, T)

    # 3. Griffin-Lim
    print("[inference] Running Griffin-Lim reconstruction...")
    waveform = griffin_lim(magnitude, n_fft=n_fft, hop_length=hop_length, n_iter=60)

    # Normalize
    waveform = waveform / np.max(np.abs(waveform)) * 0.8
    return waveform


def main():
    parser = argparse.ArgumentParser(description="Tacotron2 Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, default="../outputs/neko_output.wav")
    parser.add_argument("--max-len", type=int, default=500)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "tokenizer_chars" in ckpt:
        tokenizer = CharTokenizer(ckpt["tokenizer_chars"])
        model_state = ckpt.get("model_state_dict", ckpt)
    else:
        tokenizer = CharTokenizer.from_texts([args.text])
        model_state = ckpt

    model = Tacotron2(vocab_size=tokenizer.vocab_size, mel_dim=80).to(device)
    model.load_state_dict(model_state)
    model.eval()
    print(f"[load] Model loaded from {args.checkpoint}")

    # Encode text
    text_ids = tokenizer.encode(args.text)
    text_tensor = torch.LongTensor([text_ids]).to(device)
    print(f"[input] Text: {args.text}")
    print(f"[input] Tokens: {text_ids}")

    # Inference
    with torch.no_grad():
        mel_before, mel_after = model.inference(text_tensor, max_len=args.max_len)

    mel = mel_after[0].cpu().numpy()  # (T, 80)
    print(f"[output] Generated mel shape: {mel.shape}")

    # Mel → Waveform (Griffin-Lim)
    waveform = mel_to_waveform(mel, sr=args.sr)

    # Save
    sf.write(args.output, waveform, args.sr)
    print(f"[save] Audio saved: {args.output}")


if __name__ == "__main__":
    main()
