"""
Ch04: FastSpeech2 Inference (Parallel Generation)

Usage:
    python inference.py \
        --checkpoint ../checkpoints/fs2_final.pt \
        --text "你好，我是猫娘。" \
        --output ../outputs/fs2_output.wav

Compared to Tacotron2 (Ch02), inference is fully parallel:
all mel frames are generated in a single forward pass.
"""

import argparse
import time

import numpy as np
import soundfile as sf
import torch

from model import FastSpeech2


# --------------------------------------------------------
# Tokenizer (must match training)
# --------------------------------------------------------

class CharTokenizer:
    def __init__(self, chars):
        self.chars = chars
        self.vocab = {c: i + 1 for i, c in enumerate(chars)}
        self.pad_id = 0
        self.vocab_size = len(self.vocab) + 1

    def encode(self, text):
        return [self.vocab.get(c, self.pad_id) for c in text]


# --------------------------------------------------------
# Griffin-Lim Vocoder (placeholder — Ch03/Ch05 has better ones)
# --------------------------------------------------------

def mel_to_waveform(mel, sr=16000, n_fft=1024, hop_length=256, n_mels=80, n_iter=60):
    """Convert log-mel to waveform via Griffin-Lim."""
    mel_lin = np.exp(mel)  # (T, n_mels)

    # Mel filterbank pseudo-inverse
    import librosa
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    fb_inv = np.linalg.pinv(fb)
    magnitude = np.maximum(0, fb_inv @ mel_lin.T)  # (n_freqs, T)

    # Griffin-Lim phase recovery
    from scipy import signal
    angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    stft = magnitude * angles
    for _ in range(n_iter):
        _, wav_tmp = signal.istft(stft, fs=sr, nperseg=n_fft,
                                   noverlap=n_fft - hop_length, window="hamming")
        _, _, stft_new = signal.stft(wav_tmp, fs=sr, nperseg=n_fft,
                                      noverlap=n_fft - hop_length, window="hamming")
        if stft_new.shape != magnitude.shape:
            mn = min(stft_new.shape[1], magnitude.shape[1])
            stft_new = stft_new[:, :mn]
            magnitude = magnitude[:, :mn]
        angles = stft_new / (np.abs(stft_new) + 1e-10)
        stft = magnitude * angles

    _, wav = signal.istft(stft, fs=sr, nperseg=n_fft,
                           noverlap=n_fft - hop_length, window="hamming")
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav / peak * 0.8
    return wav


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FastSpeech2 Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, default="你好，我是猫娘。")
    parser.add_argument("--output", type=str, default="../outputs/fs2_output.wav")
    parser.add_argument("--pitch-scale", type=float, default=1.0,
                        help="Pitch multiplier (>1 = higher)")
    parser.add_argument("--energy-scale", type=float, default=1.0,
                        help="Energy multiplier (>1 = louder)")
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "tokenizer_chars" in ckpt:
        tokenizer = CharTokenizer(ckpt["tokenizer_chars"])
        state = ckpt.get("model_state_dict", ckpt)
    else:
        tokenizer = CharTokenizer(args.text)
        state = ckpt

    model = FastSpeech2(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"[load] checkpoint: {args.checkpoint}")

    # Encode text
    ids = tokenizer.encode(args.text)
    text = torch.LongTensor([ids]).to(device)
    text_lens = torch.LongTensor([len(ids)]).to(device)
    print(f"[input] \"{args.text}\"  ({len(ids)} tokens)")

    # ---- Parallel inference ----
    t0 = time.time()
    with torch.no_grad():
        mel = model.inference(
            text, text_lens,
            pitch_scale=args.pitch_scale,
            energy_scale=args.energy_scale,
        )
    t_model = time.time() - t0

    mel_np = mel[0].cpu().numpy()        # (80, T_mel)
    n_frames = mel_np.shape[1]
    audio_dur = n_frames * 256 / args.sr  # hop_length=256
    rtf = t_model / audio_dur if audio_dur > 0 else float("inf")

    print(f"[mel]   shape={mel_np.shape}")
    print(f"[time]  model inference: {t_model:.4f}s")
    print(f"[time]  audio duration:  {audio_dur:.2f}s")
    print(f"[time]  RTF = {rtf:.4f}  ({1/rtf:.0f}x realtime)" if rtf > 0 else "")

    # ---- Vocoder (Griffin-Lim) ----
    t1 = time.time()
    wav = mel_to_waveform(mel_np.T, sr=args.sr)  # mel_np.T -> (T, 80)
    t_vocoder = time.time() - t1
    print(f"[time]  Griffin-Lim:     {t_vocoder:.2f}s")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, wav, args.sr)
    print(f"[save]  {args.output}  ({len(wav)/args.sr:.2f}s)")

    # Summary comparison with Tacotron2
    print()
    print("=" * 50)
    print("FastSpeech2 vs Tacotron2 (Ch02) inference:")
    print(f"  FastSpeech2 model time:  {t_model:.4f}s  (parallel)")
    print(f"  Tacotron2 would need ~{n_frames} autoregressive steps")
    print(f"  Speedup: roughly {n_frames / max(1, int(t_model * 100)):.0f}x "
          f"fewer forward passes")
    print("=" * 50)


from pathlib import Path

if __name__ == "__main__":
    main()
