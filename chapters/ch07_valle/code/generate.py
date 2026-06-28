"""
Ch07: VALL-E Zero-Shot Generation Pipeline

Given:
    - A trained VALL-E model (AR + NAR)
    - A trained neural codec
    - A reference audio file (voice to clone)
    - Target text

Produces:
    - Generated audio that speaks the text in the reference voice

Pipeline:
    1. Load reference audio → compute mel spectrogram
    2. Encode mel → codec tokens (the "voice prompt")
    3. Encode target text → character tokens
    4. AR model: text + prompt_level0 → generated level-0 tokens
    5. NAR model: text + level0 → levels 1..3 tokens
    6. Decode all tokens → mel spectrogram
    7. Mel → waveform via Griffin-Lim vocoder

Usage:
    python generate.py \
        --codec-checkpoint ../checkpoints/codec_final.pt \
        --valle-checkpoint ../checkpoints/valle_final.pt \
        --reference-audio ../reference/neko_sample.wav \
        --text "你好，我是猫娘。" \
        --output ../outputs/valle_generated.wav
"""

import argparse
import os
import sys
import json

import torch
import numpy as np
import soundfile as sf

from codec import NeuralCodec
from valle import VALLE


# ---------------------------------------------------------------------------
# Audio Utilities
# ---------------------------------------------------------------------------

def load_audio(path, target_sr=16000):
    """Load audio file, convert to mono, resample if needed."""
    data, sr = sf.read(path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)  # stereo → mono
    if sr != target_sr:
        # Simple resample via interpolation
        ratio = target_sr / sr
        new_len = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, new_len)
        data = np.interp(indices, np.arange(len(data)), data)
    return data, target_sr


def mel_spectrogram(waveform, sr=16000, n_fft=1024, hop_size=256, n_mels=80):
    """Compute log-mel spectrogram from waveform."""
    # STFT
    window = torch.hann_window(n_fft)
    stft = torch.stft(
        waveform, n_fft, hop_size, window=window,
        return_complex=True,
    )
    magnitude = stft.abs()  # (n_fft//2+1, T)

    # Mel filterbank
    mel_basis = torch.from_numpy(
        _mel_filterbank(sr, n_fft, n_mels)
    ).float()
    mel = mel_basis @ magnitude.pow(2)  # (n_mels, T)

    # Log scaling
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel


def _mel_filterbank(sr, n_fft, n_mels, fmin=0, fmax=None):
    """Create mel filterbank matrix."""
    if fmax is None:
        fmax = sr / 2

    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freqs)

    filters = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        low, center, high = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        for j, f in enumerate(freqs):
            if low <= f <= center:
                filters[i, j] = (f - low) / (center - low + 1e-10)
            elif center < f <= high:
                filters[i, j] = (high - f) / (high - center + 1e-10)

    return filters


def griffin_lim(mel_spec, sr=16000, n_fft=1024, hop_size=256, n_mels=80, n_iter=32):
    """
    Griffin-Lim vocoder: reconstruct waveform from mel spectrogram.
    This is the same approach used in Ch02 (Tacotron).
    """
    mel_basis = _mel_filterbank(sr, n_fft, n_mels)
    mel_basis_pinv = np.linalg.pinv(mel_basis)

    # Convert mel to linear magnitude (approximate)
    mel_linear = np.exp(mel_spec)
    magnitude = np.maximum(mel_basis_pinv @ mel_linear, 0)

    # Griffin-Lim iterations
    window = np.hanning(n_fft)
    phase = np.random.randn(*magnitude.shape) * 2 * np.pi
    complex_spec = magnitude * np.exp(1j * phase)

    for _ in range(n_iter):
        waveform = _istft(complex_spec, n_fft, hop_size, window)
        new_spec = _stft(waveform, n_fft, hop_size, window)
        # Keep original magnitude, update phase
        phase = np.angle(new_spec)
        complex_spec = magnitude * np.exp(1j * phase)

    return _istft(complex_spec, n_fft, hop_size, window)


def _stft(signal, n_fft, hop_size, window):
    """NumPy STFT."""
    pad_len = n_fft // 2
    signal = np.pad(signal, (pad_len, pad_len))
    n_frames = (len(signal) - n_fft) // hop_size + 1
    frames = np.array([signal[i * hop_size:i * hop_size + n_fft] * window
                       for i in range(n_frames)]).T
    return np.fft.rfft(frames, axis=0)


def _istft(spec, n_fft, hop_size, window):
    """NumPy ISTFT."""
    frames = np.fft.irfft(spec, axis=0)
    n_frames = frames.shape[1]
    out_len = n_fft + (n_frames - 1) * hop_size
    waveform = np.zeros(out_len)
    window_sum = np.zeros(out_len)
    for i in range(n_frames):
        start = i * hop_size
        waveform[start:start + n_fft] += frames[:, i] * window
        window_sum[start:start + n_fft] += window ** 2
    nonzero = window_sum > 1e-10
    waveform[nonzero] /= window_sum[nonzero]
    return waveform[n_fft // 2:-n_fft // 2]


# ---------------------------------------------------------------------------
# Text Tokenizer
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """
    Character-level tokenizer. Builds vocabulary from data or loads from config.
    Compatible with the tokenizer used in training.
    """

    def __init__(self, vocab=None):
        if vocab is None:
            # Default: build from common chars
            chars = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:'-/")
            # Add common Chinese characters range
            chars += [chr(i) for i in range(0x4e00, 0x9fff + 1)]
            vocab = {c: i + 2 for i, c in enumerate(chars)}  # 0=pad, 1=unk
        self.char2idx = vocab
        self.idx2char = {v: k for k, v in vocab.items()}

    def encode(self, text):
        return [self.char2idx.get(c, 1) for c in text]  # 1 = unknown

    def decode(self, ids):
        return ''.join(self.idx2char.get(i, '?') for i in ids)

    @property
    def vocab_size(self):
        return max(self.char2idx.values()) + 1

    @classmethod
    def load(cls, path):
        """Load tokenizer from JSON config."""
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(vocab=config['char2idx'])

    def save(self, path):
        """Save tokenizer config."""
        with open(path, 'w') as f:
            json.dump({'char2idx': self.char2idx}, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Generation Pipeline
# ---------------------------------------------------------------------------

def generate_speech(
    valle_model,
    codec_model,
    tokenizer,
    reference_audio_path,
    text,
    output_path,
    max_new_tokens=200,
    temperature=1.0,
    top_k=50,
    sr=16000,
    device='cpu',
):
    """
    Full zero-shot generation pipeline.

    Args:
        valle_model: trained VALL-E model
        codec_model: trained neural codec
        tokenizer: character tokenizer
        reference_audio_path: path to reference audio (voice to clone)
        text: target text to synthesize
        output_path: where to save the generated audio
        max_new_tokens: number of codec frames to generate
        temperature: AR sampling temperature
        top_k: AR top-k sampling
        sr: sample rate
        device: torch device
    """
    valle_model.eval()
    codec_model.eval()

    print(f"[1/7] Loading reference audio: {reference_audio_path}")
    audio, _ = load_audio(reference_audio_path, sr)
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)

    print("[2/7] Computing mel spectrogram...")
    mel = mel_spectrogram(audio_tensor[0], sr=sr)  # (n_mels, T)
    mel = mel.unsqueeze(0).to(device)               # (1, n_mels, T)

    print("[3/7] Encoding reference audio → codec tokens...")
    with torch.no_grad():
        prompt_codes = codec_model.encode(mel)  # (1, num_levels, T_latent)
    print(f"       Prompt tokens shape: {prompt_codes.shape}")
    print(f"       ({prompt_codes.shape[2]} codec frames from ~{len(audio)/sr:.1f}s audio)")

    print(f"[4/7] Tokenizing text: \"{text}\"")
    text_ids = tokenizer.encode(text)
    text_tensor = torch.LongTensor([text_ids]).to(device)
    print(f"       Text tokens: {len(text_ids)} characters")

    print(f"[5/7] AR generation: text + prompt → level-0 tokens...")
    with torch.no_grad():
        all_codes = valle_model.generate(
            text_tensor, prompt_codes,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )  # (1, num_levels, T_prompt + max_new_tokens)
    T_total = all_codes.shape[2]
    T_new = T_total - prompt_codes.shape[2]
    print(f"       Generated {T_new} new codec frames (total: {T_total})")

    print(f"[6/7] NAR fill-in: level-0 → levels 1-3...")
    print(f"       (Already done inside valle_model.generate)")

    print(f"[7/7] Decoding tokens → mel → waveform...")
    with torch.no_grad():
        mel_hat = codec_model.decode(all_codes)  # (1, n_mels, T_mel)

    # Griffin-Lim vocoder
    mel_np = mel_hat[0].cpu().numpy()
    waveform = griffin_lim(mel_np, sr=sr)

    # Normalize
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak * 0.95

    sf.write(output_path, waveform, sr)
    duration = len(waveform) / sr
    print(f"\n✓ Generated audio saved to: {output_path}")
    print(f"  Duration: {duration:.2f}s, Sample rate: {sr}Hz")

    return waveform


# ---------------------------------------------------------------------------
# Quick Demo (no training needed)
# ---------------------------------------------------------------------------

def demo_random_model():
    """
    Run the generation pipeline with randomly initialized models.
    This verifies the full pipeline works end-to-end, even though
    the output will be noise (models are not trained).
    """
    print("=" * 60)
    print("  VALL-E Zero-Shot Generation Demo")
    print("  (Random weights — output will be noise)")
    print("=" * 60)
    print()

    device = 'cpu'

    # Create models with small parameters for fast demo
    codec = NeuralCodec(
        mel_bins=80, hidden_dim=64,
        codebook_size=128, num_levels=4,
    ).to(device)

    valle = VALLE(
        vocab_size=128,
        audio_codebook_size=128,
        dim=128,
        num_heads=4,
        ar_layers=3,
        nar_layers=3,
        num_levels=4,
    ).to(device)

    tokenizer = SimpleTokenizer()

    # Create a synthetic reference audio (sine wave, 1 second)
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    ref_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine

    ref_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(ref_dir, exist_ok=True)
    ref_path = os.path.join(ref_dir, 'demo_reference.wav')
    sf.write(ref_path, ref_audio, sr)
    print(f"Created synthetic reference audio: {ref_path}")
    print()

    output_path = os.path.join(ref_dir, 'demo_generated.wav')

    generate_speech(
        valle_model=valle,
        codec_model=codec,
        tokenizer=tokenizer,
        reference_audio_path=ref_path,
        text="hello world",
        output_path=output_path,
        max_new_tokens=50,
        temperature=1.0,
        top_k=30,
        sr=sr,
        device=device,
    )

    print()
    print("Note: The output is noise because models are randomly initialized.")
    print("After training (see train.py), the output will be intelligible speech")
    print("that matches the reference voice.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VALL-E Zero-Shot Generation")
    parser.add_argument('--codec-checkpoint', type=str, default=None,
                        help='Path to trained codec checkpoint')
    parser.add_argument('--valle-checkpoint', type=str, default=None,
                        help='Path to trained VALL-E checkpoint')
    parser.add_argument('--reference-audio', type=str, default=None,
                        help='Path to reference audio file (voice to clone)')
    parser.add_argument('--text', type=str, default='hello world',
                        help='Text to synthesize')
    parser.add_argument('--output', type=str, default='../outputs/valle_generated.wav',
                        help='Output audio path')
    parser.add_argument('--max-new-tokens', type=int, default=200,
                        help='Number of codec frames to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='AR sampling temperature (higher = more diverse)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling for AR model')
    parser.add_argument('--tokenizer-config', type=str, default=None,
                        help='Path to tokenizer config JSON')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with random models (pipeline test)')
    args = parser.parse_args()

    if args.demo:
        demo_random_model()
        return

    if not args.reference_audio:
        print("Error: --reference-audio is required (or use --demo)")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load tokenizer
    if args.tokenizer_config:
        tokenizer = SimpleTokenizer.load(args.tokenizer_config)
    else:
        tokenizer = SimpleTokenizer()

    # Load models
    codec = NeuralCodec().to(device)
    valle = VALLE().to(device)

    if args.codec_checkpoint:
        codec.load_state_dict(torch.load(args.codec_checkpoint, map_location=device))
        print(f"Loaded codec from: {args.codec_checkpoint}")

    if args.valle_checkpoint:
        valle.load_state_dict(torch.load(args.valle_checkpoint, map_location=device))
        print(f"Loaded VALL-E from: {args.valle_checkpoint}")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    generate_speech(
        valle_model=valle,
        codec_model=codec,
        tokenizer=tokenizer,
        reference_audio_path=args.reference_audio,
        text=args.text,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )


if __name__ == "__main__":
    main()
