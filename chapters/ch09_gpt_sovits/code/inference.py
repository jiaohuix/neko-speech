"""
Ch09: GPT-SoVITS -- Inference Script

End-to-end inference pipeline:
    1. Encode reference audio -> speaker embedding + prompt semantic tokens
    2. AR model: text phonemes + prompt -> predicted semantic tokens
    3. SoVITS vocoder: semantic tokens + text + ref mel -> waveform
    4. Save output WAV + benchmark RTF

Usage:
    # Basic inference (with random/untrained model for testing)
    python inference.py --text "hello world" --output output.wav

    # With trained checkpoint
    python inference.py \
        --ar-checkpoint ../checkpoints/ar_model.pt \
        --sovits-checkpoint ../checkpoints/sovits_model.pt \
        --text "hello world" \
        --ref-audio ref.wav \
        --output output.wav

    # Benchmark RTF
    python inference.py --text "hello world" --output output.wav --benchmark
"""

import argparse
import os
import sys
import time

import torch
import numpy as np

# Add parent directory to path for neko imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from model import (
    SimpleAR, GPTSoVITS, SimpleRVQ,
)

try:
    from neko.utils import save_audio, mel_spectrogram, count_parameters, set_seed
except ImportError:
    # Fallback if neko package not available
    def save_audio(waveform, path, sr=32000):
        """Save waveform to WAV file."""
        import soundfile as sf
        from pathlib import Path
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        wav_np = waveform.cpu().numpy()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, wav_np, sr)

    def mel_spectrogram(waveform, n_fft=2048, hop_length=640, win_length=2048,
                        n_mels=128, sample_rate=32000):
        """Compute mel spectrogram."""
        import librosa
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        window = torch.hann_window(win_length, device=waveform.device)
        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=window, return_complex=True)
        magnitude = torch.abs(stft)
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        mel_basis = torch.from_numpy(mel_basis).float().to(waveform.device)
        mel = torch.matmul(mel_basis, magnitude)
        return torch.log(torch.clamp(mel, min=1e-5))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def set_seed(seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)


# ===================================================================
# Phoneme conversion (simplified)
# ===================================================================

# Simple character-to-phoneme mapping for demonstration.
# In a real system, this would use a G2P (grapheme-to-phoneme) model.
CHAR_TO_PHONEME = {}
for i, c in enumerate('abcdefghijklmnopqrstuvwxyz '):
    CHAR_TO_PHONEME[c] = i + 1  # 0 is padding


def text_to_phoneme_ids(text: str, max_vocab=512) -> torch.Tensor:
    """
    Convert text to phoneme IDs (simplified character-level mapping).

    In the real GPT-SoVITS, this would use a G2P system:
      - Chinese: pypinyin -> phonemes
      - English: CMU dict or espeak -> phonemes

    For our simplified version, we map characters directly to IDs.

    Args:
        text: input text string
        max_vocab: maximum phoneme vocabulary size

    Returns:
        phoneme_ids: (1, T_text) tensor
    """
    text_lower = text.lower()
    ids = [CHAR_TO_PHONEME.get(c, 0) for c in text_lower]
    # Clamp to vocab range
    ids = [min(i, max_vocab - 1) for i in ids]
    return torch.tensor([ids], dtype=torch.long)


# ===================================================================
# Reference audio processing
# ===================================================================

def process_reference_audio(audio_path: str, model: GPTSoVITS, device: str = 'cpu'):
    """
    Process reference audio to extract:
      1. Speaker embedding (from ReferenceEncoder)
      2. Prompt semantic tokens (from RVQ quantiser)
      3. Reference mel spectrogram

    In the real system, this would use a frozen HuBERT model to extract
    SSL features, then quantise them.  We simulate this with random features.

    Args:
        audio_path: path to reference audio WAV file
        model: GPTSoVITS model (for quantiser and ref_encoder)
        device: torch device

    Returns:
        dict with speaker_emb, prompt_semantic_ids, ref_mel
    """
    try:
        import librosa
        wav, _ = librosa.load(audio_path, sr=32000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    except Exception:
        # Fallback: generate random reference
        wav_tensor = torch.randn(1, 32000 * 3, device=device)  # 3s random

    # Compute mel spectrogram for reference encoder
    ref_mel = mel_spectrogram(wav_tensor.squeeze(0), n_fft=2048, hop_length=640,
                              win_length=2048, n_mels=128, sample_rate=32000)
    ref_mel = ref_mel.unsqueeze(0).to(device)  # (1, 128, T)

    # Speaker embedding
    with torch.no_grad():
        speaker_emb = model.ref_encoder(ref_mel)  # (1, 256)

    # Simulate HuBERT features -> quantise to semantic tokens
    # In reality: HuBERT(audio_16k) -> 768-dim features -> RVQ.encode()
    n_frames = ref_mel.size(2)  # roughly same frame rate
    ssl_features = torch.randn(1, 768, n_frames, device=device) * 0.1
    prompt_semantic_ids = model.quantizer.encode(ssl_features)  # (1, T)

    return {
        'speaker_emb': speaker_emb,
        'prompt_semantic_ids': prompt_semantic_ids,
        'ref_mel': ref_mel,
        'wav': wav_tensor,
    }


def generate_reference_data(model: GPTSoVITS, device: str = 'cpu',
                            prompt_len: int = 25, ref_mel_len: int = 20):
    """
    Generate simulated reference data when no real audio is available.

    Returns the same dict structure as process_reference_audio.
    """
    ref_mel = torch.randn(1, 128, ref_mel_len, device=device) * 0.5
    with torch.no_grad():
        speaker_emb = model.ref_encoder(ref_mel)

    ssl_features = torch.randn(1, 768, prompt_len, device=device) * 0.1
    prompt_semantic_ids = model.quantizer.encode(ssl_features)

    return {
        'speaker_emb': speaker_emb,
        'prompt_semantic_ids': prompt_semantic_ids,
        'ref_mel': ref_mel,
        'wav': torch.randn(1, 32000, device=device),
    }


# ===================================================================
# Inference pipeline
# ===================================================================

@torch.no_grad()
def synthesize(
    model: GPTSoVITS,
    text: str,
    ref_data: dict,
    device: str = 'cpu',
    top_k: int = 5,
    temperature: float = 1.0,
    max_tokens: int = 500,
    noise_scale: float = 0.667,
):
    """
    Full inference pipeline: text + reference -> waveform.

    Steps:
        1. Convert text to phoneme IDs
        2. AR model generates semantic tokens (autoregressive, top-k sampling)
        3. SoVITS vocoder converts semantic tokens to waveform

    Args:
        model: GPTSoVITS model
        text: input text
        ref_data: reference audio data dict
        device: torch device
        top_k: top-k sampling width
        temperature: sampling temperature
        max_tokens: maximum tokens to generate
        noise_scale: noise scale for prior sampling

    Returns:
        waveform: (1, T) tensor
        info: dict with timing and token info
    """
    model.eval()
    info = {}

    # Step 1: Text -> phoneme IDs
    t0 = time.time()
    phoneme_ids = text_to_phoneme_ids(text).to(device)  # (1, T_text)
    info['text_len'] = phoneme_ids.size(1)

    # Step 2: AR model -> semantic tokens
    t1 = time.time()
    prompt_ids = ref_data['prompt_semantic_ids']  # (1, T_prompt)

    # Autoregressive generation with KV-cache (simplified: no explicit cache)
    generated_ids = model.ar.generate(
        phoneme_ids, prompt_ids,
        max_new_tokens=max_tokens,
        top_k=top_k,
        temperature=temperature,
    )
    info['n_tokens'] = generated_ids.size(1)
    t2 = time.time()
    info['ar_time'] = t2 - t1

    # Step 3: SoVITS vocoder -> waveform
    # Build full sequence: prompt + generated
    all_semantic_ids = torch.cat([prompt_ids, generated_ids], dim=1)

    # Quantiser decode: semantic IDs -> continuous features
    quantised = model.quantizer.decode(all_semantic_ids)  # (1, 768, T)

    # Use TextEncoder to get prior distribution
    speaker_emb = ref_data['speaker_emb']
    m_p, logs_p = model.text_encoder(phoneme_ids, speaker_emb)

    # Sample from prior (noise_scale controls randomness)
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

    # Inverse flow: z_p -> z (map from prior to decoder space)
    z, _ = model.flow(z_p, reverse=True)

    # Generator: z -> waveform
    waveform = model.generator(z)  # (1, 1, T_wav)
    waveform = waveform.squeeze(1)  # (1, T_wav)

    t3 = time.time()
    info['vocoder_time'] = t3 - t2
    info['total_time'] = t3 - t0

    return waveform, info


# ===================================================================
# RTF Benchmark
# ===================================================================

def benchmark_rtf(model, text, ref_data, device, n_runs=5):
    """
    Benchmark Real-Time Factor (RTF).

    RTF = generation_time / audio_duration
    RTF < 1.0 means faster than real-time.

    Args:
        model: GPTSoVITS model
        text: input text
        ref_data: reference data dict
        device: torch device
        n_runs: number of benchmark runs

    Returns:
        dict with timing statistics
    """
    sr = 32000
    times = []
    durations = []

    for i in range(n_runs):
        # Warmup on first run
        waveform, info = synthesize(model, text, ref_data, device)
        if i > 0:  # Skip warmup
            times.append(info['total_time'])
            audio_duration = waveform.size(1) / sr
            durations.append(audio_duration)

    avg_time = np.mean(times)
    avg_dur = np.mean(durations)
    rtf = avg_time / avg_dur if avg_dur > 0 else float('inf')

    return {
        'avg_generation_time': avg_time,
        'avg_audio_duration': avg_dur,
        'rtf': rtf,
        'n_runs': n_runs - 1,
    }


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description='GPT-SoVITS Inference')
    parser.add_argument('--text', type=str, default='hello world',
                        help='Input text to synthesize')
    parser.add_argument('--ar-checkpoint', type=str, default=None,
                        help='Path to AR model checkpoint')
    parser.add_argument('--sovits-checkpoint', type=str, default=None,
                        help='Path to SoVITS model checkpoint')
    parser.add_argument('--ref-audio', type=str, default=None,
                        help='Path to reference audio (3-5s WAV)')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'outputs', 'output.wav'),
                        help='Output WAV file path')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Top-k sampling width (default: 5)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--max-tokens', type=int, default=200,
                        help='Maximum semantic tokens to generate')
    parser.add_argument('--noise-scale', type=float, default=0.667,
                        help='Noise scale for prior sampling')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run RTF benchmark')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Build model
    print("Building GPT-SoVITS model...")
    model = GPTSoVITS().to(device)
    total_params = count_parameters(model)
    print(f"  Trainable parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Load checkpoints if available
    if args.ar_checkpoint and os.path.exists(args.ar_checkpoint):
        print(f"  Loading AR checkpoint: {args.ar_checkpoint}")
        ckpt = torch.load(args.ar_checkpoint, map_location=device, weights_only=True)
        model.ar.load_state_dict(ckpt['model_state_dict'])

    if args.sovits_checkpoint and os.path.exists(args.sovits_checkpoint):
        print(f"  Loading SoVITS checkpoint: {args.sovits_checkpoint}")
        ckpt = torch.load(args.sovits_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])

    # Process reference audio
    print("\nProcessing reference audio...")
    if args.ref_audio and os.path.exists(args.ref_audio):
        ref_data = process_reference_audio(args.ref_audio, model, device)
        print(f"  Loaded reference: {args.ref_audio}")
    else:
        ref_data = generate_reference_data(model, device)
        print("  Using simulated reference (no real audio provided)")

    print(f"  Prompt tokens: {ref_data['prompt_semantic_ids'].size(1)}")
    print(f"  Ref mel frames: {ref_data['ref_mel'].size(2)}")

    # Synthesize
    print(f"\nSynthesizing: \"{args.text}\"")
    waveform, info = synthesize(
        model, args.text, ref_data, device,
        top_k=args.top_k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        noise_scale=args.noise_scale,
    )

    print(f"  Generated tokens: {info['n_tokens']}")
    print(f"  AR time:     {info['ar_time']:.3f}s")
    print(f"  Vocoder time: {info['vocoder_time']:.3f}s")
    print(f"  Total time:  {info['total_time']:.3f}s")

    audio_duration = waveform.size(1) / 32000
    print(f"  Audio duration: {audio_duration:.2f}s")
    rtf = info['total_time'] / audio_duration if audio_duration > 0 else float('inf')
    print(f"  RTF: {rtf:.3f}")

    # Save output
    save_audio(waveform, args.output, sr=32000)
    print(f"\nSaved output to: {args.output}")

    # Benchmark
    if args.benchmark:
        print("\n--- RTF Benchmark ---")
        results = benchmark_rtf(model, args.text, ref_data, device, n_runs=6)
        print(f"  Runs: {results['n_runs']}")
        print(f"  Avg generation time: {results['avg_generation_time']:.3f}s")
        print(f"  Avg audio duration:  {results['avg_audio_duration']:.2f}s")
        print(f"  RTF: {results['rtf']:.3f}")
        if results['rtf'] < 1.0:
            print("  -> Faster than real-time!")
        else:
            print("  -> Slower than real-time (expected for unoptimized AR)")


if __name__ == '__main__':
    main()
