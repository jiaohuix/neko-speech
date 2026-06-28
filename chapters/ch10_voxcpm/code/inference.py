"""
SimpleVoxCPM Inference
======================

Load a trained SimpleVoxCPM checkpoint and generate speech from text.

Usage:
    python inference.py --checkpoint checkpoints/voxcpm.pt --text "hello world"
    python inference.py --checkpoint checkpoints/voxcpm.pt --n-steps 25 --output output.wav

Demonstrates:
  - Text tokenization (character-level for simplicity)
  - Autoregressive latent generation (n_steps patches, each via 10 CFM steps)
  - Waveform decoding
  - Real-time factor (RTF) benchmarking
"""

import argparse
import time
import sys
import os

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SimpleVoxCPM


# ---------------------------------------------------------------------------
# Simple character-level tokenizer
# ---------------------------------------------------------------------------

def tokenize_text(text: str, vocab_size: int = 256) -> torch.Tensor:
    """Encode text as a sequence of byte values (simple char-level tokenizer)."""
    ids = [ord(c) % vocab_size for c in text]
    return torch.tensor([ids], dtype=torch.long)              # (1, L)


# ---------------------------------------------------------------------------
# WAV saving (no soundfile dependency)
# ---------------------------------------------------------------------------

def save_wav(waveform: np.ndarray, path: str, sample_rate: int = 16000):
    """Write a mono 16-bit PCM WAV file using the wave module."""
    import wave
    import struct

    # Normalize to [-1, 1] and convert to 16-bit int
    wav = np.clip(waveform, -1.0, 1.0)
    int_samples = (wav * 32767).astype(np.int16)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_speech(
    model: SimpleVoxCPM,
    text: str,
    n_steps: int = 25,
    temperature: float = 1.0,
    device: str = "cpu",
) -> tuple:
    """Generate speech waveform from text.

    Returns:
        (waveform_np, elapsed_seconds, rtf)
    """
    model.eval()

    # Tokenize
    text_tokens = tokenize_text(text).to(device)
    print(f"Text: {text!r}")
    print(f"Tokens: {text_tokens.shape[1]} chars → shape {tuple(text_tokens.shape)}")
    print(f"Generating {n_steps} latent patches "
          f"({n_steps * model.audio_vae.chunk_size} samples, "
          f"{n_steps * model.audio_vae.chunk_size / 16000:.2f}s of audio)")
    print(f"CFM steps per patch: {model.cfm.n_steps}")
    print(f"Total diffusion steps: {n_steps * model.cfm.n_steps}")

    # Warmup (for CUDA)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        waveform = model.generate(
            text_tokens, n_steps=n_steps, temperature=temperature,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    wav_np = waveform.squeeze(0).cpu().numpy()
    audio_duration = len(wav_np) / 16000.0
    rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")

    return wav_np, elapsed, rtf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SimpleVoxCPM inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to checkpoint (default: random weights)")
    parser.add_argument("--text", type=str, default="hello neko world",
                        help="text to synthesize (default: 'hello neko world')")
    parser.add_argument("--n-steps", type=int, default=10,
                        help="number of latent patches to generate (default: 10)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="CFM initial noise temperature (default: 1.0)")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="output WAV path (default: output.wav)")
    parser.add_argument("--device", type=str, default=None,
                        help="device (default: auto-detect)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Build model ---
    model = SimpleVoxCPM(
        vocab_size=256,
        encoder_dim=64,
        latent_dim=32,
        decoder_dim=256,
        patch_size=1,
        loc_enc_hidden=512,
        loc_enc_layers=2,
        tslm_hidden=512,
        tslm_layers=8,
        tslm_heads=8,
        tslm_ffn=2048,
        fsq_latent=128,
        fsq_scale=9,
        ralm_hidden=512,
        ralm_layers=4,
        ralm_heads=8,
        ralm_ffn=2048,
        dit_hidden=256,
        dit_layers=4,
        dit_heads=4,
        dit_ffn=1024,
        cfm_steps=10,
    ).to(device)

    # --- Load checkpoint ---
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint} "
              f"(epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')})\n")
    else:
        print("No checkpoint found — using randomly initialized weights.\n")

    # --- Generate ---
    print("=" * 60)
    print("Generating speech...")
    print("=" * 60)

    wav_np, elapsed, rtf = generate_speech(
        model, args.text,
        n_steps=args.n_steps,
        temperature=args.temperature,
        device=device,
    )

    # --- Save ---
    save_wav(wav_np, args.output, sample_rate=16000)

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Output:          {args.output}")
    print(f"Waveform length: {len(wav_np):,} samples ({len(wav_np)/16000:.3f}s)")
    print(f"Generation time: {elapsed:.3f}s")
    print(f"RTF:             {rtf:.3f}  "
          f"({'real-time' if rtf < 1 else 'slower than real-time'})")
    print(f"WAV range:       [{wav_np.min():.4f}, {wav_np.max():.4f}]")

    # --- Show diffusion process ---
    print(f"\n{'='*60}")
    print(f"Diffusion process summary")
    print(f"{'='*60}")
    print(f"  Model generates {args.n_steps} latent patches autoregressively")
    print(f"  Each patch: {model.cfm.n_steps} Euler ODE steps (t=1 → t=0)")
    print(f"  Each ODE step: noise → DiT forward → velocity prediction → Euler update")
    print(f"  Total forward passes through DiT: {args.n_steps * model.cfm.n_steps}")
    print(f"  Then AudioVAE decodes all {args.n_steps} patches → waveform")


if __name__ == "__main__":
    main()
