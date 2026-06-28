"""
Neko Speech -- sherpa-onnx Integration Demo

sherpa-onnx (https://github.com/k2-fsa/sherpa-onnx) is a cross-platform
speech toolkit that supports TTS inference using ONNX models.

Supported TTS models in sherpa-onnx:
  - VITS (various checkpoints, multilingual)
  - Piper (fast, lightweight)
  - Kokoro-82M (high quality)
  - Matcha-TTS

This script demonstrates:
  1. How to use sherpa-onnx's built-in VITS TTS
  2. How our educational models relate to sherpa-onnx models
  3. Desktop TTS inference with sherpa-onnx

Usage:
    # Install sherpa-onnx first:
    pip install sherpa-onnx

    # Download a pre-trained model:
    # See: https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

    # Run demo:
    python sherpa_onnx_demo.py --text "Hello, this is a test"
    python sherpa_onnx_demo.py --text "猫娘老师，你好！" --model_path ./model.onnx

Note: Our educational models (ch02, ch04, ch05) use simplified architectures
that are not directly compatible with sherpa-onnx's expected format.
To use sherpa-onnx in production, use their pre-trained VITS models.
For learning, our export_onnx.py + ONNX Runtime is the way to go.
"""

import argparse
import os
import sys
import time


def check_sherpa_onnx():
    """Check if sherpa-onnx is installed."""
    try:
        import sherpa_onnx
        print(f"sherpa-onnx version: {sherpa_onnx.__version__}")
        return True
    except ImportError:
        print("[!] sherpa-onnx is not installed.")
        print("    Install with: pip install sherpa-onnx")
        print()
        print("    Available from PyPI for:")
        print("    - Linux (x86_64, aarch64)")
        print("    - macOS (x86_64, arm64)")
        print("    - Windows (x86_64)")
        print("    - Android (via JNI)")
        print("    - iOS (via framework)")
        return False


def list_available_models():
    """List sherpa-onnx TTS models available for download."""
    models = [
        {
            "name": "vits-ljs",
            "description": "VITS trained on LJSpeech (English, female)",
            "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-ljs.tar.bz2",
            "size_mb": 104,
            "lang": "en",
        },
        {
            "name": "vits-zh-hf-theresa",
            "description": "VITS trained on Chinese female voice",
            "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-vits-zh-hf-theresa.tar.bz2",
            "size_mb": 110,
            "lang": "zh",
        },
        {
            "name": "vits-melo-tts-zh_en",
            "description": "MeloTTS VITS for Chinese+English",
            "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2",
            "size_mb": 150,
            "lang": "zh+en",
        },
        {
            "name": "kokoro-en-v0_19",
            "description": "Kokoro 82M (English, high quality)",
            "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-kokoro-en-v0_19.tar.bz2",
            "size_mb": 170,
            "lang": "en",
        },
        {
            "name": "piper-en_US-lessac",
            "description": "Piper TTS (English, fast, lightweight)",
            "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-lessac-medium.tar.bz2",
            "size_mb": 60,
            "lang": "en",
        },
    ]

    print("\nAvailable sherpa-onnx TTS Models:")
    print("=" * 70)
    for m in models:
        print(f"  {m['name']}")
        print(f"    {m['description']}")
        print(f"    Language: {m['lang']}, Size: ~{m['size_mb']} MB")
        print(f"    URL: {m['url']}")
        print()

    return models


def download_model(url, output_dir="."):
    """Download and extract a sherpa-onnx model."""
    import tarfile
    import urllib.request

    filename = url.split("/")[-1]
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"  Already downloaded: {filepath}")
    else:
        print(f"  Downloading: {url}")
        urllib.request.urlretrieve(url, filepath)
        print(f"  Downloaded: {filepath}")

    # Extract
    extract_dir = os.path.join(output_dir, filename.replace(".tar.bz2", ""))
    if not os.path.exists(extract_dir):
        print(f"  Extracting to: {extract_dir}")
        with tarfile.open(filepath, "r:bz2") as tar:
            tar.extractall(output_dir)

    return extract_dir


def demo_vits_tts(model_dir, text, output_wav="output.wav", speed=1.0, sid=0):
    """
    Run VITS TTS using sherpa-onnx.

    Args:
        model_dir: Directory containing the sherpa-onnx VITS model
        text: Input text
        output_wav: Output WAV file path
        speed: Speech speed (1.0 = normal)
        sid: Speaker ID (for multi-speaker models)
    """
    import sherpa_onnx

    # Find model files
    model_path = None
    lexicon_path = None
    tokens_path = None
    dict_dir = None

    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".onnx") and "generator" not in f.lower():
                model_path = os.path.join(root, f)
            elif f == "tokens.txt":
                tokens_path = os.path.join(root, f)
            elif f.startswith("lexicon") and f.endswith(".txt"):
                lexicon_path = os.path.join(root, f)
            elif f.endswith(".dict"):
                if dict_dir is None:
                    dict_dir = root

    if model_path is None:
        print("[!] No ONNX model found in", model_dir)
        return False

    if tokens_path is None:
        print("[!] No tokens.txt found in", model_dir)
        return False

    print(f"  Model:  {model_path}")
    print(f"  Tokens: {tokens_path}")
    if lexicon_path:
        print(f"  Lexicon: {lexicon_path}")

    # Create TTS config
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=model_path,
                lexicon=lexicon_path or "",
                tokens=tokens_path,
            ),
            provider="cpu",
            num_threads=2,
        ),
        max_num_sentences=1,
        speed=speed,
    )

    # Create TTS
    tts = sherpa_onnx.OfflineTts(tts_config)

    # Generate
    print(f"\n  Generating speech: \"{text}\"")
    t0 = time.perf_counter()

    def callback(samples, sample_rate):
        """Audio callback -- called as audio is generated."""
        pass

    audio = tts.generate(text, sid=sid, speed=speed)

    t1 = time.perf_counter()

    # Save to WAV
    sample_rate = audio.sample_rate
    samples = audio.samples

    import numpy as np
    try:
        import soundfile as sf
        sf.write(output_wav, samples, sample_rate)
    except ImportError:
        # Fallback: raw WAV writing
        import wave
        import struct
        with wave.open(output_wav, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for s in samples:
                wf.writeframes(struct.pack("<h", int(s * 32767)))

    audio_length_s = len(samples) / sample_rate
    gen_time_s = t1 - t0
    rtf = gen_time_s / audio_length_s

    print(f"  Audio length: {audio_length_s:.2f}s")
    print(f"  Gen time:     {gen_time_s:.3f}s")
    print(f"  RTF:          {rtf:.3f}")
    print(f"  Sample rate:  {sample_rate} Hz")
    print(f"  Output:       {output_wav}")

    return True


def explain_architecture_mapping():
    """Explain how our educational models map to sherpa-onnx."""
    print("""
============================================================
Architecture Mapping: Neko Speech -> sherpa-onnx
============================================================

Our educational models (simplified) vs sherpa-onnx production models:

┌─────────────────┬────────────────────┬────────────────────────┐
│ Chapter          │ Our Model           │ sherpa-onnx Equivalent  │
├─────────────────┼────────────────────┼────────────────────────┤
│ ch02 Tacotron2   │ Encoder+Decoder     │ Not supported           │
│                  │ (autoregressive)    │ (too slow for edge)     │
├─────────────────┼────────────────────┼────────────────────────┤
│ ch04 FastSpeech2 │ FFT+LengthRegulator │ Not directly supported  │
│                  │ (non-autoregressive)│ (needs vocoder)         │
├─────────────────┼────────────────────┼────────────────────────┤
│ ch05 VITS        │ TextEnc+Flow+HiFi  │ ✓ VITS models           │
│                  │ (end-to-end)        │   (same architecture!)  │
└─────────────────┴────────────────────┴────────────────────────┘

Key insight: VITS (ch05) is the model that sherpa-onnx supports natively.
Our ch05 implementation follows the same architecture, so the concepts
transfer directly. The main differences:

1. sherpa-onnx uses production-quality VITS with full SDP + Flow
2. Our ch05 is simplified for education (fewer layers, smaller hidden dim)
3. sherpa-onnx includes text frontend (G2P, lexicon, phoneme processing)
4. Our code assumes phoneme IDs are pre-computed

To use our VITS with sherpa-onnx:
  1. Train our VITS on real data (e.g., LJSpeech)
  2. Export to ONNX using export_onnx.py
  3. Convert ONNX to sherpa-onnx format (add text frontend)
  4. Or: use sherpa-onnx's pre-trained VITS directly

For production deployment, sherpa-onnx's pre-trained models are
recommended. Our code is for understanding the architecture.
""")


def main():
    parser = argparse.ArgumentParser(description="Neko Speech sherpa-onnx Demo")
    parser.add_argument("--text", type=str, default="Hello, this is a test of text to speech.",
                        help="Text to synthesize")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to sherpa-onnx VITS model directory")
    parser.add_argument("--output", type=str, default="sherpa_output.wav",
                        help="Output WAV file")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed (1.0 = normal)")
    parser.add_argument("--list_models", action="store_true",
                        help="List available models")
    parser.add_argument("--download", type=str, default=None,
                        help="Download a model by name")
    parser.add_argument("--explain", action="store_true",
                        help="Explain architecture mapping")
    args = parser.parse_args()

    print("Neko Speech -- sherpa-onnx Integration Demo")
    print("=" * 60)

    if args.explain:
        explain_architecture_mapping()
        return

    if args.list_models:
        list_available_models()
        return

    if not check_sherpa_onnx():
        print("\n[!] Cannot run TTS demo without sherpa-onnx.")
        print("    Install: pip install sherpa-onnx")
        print()
        explain_architecture_mapping()
        return

    if args.download:
        models = list_available_models()
        target = [m for m in models if args.download in m["name"]]
        if target:
            download_model(target[0]["url"])
        else:
            print(f"Model not found: {args.download}")
            list_available_models()
        return

    if args.model_dir:
        demo_vits_tts(args.model_dir, args.text, args.output, args.speed)
    else:
        print("\nNo model specified. Options:")
        print("  1. --list_models   Show available models")
        print("  2. --download NAME Download a model")
        print("  3. --model_dir DIR Use a local model")
        print("  4. --explain       Architecture mapping")
        print()
        explain_architecture_mapping()


if __name__ == "__main__":
    main()
