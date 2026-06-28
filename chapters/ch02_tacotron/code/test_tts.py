"""
End-to-end TTS test script for Tacotron2.

Usage:
    # With trained checkpoint
    python test_tts.py \
        --checkpoint ../checkpoints/tacotron_epoch_50.pt \
        --text "你好，我是猫娘。" \
        --output ../outputs/test_output.wav

    # Quick test with random-init model (just to verify pipeline)
    python test_tts.py \
        --text "测试一下" \
        --output ../outputs/test_random.wav
"""

import argparse
import time

import numpy as np
import soundfile as sf
import torch

from model import Tacotron2
from inference import mel_to_waveform
from train import CharTokenizer


def text_to_speech(model, text, device, max_len=500, sr=16000):
    """
    Full pipeline: Text → Tacotron2 → Mel → Griffin-Lim → Waveform.

    Returns:
        waveform: np.ndarray
        mel: np.ndarray (T, 80)
        gen_time: float (seconds)
        vocoder_time: float (seconds)
    """
    tokenizer = CharTokenizer.from_texts([text])
    text_ids = tokenizer.encode(text)
    text_tensor = torch.LongTensor([text_ids]).to(device)

    model.eval()
    with torch.no_grad():
        t0 = time.time()
        mel_before, mel_after = model.inference(text_tensor, max_len=max_len)
        gen_time = time.time() - t0

    mel = mel_after[0].cpu().numpy()  # (T, 80)

    # Griffin-Lim vocoder
    t0 = time.time()
    waveform = mel_to_waveform(mel, sr=sr)
    vocoder_time = time.time() - t0

    return waveform, mel, gen_time, vocoder_time


def main():
    parser = argparse.ArgumentParser(description="End-to-end TTS test")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. If None, uses random-init model.")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize")
    parser.add_argument("--output", type=str, default="../outputs/test_output.wav",
                        help="Output wav path")
    parser.add_argument("--max-len", type=int, default=500)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device}")

    # Load model
    if args.checkpoint:
        print(f"[load] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "tokenizer_chars" in ckpt:
            tokenizer = CharTokenizer(ckpt["tokenizer_chars"])
            model_state = ckpt.get("model_state_dict", ckpt)
        else:
            tokenizer = CharTokenizer.from_texts([args.text])
            model_state = ckpt
        model = Tacotron2(vocab_size=tokenizer.vocab_size, mel_dim=80).to(device)
        model.load_state_dict(model_state)
    else:
        tokenizer = CharTokenizer.from_texts([args.text])
        model = Tacotron2(vocab_size=tokenizer.vocab_size, mel_dim=80).to(device)
        print("[load] No checkpoint provided — using random-init model (expect noise!)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Parameters: {total_params:,}")
    print(f"[input] Text: {args.text}")

    # Synthesize
    waveform, mel, gen_time, vocoder_time = text_to_speech(
        model, args.text, device, max_len=args.max_len, sr=args.sr
    )

    # Save
    sf.write(args.output, waveform, args.sr)

    # Stats
    duration = len(waveform) / args.sr
    rtf = (gen_time + vocoder_time) / duration if duration > 0 else 0

    print("\n" + "=" * 50)
    print("TTS Results")
    print("=" * 50)
    print(f"  Output file    : {args.output}")
    print(f"  Audio duration : {duration:.2f}s")
    print(f"  Mel shape      : {mel.shape}")
    print(f"  Tacotron time  : {gen_time:.2f}s")
    print(f"  Vocoder time   : {vocoder_time:.2f}s")
    print(f"  Total time     : {gen_time + vocoder_time:.2f}s")
    print(f"  RTF (real-time): {rtf:.2f}x")
    print("=" * 50)

    if rtf > 1.0:
        print("[note] RTF > 1.0 means slower than real-time (expected for CPU + Griffin-Lim).")
    print("Done!")


if __name__ == "__main__":
    main()
