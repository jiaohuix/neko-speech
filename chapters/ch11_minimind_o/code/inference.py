"""
SimpleOmni Inference Demo
=========================

Demonstrates text-to-speech, speech-to-text, and image+text-to-speech
generation using the trained SimpleOmni model.

Modes:
  1. Text-to-Speech (T2A): Type text -> get speech output (mel spectrogram)
  2. Speech-to-Text (A2T): Provide audio -> get text response
  3. Image+Text-to-Speech (I2A): Provide image + text -> get speech
  4. Streaming demo: demonstrates delay pattern visualization

Usage:
    python inference.py --mode t2a --text "Hello, I am Neko."
    python inference.py --mode a2t --mel_file input_mel.pt
    python inference.py --mode i2a --text "What is in this image?" --image_file img.pt
    python inference.py --mode stream_demo
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np

from model import SimpleOmni, SimpleOmniConfig, count_parameters


# ===========================================================================
# Model Loading
# ===========================================================================

def load_model(weight_path: str, config: SimpleOmniConfig = None,
               device: str = "cuda") -> SimpleOmni:
    """Load trained SimpleOmni model."""
    config = config or SimpleOmniConfig()
    model = SimpleOmni(config)

    if weight_path and os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weight_path}")

    model = model.eval().to(device)
    count_parameters(model, verbose=False)
    return model


# ===========================================================================
# Text-to-Speech Generation
# ===========================================================================

def text_to_speech(model: SimpleOmni, text: str, max_new_tokens=64,
                   temperature=0.75, top_p=0.9, device="cuda"):
    """Generate speech (mel spectrogram) from text input.

    Pipeline:
    1. Create dummy text tokens (placeholder for real tokenizer)
    2. model.generate() -> text tokens + audio codes + mel spectrogram
    3. Save mel spectrogram as output

    In production, replace dummy tokens with real tokenizer output
    and decode mel via vocoder (or Mimi codes via Mimi decoder).
    """
    config = model.config
    # Placeholder tokenizer: map chars to token IDs
    text_ids = [config.bos_token_id]
    for ch in text[:64]:
        text_ids.append(max(3, ord(ch) % config.vocab_size))

    input_ids = torch.tensor([text_ids], dtype=torch.long, device=device)

    print(f"[Thinker]: Generating response to '{text}'...")
    result = model.generate(
        input_ids, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p,
    )

    gen_text_len = result["text_ids"].shape[1]
    mel_shape = result["mel"].shape
    audio_shape = result["audio_codes"].shape

    print(f"  Generated text tokens: {gen_text_len}")
    print(f"  Audio codes shape: {audio_shape} "
          f"({audio_shape[1]} codebooks x {audio_shape[2]} frames)")
    print(f"  Mel spectrogram shape: {mel_shape} "
          f"({mel_shape[1]} frames x {mel_shape[2]} mel bins)")

    return result


# ===========================================================================
# Speech-to-Text (Audio Input -> Text Output)
# ===========================================================================

def speech_to_text(model: SimpleOmni, mel_input: torch.Tensor,
                   max_new_tokens=64, temperature=0.75, device="cuda"):
    """Generate text response from speech input.

    Pipeline:
    1. Encode speech via SimpleSpeechEncoder
    2. Project into Thinker hidden space
    3. Thinker processes audio features + generates text tokens
    """
    config = model.config
    mel_input = mel_input.unsqueeze(0).to(device)  # add batch dim

    # Prompt: "respond to the audio"
    prompt_ids = [config.bos_token_id, 10, 20, 30]  # placeholder tokens
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print("[Thinker]: Processing speech input...")
    result = model.generate(
        input_ids, max_new_tokens=max_new_tokens,
        temperature=temperature, mel_input=mel_input,
    )

    gen_text_len = result["text_ids"].shape[1]
    print(f"  Generated {gen_text_len} text tokens in response")

    return result


# ===========================================================================
# Image+Text-to-Speech
# ===========================================================================

def image_text_to_speech(model: SimpleOmni, text: str,
                          image: torch.Tensor,
                          max_new_tokens=64, temperature=0.75, device="cuda"):
    """Generate speech response from image + text input.

    Pipeline:
    1. Encode image via SimpleImageEncoder -> patch features
    2. Thinker processes image features + text tokens
    3. Talker generates audio codes
    4. Decode to mel spectrogram
    """
    config = model.config
    image = image.unsqueeze(0).to(device)  # add batch dim

    text_ids = [config.bos_token_id]
    for ch in text[:64]:
        text_ids.append(max(3, ord(ch) % config.vocab_size))
    input_ids = torch.tensor([text_ids], dtype=torch.long, device=device)

    print(f"[Thinker]: Processing image + text '{text}'...")
    result = model.generate(
        input_ids, max_new_tokens=max_new_tokens,
        temperature=temperature, image_input=image,
    )

    mel_shape = result["mel"].shape
    print(f"  Mel spectrogram: {mel_shape}")

    return result


# ===========================================================================
# Streaming Generation Demo
# ===========================================================================

def streaming_demo(model: SimpleOmni, text: str = "Hello world",
                   max_new_tokens=32, device="cuda"):
    """Demonstrate streaming generation with delay pattern.

    Shows how audio frames become available incrementally as
    codebooks finish their staggered generation.
    """
    config = model.config
    text_ids = [config.bos_token_id]
    for ch in text[:32]:
        text_ids.append(max(3, ord(ch) % config.vocab_size))
    input_ids = torch.tensor([text_ids], dtype=torch.long, device=device)

    print(f"\n[Streaming Demo] Generating response to '{text}'")
    print(f"  Codebooks: {config.num_codebooks}")
    print(f"  Delay to first frame: {config.num_codebooks} steps")
    print()

    frame_count = 0
    for step, (text_tok, audio_frame) in enumerate(
            model.stream_generate(input_ids, max_new_tokens=max_new_tokens)):
        if audio_frame is not None:
            frame_count += 1
            print(f"  Step {step:3d}: text_token={text_tok.item():5d}, "
                  f"audio_frame={audio_frame}")
        else:
            print(f"  Step {step:3d}: text_token={text_tok.item():5d}, "
                  f"audio_frame=None (waiting for delay pattern)")

    print(f"\n  Total frames generated: {frame_count}")
    print(f"  Latency to first frame: {config.num_codebooks} steps")


# ===========================================================================
# Delay Pattern Visualization
# ===========================================================================

def visualize_delay_pattern(num_codebooks=4, total_steps=12):
    """Print a visual representation of the delay pattern.

    Shows when each codebook starts generating and when complete
    frames become available for decoding.
    """
    print("\nDelay Pattern Visualization")
    print("=" * 60)
    print(f"  Codebooks: {num_codebooks}, Steps: {total_steps}")
    print()

    # Header
    header = "  Step: " + " ".join(f"{s:3d}" for s in range(total_steps))
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Each codebook row
    for cb in range(num_codebooks):
        row = f"  CB-{cb}:  "
        for step in range(total_steps):
            if step >= cb:
                row += "  * "  # active
            else:
                row += "  . "  # waiting
        print(row)

    # Complete frame row
    print("  " + "-" * (len(header) - 2))
    row = "  Frame: "
    for step in range(total_steps):
        if step >= num_codebooks - 1:
            row += f"{step - num_codebooks + 1:3d}"
        else:
            row += "  . "
    print(row)
    print()
    print(f"  First complete frame at step {num_codebooks - 1}")
    print(f"  Frame rate: ~12.5 Hz (one frame per step)")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimpleOmni Inference")
    parser.add_argument("--mode", type=str, default="t2a",
                        choices=["t2a", "a2t", "i2a", "stream", "delay_pattern"],
                        help="Inference mode")
    parser.add_argument("--weight", type=str, default=None,
                        help="Path to trained model weights")
    parser.add_argument("--text", type=str, default="Hello, I am Neko.",
                        help="Input text")
    parser.add_argument("--mel_file", type=str, default=None,
                        help="Path to mel tensor (.pt) for a2t mode")
    parser.add_argument("--image_file", type=str, default=None,
                        help="Path to image tensor (.pt) for i2a mode")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.mode == "delay_pattern":
        visualize_delay_pattern(num_codebooks=4, total_steps=16)
    else:
        config = SimpleOmniConfig()
        model = load_model(args.weight, config, args.device)

        if args.mode == "t2a":
            result = text_to_speech(model, args.text, args.max_new_tokens,
                                    args.temperature, args.top_p, args.device)
        elif args.mode == "a2t":
            if args.mel_file and os.path.exists(args.mel_file):
                mel = torch.load(args.mel_file)
            else:
                # Generate random mel for demo
                mel = torch.randn(config.n_mels, 256)
                print("Using random mel spectrogram for demo")
            result = speech_to_text(model, mel, args.max_new_tokens,
                                    args.temperature, args.device)
        elif args.mode == "i2a":
            if args.image_file and os.path.exists(args.image_file):
                image = torch.load(args.image_file)
            else:
                image = torch.randn(3, config.image_size, config.image_size)
                print("Using random image for demo")
            result = image_text_to_speech(model, args.text, image,
                                          args.max_new_tokens,
                                          args.temperature, args.device)
        elif args.mode == "stream":
            streaming_demo(model, args.text, args.max_new_tokens, args.device)
