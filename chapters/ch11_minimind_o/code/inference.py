"""
SimpleOmni Inference Demo
=========================

Demonstrates text-to-speech and audio-to-audio generation
using the trained SimpleOmni model.

Modes:
  1. Text-to-Speech (T2A): Type text → get speech output
  2. Audio-to-Audio (A2A): Speak into microphone → get speech response
  3. Voice Cloning: Use reference audio to control output voice

Based on eval_omni.py from the MiniMind-O repository.

Usage:
    # Text-to-Speech
    python inference.py --mode t2a --weight ./checkpoints/a2a_epoch1.pth

    # Audio-to-Audio (with reference audio)
    python inference.py --mode a2a --weight ./checkpoints/a2a_epoch1.pth --ref_audio ref.wav

    # Voice cloning
    python inference.py --mode clone --weight ./checkpoints/a2a_epoch1.pth --ref_voice speaker.wav
"""

import os
import argparse
import torch
import numpy as np

from model import SimpleOmni, SimpleOmniConfig, count_parameters


# ===========================================================================
# Model Loading
# ===========================================================================

def load_model(weight_path: str, config: SimpleOmniConfig = None,
               device: str = "cuda") -> tuple:
    """Load trained SimpleOmni model.

    In a complete implementation, this would also load:
    - SenseVoice (frozen audio encoder) for speech input
    - Mimi (frozen audio codec) for speech output decoding
    - Tokenizer for text processing
    """
    config = config or SimpleOmniConfig()
    model = SimpleOmni(config)

    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weight_path}")

    model = model.half().eval().to(device)
    count_parameters(model, verbose=False)
    return model


# ===========================================================================
# Text-to-Speech Generation
# ===========================================================================

def text_to_speech(model, text: str, tokenizer=None, max_new_tokens=256,
                   temperature=0.75, top_p=0.9, device="cuda"):
    """Generate speech from text input.

    Pipeline:
    1. Tokenize text → input_ids
    2. Model forward → text_logits + audio_logits
    3. Sample text tokens autoregressively
    4. Sample audio codes with delay pattern
    5. Decode Mimi codes → 24kHz waveform

    Args:
        model: trained SimpleOmni model
        text: input text string
        tokenizer: text tokenizer (TODO: integrate)
        max_new_tokens: max tokens to generate
        temperature: sampling temperature
        top_p: nucleus sampling threshold
        device: compute device

    Returns:
        dict with 'text_output' (str) and 'audio' (numpy array at 24kHz)
    """
    # TODO: implement with tokenizer
    # For now, demonstrate the generation loop structure

    print(f"[Thinker]: Generating response to '{text}'...")

    # Placeholder: in real implementation, tokenize and generate
    # input_ids = tokenizer.encode(text)
    # for step in range(max_new_tokens):
    #     out = model(input_ids)
    #     text_token = sample(out['text_logits'])
    #     audio_codes = sample_mtp(out['audio_logits'])
    #     if audio_step >= num_codebooks:
    #         frame = read_diagonal(audio_codes)
    #         audio_chunk = mimi.decode(frame)
    #         play(audio_chunk)  # streaming!

    return {
        "text_output": "TODO: implement generation",
        "audio": None,
    }


# ===========================================================================
# Audio-to-Audio Generation
# ===========================================================================

def audio_to_audio(model, audio_path: str, tokenizer=None,
                   max_new_tokens=256, device="cuda"):
    """Generate speech response to speech input.

    Pipeline:
    1. Load audio → resample to 16kHz
    2. SenseVoice encoder → audio features
    3. Audio projector → inject into Thinker sequence
    4. Generate response (same as T2A from here)

    Args:
        model: trained SimpleOmni model
        audio_path: path to input audio file (wav/mp3)
        tokenizer: text tokenizer
        max_new_tokens: max tokens to generate
        device: compute device
    """
    print(f"[A2A]: Processing audio from '{audio_path}'...")

    # TODO: implement
    # wav, sr = sf.read(audio_path)
    # if sr != 16000: wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    # features = sensevoice.encode(wav)
    # projected = model.audio_proj(features)
    # ... generate response ...

    return {
        "text_output": "TODO: implement",
        "audio": None,
    }


# ===========================================================================
# Voice Cloning
# ===========================================================================

def voice_clone(model, text: str, reference_audio: str,
                tokenizer=None, max_new_tokens=256, device="cuda"):
    """Generate speech in a target voice from reference audio.

    Pipeline:
    1. Encode reference audio → ref_codes (Mimi) + spk_emb (CAM++)
    2. Inject ref_codes and spk_emb into generation
    3. Generate speech that mimics the reference voice

    The model uses two voice signals:
    - ref_codes: Mimi-encoded audio placed before the response (context)
    - spk_emb: CAM++ speaker embedding projected into Talker hidden space

    During training, ref_codes are dropped 50% of the time, so the model
    learns to rely on spk_emb as a stable voice anchor.
    """
    print(f"[Clone]: Generating '{text}' in voice from '{reference_audio}'...")

    # TODO: implement
    # ref_wav = load_audio(reference_audio, sr=24000)
    # ref_codes = mimi.encode(ref_wav)  # (8, T_ref) or (4, T_ref)
    # spk_emb = campp.encode(ref_wav)   # (192,)
    # ... generate with conditioning ...

    return {
        "text_output": "TODO: implement",
        "audio": None,
    }


# ===========================================================================
# Streaming Playback (Conceptual)
# ===========================================================================

def streaming_playback_demo():
    """Demonstrate how streaming audio playback works.

    The key idea: Mimi can decode partial code sequences incrementally.
    As soon as all codebooks have produced codes for a time step,
    that frame can be decoded and played.

    Timeline for 4 codebooks (teaching version):
    Step 0: CB-0 produces code → wait
    Step 1: CB-1 produces code → wait
    Step 2: CB-2 produces code → wait
    Step 3: CB-3 produces code → FRAME 0 complete! → decode & play
    Step 4: CB-0 produces code → wait
    Step 5: CB-1 produces code → wait
    Step 6: CB-2 produces code → wait
    Step 7: CB-3 produces code → FRAME 1 complete! → decode & play
    ...

    Latency = num_codebooks / text_generation_rate
    For 4 codebooks at ~20 tokens/sec: 4/20 = 0.2 sec to first audio
    For 8 codebooks at ~20 tokens/sec: 8/20 = 0.4 sec to first audio
    """
    print("Streaming playback demo (conceptual)")
    print("=" * 50)
    num_codebooks = 4
    for step in range(12):
        active_cbs = [i for i in range(num_codebooks) if step >= i]
        frame_ready = step >= num_codebooks - 1 and (step - num_codebooks + 1) % 1 == 0
        print(f"  Step {step:2d}: active CBs = {active_cbs}, "
              f"frame ready = {frame_ready}")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimpleOmni Inference")
    parser.add_argument("--mode", type=str, default="t2a",
                        choices=["t2a", "a2a", "clone", "stream_demo"],
                        help="Inference mode")
    parser.add_argument("--weight", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--text", type=str, default="Hello, I am Neko.",
                        help="Input text (for t2a and clone modes)")
    parser.add_argument("--audio", type=str, default=None,
                        help="Input audio file (for a2a mode)")
    parser.add_argument("--ref_voice", type=str, default=None,
                        help="Reference voice for cloning")
    parser.add_argument("--output", type=str, default="./output.wav",
                        help="Output audio file path")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.mode == "stream_demo":
        streaming_playback_demo()
    else:
        config = SimpleOmniConfig()
        model = load_model(args.weight, config, args.device)

        if args.mode == "t2a":
            result = text_to_speech(model, args.text, device=args.device)
        elif args.mode == "a2a":
            if not args.audio:
                print("Error: --audio required for a2a mode")
            else:
                result = audio_to_audio(model, args.audio, device=args.device)
        elif args.mode == "clone":
            if not args.ref_voice:
                print("Error: --ref_voice required for clone mode")
            else:
                result = voice_clone(model, args.text, args.ref_voice, device=args.device)
