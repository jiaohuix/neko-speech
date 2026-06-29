#!/usr/bin/env python3
"""
Verify trained models and generate test audio.
Checks training completion, generates audio, verifies quality.
"""

import json
import subprocess
import sys
from pathlib import Path

import torch
import torchaudio

def check_training_complete(model_name, save_dir):
    """Check if training completed (final checkpoint exists)."""
    save_dir = Path(save_dir)

    checkpoint_patterns = {
        "ch02_tacotron": ["tacotron_final.pt"],
        "ch03_wavenet": ["wavenet_final.pt"],
        "ch04_fastspeech": ["fs2_final.pt"],
        "ch05_vits": ["vits_final.pt"],
    }

    for pattern in checkpoint_patterns.get(model_name, []):
        if (save_dir / pattern).exists():
            return True
    return False


def get_audio_info(audio_path):
    """Get audio file info (duration, sample rate)."""
    try:
        waveform, sr = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sr
        return {
            "duration_sec": duration,
            "sample_rate": sr,
            "channels": waveform.shape[0],
            "samples": waveform.shape[1],
        }
    except Exception as e:
        return {"error": str(e)}


def generate_test_audio(model_name, checkpoint_dir, output_dir):
    """Generate test audio using trained model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model-specific inference commands
    commands = {
        "ch02_tacotron": [
            "python", "chapters/ch02_tacotron/code/inference.py",
            "--checkpoint", f"{checkpoint_dir}/tacotron_final.pt",
            "--text", "你好，这是一只可爱的猫娘在说话。",
            "--output", f"{output_dir}/tacotron_test.wav",
        ],
        "ch03_wavenet": [
            "python", "chapters/ch03_wavenet/code/inference.py",
            "--checkpoint", f"{checkpoint_dir}/wavenet_final.pt",
            "--text", "喵喵喵，我是猫娘。",
            "--output", f"{output_dir}/wavenet_test.wav",
        ],
        "ch04_fastspeech": [
            "python", "chapters/ch04_fastspeech/code/inference.py",
            "--checkpoint", f"{checkpoint_dir}/fs2_final.pt",
            "--text", "今天天气真好，适合出去玩。",
            "--output", f"{output_dir}/fastspeech_test.wav",
        ],
        "ch05_vits": [
            "python", "chapters/ch05_vits/code/inference.py",
            "--checkpoint", f"{checkpoint_dir}/vits_final.pt",
            "--text", "欢迎来到猫娘语音合成系统。",
            "--output", f"{output_dir}/vits_test.wav",
        ],
    }

    cmd = commands.get(model_name)
    if not cmd:
        return {"error": f"Unknown model: {model_name}"}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/home/jhx/Projects/AIGC/neko-speech"
        )

        output_file = Path(cmd[-1])
        if output_file.exists():
            info = get_audio_info(output_file)
            info["returncode"] = result.returncode
            info["stdout"] = result.stdout[:500]
            return info
        else:
            return {
                "error": "Output file not created",
                "returncode": result.returncode,
                "stderr": result.stderr[:500],
            }
    except subprocess.TimeoutExpired:
        return {"error": "Inference timed out"}
    except Exception as e:
        return {"error": str(e)}


def main():
    """Main verification loop."""
    models = [
        ("ch02_tacotron", "experiments/logs/ch02_tacotron_20ep"),
        ("ch03_wavenet", "experiments/logs/ch03_wavenet_20ep"),
        ("ch04_fastspeech", "experiments/logs/ch04_fastspeech_20ep"),
        ("ch05_vits", "experiments/logs/ch05_vits_20ep"),
    ]

    output_dir = Path("experiments/verified_audio")
    output_dir.mkdir(exist_ok=True)

    results = {}

    for model_name, checkpoint_dir in models:
        print(f"\n{'='*60}")
        print(f"Processing {model_name}...")
        print(f"{'='*60}")

        # Check if training complete
        if not check_training_complete(model_name, checkpoint_dir):
            print(f"  ⏳ Training not complete yet")
            results[model_name] = {"status": "training"}
            continue

        print(f"  ✓ Training complete")

        # Generate test audio
        print(f"  Generating test audio...")
        audio_info = generate_test_audio(model_name, checkpoint_dir, output_dir)

        if "error" in audio_info:
            print(f"  ✗ Error: {audio_info['error']}")
            results[model_name] = {"status": "error", "error": audio_info["error"]}
        else:
            duration = audio_info.get("duration_sec", 0)
            print(f"  ✓ Generated audio: {duration:.2f}s, {audio_info.get('sample_rate', 0)}Hz")

            # Verify quality
            if duration < 0.1:
                print(f"  ✗ Audio too short (<0.1s) - likely invalid")
                results[model_name] = {"status": "invalid", "duration": duration}
            elif duration < 0.5:
                print(f"  ⚠ Audio short ({duration:.2f}s) - may be incomplete")
                results[model_name] = {"status": "short", "duration": duration}
            else:
                print(f"  ✓ Audio quality OK")
                results[model_name] = {"status": "ok", "duration": duration}

    # Save results
    results_file = output_dir / "verification_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Verification Summary")
    print(f"{'='*60}")

    for model, info in results.items():
        status = info.get("status", "unknown")
        if status == "ok":
            print(f"  ✓ {model}: {info['duration']:.2f}s")
        elif status == "training":
            print(f"  ⏳ {model}: Still training...")
        elif status == "error":
            print(f"  ✗ {model}: {info.get('error', 'Unknown error')}")
        else:
            print(f"  ? {model}: {status}")

    print(f"\nResults saved to: {results_file}")

    # Return non-zero if any model failed
    failed = [m for m, info in results.items() if info.get("status") in ["error", "invalid"]]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
