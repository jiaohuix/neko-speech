"""
Download and prepare Neko Audio dataset from HuggingFace.

Uses hf-mirror.co for China mainland access.

Usage:
    # Download full dataset (80K samples)
    HF_ENDPOINT=https://hf-mirror.com python prepare.py --output-dir ./processed

    # Download a subset for quick testing
    HF_ENDPOINT=https://hf-mirror.com python prepare.py --num-samples 1000 --output-dir ./processed_1k

Output format (GPT-SoVITS compatible):
    processed/
    ├── wavs/
    │   ├── 000001.wav
    │   ├── 000002.wav
    │   └── ...
    ├── train.list          # manifest: wav_path|speaker|language|text
    ├── metadata.csv        # structured metadata
    └── dataset_info.json   # dataset stats
"""

import argparse
import json
import os
import random
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm


# --------------------------------------------------------
# Config
# --------------------------------------------------------
DATASET_NAME = "liumindmind/Neko_Audio-80K_Short"
SPLIT = "train"
DEFAULT_OUTPUT = "processed"


# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def setup_hf_mirror():
    """Use hf-mirror.co if in mainland China."""
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"[info] Set HF_ENDPOINT={os.environ['HF_ENDPOINT']}")


def guess_columns(dataset):
    """Auto-detect text/audio/speaker columns."""
    features = dataset.features
    candidates = {
        "text": ["text", "sentence", "transcript", "transcription", "caption"],
        "audio": ["audio", "wav", "speech"],
        "speaker": ["speaker", "speaker_id", "spk"],
        "language": ["language", "lang"],
    }
    detected = {}
    for key, opts in candidates.items():
        for opt in opts:
            if opt in features:
                detected[key] = opt
                break
        if key not in detected:
            detected[key] = None
    return detected


def inspect_dataset(dataset, detected):
    """Print dataset info."""
    print("=" * 60)
    print("Dataset Schema")
    print("=" * 60)
    for k, v in dataset.features.items():
        marker = ""
        for role, col in detected.items():
            if col == k:
                marker = f" <-- {role}"
        print(f"  {k:20s}: {v}{marker}")
    print(f"  {'total_samples':20s}: {len(dataset)}")
    print("=" * 60)


# --------------------------------------------------------
# Export
# --------------------------------------------------------

def export_dataset(dataset, output_dir, detected, num_samples=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(exist_ok=True)

    audio_key = detected["audio"]
    text_key = detected.get("text")
    speaker_key = detected.get("speaker")

    # Ensure audio column is decoded
    dataset = dataset.cast_column(audio_key, Audio())

    manifest_path = output_dir / "train.list"
    metadata_rows = []

    with open(manifest_path, "w", encoding="utf-8") as f_manifest:
        for idx, sample in enumerate(tqdm(dataset, desc="Exporting")):
            if num_samples and idx >= num_samples:
                break

            audio = sample[audio_key]
            waveform = audio["array"]
            sr = audio["sampling_rate"]

            wav_name = f"{idx + 1:06d}.wav"
            wav_path = wav_dir / wav_name
            sf.write(wav_path, waveform, sr)

            text = str(sample[text_key]).replace("\n", " ") if text_key else ""
            speaker = str(sample[speaker_key]) if speaker_key else "neko"
            language = "zh"

            # GPT-SoVITS manifest format: wav_path|speaker|language|text
            f_manifest.write(f"wavs/{wav_name}|{speaker}|{language}|{text}\n")

            metadata_rows.append({
                "wav_path": f"wavs/{wav_name}",
                "speaker": speaker,
                "language": language,
                "text": text,
                "sample_rate": sr,
                "duration": len(waveform) / sr,
            })

    # Save metadata.csv
    import csv
    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_path", "speaker", "language", "text", "sample_rate", "duration"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Save dataset_info.json
    info = {
        "dataset": DATASET_NAME,
        "num_samples": len(metadata_rows),
        "audio_key": audio_key,
        "text_key": text_key,
        "speaker_key": speaker_key,
        "columns": detected,
    }
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n[done] Exported {len(metadata_rows)} samples to {output_dir}")
    print(f"       wavs: {wav_dir}")
    print(f"       manifest: {manifest_path}")
    print(f"       metadata: {csv_path}")


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare Neko Audio dataset")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=None, help="Subset size (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_hf_mirror()
    random.seed(args.seed)

    print(f"Loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)

    detected = guess_columns(dataset)
    inspect_dataset(dataset, detected)

    if detected["audio"] is None:
        raise RuntimeError("No audio column found in dataset!")

    # If sampling subset, shuffle first
    if args.num_samples and args.num_samples < len(dataset):
        print(f"\nSampling {args.num_samples} / {len(dataset)} samples...")
        indices = random.sample(range(len(dataset)), args.num_samples)
        dataset = dataset.select(indices)

    export_dataset(dataset, args.output_dir, detected, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
