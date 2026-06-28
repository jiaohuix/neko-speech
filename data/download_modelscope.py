"""
Download Neko Audio dataset from ModelScope (国内镜像).

Usage:
    python download_modelscope.py --num-samples 1000 --output-dir ./processed

Requires:
    pip install modelscope
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import soundfile as sf
from modelscope.msdatasets import MsDataset
from tqdm import tqdm


DATASET_NAME = "liumindmind/Neko_Audio-80K_Short"


def inspect_dataset(ds):
    print("=" * 60)
    print("Dataset Info")
    print("=" * 60)
    print(f"  Total samples: {len(ds)}")
    sample = ds[0]
    print(f"  Keys: {list(sample.keys())}")
    for k, v in sample.items():
        if k == "audio":
            print(f"  {k}: dict with keys {list(v.keys())}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)


def export_subset(ds, output_dir, num_samples=None, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(exist_ok=True)

    random.seed(seed)
    indices = list(range(len(ds)))
    if num_samples and num_samples < len(ds):
        indices = random.sample(indices, num_samples)

    manifest_path = output_dir / "train.list"
    metadata_rows = []

    with open(manifest_path, "w", encoding="utf-8") as f_manifest:
        for idx in tqdm(indices, desc="Exporting"):
            sample = ds[idx]

            # Audio
            audio = sample["audio"]
            waveform = audio["array"]
            sr = audio["sampling_rate"]

            wav_name = f"{idx + 1:06d}.wav"
            wav_path = wav_dir / wav_name
            sf.write(wav_path, waveform, sr)

            # Text
            text = str(sample.get("text", "")).replace("\n", " ")

            # Speaker / Language
            speaker = str(sample.get("speaker", "neko"))
            language = str(sample.get("language", "zh"))

            # GPT-SoVITS manifest format
            f_manifest.write(f"wavs/{wav_name}|{speaker}|{language}|{text}\n")

            metadata_rows.append({
                "wav_path": f"wavs/{wav_name}",
                "speaker": speaker,
                "language": language,
                "text": text,
                "sample_rate": sr,
                "duration": len(waveform) / sr,
            })

    # metadata.csv
    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_path", "speaker", "language", "text", "sample_rate", "duration"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    # dataset_info.json
    info = {
        "dataset": DATASET_NAME,
        "source": "modelscope",
        "num_samples": len(metadata_rows),
        "total_original": len(ds),
    }
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n[done] Exported {len(metadata_rows)} samples to {output_dir}")
    print(f"       wavs: {wav_dir}")
    print(f"       manifest: {manifest_path}")
    print(f"       metadata: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Download Neko Audio from ModelScope")
    parser.add_argument("--output-dir", type=str, default="processed")
    parser.add_argument("--num-samples", type=int, default=None, help="Subset size (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading dataset from ModelScope: {DATASET_NAME}...")
    ds = MsDataset.load(DATASET_NAME)

    inspect_dataset(ds)

    export_subset(ds, args.output_dir, num_samples=args.num_samples, seed=args.seed)


if __name__ == "__main__":
    main()
