"""
Download 1000 Neko Audio samples from ModelScope.

Strategy:
    1. metadata.jsonl already downloaded (contains file_name, talk text)
    2. Parse and sample 1000 entries
    3. Download .wav using file_name directly (e.g. '00/0.wav')

Usage:
    python download_neko_1k.py --output-dir ../data/processed
"""

import argparse
import csv
import json
import random
from pathlib import Path

import requests
import soundfile as sf
from tqdm import tqdm

DATASET_ID = "liumindmind/Neko_Audio-80K_Short"
MODELSCOPE_ENDPOINT = "https://www.modelscope.cn"


def download_file(repo_file_path, local_path):
    """Download a single file from ModelScope dataset repo."""
    url = f"{MODELSCOPE_ENDPOINT}/api/v1/datasets/{DATASET_ID}/repo?Revision=master&FilePath={repo_file_path}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(r.content)
    return local_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="../data/processed")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(exist_ok=True)

    meta_path = output_dir / "metadata.jsonl"
    if not meta_path.exists():
        print(f"[error] {meta_path} not found. Please download metadata first.")
        return

    # Parse metadata
    print(f"[1/2] Parsing metadata and sampling {args.num_samples} entries...")
    all_entries = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                all_entries.append(entry)
            except json.JSONDecodeError:
                continue

    print(f"       Total entries: {len(all_entries)}")

    random.seed(args.seed)
    selected = random.sample(all_entries, min(args.num_samples, len(all_entries)))

    # Check existing files for incremental download
    existing_wavs = sorted(wav_dir.glob("*.wav"))
    start_idx = len(existing_wavs)
    print(f"[info] Found {start_idx} existing wav files, continuing from {start_idx+1}")

    # Download wav files
    print(f"[2/2] Downloading {len(selected)} wav files...")
    manifest_path = output_dir / "train.list"
    csv_rows = []
    success_count = 0

    # Append mode for manifest
    manifest_mode = "a" if start_idx > 0 else "w"
    with open(manifest_path, manifest_mode, encoding="utf-8") as f_manifest:
        for i, entry in enumerate(tqdm(selected)):
            # file_name is like "00/0.wav" - use directly
            repo_path = entry.get("file_name", "")
            if not repo_path:
                continue

            # Use 'talk' field for TTS text (shorter, more natural)
            text = str(entry.get("talk", "")).replace("\n", " ")
            if not text:
                text = str(entry.get("output", "")).replace("\n", " ")

            speaker = "neko"
            language = "zh"

            wav_name = f"{start_idx + i + 1:06d}.wav"
            local_wav = wav_dir / wav_name

            # Skip if already exists
            if local_wav.exists():
                success_count += 1
                continue

            try:
                download_file(repo_path, local_wav)
            except Exception as e:
                tqdm.write(f"       Skip {repo_path}: {e}")
                continue

            # Verify audio
            try:
                info = sf.info(local_wav)
                sr = info.samplerate
                duration = info.duration
            except Exception:
                sr = 0
                duration = 0

            f_manifest.write(f"wavs/{wav_name}|{speaker}|{language}|{text}\n")

            csv_rows.append({
                "wav_path": f"wavs/{wav_name}",
                "speaker": speaker,
                "language": language,
                "text": text,
                "sample_rate": sr,
                "duration": duration,
            })
            success_count += 1

    # Write metadata.csv (append mode)
    csv_path = output_dir / "metadata.csv"
    csv_exists = csv_path.exists()
    with open(csv_path, "a" if csv_exists else "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_path", "speaker", "language", "text", "sample_rate", "duration"])
        if not csv_exists:
            writer.writeheader()
        writer.writerows(csv_rows)

    # Write dataset_info.json
    import json as json_mod
    total_wavs_now = len(list(wav_dir.glob("*.wav")))
    info = {
        "dataset": DATASET_ID,
        "source": "modelscope",
        "num_samples": total_wavs_now,
        "newly_downloaded": success_count,
        "requested": len(selected),
        "total_original": len(all_entries),
    }
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json_mod.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n[done] Successfully downloaded {success_count}/{len(selected)} new samples")
    print(f"       Total wavs: {total_wavs_now}")
    print(f"       wavs: {wav_dir}")
    print(f"       manifest: {manifest_path}")
    print(f"       metadata: {csv_path}")


if __name__ == "__main__":
    main()
