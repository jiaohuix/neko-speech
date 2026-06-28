"""
Generate Neko Teacher illustrations for Ch01: Audio Fundamentals.

Usage:
    cd skills/image-gen
    python gen_ch01_figures.py

Output:
        ../../chapters/ch01_audio_fundamentals/figures/
"""

import os
import base64
import time
from pathlib import Path

from openai import OpenAI

# Load API key from .env
env_path = Path(__file__).parent.parent.parent / ".env"
API_KEY = ""
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            if line.strip().startswith("IMAGEN_KEY"):
                API_KEY = line.split("=")[1].strip().strip('"')
                break

if not API_KEY:
    raise RuntimeError("IMAGEN_KEY not found in .env")

BASE_URL = "https://api.viviai.cc/v1"
MODEL = "gpt-image-2"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "chapters" / "ch01_audio_fundamentals" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# Universal style prefix (from STYLE_GUIDE.md)
STYLE_PREFIX = (
    "A cute chibi anime catgirl teacher with pink hair, cat ears, glasses, "
    "wearing a white lab coat with pink bow, explaining technical concepts in a friendly way, "
    "soft hand-drawn illustration style mixed with professional academic diagram, "
    "clear labels and arrows, pastel pink-purple-blue color palette, "
    "kawaii 2D anime style with light watercolor and marker texture, "
    "clean educational layout, high clarity, professional yet adorable, "
    "white background with subtle sparkles and cherry blossom elements. "
)


PROMPTS = {
    "01_waveform": (
        STYLE_PREFIX +
        "The catgirl is pointing at a large diagram showing a sine wave on an oscilloscope screen, "
        "with small dots along the wave labeled 'sample points', an arrow showing 'sampling rate = 16000 Hz', "
        "and a zoomed inset showing discrete digital values. "
        "Chinese text labels: '波形' (waveform), '采样' (sampling), '数字化'. "
        "Hand-drawn academic style, soft lines, adorable and precise."
    ),
    "02_fft": (
        STYLE_PREFIX +
        "The catgirl is standing between two large panels. Left panel shows a time-domain waveform with label '时域'. "
        "Right panel shows a frequency spectrum with peaks labeled '440Hz', '1320Hz' and a formula 'X(ω) = Σ x[n]e^(-jωn)'. "
        "A big arrow between them labeled 'FFT'. Chinese text labels mixed with English. "
        "Hand-drawn academic style, colorful spectrum bars, pastel tones."
    ),
    "03_stft": (
        STYLE_PREFIX +
        "The catgirl is pointing at a spectrogram displayed on a screen, showing a diagonal line from bottom-left to top-right "
        "representing a chirp signal going from low to high frequency. "
        "Labels: '时间 (Time)' on x-axis, '频率 (Frequency)' on y-axis, 'STFT' as title. "
        "Small inset showing overlapping window frames sliding across a waveform. "
        "Colorful heatmap in magenta-purple gradient, hand-drawn academic style."
    ),
    "04_mel": (
        STYLE_PREFIX +
        "The catgirl is pointing at two frequency axes side by side. "
        "Left: linear frequency axis 'Hz' with evenly spaced tick marks. "
        "Right: Mel frequency axis with tick marks crowded at the bottom (low freq) and sparse at the top (high freq). "
        "Formula 'mel = 2595·log10(1+f/700)' floating between them. "
        "Below: overlapping triangular filter shapes labeled 'Mel Filter Bank'. "
        "Hand-drawn academic diagram, clear Chinese labels, pastel colors."
    ),
    "05_griffinlim": (
        STYLE_PREFIX +
        "The catgirl is showing a circular arrow diagram: 'Magnitude Spectrum' → 'Random Phase' → 'iSTFT' → 'Waveform' → 'STFT' → back to start. "
        "Labeled 'Griffin-Lim Iteration' with 'Iteration 1, 2, ... 60' shown as small step icons. "
        "Speech bubble from Neko saying '相位丢失了，只能迭代估计！' (Phase is lost, can only estimate iteratively!). "
        "Hand-drawn academic style, cute chibi elements, clear labels in Chinese."
    ),
}


def save_image(data, save_path):
    url = getattr(data, "url", None)
    b64_json = getattr(data, "b64_json", None)
    if url:
        import requests
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        return
    if b64_json:
        with open(save_path, "wb") as f:
            f.write(base64.b64decode(b64_json))
        return
    raise ValueError("No image data found")


def generate_image(prompt, tag):
    print(f"[generate] {tag} ...")
    resp = client.images.generate(model=MODEL, prompt=prompt, size="1024x1024")
    save_path = OUTPUT_DIR / f"{tag}_{int(time.time())}.png"
    save_image(resp.data[0], save_path)
    print(f"[save] {save_path}")
    return save_path


def main():
    print(f"Output directory: {OUTPUT_DIR}")
    for tag, prompt in PROMPTS.items():
        try:
            generate_image(prompt, tag)
            print()
        except Exception as e:
            print(f"[error] {tag}: {e}\n")
            continue
    print("Done!")


if __name__ == "__main__":
    main()
