"""
Ch08: Inference Script — Generate Speech with All Three Models

Demonstrates:
    1. F5-TTS:      Flow Matching sampling (ODE integration)
    2. CosyVoice:   Zero-shot voice cloning (3s reference → new voice)
    3. IndexTTS:    Pinyin/tone control (change tone → hear difference)

Usage:
    python inference.py --model f5_tts
    python inference.py --model cosyvoice
    python inference.py --model indextts
    python inference.py --model all

Note: These use randomly initialized weights → output is noise.
      For real audio, load trained checkpoints.
"""

import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from f5_tts import F5TTS, SimpleTextEncoder
from cosyvoice import CosyVoice
from indextts import IndexTTS


def save_mel(mel, path, title="Generated Mel Spectrogram"):
    """Save mel spectrogram as image."""
    fig, ax = plt.subplots(figsize=(10, 4))
    # mel: (1, T, mel_dim) → (mel_dim, T)
    m = mel[0].cpu().numpy().T
    ax.imshow(m, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Mel bins")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def demo_f5_tts():
    """F5-TTS: Flow Matching inference with Euler ODE integration."""
    print("\n" + "=" * 60)
    print("F5-TTS: Flow Matching Inference")
    print("=" * 60)

    mel_dim, text_dim, dim = 80, 256, 256
    text_enc = SimpleTextEncoder(text_dim=text_dim)
    model = F5TTS(
        mel_dim=mel_dim, text_dim=text_dim, dim=dim,
        heads=4, n_layers=4, n_inference_steps=20,
    )
    model.eval()

    # Input: "Hello Neko" (dummy token ids)
    text_ids = torch.tensor([[72, 101, 108, 108, 111, 32, 78, 101, 107, 111]])
    text_mask = torch.zeros(1, text_ids.size(1), dtype=torch.bool)
    ref_mel = torch.randn(1, 50, mel_dim)   # 0.5s reference audio
    ref_mask = torch.zeros(1, 50, dtype=torch.bool)

    text_emb = text_enc(text_ids)

    # Generate with different number of ODE steps
    for n_steps in [5, 10, 20]:
        mel = model.sample(
            n_frames=100,
            text_emb=text_emb, text_mask=text_mask,
            ref_mel=ref_mel, ref_mask=ref_mask,
            n_steps=n_steps,
        )
        print(f"  n_steps={n_steps:2d} → mel shape: {tuple(mel.shape)}, "
              f"range: [{mel.min():.2f}, {mel.max():.2f}]")

    save_mel(mel, "outputs/f5_tts_mel.png",
             "F5-TTS: Flow Matching (20 Euler steps)")
    print("  Key insight: Flow Matching needs only 10-20 ODE steps")
    print("  (vs 50-1000 for diffusion) because trajectories are straight lines.")


def demo_cosyvoice():
    """CosyVoice: Zero-shot voice cloning demo."""
    print("\n" + "=" * 60)
    print("CosyVoice: Zero-Shot Voice Cloning")
    print("=" * 60)

    mel_dim, text_dim, spk_dim, dim = 80, 256, 128, 256
    model = CosyVoice(
        mel_dim=mel_dim, text_dim=text_dim,
        spk_dim=spk_dim, dim=dim, heads=4, n_layers=4,
    )
    model.eval()

    text_enc = SimpleTextEncoder(text_dim=text_dim)
    text_ids = torch.tensor([[72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]])
    text_mask = torch.zeros(1, text_ids.size(1), dtype=torch.bool)
    text_emb = text_enc(text_ids)

    # Two different speakers' reference audio (3 seconds = ~300 frames at 100Hz)
    torch.manual_seed(1)
    speaker_a_ref = torch.randn(1, 300, mel_dim)
    torch.manual_seed(2)
    speaker_b_ref = torch.randn(1, 300, mel_dim)
    dummy_mask = torch.zeros(1, 300, dtype=torch.bool)

    # Clone speaker A
    mel_a = model.sample(
        n_frames=100,
        text_emb=text_emb, text_mask=text_mask,
        ref_mel=speaker_a_ref, ref_mask=dummy_mask,
    )

    # Clone speaker B (same text, different reference)
    mel_b = model.sample(
        n_frames=100,
        text_emb=text_emb, text_mask=text_mask,
        ref_mel=speaker_b_ref, ref_mask=dummy_mask,
    )

    # Compare: same text, different reference → different voice
    diff = (mel_a - mel_b).abs().mean().item()
    print(f"  Speaker A mel shape: {tuple(mel_a.shape)}")
    print(f"  Speaker B mel shape: {tuple(mel_b.shape)}")
    print(f"  Mean |mel_A - mel_B|: {diff:.4f}")
    print("  → Same text + different 3s reference = different voice!")
    print("  This is zero-shot voice cloning: no retraining needed.")

    save_mel(mel_a, "outputs/cosyvoice_speaker_a.png",
             "CosyVoice: Speaker A (3s reference)")
    save_mel(mel_b, "outputs/cosyvoice_speaker_b.png",
             "CosyVoice: Speaker B (3s reference)")


def demo_indextts():
    """IndexTTS: Pinyin tone control demo."""
    print("\n" + "=" * 60)
    print("IndexTTS: Pinyin/Tone Control")
    print("=" * 60)

    mel_dim, pinyin_dim, dim = 80, 128, 128
    model = IndexTTS(
        mel_dim=mel_dim, pinyin_dim=pinyin_dim, dim=dim,
        heads=4, n_layers=4,
    )
    model.eval()

    # "ma1 ma2 ma3 ma4" — same syllable, 4 tones
    # initial: m=12 (arbitrary), final: a=0, tones: 1,2,3,4
    N = 4
    initials = torch.tensor([[12, 12, 12, 12]])
    finals = torch.tensor([[0, 0, 0, 0]])
    syl_mask = torch.zeros(1, N, dtype=torch.bool)

    # Fixed duration: 15 frames per syllable
    durations = torch.ones(1, N) * 15.0

    print("  Generating 'ma' in 4 tones: ma1(妈) ma2(麻) ma3(马) ma4(骂)")
    with torch.no_grad():
        mels = []
        for tone in [1, 2, 3, 4]:
            tones = torch.full((1, N), tone)
            mel, dur_pred, frame_lens = model(
                initials, finals, tones, syl_mask, durations=durations
            )
            mels.append(mel)
            print(f"    tone={tone} → mel shape: {tuple(mel.shape)}, "
                  f"total frames: {frame_lens[0].item()}")

    # Show that different tones produce different mel
    for i in range(len(mels) - 1):
        diff = (mels[i] - mels[i + 1]).abs().mean().item()
        print(f"  |tone{i+1} - tone{i+2}|: {diff:.4f}")

    save_mel(mels[0], "outputs/indextts_tone1_ma.png",
             "IndexTTS: ma1 (妈, high level)")
    save_mel(mels[2], "outputs/indextts_tone3_ma.png",
             "IndexTTS: ma3 (马, dipping)")

    print("  Key insight: explicit tone control solves multi-pronunciation")
    print("  characters (多音字) — a major pain point in Chinese TTS.")


# --------------------------------------------------------
# Main
# --------------------------------------------------------

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["f5_tts", "cosyvoice", "indextts", "all"],
                        default="all")
    args = parser.parse_args()

    print("Ch08 Inference Demo")
    print("(Using randomly initialized weights — output is noise,")
    print(" but the pipeline is real. Load checkpoints for real audio.)")

    if args.model in ("f5_tts", "all"):
        demo_f5_tts()
    if args.model in ("cosyvoice", "all"):
        demo_cosyvoice()
    if args.model in ("indextts", "all"):
        demo_indextts()

    print("\nDone. Check outputs/ for mel spectrograms.")
