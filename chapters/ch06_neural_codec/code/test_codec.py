"""
Ch06: Test EnCodec Mini — Codec 质量评估

Usage:
    # 测试未训练模型 (随机权重，观察输出)
    python test_codec.py

    # 测试训练好的模型
    python test_codec.py --ckpt ../checkpoints/codec_best.pt

    # 测试真实音频
    python test_codec.py --ckpt ../checkpoints/codec_best.pt \
        --audio ../../../data/processed/wavs/000001.wav

输出到 outputs/:
    - original.wav     原始音频
    - reconstructed.wav 重建音频
    - tokens.npy       码本索引
    - comparison.png   波形 + 频谱对比图

评估指标:
    - SNR (信噪比): 越高越好，>10dB 表示基本可用
    - LSD (对数谱失真): 越低越好，<2dB 表示质量较好
    - Codebook usage: 码本使用率
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from codec import EnCodecMini


# ============================================================
# Metrics
# ============================================================

def compute_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    信噪比 (Signal-to-Noise Ratio).

    SNR = 10 * log10(||signal||^2 / ||noise||^2)
    越高越好。>10dB 基本可用，>20dB 质量较好。

    未训练模型通常 < 0dB（噪声比信号还大）。
    """
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    noise = original - reconstructed
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def compute_lsd(original: np.ndarray, reconstructed: np.ndarray,
                sample_rate: int = 24000) -> float:
    """
    对数谱失真 (Log Spectral Distortion).

    LSD = mean(sqrt(mean((log10(S_orig/S_recon))^2)))
    越低越好。<2dB 质量较好，>5dB 明显可闻。

    这是评估音频 codec 最常用的客观指标。
    """
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    n_fft = 2048
    hop = 512

    # STFT
    orig_spec = np.abs(np.fft.rfft(
        np.lib.stride_tricks.as_strided(
            np.pad(original, n_fft // 2),
            shape=((len(original) - n_fft) // hop + 1, n_fft),
            strides=(hop * 8, 8),
        ), axis=1
    ))

    recon_spec = np.abs(np.fft.rfft(
        np.lib.stride_tricks.as_strided(
            np.pad(reconstructed, n_fft // 2),
            shape=((len(reconstructed) - n_fft) // hop + 1, n_fft),
            strides=(hop * 8, 8),
        ), axis=1
    ))

    # 避免 log(0)
    floor = 1e-7
    orig_spec = np.maximum(orig_spec, floor)
    recon_spec = np.maximum(recon_spec, floor)

    # LSD in dB
    lsd = np.mean(np.sqrt(np.mean((20 * np.log10(orig_spec / recon_spec)) ** 2, axis=1)))
    return lsd


def compute_pesq_like(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    简化的 "伪 PESQ" 分数.

    不是真正的 PESQ (需要 pesq 库)，而是用相关系数近似。
    范围 [-1, 1]，越高越好。
    """
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    # 归一化
    o = original - original.mean()
    r = reconstructed - reconstructed.mean()

    corr = np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-10)
    return float(corr)


# ============================================================
# Visualization
# ============================================================

def plot_comparison(original, reconstructed, sample_rate, output_path):
    """绘制波形和频谱对比图."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        min_len = min(len(original), len(reconstructed))
        original = original[:min_len]
        reconstructed = reconstructed[:min_len]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # 1. Waveform comparison
        t = np.arange(min_len) / sample_rate
        axes[0].plot(t, original, alpha=0.7, label="Original", linewidth=0.5)
        axes[0].plot(t, reconstructed, alpha=0.7, label="Reconstructed", linewidth=0.5)
        axes[0].set_title("Waveform: Original vs Reconstructed")
        axes[0].set_xlabel("Time (s)")
        axes[0].legend()

        # 2. Original spectrogram
        axes[1].specgram(original, NFFT=1024, Fs=sample_rate, noverlap=768, cmap="magma")
        axes[1].set_title("Original Spectrogram")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Frequency (Hz)")

        # 3. Reconstructed spectrogram
        axes[2].specgram(reconstructed, NFFT=1024, Fs=sample_rate, noverlap=768, cmap="magma")
        axes[2].set_title("Reconstructed Spectrogram")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Frequency (Hz)")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Comparison plot saved to {output_path}")
    except ImportError:
        print("  matplotlib not available, skipping plot")


# ============================================================
# Test Functions
# ============================================================

def generate_test_audio(sample_rate=24000, duration=3.0):
    """生成测试音频: 正弦波 + 谐波 + 噪声."""
    t = np.arange(int(sample_rate * duration)) / sample_rate

    # 基频 220Hz (A3) + 谐波
    wav = np.zeros_like(t)
    f0 = 220
    for h, amp in [(1, 0.5), (2, 0.3), (3, 0.15), (4, 0.08), (5, 0.04)]:
        wav += amp * np.sin(2 * np.pi * f0 * h * t)

    # 加一点颤音 (vibrato)
    vibrato = 0.003 * np.sin(2 * np.pi * 5 * t)
    wav *= (1 + vibrato)

    # 轻微的噪声
    wav += np.random.randn(len(wav)) * 0.005

    # 归一化
    wav = wav / np.abs(wav).max() * 0.9

    return wav.astype(np.float32), sample_rate


def load_model(ckpt_path=None, device="cpu"):
    """加载模型."""
    model = EnCodecMini()

    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if "config" in ckpt:
            config = ckpt["config"]
            model = EnCodecMini(
                base_dim=config.get("base_dim", 32),
                latent_dim=config.get("latent_dim", 128),
                num_codebooks=config.get("num_codebooks", 8),
                num_codes=config.get("num_codes", 1024),
            )
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "?")
        print(f"  Loaded checkpoint: epoch {epoch}")
        return model, True
    else:
        print("  No checkpoint found, using random weights")
        return model, False


def test_codec(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    print("\n[1/4] Loading model...")
    model, trained = load_model(args.ckpt, device)
    model = model.to(device)
    model.eval()
    print(f"  Model: {model.n_params():,} parameters")
    print(f"  Hop length: {model.hop_length}")

    # ---- Load/generate test audio ----
    print("\n[2/4] Loading test audio...")
    if args.audio and Path(args.audio).exists():
        wav_np, sr = sf.read(args.audio, dtype="float32")
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)
        # 裁剪到 5 秒
        max_len = sr * 5
        if len(wav_np) > max_len:
            wav_np = wav_np[:max_len]
        # 补齐到 hop_length 的整数倍
        remainder = len(wav_np) % model.hop_length
        if remainder > 0:
            wav_np = np.pad(wav_np, (0, model.hop_length - remainder))
        print(f"  Loaded: {args.audio} ({len(wav_np)/sr:.2f}s, {sr}Hz)")
    else:
        wav_np, sr = generate_test_audio(sample_rate=24000, duration=3.0)
        print(f"  Generated: synthetic tone (220Hz + harmonics, 3s)")

    sample_rate = sr if args.audio else 24000

    # ---- Encode & Decode ----
    print("\n[3/4] Encoding and decoding...")
    with torch.no_grad():
        wav_tensor = torch.FloatTensor(wav_np).unsqueeze(0).unsqueeze(0).to(device)

        # Encode
        tokens = model.encode(wav_tensor)
        print(f"  Tokens shape: {tokens.shape}")
        print(f"  Compression: {wav_tensor.shape[-1]} samples → {tokens.shape[-1]} frames")
        print(f"  Bitrate: {tokens.shape[1] * 10} bit/frame × "
              f"{sample_rate / model.hop_length:.0f} fps = "
              f"{tokens.shape[1] * 10 * sample_rate / model.hop_length:.0f} bps")

        # Decode
        recon = model.decode(tokens)
        recon_np = recon.squeeze().cpu().numpy()

    # ---- Evaluate ----
    print("\n[4/4] Evaluating reconstruction quality...")

    # 对齐长度
    min_len = min(len(wav_np), len(recon_np))
    wav_eval = wav_np[:min_len]
    recon_eval = recon_np[:min_len]

    snr = compute_snr(wav_eval, recon_eval)
    lsd = compute_lsd(wav_eval, recon_eval, sample_rate)
    corr = compute_pesq_like(wav_eval, recon_eval)

    print(f"\n  {'Metric':<25} {'Value':<15} {'Quality'}")
    print(f"  {'─' * 60}")
    print(f"  {'SNR (dB)':<25} {snr:<15.2f} "
          f"{'Good' if snr > 15 else 'OK' if snr > 5 else 'Poor' if snr > 0 else 'Very Poor'}")
    print(f"  {'LSD (dB)':<25} {lsd:<15.2f} "
          f"{'Good' if lsd < 2 else 'OK' if lsd < 5 else 'Poor'}")
    print(f"  {'Waveform Correlation':<25} {corr:<15.4f} "
          f"{'Good' if corr > 0.8 else 'OK' if corr > 0.5 else 'Poor'}")

    if not trained:
        print("\n  Note: 未训练模型的指标很差是正常的。")
        print("  训练后 SNR 应 > 10dB, Correlation 应 > 0.8")

    # ---- Codebook stats ----
    print("\n  Codebook usage:")
    num_codes = model.quantizer.quantizers[0].num_codes
    for k in range(tokens.shape[1]):
        unique = tokens[0, k, :].unique().numel()
        print(f"    Layer {k}: {unique}/{num_codes} codes used "
              f"({unique / num_codes * 100:.1f}%)")

    # ---- Save outputs ----
    print(f"\n  Saving outputs to {output_dir}/")

    sf.write(output_dir / "original.wav", wav_np, sample_rate)
    sf.write(output_dir / "reconstructed.wav", recon_np, sample_rate)
    np.save(output_dir / "tokens.npy", tokens.cpu().numpy())

    plot_comparison(wav_np, recon_np, sample_rate, output_dir / "comparison.png")

    # ---- Token visualization ----
    _plot_tokens(tokens.cpu().numpy(), output_dir / "tokens.png")

    print(f"\nDone! Listen to outputs/original.wav and outputs/reconstructed.wav")


def _plot_tokens(tokens: np.ndarray, output_path: Path):
    """可视化码本索引."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # tokens: [1, K, T]
        tokens = tokens[0]  # [K, T]
        K, T = tokens.shape

        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(tokens, aspect="auto", cmap="tab20", interpolation="nearest")
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Codebook layer")
        ax.set_title(f"RVQ Tokens ({K} codebooks × {T} frames)")
        ax.set_yticks(range(K))
        ax.set_yticklabels([f"Layer {k}" for k in range(K)])
        plt.colorbar(im, ax=ax, label="Code index")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Token visualization saved to {output_path}")
    except ImportError:
        print("  matplotlib not available, skipping token plot")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Test EnCodec Mini")
    parser.add_argument("--ckpt", type=str, default="../checkpoints/codec_best.pt",
                        help="Path to checkpoint")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to test audio file")
    parser.add_argument("--output-dir", type=str, default="../outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_codec(args)
