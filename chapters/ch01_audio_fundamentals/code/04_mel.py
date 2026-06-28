"""
Ch01-04: Mel Spectrogram — 梅尔频谱

运行:
    python 04_mel.py --input path/to/audio.wav
    python 04_mel.py --generate

输出:
    - mel.png : 梅尔频谱图

核心思想：
    人耳对低频敏感、高频不敏感。
    Mel 刻度模拟人耳感知：mel = 2595 * log10(1 + f/700)
    在 Mel 频谱上，相同的"感知距离"对应相同的频率变化。
"""

import argparse
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_audio(path):
    wave, sr = sf.read(path)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    return wave, sr


def generate_speech_like(duration=2.0, sr=16000, out_path="mel_demo.wav"):
    """
    生成一个类似语音的信号（多个变化的共振峰），用于展示 Mel 频谱。
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = np.zeros_like(t)
    # 模拟基频 + 几个共振峰
    f0 = 150  # 基频（类似女声）
    wave += 0.4 * np.sin(2 * np.pi * f0 * t)
    wave += 0.2 * np.sin(2 * np.pi * (f0 * 3) * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
    wave += 0.15 * np.sin(2 * np.pi * (f0 * 5) * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    # 加一些噪声
    wave += 0.05 * np.random.randn(len(t))
    wave = wave / np.max(np.abs(wave)) * 0.8
    sf.write(out_path, wave, sr)
    print(f"[generate] 已生成: {out_path} (模拟语音特征)")
    return wave, sr


def stft(wave, n_fft=1024, hop_length=256):
    """纯 NumPy STFT（复用 03_stft.py）"""
    window = np.hamming(n_fft)
    pad_len = n_fft // 2
    wave = np.pad(wave, (pad_len, pad_len), mode="constant")
    n_frames = 1 + (len(wave) - n_fft) // hop_length
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        start = i * hop_length
        frame = wave[start:start + n_fft]
        if len(frame) < n_fft:
            break
        stft_matrix[:, i] = np.fft.rfft(frame * window, n=n_fft)
    return stft_matrix


def hz_to_mel(hz):
    """频率(Hz) → Mel"""
    return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel):
    """Mel → 频率(Hz)"""
    return 700 * (10 ** (mel / 2595.0) - 1)


def mel_filter_bank(n_mels=80, n_fft=1024, sr=16000, f_min=0, f_max=None):
    """
    构建 Mel 滤波器组。

    Returns:
        mel_filter: (n_mels, n_fft//2 + 1) 的三角滤波器矩阵
    """
    if f_max is None:
        f_max = sr // 2

    # FFT 对应的频率轴
    fft_freqs = np.linspace(0, sr // 2, n_fft // 2 + 1)

    # Mel 刻度上的均匀分布点（包括两端）
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # 构建三角滤波器
    mel_filter = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(n_mels):
        # 第 i 个三角：左中右三个点
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        # 上升沿
        up = (fft_freqs - left) / (center - left)
        # 下降沿
        down = (right - fft_freqs) / (right - center)

        # 取 max(0, min(up, down))
        mel_filter[i] = np.maximum(0, np.minimum(up, down))

    return mel_filter


def compute_mel_spectrogram(wave, sr=16000, n_fft=1024, hop_length=256, n_mels=80):
    """
    计算梅尔频谱。

    流程:
        wave → STFT → 幅度谱 → Mel 滤波器组 → 取对数
    """
    # 1. STFT
    stft_matrix = stft(wave, n_fft=n_fft, hop_length=hop_length)
    # 2. 幅度谱
    magnitude = np.abs(stft_matrix)
    # 3. Mel 滤波器组
    mel_filter = mel_filter_bank(n_mels=n_mels, n_fft=n_fft, sr=sr)
    # 4. 应用滤波器：(n_mels, n_fft//2+1) @ (n_fft//2+1, n_frames)
    mel_spec = mel_filter @ magnitude
    # 5. 取对数（加 epsilon 防止 log(0)）
    log_mel_spec = np.log(mel_spec + 1e-10)
    return log_mel_spec, mel_spec


def plot_mel_spectrogram(log_mel_spec, sr, hop_length, out_path="mel.png"):
    """绘制梅尔频谱图。"""
    n_frames = log_mel_spec.shape[1]
    duration = n_frames * hop_length / sr
    times = np.linspace(0, duration, n_frames)

    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.imshow(
        log_mel_spec,
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, log_mel_spec.shape[0]],
        cmap="magma",
    )
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Mel Frequency Bin", fontsize=12)
    ax.set_title("Log-Mel Spectrogram (80 bins)", fontsize=14)
    plt.colorbar(im, ax=ax, format="%.1f")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] 已保存 Mel 频谱图: {out_path}")


def plot_mel_filter_bank(mel_filter, sr, n_fft, out_path="mel_filters.png"):
    """可视化 Mel 滤波器组的形状。"""
    fft_freqs = np.linspace(0, sr // 2, n_fft // 2 + 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(mel_filter.shape[0]):
        ax.plot(fft_freqs, mel_filter[i], alpha=0.7)
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.set_title("Mel Filter Bank (80 triangular filters)", fontsize=14)
    ax.set_xlim(0, 8000)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] 已保存 Mel 滤波器图: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Ch01-04: Mel Spectrogram")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--output", type=str, default="../outputs/mel.png")
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=80)
    args = parser.parse_args()

    if args.generate:
        wave, sr = generate_speech_like()
    elif args.input:
        wave, sr = load_audio(args.input)
    else:
        print("请提供 --input <audio.wav> 或使用 --generate")
        return

    print(f"[mel] 音频: {len(wave)/sr:.3f}s, n_fft={args.n_fft}, n_mels={args.n_mels}")

    # 计算 Mel 频谱
    log_mel_spec, mel_spec = compute_mel_spectrogram(
        wave, sr=sr, n_fft=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels
    )
    print(f"[mel] Mel 频谱 shape: {log_mel_spec.shape} (n_mels, n_frames)")

    # 绘制 Mel 滤波器组
    mel_filter = mel_filter_bank(n_mels=args.n_mels, n_fft=args.n_fft, sr=sr)
    plot_mel_filter_bank(mel_filter, sr, args.n_fft)

    # 绘制 Mel 频谱图
    plot_mel_spectrogram(log_mel_spec, sr, args.hop_length, out_path=args.output)
    print("Done!")


if __name__ == "__main__":
    main()
