"""
Ch01-03: STFT — 短时傅里叶变换与频谱图

运行:
    python 03_stft.py --input path/to/audio.wav
    python 03_stft.py --generate

输出:
    - spectrogram.png : STFT 幅度谱（线性 + 对数 dB）

核心思想：
    语音是非稳态信号，频率成分随时间变化。
    STFT = 对音频加窗 → 逐帧做 FFT → 得到时频表示。
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


def generate_chirp(duration=2.0, sr=16000, out_path="stft_demo.wav"):
    """生成一个频率随时间变化的 chirp 信号，完美展示时频分析的意义。"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 频率从 200Hz 线性增加到 2000Hz
    f0, f1 = 200, 2000
    wave = 0.5 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    wave = wave / np.max(np.abs(wave)) * 0.8
    sf.write(out_path, wave, sr)
    print(f"[generate] 已生成 chirp: {out_path} ({f0}Hz → {f1}Hz)")
    return wave, sr


def stft(wave, n_fft=1024, hop_length=256, win_length=None):
    """
    纯 NumPy 实现 STFT。

    Args:
        wave: 1D 音频数组
        n_fft: FFT 点数
        hop_length: 帧移（相邻两帧的距离）
        win_length: 窗长，默认等于 n_fft

    Returns:
        stft_matrix: (1 + n_fft/2, n_frames) 复数矩阵
    """
    if win_length is None:
        win_length = n_fft

    # 汉明窗
    window = np.hamming(win_length)

    # 补零，使第一帧中心在 0 时刻
    pad_len = n_fft // 2
    wave = np.pad(wave, (pad_len, pad_len), mode="constant")

    # 帧数
    n_frames = 1 + (len(wave) - win_length) // hop_length

    # 预分配
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)

    for i in range(n_frames):
        start = i * hop_length
        frame = wave[start:start + win_length]
        if len(frame) < win_length:
            break
        windowed = frame * window
        fft_frame = np.fft.rfft(windowed, n=n_fft)
        stft_matrix[:, i] = fft_frame

    return stft_matrix


def plot_spectrogram(stft_matrix, sr, hop_length, out_path="spectrogram.png"):
    """绘制 STFT 频谱图。"""
    magnitude = np.abs(stft_matrix)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    # 时间轴和频率轴
    n_frames = stft_matrix.shape[1]
    duration = n_frames * hop_length / sr
    times = np.linspace(0, duration, n_frames)
    freqs = np.linspace(0, sr // 2, stft_matrix.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 上图：线性幅度
    ax1 = axes[0]
    im1 = ax1.pcolormesh(times, freqs, magnitude, shading="gouraud", cmap="magma")
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Frequency (Hz)", fontsize=11)
    ax1.set_title("STFT Magnitude Spectrogram (Linear)", fontsize=13)
    ax1.set_ylim(0, 8000)
    plt.colorbar(im1, ax=ax1, format="%.1f")

    # 下图：对数 dB（更常用）
    ax2 = axes[1]
    im2 = ax2.pcolormesh(times, freqs, magnitude_db, shading="gouraud", cmap="magma")
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Frequency (Hz)", fontsize=11)
    ax2.set_title("STFT Magnitude Spectrogram (dB)", fontsize=13)
    ax2.set_ylim(0, 8000)
    plt.colorbar(im2, ax=ax2, format="%.0f dB")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] 已保存频谱图: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Ch01-03: STFT spectrogram")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--output", type=str, default="../outputs/spectrogram.png")
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    args = parser.parse_args()

    if args.generate:
        wave, sr = generate_chirp()
    elif args.input:
        wave, sr = load_audio(args.input)
    else:
        print("请提供 --input <audio.wav> 或使用 --generate")
        return

    print(f"[stft] 音频长度: {len(wave)/sr:.3f}s, n_fft={args.n_fft}, hop={args.hop_length}")

    stft_matrix = stft(wave, n_fft=args.n_fft, hop_length=args.hop_length)
    print(f"[stft] 输出 shape: {stft_matrix.shape} (freq_bins, frames)")

    plot_spectrogram(stft_matrix, sr, args.hop_length, out_path=args.output)
    print("Done!")


if __name__ == "__main__":
    main()
