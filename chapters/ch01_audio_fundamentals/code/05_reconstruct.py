"""
Ch01-05: Griffin-Lim — 从频谱重建波形

运行:
    python 05_reconstruct.py --input path/to/audio.wav

输出:
    - original_vs_reconstructed.png : 对比图
    - reconstructed.wav             : 重建的音频

核心思想：
    STFT 丢失了相位信息，但 Griffin-Lim 算法可以通过迭代
    从幅度谱（或 Mel 谱反变换得到的幅度谱）中估计相位，
    从而重建出可听的波形。

    这也是 Tacotron 早期使用的声码器方案（后来才被 WaveNet 替代）。
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


def stft(wave, n_fft=1024, hop_length=256):
    """纯 NumPy STFT。"""
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


def istft(stft_matrix, n_fft=1024, hop_length=256):
    """
    逆 STFT（inverse STFT）。

    使用 overlap-add 方法重建时域信号。
    """
    window = np.hamming(n_fft)
    n_frames = stft_matrix.shape[1]
    expected_len = n_fft + (n_frames - 1) * hop_length
    output = np.zeros(expected_len)
    window_sum = np.zeros(expected_len)

    for i in range(n_frames):
        start = i * hop_length
        # 逆 FFT
        frame = np.fft.irfft(stft_matrix[:, i], n=n_fft)
        # overlap-add
        output[start:start + n_fft] += frame * window
        window_sum[start:start + n_fft] += window ** 2

    # 归一化（避免重叠区域能量叠加）
    output = output / np.maximum(window_sum, 1e-10)
    # 去掉 padding
    pad_len = n_fft // 2
    output = output[pad_len:pad_len + expected_len - 2 * pad_len]
    return output


def griffin_lim(magnitude, n_fft=1024, hop_length=256, n_iter=60):
    """
    Griffin-Lim 算法：从幅度谱重建相位，恢复波形。

    算法步骤：
        1. 随机初始化相位
        2. 逆 STFT 得到时域信号
        3. 正 STFT 得到新的复数谱
        4. 保留新相位，替换回原始幅度
        5. 重复 2-4 直到收敛

    Args:
        magnitude: (n_freqs, n_frames) 幅度谱
        n_iter: 迭代次数（通常 30-100 次）

    Returns:
        重建的复数 STFT 矩阵
    """
    # 随机初始化相位
    angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    stft_reconstructed = magnitude * angles

    for i in range(n_iter):
        # 逆 STFT → 时域
        wave = istft(stft_reconstructed, n_fft=n_fft, hop_length=hop_length)
        # 正 STFT → 新复数谱
        stft_new = stft(wave, n_fft=n_fft, hop_length=hop_length)
        # 保留新相位，替换幅度
        angles = stft_new / (np.abs(stft_new) + 1e-10)
        stft_reconstructed = magnitude * angles

        if (i + 1) % 10 == 0:
            print(f"[griffin-lim] 迭代 {i+1}/{n_iter}")

    return stft_reconstructed


def plot_comparison(original, reconstructed, sr, out_path="original_vs_reconstructed.png"):
    """对比原始波形和重建波形。"""
    # 截断到相同长度
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    duration = min_len / sr
    time = np.linspace(0, duration, min_len)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8))

    # 原始波形
    ax1 = axes[0]
    ax1.plot(time, original, color="#E88AC0", linewidth=0.5)
    ax1.set_title("Original Waveform", fontsize=13)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, duration)
    ax1.grid(True, alpha=0.3)

    # 重建波形
    ax2 = axes[1]
    ax2.plot(time, reconstructed, color="#A27BD8", linewidth=0.5)
    ax2.set_title("Reconstructed Waveform (Griffin-Lim)", fontsize=13)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlim(0, duration)
    ax2.grid(True, alpha=0.3)

    # 差异
    ax3 = axes[2]
    diff = original - reconstructed
    ax3.plot(time, diff, color="#FF6B6B", linewidth=0.5)
    ax3.set_title(f"Difference (MSE={np.mean(diff**2):.6f})", fontsize=13)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.set_xlim(0, duration)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] 已保存对比图: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Ch01-05: Griffin-Lim reconstruction")
    parser.add_argument("--input", type=str, required=True, help="Input audio file")
    parser.add_argument("--output-wav", type=str, default="../outputs/reconstructed.wav")
    parser.add_argument("--output-png", type=str, default="../outputs/original_vs_reconstructed.png")
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-iter", type=int, default=60, help="Griffin-Lim iterations")
    args = parser.parse_args()

    # 读取原始音频
    original, sr = load_audio(args.input)
    print(f"[load] 原始音频: {len(original)/sr:.3f}s, sr={sr}Hz")

    # 正 STFT → 幅度谱
    stft_orig = stft(original, n_fft=args.n_fft, hop_length=args.hop_length)
    magnitude = np.abs(stft_orig)
    print(f"[stft] 幅度谱 shape: {magnitude.shape}")

    # Griffin-Lim 重建
    print(f"[griffin-lim] 开始重建，迭代 {args.n_iter} 次...")
    stft_recon = griffin_lim(magnitude, n_fft=args.n_fft, hop_length=args.hop_length, n_iter=args.n_iter)

    # 逆 STFT
    reconstructed = istft(stft_recon, n_fft=args.n_fft, hop_length=args.hop_length)
    print(f"[istft] 重建波形长度: {len(reconstructed)}")

    # 保存
    sf.write(args.output_wav, reconstructed, sr)
    print(f"[save] 已保存重建音频: {args.output_wav}")

    # 对比图
    plot_comparison(original, reconstructed, sr, out_path=args.output_png)
    print("Done!")


if __name__ == "__main__":
    main()
