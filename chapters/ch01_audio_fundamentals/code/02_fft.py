"""
Ch01-02: FFT — 从时域到频域

运行:
    python 02_fft.py --input path/to/audio.wav
    python 02_fft.py --generate

输出:
    - fft.png : 频谱图（单帧 FFT 幅度谱）
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


def generate_sine_wave(freq=440.0, duration=1.0, sr=16000, out_path="fft_demo.wav"):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    wave += 0.3 * np.sin(2 * np.pi * (freq * 3) * t)  # 3次谐波
    wave = wave / np.max(np.abs(wave)) * 0.8
    sf.write(out_path, wave, sr)
    print(f"[generate] 已生成: {out_path} (基频{freq}Hz + 3次谐波)")
    return wave, sr


def compute_fft(wave, sr):
    """
    对整段音频做 FFT，返回频率轴和幅度谱。

    注意：这里假设音频是"稳态"的（频率成分不随时间变化）。
    实际语音是非稳态信号，需要用 STFT（见 03_stft.py）。
    """
    n = len(wave)
    # FFT
    fft_vals = np.fft.rfft(wave)
    # 幅度谱（取绝对值）
    magnitude = np.abs(fft_vals)
    # 频率轴
    freqs = np.fft.rfftfreq(n, d=1.0/sr)
    return freqs, magnitude


def plot_fft(freqs, magnitude, sr, out_path="fft.png"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # 上图：线性频率轴
    ax1 = axes[0]
    ax1.plot(freqs, magnitude, color="#E88AC0", linewidth=0.8)
    ax1.set_xlabel("Frequency (Hz)", fontsize=11)
    ax1.set_ylabel("Magnitude", fontsize=11)
    ax1.set_title(f"FFT Magnitude Spectrum (Linear Scale) | sr={sr}Hz", fontsize=13)
    ax1.set_xlim(0, sr // 2)
    ax1.grid(True, alpha=0.3)

    # 下图：对数幅度（dB），且限制在 0-8000Hz（语音主要频段）
    ax2 = axes[1]
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    ax2.plot(freqs, magnitude_db, color="#A27BD8", linewidth=0.8)
    ax2.set_xlabel("Frequency (Hz)", fontsize=11)
    ax2.set_ylabel("Magnitude (dB)", fontsize=11)
    ax2.set_title("FFT Magnitude Spectrum (Log Scale / dB) — Zoomed to 8kHz", fontsize=13)
    ax2.set_xlim(0, 8000)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] 已保存 FFT 图: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Ch01-02: FFT demo")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--output", type=str, default="../outputs/fft.png")
    args = parser.parse_args()

    if args.generate:
        wave, sr = generate_sine_wave()
    elif args.input:
        wave, sr = load_audio(args.input)
    else:
        print("请提供 --input <audio.wav> 或使用 --generate")
        return

    # 对整段音频做 FFT（适合稳态信号演示）
    freqs, magnitude = compute_fft(wave, sr)

    # 找到最大能量对应的频率
    peak_idx = np.argmax(magnitude[1:]) + 1  # 跳过 DC
    print(f"[fft] 峰值频率: {freqs[peak_idx]:.1f} Hz, 幅度: {magnitude[peak_idx]:.2f}")

    plot_fft(freqs, magnitude, sr, out_path=args.output)
    print("Done!")


if __name__ == "__main__":
    main()
