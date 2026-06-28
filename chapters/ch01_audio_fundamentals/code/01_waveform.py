"""
Ch01-01: Waveform — 认识数字音频的时域表示

运行:
    python 01_waveform.py --input path/to/audio.wav
    python 01_waveform.py --generate          # 生成示例正弦波

输出:
    - waveform.png  : 时域波形图
    - waveform.wav  : (若 --generate) 示例音频
"""

import argparse
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_sine_wave(freq=440.0, duration=2.0, sr=16000, out_path="waveform.wav"):
    """生成一个正弦波示例音频，建立"我能合成声音"的信心。"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 叠加两个频率，让声音更"丰富"一点
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    wave += 0.3 * np.sin(2 * np.pi * (freq * 2) * t)
    # 防止削波
    wave = wave / np.max(np.abs(wave)) * 0.8
    sf.write(out_path, wave, sr)
    print(f"[generate] 已生成示例音频: {out_path} ({freq}Hz + {freq*2}Hz, {duration}s, {sr}Hz)")
    return wave, sr


def load_audio(path):
    """读取音频文件，返回波形数组和采样率。"""
    wave, sr = sf.read(path)
    # 如果是立体声，取单声道
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    return wave, sr


def plot_waveform(wave, sr, out_path="waveform.png", title="Waveform"):
    """绘制时域波形图。"""
    duration = len(wave) / sr
    time = np.linspace(0, duration, len(wave))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # 上图：完整波形
    ax1 = axes[0]
    ax1.plot(time, wave, color="#E88AC0", linewidth=0.5)
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Amplitude", fontsize=11)
    ax1.set_title(f"{title} — Full View | sr={sr}Hz, duration={duration:.2f}s", fontsize=13)
    ax1.set_xlim(0, duration)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    # 下图：放大看 20ms 的细节（约 3 个周期，如果是 440Hz）
    zoom_samples = int(sr * 0.02)  # 20ms
    ax2 = axes[1]
    ax2.plot(time[:zoom_samples], wave[:zoom_samples], color="#A27BD8", linewidth=1.2)
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Amplitude", fontsize=11)
    ax2.set_title("Zoomed View — First 20ms (notice the periodicity)", fontsize=13)
    ax2.set_xlim(0, zoom_samples / sr)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] 已保存波形图: {out_path}")


def print_stats(wave, sr):
    """打印波形的基本统计信息。"""
    print("=" * 50)
    print("Waveform Statistics")
    print("=" * 50)
    print(f"  Sample rate : {sr} Hz")
    print(f"  Total samples: {len(wave)}")
    print(f"  Duration    : {len(wave) / sr:.3f} s")
    print(f"  Min amplitude: {wave.min():.4f}")
    print(f"  Max amplitude: {wave.max():.4f}")
    print(f"  Mean         : {wave.mean():.6f}")
    print(f"  RMS energy   : {np.sqrt(np.mean(wave**2)):.4f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Ch01-01: Waveform visualization")
    parser.add_argument("--input", type=str, default=None, help="Path to input audio file")
    parser.add_argument("--generate", action="store_true", help="Generate a sine wave example")
    parser.add_argument("--output-png", type=str, default="../outputs/waveform.png", help="Output image path")
    parser.add_argument("--output-wav", type=str, default="../outputs/waveform.wav", help="Output audio path (for --generate)")
    args = parser.parse_args()

    if args.generate:
        wave, sr = generate_sine_wave(out_path=args.output_wav)
    elif args.input:
        wave, sr = load_audio(args.input)
    else:
        print("请提供 --input <audio.wav> 或使用 --generate 生成示例音频")
        return

    print_stats(wave, sr)
    plot_waveform(wave, sr, out_path=args.output_png)
    print("Done!")


if __name__ == "__main__":
    main()
