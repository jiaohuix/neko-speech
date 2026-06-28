# Ch01: Audio Fundamentals — Neko 的第一课：声音是什么？

> 在让 Neko 学会说话之前，她需要先理解"声音"本身。
>
> 这一章建立所有现代 Audio AI 模型共同依赖的基础。

## 本章导学

### 为什么学这一章？

Tacotron、VITS、GPT-SoVITS、VALL-E、Omni……这些模型的输入或输出，几乎都离不开同一个东西：**Mel Spectrogram（梅尔频谱）**。

如果你跳过这一章直接学 Tacotron，你会遇到：
- 为什么模型输出的是 80 维向量而不是波形？
- Griffin-Lim 是什么？为什么音质这么差？
- VITS 为什么说"不需要声码器"？

这些问题都会迫使你**不断回头补基础**。所以，我们先一次性把音频表示的完整链条走完：

```
Waveform ──→ FFT ──→ STFT ──→ Mel Spectrogram ──→ Griffin-Lim Reconstruction
  ↑                                                          ↓
  └──────────────── 音频的完整表示与重建循环 ─────────────────────┘
```

### 学习路线

| 节 | 内容 | 目标 |
|---|------|------|
| 1.1 | 波形与采样 | 理解数字音频的时域表示 |
| 1.2 | FFT：从时域到频域 | 理解频率分解 |
| 1.3 | STFT：时频分析 | 理解语音是非稳态信号 |
| 1.4 | Mel 频谱 | 理解人耳感知与非线性频率刻度 |
| 1.5 | Griffin-Lim 重建 | 理解"从频谱回到波形"的问题 |
| 1.6 | 动手实验 | 运行全部 Demo，眼见为实 |

---

## 1.1 波形与采样：声音的数字化身

### 1.1.1 什么是声音？

声音是**空气压力的振动**。当声带振动、琴弦颤动、或扬声器膜片推动空气时，周围的空气分子就会形成疏密相间的波动。

人耳听到的是**压力随时间的变化**。如果我们把压力变化记录下来，就得到了**波形（Waveform）**。

### 1.1.2 采样与数字化

真实世界的声音是**连续**的，但计算机只能处理**离散**的数字。所以需要做两步：

**采样（Sampling）**：每隔固定时间测一次压力值。
- 采样率（Sample Rate）：每秒采多少个点。语音常用 16kHz（每秒 16000 个样本）。
- 奈奎斯特定理：采样率必须 > 2× 最高频率，才能完整还原信号。人耳上限约 20kHz，所以 CD 用 44.1kHz。

**量化（Quantization）**：把每个采样值映射到有限精度的整数。
- 16-bit PCM：每个样本用 16 位表示，范围 [-32768, 32767]。

> **Neko 笔记**：你可以把采样想象成给波形"拍照"——拍得越密（采样率越高）、像素越好（量化位数越高），还原得越真。

### 1.1.3 动手：生成并观察波形

运行代码，生成一个 440Hz 的正弦波（A4 音，钢琴中央 A）：

```bash
cd chapters/ch01_audio_fundamentals/code
python 01_waveform.py --generate
```

输出到 `../outputs/`：
- `waveform.wav` — 可播放的音频
- `waveform.png` — 时域波形图

**观察重点**：
- 上图是完整波形，下图放大 20ms。你会发现波形是**周期性**的——这就是"音高"的来源。

---

## 1.2 FFT：从时域到频域

### 1.2.1 为什么需要 FFT？

波形告诉我们"什么时候声音大"，但没告诉我们"声音由哪些频率组成"。

**傅里叶变换**的核心思想：任何信号都可以表示为不同频率正弦波的叠加。

$$X(\omega) = \sum_{n=-\infty}^{\infty} x[n] \, e^{-j\omega n}$$

### 1.2.2 快速傅里叶变换（FFT）

FFT 是傅里叶变换的高效算法，把 $O(N^2)$ 降到 $O(N \log N)$。

输入：时域波形（N 个点）
输出：频域幅度谱（N/2+1 个频率 bins）

### 1.2.3 动手：观察频谱

```bash
python 02_fft.py --generate
```

输出：`../outputs/fft.png`

**观察重点**：
- 上图是线性幅度谱，你会看到 440Hz 和 1320Hz（3次谐波）两个尖峰。
- 下图是对数刻度（dB），更适合观察动态范围大的信号。

> **Neko 笔记**：FFT 假设信号是**稳态**的——频率成分不随时间变化。但语音不是稳态的！这就需要 STFT。

---

## 1.3 STFT：短时傅里叶变换

### 1.3.1 语音是非稳态信号

说"你好"时，"n"、"i"、"h"、"a"、"o" 的发音频率完全不同。FFT 只能告诉你整段音频里有哪些频率，但**不知道这些频率什么时候出现**。

### 1.3.2 STFT 的核心思想

**加窗 → 逐帧 FFT → 拼接**

1. 用一个窗函数（通常是汉明窗）截取一小段音频（20-50ms）
2. 对这一小段做 FFT
3. 窗向右滑动（hop length，通常 10ms）
4. 重复，把所有帧的频谱竖着拼起来

$$X[m, \omega] = \sum_n x[n] \, w[n-m] \, e^{-j\omega n}$$

其中 $w[n-m]$ 就是滑动的窗。

### 1.3.3 动手：观察时频图

```bash
python 03_stft.py --generate
```

输出：`../outputs/spectrogram.png`

**观察重点**：
- 生成的是一个 chirp 信号（频率从 200Hz 线性增加到 2000Hz）。
- 频谱图上应该看到一条**斜线**——这是时频分析的意义：你能"看到"频率随时间的变化。

---

## 1.4 Mel 频谱：模拟人耳的感知

### 1.4.1 人耳的非线性听觉

人耳对低频敏感、对高频不敏感。
- 100Hz → 200Hz，你明显感到音高翻倍了。
- 8000Hz → 8100Hz，你几乎听不出区别。

### 1.4.2 Mel 刻度

Mel 刻度模拟了人耳的这种非线性感知：

$$\text{mel} = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$

- 低频区：Mel 值增长快（人耳敏感）
- 高频区：Mel 值增长慢（人耳不敏感）

### 1.4.3 Mel 滤波器组

STFT 给出的是**线性频率轴**的频谱。Mel 频谱把它转换为 Mel 刻度：

1. 在 Mel 刻度上均匀放置若干三角滤波器（通常 80 个）
2. 每个滤波器覆盖一段频率范围，中心在 Mel 刻度上等距
3. 用这些滤波器对线性频谱做加权求和

### 1.4.4 Log-Mel Spectrogram

最后取对数：

$$\text{Log-Mel} = \log(\text{MelFilter} \cdot |\text{STFT}| + \epsilon)$$

为什么取对数？
- 人耳对响度的感知也是对数的（分贝刻度）
- 压缩动态范围，让数值更稳定

> **这就是 Tacotron/VITS 等模型的标准输入！** 80-bin Log-Mel，帧移 256 样本（约 16ms @ 16kHz）。

### 1.4.5 动手：观察 Mel 频谱

```bash
python 04_mel.py --generate
```

输出：`../outputs/mel.png`、`../outputs/mel_filters.png`

**观察重点**：
- `mel_filters.png`：三角滤波器在低频密集、高频稀疏。
- `mel.png`：模拟语音的 Mel 频谱，可以看到共振峰（formants）——这些是区分元音的关键。

---

## 1.5 Griffin-Lim：从频谱重建波形

### 1.5.1 问题：相位丢失

STFT 把时域信号变成复数谱：$X = |X| \cdot e^{j\phi}$。

我们保存/使用的通常是**幅度谱** $|X|$，**相位 $\phi$ 被丢掉了**。

从幅度谱逆变换回时域，如果直接用随机相位，声音会像"金属噪音"。

### 1.5.2 Griffin-Lim 算法

Griffin-Lim 是一个迭代算法，通过约束"逆变换后的信号再正变换，其幅度应该接近原幅度"来估计相位：

1. 随机初始化相位
2. 逆 STFT → 时域信号
3. 正 STFT → 新复数谱
4. 保留新相位，替换回原始幅度
5. 重复 30-100 次

### 1.5.3 局限性

Griffin-Lim 能重建出可听的语音，但：
- 有"金属感"或"嗡嗡声"
- 高频细节丢失
- 音质远不如原始录音

**这就是为什么需要神经声码器（WaveNet / HiFi-GAN）的原因。**

### 1.5.4 动手：对比原始与重建

```bash
python 05_reconstruct.py --input ../outputs/waveform.wav --n-iter 30
```

输出：`../outputs/reconstructed.wav`、`../outputs/original_vs_reconstructed.png`

**观察重点**：
- 听 `waveform.wav` 和 `reconstructed.wav`，对比差异。
- 看对比图：差异（最下图）不是零——这就是相位信息丢失造成的损失。

---

## 1.6 本章小结

### 核心链条

```
模拟声波 → 采样(16kHz) → 量化(16bit) → 数字波形
                                    ↓
                              加窗 + 逐帧 FFT (STFT)
                                    ↓
                              线性频谱 → Mel 滤波器组
                                    ↓
                              80-bin Log-Mel Spectrogram
                                    ↓
                         ┌──────────┴──────────┐
                         ↓                     ↓
                   Griffin-Lim           神经声码器
                   (金属感)              (WaveNet/HiFi-GAN)
                         ↓                     ↓
                      波形                  高质量波形
```

### 关键概念

| 概念 | 作用 | 本章代码 |
|------|------|----------|
| 采样率 | 决定可还原的最高频率 | `01_waveform.py` |
| FFT | 时域 → 频域 | `02_fft.py` |
| STFT | 时域 → 时频表示 | `03_stft.py` |
| Mel Filter | 模拟人耳感知 | `04_mel.py` |
| Griffin-Lim | 幅度谱 → 波形（估计相位） | `05_reconstruct.py` |

### 遗留问题

Griffin-Lim 重建音质差。Tacotron 论文最初就是用 Griffin-Lim 作为声码器，音质自然不够好。**下一章，Neko 将学会用 WaveNet 生成更自然的波形。**

### 参考文献

- [1] 语音合成基础(3)——关于梅尔频谱你想知道的都在这里. 知乎.
- [2] Wang et al., 2018. *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions*.
- [3] Griffin & Lim, 1984. *Signal Estimation from Modified Short-Time Fourier Transform*.

---

## 习题

1. 如果把采样率从 16kHz 降到 8kHz，FFT 的最高频率会发生什么变化？对语音质量有什么影响？
2. 为什么 STFT 的窗长通常是 20-50ms？窗太长或太短分别有什么问题？
3. Mel 刻度公式中，为什么是 $f/700$ 而不是 $f/1000$？这个 700 有什么物理意义？
4. 运行 `04_mel.py`，观察 `mel_filters.png`。为什么低频区的三角滤波器更密集？
5. 把 `05_reconstruct.py` 的 `--n-iter` 从 30 改为 5 和 100，对比重建质量。迭代次数越多越好吗？

---

## 目录结构

```
ch01_audio_fundamentals/
├── README.md          # 本章教程（你正在看的文件）
├── code/              # 代码
│   ├── 01_waveform.py
│   ├── 02_fft.py
│   ├── 03_stft.py
│   ├── 04_mel.py
│   └── 05_reconstruct.py
└── outputs/           # 运行输出（.gitignore）
```
