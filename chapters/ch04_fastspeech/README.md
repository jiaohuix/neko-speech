# Ch04: FastSpeech2 -- Neko 学会了快速说话

> Ch02 的 Tacotron2 能用，但太慢了。
>
> 这一章，我们用**并行生成**取代自回归，让 TTS 快 10-100 倍。

## 本章导学

### 为什么需要 FastSpeech2？

Ch02 的 Tacotron2 是**自回归**模型：生成 Mel 频谱时，必须一帧一帧地预测。

```
Tacotron2 推理：
  第1帧 -> 第2帧 -> 第3帧 -> ... -> 第N帧
  每帧需要一次 LSTM 前向传播
  1秒语音 ≈ 87帧 (hop=256, sr=22050)
  → 1秒语音需要 87 次串行计算
```

这带来两个问题：

1. **推理慢**：生成 3 秒语音需要 200+ 次前向传播，RTF（实时率）> 1，不能实时
2. **鲁棒性差**：自回归的误差会累积——某一帧预测偏了，后面全跟着偏（跳字、重复）

FastSpeech2 的核心思想：**不要一帧一帧生成，而是一次性生成所有帧**。

### 学习路线

| 节 | 内容 | 目标 |
|---|------|------|
| 4.1 | 架构总览 | 理解 Encoder -> Length Regulator -> Decoder |
| 4.2 | Duration Predictor | 时长建模：每个音素持续多少帧 |
| 4.3 | Length Regulator | 根据时长拉伸序列 |
| 4.4 | Pitch & Energy | 方差适配器：控制韵律 |
| 4.5 | FFT Blocks | 前馈 Transformer 模块 |
| 4.6 | 训练 | 时长标签 + 多任务损失 |
| 4.7 | 推理 | 并行生成 + 速度对比 |

---

## 4.1 架构总览

```
Text (字符序列)
    |
Embedding + Positional Encoding
    |
FFT Encoder (Self-Attention + FFN, x4)
    |  (B, T_text, d_model)
    +-- Duration Predictor -> durations (每音素帧数)
    +-- Pitch Predictor    -> pitch (音高)
    +-- Energy Predictor   -> energy (响度)
    |
Length Regulator (按 durations 拉伸)
    |  (B, T_mel, d_model)
    |
FFT Decoder (Self-Attention + FFN, x4)
    |
Linear -> Mel Spectrogram (B, 80, T_mel)
    |
Vocoder (HiFi-GAN / Griffin-Lim) -> Waveform
```

### 与 Tacotron2 的关键区别

| | Tacotron2 (Ch02) | FastSpeech2 (本章) |
|---|---|---|
| 解码方式 | 自回归（逐帧） | 并行（一次性） |
| 对齐机制 | Attention（隐式） | Duration Predictor（显式） |
| 推理速度 | 慢（RTF > 1） | 快（RTF < 0.01） |
| 鲁棒性 | 可能跳字/重复 | 不会（确定性映射） |
| 可控性 | 难以控制语速/音高 | 可直接调节 |

---

## 4.2 Duration Predictor：每个音素说多久？

### 4.2.1 为什么需要时长信息？

Tacotron2 用 Attention 自动学习"文本 -> 语音"的对齐。但 Attention 有两个问题：
- 训练不稳定（需要大量数据才能学到好的对角线注意力）
- 推理时可能出错（跳字、重复）

FastSpeech2 的方案：**显式预测每个音素的持续帧数**。

```
输入: "你好世界"  (4 个字符)
Duration Predictor 预测: [12, 8, 15, 10]  (帧数)
总和 = 45 帧 = Mel 频谱长度
```

### 4.2.2 预测器结构

```python
# 2 层 Conv1d + LayerNorm + ReLU
Encoder Output (B, T_text, d)
    -> Conv1d -> LN -> ReLU -> Dropout
    -> Conv1d -> LN -> ReLU -> Dropout
    -> Linear -> (B, T_text)   # 每个音素一个标量
```

输出在 **log 域**：因为时长分布高度偏斜（有的音素 1 帧，有的 50 帧），取 log 后更接近正态分布，MSE 损失更好优化。

### 4.2.3 时长标签从哪来？

这是 FastSpeech2 的核心难题。三种常见方案：

1. **Teacher Model 对齐**：先训练一个 Tacotron2，提取其 Attention 矩阵，每列的 argmax 就是时长。这是原论文的方法。
2. **Montreal Forced Aligner (MFA)**：独立的语音对齐工具，基于 HMM，不需要训练 TTS 模型。
3. **均匀分配**（本章简化方案）：假设每个字符持续相同帧数 `mel_len / text_len`。

```python
# 均匀分配（本教程使用）
def estimate_uniform_durations(text_len, mel_len):
    base = mel_len // text_len
    rem = mel_len - base * text_len
    durs = [base] * text_len
    for i in range(rem):
        durs[i] += 1
    return durs
```

> **Neko 笔记**：均匀分配是简化方案，会导致语速不自然。生产中应使用 MFA 或 Tacotron2 对齐。但模型结构是完全一样的，只是训练数据质量不同。

---

## 4.3 Length Regulator：把文本拉伸到语音长度

### 4.3.1 问题

Encoder 输出 T_text 个向量，但 Mel 频谱有 T_mel 帧。需要把 T_text 扩展到 T_mel。

### 4.3.2 方案：按持续时间重复

```python
# 核心操作：repeat_interleave
encoder_out = [h1, h2, h3]       # 3 个音素
durations   = [2,  3,  1]         # 每个音素的帧数
expanded    = [h1, h1, h2, h2, h2, h3]  # 6 帧
```

就是这么简单！**每个音素的向量重复其预测的时长次**。

### 4.3.3 为什么有效？

一个音素在时间上持续多帧，但其"语义内容"不变。重复同一个向量相当于说：**这几帧都属于同一个音素，内容相同**。

Decoder 再通过 Self-Attention 看到"相邻帧来自不同音素"，自动学到过渡和韵律。

---

## 4.4 Pitch & Energy：控制韵律

### 4.4.1 为什么需要？

Duration 只决定了"每个音素说多久"，但没说"以什么音调说"、"以多大声音说"。

- **Pitch（音高）**：疑问句的末尾音高上扬，陈述句平稳
- **Energy（能量/响度）**：强调某个词时响度变大

### 4.4.2 Variance Adapter

每个都有一个 Predictor（结构和 Duration Predictor 相同）+ 一个 Embedding 层：

```python
# Pitch Predictor: encoder_out -> (B, T_text) pitch values
# Pitch Embedding: pitch -> (B, T_text, d_model)
# Add to encoder output
x = x + pitch_embed(pitch.unsqueeze(-1))
x = x + energy_embed(energy.unsqueeze(-1))
```

### 4.4.3 推理时的控制

因为 pitch 和 energy 是独立预测的，我们可以在推理时直接缩放：

```python
# 高音版
mel = model.inference(text, pitch_scale=1.2)
# 轻声版
mel = model.inference(text, energy_scale=0.8)
```

> **Neko 笔记**：本教程使用 mel 频谱的统计量（频谱质心、均值幅度）作为 pitch/energy 的代理训练目标。生产系统会用 pyworld 提取真实 F0（基频）。

---

## 4.5 FFT Block：前馈 Transformer

### 4.5.1 结构

每个 FFT Block 就是一个标准的 Pre-LN Transformer 层：

```
Input (B, T, d)
  |
  +-- LayerNorm -> Multi-Head Self-Attention -> Residual
  |
  +-- LayerNorm -> FFN (Linear->ReLU->Linear) -> Residual
  |
Output (B, T, d)
```

Encoder 和 Decoder 都是 4 层 FFT Block 的堆叠。

### 4.5.2 Encoder vs Decoder

- **Encoder**：处理文本序列，Self-Attention 让每个字符看到全文上下文
- **Decoder**：处理展开后的帧序列，Self-Attention 让每帧看到前后帧的过渡

两者结构完全相同，只是输入不同。

### 4.5.3 为什么不用 LSTM？

Tacotron2 用 LSTM，因为当时（2018）Transformer 还没在 TTS 中广泛使用。

FFT 的优势：
- **并行计算**：Self-Attention 不需要逐步处理序列
- **长程依赖**：任意两个位置都直接交互
- **训练稳定**：不存在 RNN 的梯度消失/爆炸

---

## 4.6 训练

### 4.6.1 多任务损失

FastSpeech2 同时优化 4 个目标：

$$\mathcal{L} = \mathcal{L}_{mel} + \mathcal{L}_{dur} + \mathcal{L}_{pitch} + \mathcal{L}_{energy}$$

- **Mel Loss**：预测 Mel 与 GT 的 MSE（核心目标）
- **Duration Loss**：预测 log-duration 与 GT 的 MSE（辅助目标）
- **Pitch Loss**：预测 pitch 与 GT 的 MSE
- **Energy Loss**：预测 energy 与 GT 的 MSE

### 4.6.2 数据准备

使用和 Ch02 相同的数据集：

```bash
# 下载猫娘数据集
cd data
python download_neko_1k.py --output-dir processed --num-samples 1000
```

### 4.6.3 启动训练

```bash
cd chapters/ch04_fastspeech/code
python train.py \
    --data-dir ../../../data/processed \
    --epochs 50 \
    --batch-size 8
```

**预期现象**：
- 前 5 epoch loss 快速下降（duration/pitch/energy 开始学习）
- 10-20 epoch mel loss 进入平台期
- 多任务损失中 dur_loss 下降最快（均匀时长容易学）
- 每 10 epoch 自动保存 checkpoint

### 4.6.4 与 Ch02 训练对比

| | Tacotron2 (Ch02) | FastSpeech2 (本章) |
|---|---|---|
| 每 batch 计算量 | 大（T_mel 次 LSTM 步进） | 小（全并行） |
| 训练速度 | 慢 | 快 2-3x |
| 需要的额外数据 | 无 | 时长标签（可自动估计） |
| 收敛难度 | 中（Attention 需要对齐） | 低（确定性映射） |

---

## 4.7 推理

### 4.7.1 并行推理

```bash
python inference.py \
    --checkpoint ../checkpoints/fs2_final.pt \
    --text "你好，我是猫娘。" \
    --output ../outputs/fs2_output.wav
```

FastSpeech2 推理只需 **1 次前向传播** 就生成全部 Mel 帧：

```python
# 全部 mel 帧一次性生成
mel = model.inference(text)  # (B, 80, T_mel)  一次调用
```

### 4.7.2 速度对比

假设生成 100 帧 Mel（约 1.2 秒语音）：

| 模型 | 前向传播次数 | 典型耗时 (GPU) | RTF |
|------|-------------|---------------|-----|
| Tacotron2 | 100 (串行) | ~500ms | ~0.4 |
| FastSpeech2 | 1 (并行) | ~10ms | ~0.008 |
| **加速比** | 100x | **50x** | **50x** |

> **RTF (Real-Time Factor)** = 生成时间 / 音频时长。RTF < 1 表示可以实时。

### 4.7.3 可控生成

```python
# 正常
mel = model.inference(text)
# 高音 (pitch_scale > 1)
mel_high = model.inference(text, pitch_scale=1.3)
# 慢速 (duration 自动变长)
# 注：需要修改 durations 的缩放因子
```

### 4.7.4 预期效果

- 早期训练（< 10 epoch）：输出可能是平坦的噪声
- 中期（10-30 epoch）：开始出现语调轮廓
- 充分训练后：能生成可辨识的语音（配合好的声码器）

> **声码器仍然重要**：FastSpeech2 只生成 Mel 频谱，还需要声码器转为波形。本章用 Griffin-Lim（音质差），Ch05 VITS 会学到端到端方案。

---

## 4.8 本章小结

### FastSpeech2 的核心贡献

| 问题 | 解决方案 |
|------|----------|
| 自回归慢 | 并行生成（1 次前向传播） |
| Attention 不稳定 | 显式 Duration Predictor |
| 误差累积 | 确定性映射，无累积 |
| 无法控制韵律 | Pitch/Energy Variance Adapters |

### 遗留问题

1. **仍需独立声码器**：FastSpeech2 只输出 Mel，还需要 HiFi-GAN 等声码器 -> Ch05 VITS 端到端
2. **时长标签依赖**：均匀估计不够好，需要 MFA 或 teacher model
3. **音质上限**：非自回归模型的 mel 预测精度略低于 Tacotron2（但推理快得多）

### 参考文献

- [1] Ren et al., 2021. *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech*. ICLR 2021.
- [2] Ren et al., 2019. *FastSpeech: Fast, Robust and Controllable Text to Speech*. NeurIPS 2019.
- [3] Vaswani et al., 2017. *Attention Is All You Need*. NeurIPS 2017.

---

## 习题

1. **时长建模**：为什么 Duration Predictor 在 log 域预测，而不是直接预测帧数？提示：想想时长值的分布特点。

2. **Length Regulator**：如果 Duration Predictor 预测的所有时长之和与实际 Mel 长度不匹配，会发生什么？如何缓解？

3. **并行 vs 自回归**：Tacotron2 的 Decoder 每步需要上一帧的输出，而 FastSpeech2 的 Decoder 不需要。这意味着 FastSpeech2 的 Decoder 无法建模什么信息？这对音质有什么影响？

4. **Variance Adapter**：如果去掉 Pitch Predictor 和 Energy Predictor，只保留 Duration Predictor，模型还能正常训练吗？音质会受到什么影响？

5. **进阶思考**：VITS（Ch05）把 FastSpeech2 的结构和 VAE + Normalizing Flow 结合，实现了端到端 TTS。它解决了 FastSpeech2 的什么问题？又引入了什么新的复杂度？

---

## 目录结构

```
ch04_fastspeech/
├── README.md           # 本章教程（你正在读的）
├── code/
│   ├── model.py        # FastSpeech2 模型 (~280 行)
│   ├── train.py        # 训练脚本（含均匀时长估计）
│   └── inference.py    # 并行推理 + Griffin-Lim
├── checkpoints/        # 模型权重
└── outputs/            # 生成的音频
```
