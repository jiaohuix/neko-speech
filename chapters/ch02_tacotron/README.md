# Ch02: Tacotron2 — Neko 学会说话

> 有了音频基础，Neko 终于可以尝试把文字变成声音了。
>
> 这一章，我们实现第一个端到端神经 TTS 模型。

## 本章导学

### 为什么学 Tacotron2？

在 Tacotron 之前，语音合成的路线是：

```
文本分析 → 音素序列 → HMM/GMM 声学模型 → 参数声码器 → 波形
```

这条路线的问题：
1. **文本分析依赖人工规则**（分词、注音、韵律标注）
2. **HMM 假设每个音素的状态转移是马尔可夫的**，无法捕捉长程依赖
3. **参数声码器（如 MLSA）过度平滑**，音质像"机器人"

Tacotron2 的革命性在于：**端到端**——直接从字符/音素输入，输出 Mel Spectrogram，无需人工特征工程。

### Tacotron2 之前的演进

| 模型 | 年份 | 核心改进 | 问题 |
|------|------|----------|------|
| Tacotron | 2017 | 首个端到端 Seq2Seq TTS | 对齐不稳定（跳字、重复） |
| Tacotron2 | 2018 | Location-Sensitive Attention + WaveNet 声码器 | 自回归慢，需预训练声码器 |

### 学习路线

| 节 | 内容 | 目标 |
|---|------|------|
| 2.1 | 模型架构总览 | 理解 Encoder-Attention-Decoder-PostNet 流程 |
| 2.2 | Encoder | 文本 → 时序特征 |
| 2.3 | Location-Sensitive Attention | 解决对齐问题 |
| 2.4 | Decoder | 自回归生成 Mel |
| 2.5 | PostNet | 细化频谱细节 |
| 2.6 | 训练 | 用真实数据训练 |
| 2.7 | 推理 | 生成 Mel + Griffin-Lim 声码器 |

---

## 2.1 模型架构总览

Tacotron2 的核心流程：

```
Text (字符序列)
    ↓
Encoder (Embedding + Conv + BiLSTM)
    ↓  (T_text, encoder_dim)
Location-Sensitive Attention
    ↓  (context向量，每步一个)
Decoder (PreNet + LSTM，自回归)
    ↓  (每步预测1帧Mel)
Mel Spectrogram (T_mel, 80)
    ↓
PostNet (Conv，残差修正)
    ↓
Refined Mel Spectrogram
    ↓
Vocoder (WaveNet / Griffin-Lim) → Waveform
```

### 为什么这个结构？

**Encoder**：文本是离散的，需要先变成连续的向量序列。

**Attention**：文本长度和语音长度不对等（"你好"2个字符 → 约100帧Mel）。Attention 自动学习这种对齐。

**Decoder**：语音是时序信号，天然适合自回归生成。

**PostNet**：Decoder 输出的 Mel 有"毛刺"，PostNet 用卷积平滑它。

---

## 2.2 Encoder：把文本变成"声音的特征"

### 2.2.1 结构

```
字符 Embedding (512维)
    ↓
3层 Conv1d (kernel=5) + ReLU + BN + Dropout
    ↓
BiLSTM (256×2=512维)
    ↓
Encoder Output: (B, T_text, 512)
```

### 2.2.2 为什么用 Conv1d + BiLSTM？

- **Conv1d**：提取局部上下文（类似 n-gram），比纯 RNN 更稳定
- **BiLSTM**：捕捉长程依赖，且双向编码让每个位置都能看到全文

### 2.2.3 代码位置

`code/model.py` 中的 `Encoder` 类。

---

## 2.3 Location-Sensitive Attention：让对齐不再乱跳

### 2.3.1 Tacotron1 的对齐问题

Tacotron1 使用普通 Attention，经常出现：
- **跳字**：注意力权重突然跳到很远的位置，漏掉中间的字
- **重复**：注意力在某个位置来回震荡，导致同一个字读多遍
- **漏字**：注意力从来没访问到某个位置

### 2.3.2 Location-Sensitive Attention 的解决思路

核心思想：**让 Attention 知道"上一秒我看了哪里"**。

普通 Attention：
$$\text{score} = v^T \tanh(W_{enc} \cdot enc + W_{dec} \cdot dec)$$

Location-Sensitive Attention：
$$\text{score} = v^T \tanh(W_{enc} \cdot enc + W_{dec} \cdot dec + W_{loc} \cdot \text{conv}(\alpha_{prev}))$$

其中 $\alpha_{prev}$ 是上一时刻的注意力权重，先做一个 Conv1d 提取"位置特征"，再加到 score 中。

这样 Attention 就被约束了：**只能缓慢向前移动，不能跳跃**。

### 2.3.3 可视化

训练良好的 Tacotron2，Attention 权重应该呈现一条**清晰的对角线**：

```
      时间 →
    ┌──────────
文  │  █
字  │    █
↓   │      █
    │        █
    └──────────
```

如果 Attention 是混乱的斑块，说明训练出了问题。

---

## 2.4 Decoder：自回归生成 Mel

### 2.4.1 Teacher Forcing

训练时，Decoder 的输入不是上一帧的**预测**，而是**真实的**上一帧 Mel。这称为 **Teacher Forcing**。

好处：训练稳定，梯度传播直接。
坏处：训练和推理不一致（推理时没有 ground truth）。

### 2.4.2 PreNet：信息瓶颈

Decoder 输入先过 PreNet（2层 FC + Dropout），把 80 维 Mel 压到 256 维。

> **Neko 笔记**：PreNet 强制模型压缩信息，防止 Decoder 直接"抄"上一帧。类似正则化效果。

### 2.4.3 双输出头

Decoder 每一步输出两个东西：
1. **Mel 帧**：80 维向量
2. **Stop Token**：二分类（是否结束生成）

Stop Token 让模型自己决定"这段话说完了"。

---

## 2.5 PostNet：最后的打磨

### 2.5.1 为什么需要 PostNet？

Decoder 是 LSTM，擅长时序建模，但不擅长局部细节修正。

PostNet 是 5 层 Conv1d，专门做"局部平滑"：
- 修正 Mel 频谱的小波动
- 增强谐波结构
- 残差连接：输出 = 输入 + 卷积修正

### 2.5.2 残差连接的意义

$$\text{output} = \text{input} + \text{conv}(\text{input})$$

模型只需要学习"差值"（残差），比从零学习完整映射更容易。

---

## 2.6 训练

### 2.6.1 损失函数

Tacotron2 使用三个损失：

$$\mathcal{L} = \mathcal{L}_{mel}^{before} + \mathcal{L}_{mel}^{after} + \mathcal{L}_{stop}$$

- $\mathcal{L}_{mel}^{before}$：Decoder 直接输出的 Mel 与 GT 的 MSE
- $\mathcal{L}_{mel}^{after}$：PostNet 修正后的 Mel 与 GT 的 MSE
- $\mathcal{L}_{stop}$：Stop Token 的 BCE

### 2.6.2 数据准备

```bash
# 下载猫娘数据集（ModelScope）
cd data
python download_neko_1k.py --output-dir processed --num-samples 1000
```

输出格式：
```
processed/
├── wavs/000001.wav
├── train.list          # wav_path|speaker|language|text
└── metadata.csv
```

**注意**：原始数据是 24kHz，训练脚本会自动重采样到 16kHz。超过 25 秒的音频会被过滤以避免 OOM。

### 2.6.3 启动训练

```bash
cd chapters/ch02_tacotron/code
python train.py \
    --data-dir ../../data/processed \
    --epochs 50 \
    --batch-size 4
```

**预期现象**：
- 前 10 个 epoch loss 快速下降
- 20-50 epoch Mel MSE 进入平台期
- 如果 loss 不下降，检查数据是否正确加载
- 每 10 epoch 自动保存 checkpoint 到 `../checkpoints/`

### 2.6.4 训练技巧

1. **Gradient Clipping**：max norm = 1.0，防止 RNN 梯度爆炸
2. **Teacher Forcing Ratio**：本实现使用 100% teacher forcing（简化版）
3. **Batch Size**：如果显存不够，用 4 甚至 2

---

## 2.7 推理与声码器

### 2.7.1 自回归推理

训练时用的是 Teacher Forcing。推理时，Decoder 的输入是**上一帧的预测**：

```
第 1 帧：输入 = 零向量 → 预测第 1 帧 Mel
第 2 帧：输入 = 第 1 帧预测 → 预测第 2 帧 Mel
第 3 帧：输入 = 第 2 帧预测 → ...
```

直到 Stop Token > 0.5 或达到最大长度。

### 2.7.2 Griffin-Lim 声码器

Tacotron2 论文用的是 WaveNet 声码器，但 WaveNet 很慢。

本章先用 **Griffin-Lim**（Ch01 已实现）作为占位声码器，让整条链路能跑通。

```bash
# 基础推理
python inference.py \
    --checkpoint ../checkpoints/tacotron_final.pt \
    --text "你好，我是猫娘。" \
    --output ../outputs/neko_output.wav

# 端到端测试（带计时和 RTF 统计）
python test_tts.py \
    --checkpoint ../checkpoints/tacotron_epoch_50.pt \
    --text "你好，我是猫娘。" \
    --output ../outputs/test_output.wav
```

### 2.7.3 预期效果

- v0.1（早期训练）：可能输出噪音或单个音的重复
- v0.3（训练充分）：能听出语调轮廓，但吐字不清
- v0.6（加上 WaveNet 声码器）：音质大幅提升

> **这是正常的！** Tacotron 训练需要大量数据和计算。本章的目标是理解结构和跑通流程。

---

## 2.8 本章小结

### Tacotron2 的核心贡献

| 问题 | 解决方案 |
|------|----------|
| 人工特征工程 | 端到端：字符 → Mel |
| 对齐不稳定 | Location-Sensitive Attention |
| Mel 细节差 | PostNet 残差修正 |
| 不知道何时停 | Stop Token |

### 遗留问题

1. **自回归慢**：生成 1 秒语音需要 1000 步前向传播 → **FastSpeech** (Ch04)
2. **Griffin-Lim 音质差** → **WaveNet** (Ch03)
3. **需要大量 paired 数据** → **GPT-SoVITS** (Ch06)

### 参考文献

- [1] Shen et al., 2018. *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions*.
- [2] Tachibana et al., 2018. *Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention*.

---

## 习题

1. 为什么 Encoder 用 3 层 Conv1d 而不是直接用 LSTM？Conv 相比 RNN 有什么优势？
2. Location-Sensitive Attention 中，如果对上一时刻的注意力做卷积（kernel=31），这个感受野覆盖了多少个字符位置？
3. 为什么训练时用 Teacher Forcing，但推理时不用？这种不一致会带来什么问题？
4. PostNet 的输入和输出维度都是 80，中间层是 512。这种"瓶颈-扩张-收缩"结构和 AutoEncoder 有什么相似之处？
5. 如果 Stop Token 一直预测为"不停止"，模型会输出什么？这在实际系统中怎么解决？

---

## 目录结构

```
ch02_tacotron/
├── README.md          # 本章教程
├── code/              # 代码
│   ├── model.py       # Tacotron2 模型 (~2600万参数)
│   ├── train.py       # 训练脚本（含数据重采样、时长过滤）
│   ├── inference.py   # 推理 + Griffin-Lim 声码器
│   └── test_tts.py    # 端到端 TTS 测试（带 RTF 统计）
├── checkpoints/       # 模型保存（.gitignore）
└── outputs/           # 生成音频
```
