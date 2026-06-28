# Ch06: Neural Audio Codec — Neko 的音频压缩术

> 在让 Neko 用 Token 说话之前，她需要先学会"压缩"声音。
>
> 这一章，我们实现简化的 EnCodec，理解 RVQ-VAE 如何把波形变成离散 Token。

## 本章导学

### 为什么需要 Neural Codec？

在前面的章节中，我们已经见过两种音频表示：

| 表示 | Ch01 | Ch02 | 问题 |
|------|------|------|------|
| Waveform | 原始波形 | Tacotron 的声码器输出 | 采样点太多（24kHz = 每秒 24000 个数），不适合建模 |
| Mel Spectrogram | STFT + Mel 滤波器组 | Tacotron 直接预测 | 连续值，维度固定 80，相位丢失 |

这两种表示都有一个共同的问题：**它们是为人类设计的**，不是为神经网络设计的。

- Mel Spectrogram 的 Mel 刻度基于人耳感知，但神经网络不关心人耳。
- STFT 的窗口大小是人工选择的，无法自适应不同音频。
- Griffin-Lim 从 Mel 重建波形效果很差（Ch01 实验已验证）。

**核心问题**：有没有一种表示，既紧凑（少量离散 Token），又能高保真重建波形？

### EnCodec 的回答

2022 年，Meta AI 提出了 EnCodec：一个端到端训练的**神经音频编解码器**。

```
Waveform [24000 samples/sec]
    ↓ Encoder (320× 下采样)
Continuous Latent [75 frames/sec, 128-dim]
    ↓ RVQ (8 层码本量化)
Discrete Tokens [75 frames/sec × 8 codebooks]
    ↓ Decoder (320× 上采样)
Reconstructed Waveform [24000 samples/sec]
```

**核心成果**：
- 24kHz 音频压缩到 6 kbps（75 帧/秒 × 8 码本 × 10 bit），质量接近原始
- 离散 Token 可以直接被语言模型（如 VALL-E、GPT-SoVITS）建模
- 这是所有现代 Audio LM 的基础组件

### 学习路线

| 节 | 内容 | 目标 |
|---|------|------|
| 6.1 | VQ-VAE 基础 | 理解向量量化 + 自编码器 |
| 6.2 | RVQ 残差向量量化 | 理解多层码本如何逐层逼近 |
| 6.3 | EnCodec 架构 | 理解 Encoder-Decoder 设计 |
| 6.4 | 训练与损失函数 | 理解重建损失 + commitment loss |
| 6.5 | 实验 | 运行代码，评估压缩质量 |
| 6.6 | 与 EnCodec/SoundStream 的关系 | 理解工业级实现 |

---

## 6.1 VQ-VAE 基础：用离散码本表示连续向量

### 6.1.1 为什么需要离散化？

连续向量 z ∈ R^d 有两个问题：

1. **不可数**：无法被语言模型直接建模（语言模型处理的是离散 Token）
2. **冗余**：相邻帧的 z 通常高度相关，存在压缩空间

**向量量化（Vector Quantization）** 的解法：

```
预定义一个码本 (codebook): {e_1, e_2, ..., e_K}，每个 e_k ∈ R^d

对于输入 z，找到码本中最近的向量:
    k* = argmin_k ||z - e_k||_2

用 e_{k*} 代替 z:
    z_q = e_{k*}
```

这样，连续的 d 维向量就被压缩成一个整数索引 k* ∈ {0, 1, ..., K-1}。

> **Neko 笔记**：就像用颜色编号代替描述"这个红偏橙带点紫"——只要双方都有一本相同的颜色手册（码本），一个数字就够了。

### 6.1.2 VQ-VAE 完整流程

```
x (input) → Encoder → z (continuous)
                          ↓
                    VQ: k* = argmin ||z - e_k||
                          ↓
                    z_q = e_{k*} (quantized)
                          ↓
                    Decoder → x_hat (reconstructed)
```

### 6.1.3 训练的难点：argmin 不可导

VQ 的核心操作 `argmin` 是不可导的——梯度无法从 z_q 流回 z。

**解决方案：Straight-Through Estimator (STE)**

```python
# 前向传播: 用离散值 z_q
# 反向传播: 梯度直通 z（跳过 argmin）
z_q = z + (quantize(z) - z).detach()
```

这看起来很 hack，但直觉上合理：
- 前向时，我们用最近邻代替 z（离散化）
- 反向时，我们假设 z_q ≈ z，让梯度直接更新 encoder

### 6.1.4 VQ-VAE 的三个损失

```python
# 1. 重建损失: 让 decoder 输出接近输入
loss_recon = ||x - x_hat||

# 2. 码本损失: 让码本向量靠近 encoder 输出
loss_codebook = ||sg(z) - e||^2     # sg = stop_gradient

# 3. 承诺损失: 让 encoder 输出靠近码本（防止 encoder "跑太远"）
loss_commitment = ||z - sg(e)||^2   # β = 0.25

# 总损失
loss = loss_recon + loss_codebook + β * loss_commitment
```

> **Neko 笔记**：承诺损失就像给 encoder 拴一根绳子——"你可以自由学习，但不要跑到码本找不到的地方去。"

---

## 6.2 RVQ 残差向量量化：从粗到细的多层逼近

### 6.2.1 单层 VQ 的局限

假设 K=1024（码本 1024 个条目），每帧用 log₂(1024) = 10 bit 表示。

对于 75 帧/秒的音频：10 × 75 = 750 bps。

要提高质量，要么：
- 增大 K → K=65536 → 查找表开销爆炸
- 增大 d → 计算量增大

**有没有更好的方法？**

### 6.2.2 RVQ 的核心思想：逐层量化残差

RVQ（Residual Vector Quantization）是 EnCodec 的核心创新。

```
第 1 层: z_q1 = quantize(z, codebook_1)
         残差 r1 = z - z_q1

第 2 层: z_q2 = quantize(r1, codebook_2)
         残差 r2 = r1 - z_q2

...

第 K 层: z_qK = quantize(r_{K-1}, codebook_K)

最终量化结果: z_q = z_q1 + z_q2 + ... + z_qK
```

**直觉**：
- 第 1 层捕捉 **粗粒度** 信息（类似 JPEG 的 DC 系数）
- 第 2 层捕捉 **第 1 层的误差**（类似 JPEG 的低频 AC 系数）
- 第 K 层捕捉 **高频细节**

这和 JPEG 的 DCT 逐层量化思想完全一致！

### 6.2.3 RVQ 的信息量

```
K 层码本，每层 1024 条目:
    每帧 = K × log₂(1024) = K × 10 bit

8 层码本:
    每帧 = 8 × 10 = 80 bit
    75 帧/秒 → 6000 bps = 6 kbps

相比 24kHz 16-bit PCM:
    原始 = 24000 × 16 = 384 kbps
    压缩比 = 384 / 6 = 64×
```

### 6.2.4 为什么 RVQ 比大码本更好？

| 方案 | 码本大小 | 每帧 bit | 查找复杂度 |
|------|---------|----------|-----------|
| 单层 K=1024 | 1024 | 10 | O(1024) |
| 单层 K=65536 | 65536 | 16 | O(65536) |
| RVQ 8×1024 | 8×1024=8192 | 80 | O(8×1024)=O(8192) |

RVQ 用 O(K×N) 的计算量获得 K^N 的表达力——**指数级的信息容量，线性的计算开销**。

### 6.2.5 RVQ 代码实现

核心代码在 `codec.py` 的 `ResidualVectorQuantizer` 类：

```python
class ResidualVectorQuantizer(nn.Module):
    def forward(self, z):
        residual = z           # 初始残差 = z
        z_q = 0                # 累积量化结果

        for vq in self.quantizers:
            z_q_k = vq(residual)          # 量化当前残差
            z_q += z_q_k                   # 累加
            residual = residual - z_q_k    # 更新残差

        return z_q
```

**关键点**：每层量化的是上一层的**残差**，不是原始 z。

---

## 6.3 EnCodec 架构：Encoder-Decoder 设计

### 6.3.1 Encoder：波形 → 连续隐变量

```
Waveform [B, 1, 24000]
    ↓ Conv1d(1→32, k=7, p=3)           # 初始投影
    ↓ EncoderBlock(32→64, stride=8)     # 8× 下采样
    ↓ EncoderBlock(64→128, stride=5)    # 5× 下采样
    ↓ EncoderBlock(128→256, stride=4)   # 4× 下采样
    ↓ EncoderBlock(256→512, stride=2)   # 2× 下采样
    ↓ Conv1d(512→128, k=7, p=3)        # 投影到 latent dim
Continuous Latent [B, 128, 75]
```

每个 EncoderBlock 的结构：
```
x → GELU → Conv1d(k=3) → GELU → Conv1d(k=1) → (+x)  [残差]
  → Conv1d(k=2s, stride=s)                            [下采样]
```

**为什么这些 stride？**
- 8 × 5 × 4 × 2 = 320 → 24000 / 320 = 75 帧/秒
- 与 EnCodec 论文一致
- 混合 stride（而非全用 2）减少层数，同时保持下采样效率

### 6.3.2 Decoder：连续隐变量 → 重建波形

Decoder 是 Encoder 的完美镜像：

```
Quantized Latent [B, 128, 75]
    ↓ Conv1d(128→512, k=7, p=3)         # 投影
    ↓ DecoderBlock(512→256, stride=2)    # 2× 上采样
    ↓ DecoderBlock(256→128, stride=4)    # 4× 上采样
    ↓ DecoderBlock(128→64, stride=5)     # 5× 上采样
    ↓ DecoderBlock(64→32, stride=8)      # 8× 上采样
    ↓ Conv1d(32→1, k=7, p=3)            # 投影到波形
Reconstructed Waveform [B, 1, 24000]
```

每个 DecoderBlock 用 `ConvTranspose1d` 实现上采样。

### 6.3.3 接口约定

```python
# 编码: 波形 → 离散 Token
tokens = model.encode(wav)        # [B, 1, T] → [B, K, T/320]

# 解码: 离散 Token → 重建波形
wav_hat = model.decode(tokens)    # [B, K, T/320] → [B, 1, T]
```

这个接口非常重要——它定义了 Audio Tokenizer 的标准 API，被后续的 VALL-E、GPT-SoVITS 等模型直接使用。

---

## 6.4 训练与损失函数

### 6.4.1 三类损失

```python
# 1. L1 时域损失
loss_l1 = |wav - wav_hat|

# 2. 多分辨率 STFT 损失
loss_spec = mean over resolutions:
    |STFT(wav) - STFT(wav_hat)|
# 分辨率: [(256,64), (512,128), (1024,256), (2048,512)]

# 3. VQ 损失 (来自 RVQ)
loss_vq = sum over codebooks:
    ||sg(z) - e||^2 + 0.25 × ||z - sg(e)||^2

# 总损失
loss = loss_l1 + loss_spec + loss_vq
```

### 6.4.2 为什么需要多分辨率 STFT 损失？

单个 n_fft 只能在时间分辨率和频率分辨率之间取一个平衡点：
- 小 n_fft (256): 时间分辨率高（瞬态准确），频率分辨率低
- 大 n_fft (2048): 频率分辨率高（音高准确），时间分辨率低

多分辨率组合让模型**同时学好时域细节和频域结构**。

### 6.4.3 码本崩塌 (Codebook Collapse)

训练 VQ-VAE 最常见的陷阱是**码本崩塌**：

- 大量码本条目从未被使用（"死码"）
- 只有少数条目被频繁使用
- 码本的有效容量远小于名义容量

```
理想情况:  1024 个码字均匀使用
码本崩塌:  仅 50 个码字被使用，其余 974 个是"死码"
```

缓解方法：
- EMA 更新码本（而非梯度更新）
- 码本重置：定期将死码替换为随机编码向量
- 更大的 β (commitment loss 权重)

> **Neko 笔记**：码本崩塌就像一本 1024 页的字典，你只翻了前 50 页——剩下 974 页完全浪费了。

---

## 6.5 实验

### 6.5.1 快速验证（合成数据）

```bash
cd chapters/ch06_neural_codec/code

# 3 epoch 快速训练（验证代码正确性）
python train.py --synthetic --epochs 3 --batch-size 4

# 测试编解码
python test_codec.py --ckpt ../checkpoints/codec_best.pt
```

### 6.5.2 真实训练

```bash
# 用真实数据训练（推荐 50 epochs）
python train.py \
    --data-dir ../../../data/processed \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4

# 测试重建质量
python test_codec.py \
    --ckpt ../checkpoints/codec_best.pt \
    --audio ../../../data/processed/wavs/000001.wav
```

### 6.5.3 预期结果

| 训练阶段 | SNR (dB) | Waveform Correlation | 码本使用率 |
|----------|----------|---------------------|-----------|
| 未训练 | < 0 | ~0.0 | ~1% |
| 10 epochs | 5-10 | 0.5-0.7 | 10-30% |
| 50 epochs | 15-25 | 0.85-0.95 | 50-80% |

### 6.5.4 实验观察

1. **Token 可视化**：训练后，不同音频区域的 Token 分布应有规律（元音 vs 辅音 vs 静音）
2. **码本使用率**：第一层码本使用率通常最高（捕捉粗粒度信息），深层递减
3. **重建质量**：听 `outputs/reconstructed.wav`，与 `outputs/original.wav` 对比

---

## 6.6 与 EnCodec / SoundStream 的关系

### 6.6.1 对比表

| 特性 | 本教程 EnCodec Mini | Meta EnCodec | Google SoundStream |
|------|---------------------|-------------|-------------------|
| 参数量 | ~4.5M | ~30M | ~30M |
| 下采样 | 320× | 320× | 320× |
| 码本 | 8×1024 | 8×1024 | 可变 |
| 判别器 | 无 | Multi-scale | Multi-scale |
| 损失 | L1 + STFT + VQ | L1 + STFT + 对抗 + VQ | 类似 EnCodec |
| 音质 | 可接受 | 接近原始 | 接近原始 |

### 6.6.2 我们缺了什么？

**对抗训练（Adversarial Training）**：
- EnCodec 使用多尺度判别器（Multi-Scale Discriminator）
- 判别器判断重建音频是"真实的"还是"生成的"
- 这极大提升了高频细节和感知质量
- 类似于 GAN 的思想

**更深的网络**：
- EnCodec 使用更多残差块（每个下采样层有多个残差块）
- 更大的通道数

**EMA 码本更新**：
- 用指数移动平均更新码本向量，而非梯度
- 更稳定，缓解码本崩塌

**但这些是工程优化**。核心的 RVQ-VAE 原理——我们已经完整实现了。

---

## 6.7 Codec 作为 Audio Tokenizer 的意义

### 6.7.1 从连续到离散的范式转变

EnCodec 之前，Audio AI 的主流范式：

```
文本 → TTS模型 → Mel Spectrogram → 声码器 → 波形
                (连续值, 80-dim)
```

EnCodec 之后：

```
文本 → LM → Audio Tokens → Codec Decoder → 波形
           (离散值, K×T)
```

**为什么离散 Token 更好？**

1. **可以复用语言模型的技术**：Transformer、自回归、KV Cache……
2. **可以做多模态统一**：文本 Token + 音频 Token → 同一个 LM
3. **可以做零样本学习**：给几个参考 Token → 生成新音频

### 6.7.2 Audio Token 的层次结构

RVQ 的 8 层码本自然形成了层次结构：

```
Layer 0: 语义信息（内容、语言）     ← 最重要
Layer 1: 声学信息（音高、音色）
Layer 2: 细节信息（气息、颤音）
...
Layer 7: 极细细节（环境噪声、量化噪声）
```

这个层次结构与 VALL-E 的设计直接相关——VALL-E 用自回归预测第 0 层（语义），用并行预测第 1-7 层（声学细节）。

---

## 6.8 遗留问题：如何用这些 Token 做 TTS？

到这里，我们有了一个可以把波形压缩成 Token、再从 Token 重建波形的 Codec。

但核心问题还没解决：**怎么从文本生成这些 Token？**

答案就在下一章：**VALL-E**。

```
Ch06 (本章)                    Ch07 (下一章)
─────────────────              ─────────────────
Waveform → Token               Text → Token
(压缩，已知输入)               (生成，未知输入)

学会了"读写"音频              接下来学会"说话"
```

VALL-E 的核心思想：把 TTS 建模成一个**语言模型**问题——
- 输入：文本 Token + 3 秒参考音频 Token
- 输出：目标音频 Token
- 模型：Transformer (GPT-style)

然后把我们 Ch06 训练的 Codec Decoder 接上去，就能从 Token 重建波形。

> **Neko 预告**：下一章，Neko 终于要学会说话了！而且这次，她用的是和 GPT 一样的方法——预测下一个 Token。

---

## 参考论文

- Defossez et al., "High Fidelity Neural Audio Compression", 2022 ([arXiv](https://arxiv.org/abs/2210.13438))
- van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE), 2017 ([arXiv](https://arxiv.org/abs/1711.00937))
- Zelmer et al., "SoundStream: An End-to-End Neural Audio Codec", 2021 ([arXiv](https://arxiv.org/abs/2107.03312))
- Wang et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (VALL-E), 2023 ([arXiv](https://arxiv.org/abs/2301.02111))

---

## 代码文件

```
chapters/ch06_neural_codec/
├── README.md              # 本文件
├── code/
│   ├── codec.py           # EnCodec Mini 模型 (~340 行)
│   ├── train.py           # 训练脚本
│   └── test_codec.py      # 测试与评估
├── checkpoints/           # 训练权重
└── outputs/               # 测试输出
```
