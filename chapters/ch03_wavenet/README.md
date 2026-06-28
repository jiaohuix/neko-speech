# Ch03: WaveNet — Neko 的声音变好听了

> Ch02 中，Neko 学会了说话，但声音像机器人。
>
> 罪魁祸首是 Griffin-Lim 声码器——它只能从 Mel 频谱恢复"差不多"的波形，丢失了大量细节。
>
> 这一章，我们实现 WaveNet 声码器，让 Neko 的声音变得自然。

## 本章导学

### 为什么学 WaveNet？

Ch02 的 Tacotron2 输出了 Mel 频谱，但 Mel 不是波形。我们需要一个**声码器（Vocoder）**把 Mel 变成波形。

Ch01 介绍了 Griffin-Lim，但它的缺陷很明显：
- 只能恢复幅度谱，相位信息完全丢失
- 输出声音有"金属感"，不自然
- 无法建模音频的时序依赖性

WaveNet（van den Oord et al., 2016）是第一个直接从波形学习的声码器：
- **自回归**：逐采样点生成，建模 P(y_t | y_{<t}, mel)
- **因果卷积**：保证时序因果性
- **门控激活**：比 ReLU 更强的表达能力

### 在整条链路中的位置

```
Text → Tacotron2 (Ch02) → Mel Spectrogram → Vocoder → Waveform
                                                  ↑
                                              本章：WaveNet
```

### 学习路线

| 节 | 内容 | 目标 |
|---|------|------|
| 3.1 | 为什么需要声码器 | 理解 Griffin-Lim 的缺陷 |
| 3.2 | 因果卷积 | 理解自回归序列建模的时序约束 |
| 3.3 | 扩张卷积与感受野 | 理解如何用小卷积核覆盖大时间范围 |
| 3.4 | 门控激活单元 | 理解 tanh × sigmoid 的设计动机 |
| 3.5 | WaveNet 完整架构 | 理解 Mel 条件注入和残差/跳跃连接 |
| 3.6 | 训练 | 用猫娘数据集训练 |
| 3.7 | 推理 | Mel → 波形，对比 Griffin-Lim |
| 3.8 | 遗留问题 | 自回归太慢 → 引出 Ch04 |

---

## 3.1 为什么需要声码器？

### 3.1.1 Griffin-Lim 的缺陷

在 Ch01 中，我们实现了 Griffin-Lim 相位恢复。它的核心问题：

1. **只优化幅度谱一致性**，相位靠随机初始化迭代逼近
2. **没有学习过程**——不知道"语音应该听起来像什么"
3. **输出过度平滑**——谐波结构模糊，高频细节丢失

听感上，Griffin-Lim 生成的语音有明显的"嗡嗡"声，像隔着玻璃说话。

### 3.1.2 WaveNet 的思路

WaveNet 不尝试恢复相位，而是**直接学习波形的条件分布**：

$$P(\mathbf{y} | \text{mel}) = \prod_{t=1}^{T} P(y_t | y_1, \ldots, y_{t-1}, \text{mel})$$

每个采样点的概率由**所有过去的采样点**和**Mel 条件**共同决定。

这和语言模型一样：P(下一个token | 之前所有token)。

### 3.1.3 代价：自回归慢

WaveNet 的代价是**生成速度**。16kHz 音频每秒 16000 个采样点，每个点都需要一次完整前向传播。即使前向传播只要 1ms，1 秒音频也需要 16 秒生成。

> **这就是为什么 Ch04 要学 Parallel WaveNet / HiFi-GAN。**

---

## 3.2 因果卷积：只看过去，不看未来

### 3.2.1 为什么需要因果性？

WaveNet 是自回归模型：生成 y_t 时，只能看到 y_1, ..., y_{t-1}。

如果卷积"看到"了 y_{t+1}，就是信息泄漏——训练时模型偷看了答案。

### 3.2.2 因果卷积的实现

标准 Conv1d（kernel_size=K, dilation=d）对每个位置 t 看到：

$$[x_{t - d(K-1)/2}, \ldots, x_t, \ldots, x_{t + d(K-1)/2}]$$

因果卷积通过**只在左侧 padding** 实现：

```
左侧 padding = (K-1) × d
右侧 padding = 0
```

```
标准卷积 (K=3, d=1):
  [x_{t-1}, x_t, x_{t+1}]  ← 看到未来！

因果卷积 (K=3, d=1):
  [x_{t-2}, x_{t-1}, x_t]  ← 只看过去 ✓
```

### 3.2.3 代码

```python
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))   # 左侧补零
        return self.conv(x)                # 输出长度 = 输入长度
```

**验证方法**：改变 x[t+1:] 的值，y[t] 不应该变化。`model.py` 的测试中包含了因果性验证。

---

## 3.3 扩张卷积与感受野

### 3.3.1 问题：普通卷积感受野太小

kernel_size=3 的普通卷积，每层感受野只增加 2。要看 1000 个采样点，需要 500 层！

### 3.3.2 扩张卷积的解决方案

扩张卷积（Dilated Convolution）在卷积核元素之间插入"空洞"：

```
dilation=1: [x_t, x_{t+1}, x_{t+2}]              跨度 = 3
dilation=2: [x_t, x_{t+2}, x_{t+4}]              跨度 = 5
dilation=4: [x_t, x_{t+4}, x_{t+8}]              跨度 = 9
dilation=8: [x_t, x_{t+8}, x_{t+16}]             跨度 = 17
```

### 3.3.3 感受野计算

每一层 dilation=d, kernel_size=K 的因果卷积，感受野增加 d × (K-1)。

**单层感受野**：

$$RF_{\text{single}} = \sum_{i=0}^{L-1} d_i \times (K - 1) + 1$$

**指数增长（dilation = 1, 2, 4, 8, ...）**：

$$RF = (K - 1) \times \sum_{i=0}^{L-1} 2^i + 1 = (K - 1) \times (2^L - 1) + 1$$

多个 cycle 重复（每个 cycle L 层）：

$$RF_{\text{total}} = C \times (K - 1) \times (2^L - 1) + 1$$

**默认参数**：K=3, L=10, C=3

$$RF = 3 \times 2 \times (1024 - 1) + 1 = 6139 \text{ samples}$$

16kHz 下约 0.38 秒。对语音来说，这足以覆盖一个音节。

> **直觉**：dilation 翻倍就像把"放大镜"拉远一倍，每层看到的范围加倍。

---

## 3.4 门控激活单元

### 3.4.1 为什么不用 ReLU？

ReLU 只能"截断负值"，不能动态调节信号强度。

### 3.4.2 门控激活公式

WaveNet 使用门控激活（Gated Activation Unit）：

$$z = \tanh(W_f * x + V_f * c) \odot \sigma(W_g * x + V_g * c)$$

其中：
- $\tanh(\cdot)$ 是**信息通道**（filter），值域 [-1, 1]
- $\sigma(\cdot)$ 是**门**（gate），值域 [0, 1]
- $x$ 是波形特征，$c$ 是 mel 条件
- $\odot$ 是逐元素乘法

门决定"让多少信息通过"：gate=0 时完全阻断，gate=1 时全部通过。

### 3.4.3 和 LSTM 门控的关系

| 模型 | 门控公式 | 共同点 |
|------|----------|--------|
| LSTM | $f \odot c + i \odot \tilde{c}$ | sigmoid gate × tanh candidate |
| GLU | $x \odot \sigma(Wx)$ | 自门控 |
| WaveNet | $\tanh(Wx+Vc) \odot \sigma(Wx+Vc)$ | 条件门控 |

### 3.4.4 代码

```python
h = self.dil_conv(x) + self.mel_proj(mel_cond)  # (B, 2*res, T)
h_filter, h_gate = h.chunk(2, dim=1)             # 各 (B, res, T)
h = torch.tanh(h_filter) * torch.sigmoid(h_gate) # 门控激活
```

---

## 3.5 WaveNet 完整架构

### 3.5.1 整体结构

```
Mel Spectrogram (B, 80, T_mel)
    ↓ TransposedConv (上采样 hop_length 倍)
Mel Condition (B, 80, T_wav)

Waveform (mu-law 编码, B, T_wav)
    ↓ One-Hot + Linear Projection
Features (B, res_channels, T_wav)
    ↓
┌─── ResidualBlock (dilation=1) ─── skip_1
├─── ResidualBlock (dilation=2) ─── skip_2
├─── ResidualBlock (dilation=4) ─── skip_3
├─── ...
└─── ResidualBlock (dilation=512) ── skip_10
    ↓ (Cycle 重复 3 次 = 30 个 block)
Sum of All Skips (B, skip_channels, T_wav)
    ↓ ReLU + 1×1 Conv + ReLU + 1×1 Conv
Logits (B, 256, T_wav)  ← 256 类 = mu-law 量化
```

### 3.5.2 残差连接与跳跃连接

每个残差块产生两个输出：

1. **残差（Residual）**：传回主路径 `x = x + residual`，维持信息流
2. **跳跃（Skip）**：累加到全局 skip_sum，最终用于预测

这和 ResNet 的思路一致：
- 残差连接让梯度直接流过，深层网络可训练
- 跳跃连接让输出层可以"访问"每一层的特征

### 3.5.3 Mel 条件注入

Mel 在每个残差块中通过 1×1 卷积注入：

```python
h = dilated_causal_conv(x) + conv1x1(mel_condition)
```

Mel 先被 TransposedConv 上采样到波形分辨率（每个 mel 帧扩展为 hop_length 个采样点），确保波形每个时间步都有对应的 mel 条件。

### 3.5.4 mu-law 编码

波形是连续的浮点值，直接回归困难。WaveNet 用 **mu-law 压缩** 将波形量化为 256 个类别：

$$f(x) = \text{sgn}(x) \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}, \quad \mu = 255$$

然后均匀量化到 [0, 255] 的整数。

训练目标变为 **256 分类交叉熵**，而不是 MSE：

$$\mathcal{L} = -\sum_t \sum_{k=0}^{255} y_{t,k} \log \hat{y}_{t,k}$$

比 MSE 更好：可以建模多模态分布（同一个 mel 可能对应多种合理的波形）。

---

## 3.6 训练

### 3.6.1 数据准备

使用猫娘数据集（Ch02 相同）：

```bash
# 如果还没有下载数据
cd data && python download_neko_1k.py --output-dir processed --num-samples 1000
```

### 3.6.2 启动训练

```bash
cd chapters/ch03_wavenet/code

# 基础训练（约 30 分钟 on GPU）
python train.py \
    --data-dir ../../../data/processed \
    --epochs 20 \
    --batch-size 4

# 更大模型（需要更多显存）
python train.py \
    --data-dir ../../../data/processed \
    --epochs 50 \
    --batch-size 4 \
    --res-channels 128 \
    --skip-channels 256 \
    --n-blocks 10 \
    --n-cycles 3
```

### 3.6.3 预期现象

- **前 5 epoch**：loss 快速下降（模型学习基本分布）
- **10-20 epoch**：loss 缓慢下降（细化波形建模）
- **50+ epoch**：平台期，音质缓慢提升
- 每 5 epoch 自动保存 checkpoint 到 `../checkpoints/`

### 3.6.4 训练技巧

1. **Gradient Clipping**：max norm = 1.0，防止深层网络梯度爆炸
2. **AMP**：GPU 上自动启用混合精度（FP16），加速 ~1.5x
3. **Segment Length**：默认 8192 采样点（0.5 秒），增大可提升质量但消耗更多显存

---

## 3.7 推理：Mel → 波形

### 3.7.1 从 Ground Truth Mel 推理

```bash
python inference.py \
    --checkpoint ../checkpoints/wavenet_final.pt \
    --source ../../../data/processed/wavs/000001.wav \
    --max-samples 8000 \
    --output-dir ../outputs
```

这会：
1. 从源音频计算 Mel 频谱
2. 用 WaveNet 自回归生成波形
3. 用 Griffin-Lim 生成基线
4. 保存两个音频供对比

### 3.7.2 与 Tacotron2 联动

```bash
python inference.py \
    --checkpoint ../checkpoints/wavenet_final.pt \
    --tacotron-checkpoint ../../ch02_tacotron/checkpoints/tacotron_final.pt \
    --text "你好，我是猫娘。" \
    --output-dir ../outputs
```

### 3.7.3 预期结果

| 方法 | 音质特点 | 生成速度 |
|------|----------|----------|
| Griffin-Lim | 金属感，模糊，高频伪影 | 快（迭代收敛） |
| WaveNet (训练充分) | 更自然，清晰的高频细节 | 非常慢（逐采样点） |
| WaveNet (训练不足) | 可能是噪声 | - |

> **注意**：自回归生成非常慢！8000 采样点（0.5 秒）可能需要几分钟。
> 这是因为每一步都要过一遍完整的 30 层网络。

---

## 3.8 本章小结

### WaveNet 的核心贡献

| 问题 | 解决方案 |
|------|----------|
| Griffin-Lim 相位丢失 | 直接建模波形分布，无需相位恢复 |
| 普通卷积感受野小 | 扩张卷积：指数增长的感受野 |
| 自回归时序约束 | 因果卷积：左侧 padding |
| ReLU 表达力不足 | 门控激活：tanh × sigmoid |
| 深层网络难训练 | 残差连接 + 跳跃连接 |
| 连续波形难建模 | mu-law 压缩 → 256 分类 |
| 声码器如何条件化 | Mel 上采样 + 逐层 1×1 注入 |

### 遗留问题：自回归太慢

WaveNet 最大的问题是**推理速度**。生成 1 秒 16kHz 音频需要 16000 步前向传播，每步约 1ms，总计 16 秒。

RTF (Real-Time Factor) = 16x，完全无法实时使用。

后续解决方案：

| 方法 | 思路 | 加速比 |
|------|------|--------|
| Parallel WaveNet (2018) | 知识蒸馏，并行生成 | ~1000x |
| WaveRNN (2019) | 次采样 + 轻量 RNN | ~100x |
| WaveGlow (2019) | Flow-based，非自回归 | ~50x |
| HiFi-GAN (2020) | GAN 训练，一次生成整段 | ~100x |

> **下一章（Ch04）我们实现 HiFi-GAN**——用 GAN 的方式一次性生成整段波形。

### 参考文献

- [1] van den Oord et al., 2016. *WaveNet: A Generative Model for Raw Audio*.
- [2] Mehri et al., 2017. *SampleRNN: An Unconditional End-to-End Neural Audio Generation Model* (mu-law encoding).
- [3] Shen et al., 2018. *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions* (Tacotron2 + WaveNet).
- [4] van den Oord et al., 2018. *Parallel WaveNet: Fast High-Fidelity Speech Synthesis*.

---

## 习题

1. **因果卷积验证**：如果去掉因果 padding（用对称 padding），训练 loss 会怎样变化？为什么这反而可能导致训练 loss 更低但推理效果更差？

2. **感受野计算**：
   - 当前默认配置（K=3, L=10, C=3），感受野是多少？
   - 如果把 dilation 序列改为 [1, 2, 4, 8, 16, 32]（L=6），感受野是多少？
   - 感受野小于 hop_length（256）会有什么问题？

3. **扩张卷积的效率**：dilation=512, kernel_size=3 的因果卷积，每个输出位置实际需要读取多少个输入值？这和普通 kernel_size=1025 的卷积有什么区别？

4. **门控激活 vs ReLU**：
   - 画出 tanh(x) × sigmoid(x) 的函数图像
   - 当 sigmoid 输出接近 0 时，梯度会怎样？这是否会导致"门控死区"？

5. **自回归速度**：
   - WaveNet 生成 1 秒 16kHz 音频需要多少步前向传播？
   - WaveRNN 通过 4x 次采样减少了多少步？
   - HiFi-GAN 为什么可以一次性生成整段？（提示：GAN 的生成器是非自回归的）

6. **mu-law 编码**：
   - mu-law 编码对大振幅和小振幅的量化精度有何不同？为什么这对语音有利？
   - 如果改用线性量化 256 级，[-1, 1] 的步长是多少？小振幅信号会怎样？

---

## 目录结构

```
ch03_wavenet/
├── README.md              # 本章教程
├── code/
│   ├── model.py           # WaveNet 架构 + mu-law + 形状验证 (~260 行)
│   ├── train.py           # 训练脚本 (~190 行)
│   └── inference.py       # 推理脚本 + Griffin-Lim 对比 (~170 行)
├── checkpoints/           # 模型权重 (.gitignore)
└── outputs/               # 生成的音频
```
