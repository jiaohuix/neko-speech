# Ch10: VoxCPM — 猫娘不再需要 Tokenizer

> 前几章，Neko 学会了用 VALL-E 把音频变成离散 Token，再用语言模型预测。
> 但她总觉得不对劲——把连续的声音强行切成离散的"积木块"，是不是丢了什么？
>
> 这一章，Neko 发现了一条更优雅的路：**直接在连续空间里说话**。

## 本章导学

### 为什么需要 Tokenizer-Free？

回顾 VALL-E（Ch07）和它的后代：

```
传统 TTS 流水线:
  文本 → [Neural Codec] → 离散 Token → [语言模型] → 预测 Token → [Codec 解码] → 波形
            EnCodec              AR                     Decoder
            RVQ 码本                                    RVQ 重建
```

这种"离散化"思路有几个根深蒂固的问题：

1. **信息丢失**：量化是不可逆的。声音中那些微妙的气音、音色微变、情感细节，一旦被"四舍五入"到最近的码本条目，就永远消失了。
2. **码本坍塌**：EnCodec 的码本有 30-50% 的条目根本不被使用——资源浪费。
3. **多级预测复杂**：RVQ 通常有 8-12 层，需要逐层顺序预测，推理慢。
4. **梯度阻断**：argmin 操作不可微，训练时无法端到端优化。

VoxCPM 的回答：**不用 Token，直接在连续空间里生成**。

```
VoxCPM 流水线:
  文本 → [AudioVAE 编码器] → 连续 Latent → [TSLM + RALM + CFM] → Latent → [AudioVAE 解码器] → 波形
          (因果 CNN)           (不量化!)      (语义+声学+扩散)              (因果 CNN)
```

> **VoxCPM** = Voice Conditional Patch Model（arXiv:2509.24650，OpenBMB）。
> 核心创新：用**连续 latent patch** 替代离散 audio token，用**条件流匹配（CFM）**替代交叉熵损失。

### 核心直觉：连续比离散更"保真"

Neko 用一个比喻来理解：

| 类比 | 离散 Token（VALL-E） | 连续 Latent（VoxCPM） |
|------|---------------------|---------------------|
| 画画 | 从 1024 种固定颜色的蜡笔盒里选 | 用无限渐变的调色盘 |
| 说话 | 用固定的音节拼凑 | 自然流畅地发声 |
| 信息 | 量化 = 四舍五入，有损 | 连续 = 保留所有细节 |

当然，连续也有代价——语言模型擅长预测离散分布（softmax），预测连续分布需要更复杂的生成方法（diffusion / flow matching）。

### 学习路线

| 节 | 内容 | 目标 |
|---|------|------|
| 10.1 | AudioVAE：连续潜空间 | 理解因果 CNN 如何把波形压缩成连续 latent |
| 10.2 | 为什么不要 Tokenizer | 对比离散 Token vs 连续 Latent 的利弊 |
| 10.3 | TSLM + FSQ：语义规划 | 理解 TSLM 的语义角色和 FSQ 的语义瓶颈设计 |
| 10.4 | RALM：声学细化 | 理解 RALM 如何补充声学细节 |
| 10.5 | CFM：条件流匹配 | 理解为什么 CFM 比 DDPM 更适合音频生成 |
| 10.6 | 完整代码走读 | 从零实现 MiniVoxCPM |
| 10.7 | 训练与推理 | 跑通完整流程 |
| 10.8 | 与工业界的关系 | CosyVoice 2、FishSpeech 等现代 Tokenizer-Free 模型 |

---

## 10.1 AudioVAE：把声音变成连续潜空间

### 10.1.1 问题：16kHz 波形太长了

1 秒 16kHz 音频 = 16,000 个浮点数。语言模型处理这样的序列太慢。

解决方案：用一个 **AudioVAE** 把波形压缩到低帧率的连续表示。

### 10.1.2 因果 CNN 编码器

VoxCPM 的 AudioVAE 编码器是一个**因果卷积网络**（Causal CNN）：

```
16 kHz 波形:  [s₁, s₂, s₃, ..., s₁₆₀₀₀]   (16000 个采样点)
                  ↓ Causal Conv1d (stride=2)
                  ↓ Causal Conv1d (stride=5)
                  ↓ Causal Conv1d (stride=8)
                  ↓ Causal Conv1d (stride=8)
Latent:       [z₁, z₂, ..., z₂₅]             (25 帧/秒, 每帧 32 维)
```

**总下采样率**：2 × 5 × 8 × 8 = 640。所以 16000 / 640 = 25 Hz。

**因果（Causal）**：每个卷积只看到当前帧和之前的帧，不偷看未来。这使得编码器可以**流式处理**——边来音频边编码，不需要等整段录完。

```python
# 因果卷积的实现技巧：左填充
def causal_conv1d(x, weight, kernel_size):
    x = F.pad(x, (kernel_size - 1, 0))   # 只在左边填充
    return F.conv1d(x, weight)            # 不设 padding
```

### 10.1.3 因果 CNN 解码器

解码器用 **转置卷积（ConvTranspose1d）** 上采样，镜像结构：

```
Latent (25 Hz, D=32)
    ↓ ConvTranspose1d (stride=8)
    ↓ ConvTranspose1d (stride=8)
    ↓ ConvTranspose1d (stride=5)
    ↓ ConvTranspose1d (stride=2)
Reconstructed waveform (16 kHz)
```

### 10.1.4 对比：AudioVAE vs EnCodec

| 特性 | AudioVAE (VoxCPM) | EnCodec (VALL-E) |
|------|-------------------|-----------------|
| 输出 | 连续向量 (float) | 离散码本索引 (int) |
| 量化 | 无 | RVQ（残差向量量化） |
| 帧率 | 25 Hz | 50 Hz |
| 维度 | 32-64 维 | 8-12 个码本 × 1024 条目 |
| 梯度 | 全链路可微 | 在量化处断开 |
| 重建质量 | 高（连续空间） | 受限于码本容量 |

### 10.1.5 代码位置

`code/model.py` 中的 `SimpleAudioVAE` 类：

```python
vae = SimpleAudioVAE(encoder_dim=64, latent_dim=32, decoder_dim=256)

# 编码：波形 → 连续 latent
z = vae.encode(waveform)     # (B, 32, 25)  — 1秒音频 → 25帧 × 32维

# 解码：连续 latent → 波形
wav_hat = vae.decode(z)      # (B, 16000)
```

---

## 10.2 为什么不要 Tokenizer？

### 10.2.1 离散化的信息论代价

EnCodec 把连续向量"四舍五入"到最近的码本条目：

```
连续向量 v = [0.314, -0.271, 0.577, ...]     ← 无限可能性
                    ↓ argmin 距离
码本索引 i = 42                               ← 只有 1024 种可能
```

**信息丢失量**：连续向量的信息量远大于一个整数索引。每次量化都在"截断"信息。

### 10.2.2 码本坍塌问题

训练过程中，很多码本条目"死掉"——没有任何输入被量化到它们：

```
理想：1024 个码本条目全部活跃，均匀使用
现实：~500 个活跃，~524 个从不被使用

→ 有效容量只有标称的一半
→ 需要各种 trick 来缓解（EMA 更新、码本重初始化、...）
```

### 10.2.3 RVQ 的多级误差累积

Residual VQ 逐层量化残差，每一层都引入量化误差：

```
原始向量 v
Level 0: q₀ = quantize(v)          → 误差 e₀ = v - q₀
Level 1: q₁ = quantize(e₀)         → 误差 e₁ = e₀ - q₁
Level 2: q₂ = quantize(e₁)         → 误差 e₂ = e₁ - q₂
...
Level 7: q₇ = quantize(e₆)         → 误差 e₇ = e₆ - q₇

总重建误差 = e₀ + e₁ + ... + e₇     ← 8 层误差叠加
```

VoxCPM 的 AudioVAE **没有这个问题**——连续空间，无量化误差。

### 10.2.4 Tokenizer-Free 的优势总结

1. **无信息丢失**：连续 latent 保留完整的声学信息
2. **无码本坍塌**：不需要维护码本，无死条目
3. **端到端可微**：VAE 编码器 → 语言模型 → VAE 解码器，梯度畅通
4. **更好的音色克隆**：连续的音色细节不被量化丢失
5. **更简单的 AR 模型**：预测一个连续向量 vs 顺序预测 8 层离散分布

### 10.2.5 Tokenizer-Free 的代价

当然没有免费的午餐：

1. **语言模型不擅长连续**：softmax 只能预测离散分布，连续分布需要 diffusion/flow matching
2. **推理更慢**：每个 AR 步骤需要多步扩散采样（10 步 Euler ODE）
3. **训练更复杂**：flow matching loss 比交叉熵难调

---

## 10.3 TSLM + FSQ：语义规划

### 10.3.1 TSLM（Text-Semantic Language Model）

TSLM 是 VoxCPM 的"大脑"——它决定"说什么"（语义内容），但不关心"怎么说"（声学细节）。

```
输入: [text_emb₁, ..., text_embₗ, BOS, audio_emb₁, ..., audio_embₜ₋₁]
输出: [语义表示₁, ..., 语义表示ₗ, _, 语义表示₁, ..., 语义表示ₜ₋₁]
                                                              ↑
                                              预测下一个 patch 的语义
```

在我们的教学版本中，TSLM 是一个 8 层 decoder-only Transformer（hidden=512, heads=8, ffn=2048），和 GPT-2 结构相同。

### 10.3.2 LocEnc：把 patch 压缩成一个向量

AudioVAE 输出的每个 latent patch 是 (patch_size, latent_dim) 的矩阵。LocEnc（Local Encoder）把它压缩成一个 hidden 向量：

```
Latent patch: (P=1, D=32)
    ↓ Linear projection → (hidden=512)
    ↓ + CLS token → (P+1, hidden)
    ↓ 2-layer Transformer (bidirectional)
    ↓ CLS token pooling
Hidden vector: (hidden=512)
```

CLS token 机制：在序列开头加一个可学习的特殊 token，让它通过注意力"看到"所有 patch 帧，最终取出它的表示作为整个 patch 的摘要。

### 10.3.3 FSQ：有限标量量化——温柔的语义瓶颈

这是 VoxCPM 最精妙的设计之一。

**问题**：TSLM 的输出是一个 512 维的连续向量，信息量很大。我们希望 RALM（后面的声学模型）能专注于"补充细节"，而不是重复 TSLM 的工作。

**解决方案**：用 **FSQ（Finite Scalar Quantization）** 在 TSLM 和 RALM 之间创建一个"语义瓶颈"：

```python
class SimpleFSQ:
    def forward(self, x):
        h = self.in_proj(x)           # 512 → 128 (降维)
        h = torch.tanh(h)             # 限制到 [-1, 1]
        q = round(h * 9) / 9          # 量化到 19 个离散级别
        h = h + (q - h).detach()      # 直通估计器 (STE)
        return self.out_proj(h)       # 128 → 512 (升维)
```

FSQ 的量化有多"温柔"？

```
tanh 输出: [-1.0, 1.0]
量化步长:  1/9 ≈ 0.111
可能取值:  {-1, -8/9, -7/9, ..., 0, ..., 7/9, 8/9, 1}  ← 共 19 个值

每维 19 个值 × 128 维 = 19^128 种组合（仍然天文数字）
但比连续空间的无限可能性小很多
```

**直通估计器（Straight-Through Estimator）**：

```
前向:   h → round → q       (有梯度阻断)
反向:   ∂L/∂h = ∂L/∂q       (梯度直通, 忽略 round)
```

这样，FSQ 在前向传播时是离散的（创建瓶颈），在反向传播时是可微的（允许学习）。

### 10.3.4 FSQ vs VQ 对比

| 特性 | FSQ (VoxCPM) | VQ (VQ-VAE / EnCodec) |
|------|-------------|---------------------|
| 量化对象 | LM 的 hidden state | 音频表示 |
| 量化方式 | 逐维标量四舍五入 | 码本最近邻查找 |
| 码本 | 无（数学计算） | 有（可学习参数） |
| 坍塌风险 | 无 | 高（需要 EMA 等技巧） |
| 梯度 | 直通估计器 | 直通估计器 |
| 信息保留 | 多（19 级 × 多维） | 少（码本大小限制） |

---

## 10.4 RALM：声学细化

### 10.4.1 为什么需要 RALM？

TSLM + FSQ 的输出是一个"粗略的语义蓝图"——它知道要说什么，但缺少声学细节。

RALM（Residual Acoustic Language Model）的任务是**在语义蓝图的基础上，补充声学细节**：

```
TSLM 输出（经 FSQ）: "我要说一个高音的 '喵'，时长 0.3 秒"  ← 语义
RALM 补充:           "基频 440Hz，F1=800Hz，气息声 15%，颤音 5Hz"  ← 声学
```

### 10.4.2 RALM 结构

RALM 是一个 4 层 decoder-only Transformer（和 TSLM 同样的 hidden size = 512），但更浅：

```
输入: FSQ 量化的语义表示 (因果自注意力)
输出: 声学细节残差

TSLM（8 层，深）→ 语义理解
RALM（4 层，浅）→ 声学精修
```

### 10.4.3 语义-声学融合（v1 风格）

```python
# v1: 加法融合（简单有效）
dit_cond = lm_to_dit(tslm_output) + res_to_dit(ralm_output)

# v2: 拼接 + 投影（更强但更复杂）
# dit_cond = fusion_proj(cat(lm_to_dit(tslm), res_to_dit(ralm)))
```

我们的教学版本使用 v1 的加法融合。

### 10.4.4 分层设计的直觉

```
                    TSLM (深, 8层)
                    ┌──────────────────┐
Text + Audio ───────► 语义理解         │
                    │ "说什么"          │
                    └────────┬─────────┘
                             │ FSQ (语义瓶颈)
                    ┌────────▼─────────┐
                    │ RALM (浅, 4层)    │
FSQ output ────────► 声学细化         │
                    │ "怎么说"          │
                    └────────┬─────────┘
                             │ 融合
                    ┌────────▼─────────┐
                    │ DiT + CFM        │
                    │ 生成连续 latent   │
                    └──────────────────┘
```

这种分层设计让每个模块专注自己的工作，比一个巨型模型更高效。

---

## 10.5 CFM：条件流匹配

### 10.5.1 为什么不是 DDPM？

如果要在连续空间生成 latent patch，自然想到用扩散模型（DDPM）。但 DDPM 有几个问题：

1. **需要很多步**：50-1000 步去噪，推理慢
2. **训练效率低**：需要在每个时间步求解 ODE
3. **噪声调度复杂**：β schedule 的选择很敏感

**CFM（Conditional Flow Matching）** 是一种更高效的替代方案：

### 10.5.2 CFM 的核心思想

CFM 的目标是学习一个**速度场（velocity field）** `v(x, t)`，把噪声"流动"到目标数据：

```
t=1 (噪声) ──────v(x,t)──────► t=0 (数据)
  z ~ N(0,I)                      x₁ ~ 目标 latent
```

**训练目标**：直接回归最优传输速度

```
L = E_{t, x₁, z} [ || v_θ(y_t, t, μ) - (z - x₁) ||² ]

其中:
  x₁ = 目标 latent patch (ground truth)
  z  = 随机噪声 ~ N(0, I)
  t  = 时间 ~ Uniform(0, 1)
  y_t = (1-t) · x₁ + t · z    ← 线性插值
  v_θ = 模型预测的速度场
  μ  = TSLM + RALM 的条件
```

直觉：`z - x₁` 是从数据到噪声的方向。模型学习在这个方向上"推动"，推理时反向推动（从噪声到数据）。

### 10.5.3 推理：Euler ODE 求解器

训练好速度场后，推理时用 **Euler 方法**积分：

```
x₀ = z                           ← 从纯噪声开始
x₁ = x₀ - dt · v(x₀, t=1, μ)    ← 第一步
x₂ = x₁ - dt · v(x₁, t=1-dt, μ) ← 第二步
...
xₙ = xₙ₋₁ - dt · v(xₙ₋₁, t=dt, μ) ← 最后一步 → 生成结果
```

**只需要 10 步**！（DDPM 通常需要 50-1000 步）

### 10.5.4 CFM vs DDPM 对比

| 特性 | CFM (VoxCPM) | DDPM |
|------|-------------|------|
| 训练目标 | 回归速度场 (MSE) | 预测噪声 (MSE) |
| 插值路径 | 线性 (optimal transport) | 非线性 (cosine/linear schedule) |
| 推理步数 | 10 步 | 50-1000 步 |
| 求解器 | Euler ODE | DDIM / ancestral sampling |
| 效率 | 高 | 低 |
| 质量 | 相当或更好 | 基线 |

### 10.5.5 时间调度

VoxCPM 原版使用 **log-normal 分布**采样训练时间步（偏向中间时刻），教学版使用简单的均匀分布。

推理时的 **sway sampling**：时间步不是均匀分布，而是使用余弦弯曲：

```
t_span = linspace(1, 0, n+1) + sway_coef · (cos(π/2 · t) - 1 + t)
```

这使得模型在开始和结束时走更小的步，中间走更大的步——因为数据流形的曲率在中间最大。

### 10.5.6 CFG-Zero*（可选进阶）

VoxCPM 使用 **CFG-Zero*** 来优化 classifier-free guidance：

```
标准 CFG:  v = v_uncond + cfg_scale · (v_cond - v_uncond)

CFG-Zero*: 自适应缩放
  s* = <v_cond, v_uncond> / ||v_uncond||²
  v = v_uncond · s* + cfg_scale · (v_cond - v_uncond · s*)
```

这防止了过曝（over-exposure），让生成的音频更自然。教学版暂不实现。

---

## 10.6 完整代码走读

### 10.6.1 文件结构

```
ch10_voxcpm/
├── README.md              # 本章教程
├── code/
│   ├── model.py           # 完整模型（AudioVAE + TSLM + RALM + CFM）
│   ├── train.py           # 训练脚本
│   ├── inference.py       # 推理脚本
│   └── export_onnx.py     # ONNX 导出
├── checkpoints/           # 模型权重
└── outputs/               # 生成音频
```

### 10.6.2 model.py 架构总览

```
SimpleVoxCPM
├── audio_vae (SimpleAudioVAE)
│   ├── encoder: 4× CausalConv1dBlock (strides [2,5,8,8])
│   ├── enc_to_latent: Conv1d → latent_dim=32
│   ├── latent_to_dec: Conv1d → decoder_dim=256
│   ├── decoder: 4× ConvTranspose1d (strides [8,8,5,2])
│   └── dec_to_out: Conv1d → 1 channel
│
├── text_emb: Embedding(256, 512)
│
├── loc_enc (SimpleLocEnc)
│   ├── in_proj: Linear(32 → 512)
│   ├── cls_token: learnable (1, 1, 512)
│   └── encoder: 2× TransformerEncoderLayer (bidirectional)
│
├── tslm (SimpleTransformer, 8 layers)
│   ├── input_proj, pos_emb
│   └── 8× _TransformerBlock (causal MHA + FFN)
│
├── fsq (SimpleFSQ)
│   ├── in_proj: Linear(512 → 128)
│   ├── tanh + round*scale + STE
│   └── out_proj: Linear(128 → 512)
│
├── ralm (SimpleTransformer, 4 layers)
│   └── 4× _TransformerBlock (causal MHA + FFN)
│
├── lm_to_dit: Linear(512 → 256)
├── res_to_dit: Linear(512 → 256)
│
├── dit (SimpleDiT)
│   ├── in_proj: Linear(32 → 256)
│   ├── cond_proj: Linear(512 → 256)
│   ├── time_proj: Linear(256 → 256)
│   ├── 4× DiTBlock (bidirectional MHA + AdaLN + ada_scale)
│   └── out_proj: Linear(256 → 32)
│
└── cfm (SimpleCFM)
    └── compute_loss / sample (Euler ODE solver)
```

### 10.6.3 关键模块详解

**CausalConv1dBlock**：因果卷积 + GroupNorm + SiLU + 残差连接

```python
# 因果填充: 只在左侧 pad (kernel_size - 1) 个位置
out = F.pad(x, (kernel_size - 1, 0))
out = conv(out)  # 不设 padding
```

**DiTBlock**：Diffusion Transformer 块，核心是 **AdaLN（自适应层归一化）**

```python
# AdaLN: 从条件向量生成 scale 和 shift，调制层归一化的输出
shift, scale = Linear(cond).chunk(2)
out = LayerNorm(x) * (1 + scale) + shift

# ada_scale: 最终的自适应缩放（DiT 的 signature 设计）
out = x + SiLU(Linear(cond)) * h
```

**SimpleCFM.sample**：Euler ODE 求解器

```python
def sample(self, cond, shape, temperature=1.0):
    x = torch.randn(shape) * temperature     # t=1: 纯噪声
    dt = 1.0 / self.n_steps
    for i in range(self.n_steps):
        t = 1.0 - i * dt
        v = self.estimator(x, t, cond)       # 预测速度场
        x = x - dt * v                       # Euler 步 (向数据移动)
    return x                                  # t≈0: 生成的 latent
```

### 10.6.4 参数量

| 组件 | 参数量 | 说明 |
|------|--------|------|
| AudioVAE | 2.1M | 因果 CNN 编码/解码器 |
| Text Embedding | 0.1M | 256 字符 × 512 维 |
| LocEnc | 6.3M | 2 层 Transformer |
| TSLM | 26.3M | 8 层 Transformer |
| FSQ | 0.1M | 两个线性投影 |
| RALM | 13.7M | 4 层 Transformer |
| DiT | 5.1M | 4 层 DiT + AdaLN |
| **总计** | **53.7M** | 原版 VoxCPM ~500M |

原版 VoxCPM 用 MiniCPM-4 作为 backbone（~350M），我们缩小到 54M 方便学习。

---

## 10.7 训练与推理

### 10.7.1 训练流程

```bash
# 快速测试（小数据集，1 epoch）
python code/train.py --epochs 1 --batch-size 2 --n-train 64 --audio-len 3200

# 完整训练
python code/train.py --epochs 50 --batch-size 4 --lr 1e-4 --audio-len 16000
```

训练过程：

```
对每个 batch:
  1. AudioVAE.encode(audio) → latents z       (冻结 VAE)
  2. LocEnc(latent_patches) → audio_emb
  3. text_emb + audio_emb (interleaved) → TSLM (causal) → semantic hidden
  4. FSQ(semantic hidden) → quantized
  5. quantized → RALM (causal) → acoustic hidden
  6. fusion(semantic, acoustic) → conditioning μ
  7. CFM.compute_loss(target_patch, μ):
       a. 采样 t ~ Uniform(0,1)
       b. 采样 z ~ N(0, I)
       c. y_t = (1-t)·target + t·z
       d. v_pred = DiT(y_t, t, μ)
       e. loss = MSE(v_pred, z - target)
  8. loss.backward() + optimizer.step()
```

### 10.7.2 推理流程

```bash
python code/inference.py --checkpoint checkpoints/voxcpm.pt --text "你好猫娘" --n-steps 25
```

推理过程：

```
1. Tokenize text → text_tokens
2. text_emb = Embedding(text_tokens)
3. audio_history = [BOS]

For step = 0, 1, 2, ..., n_steps-1:
  4. TSLM.forward(text_emb + audio_history) → semantic (取最后位置)
  5. FSQ(semantic) → quantized
  6. RALM.forward_step(quantized) → acoustic
  7. dit_cond = lm_to_dit(semantic) + res_to_dit(acoustic)
  8. CFM.sample(dit_cond):
       x = randn(patch_shape)               # 初始噪声
       For i in range(10):                  # 10 步 Euler
         v = DiT(x, t=1-i/10, dit_cond)
         x = x - (1/10) * v
       patch = x                             # 生成的 latent patch
  9. audio_history += LocEnc(patch)

10. z = concat(all patches)
11. AudioVAE.decode(z) → waveform
12. Save WAV
```

### 10.7.3 ONNX 导出

```bash
python code/export_onnx.py --checkpoint checkpoints/voxcpm.pt --output-dir onnx_models/
```

导出三个模型：
- `audio_vae_encoder.onnx`: 波形 → latent
- `audio_vae_decoder.onnx`: latent → 波形
- `tslm_ralm_step.onnx`: 一步 AR 推理

---

## 10.8 与工业界的关系

### 10.8.1 VoxCPM 在 TTS 发展史中的位置

```
2017  Tacotron      — 端到端 Mel + Vocoder
2019  FastSpeech    — 并行, 稳定
2021  VITS          — 端到端波形, 最佳音质
2023  VALL-E        — 语言模型 + 离散 Token, 零样本克隆
2024  CosyVoice     — 改进的 Token 方案 + 流式
2025  VoxCPM        — Tokenizer-Free, 连续空间, 流匹配
```

### 10.8.2 Tokenizer-Free 家族

| 模型 | 方法 | 连续表示 | 生成方法 |
|------|------|---------|---------|
| **VoxCPM** | AudioVAE + TSLM/RALM + CFM | 连续 latent patches | 条件流匹配 |
| **CosyVoice 2** | 改进的 speech token + LLM | 半连续 (token + flow) | Flow matching |
| **F5-TTS** | DiT + CFM (直接 Mel) | Mel spectrogram | 条件流匹配 |
| **NaturalSpeech 3** | 分解式 codec + diffusion | 连续 components | 扩散模型 |

趋势：**越来越多的模型放弃纯离散 Token，转向连续或混合表示**。

### 10.8.3 VoxCPM 的关键技术贡献

1. **证明 Tokenizer-Free 可行**：在大规模（1.8M 小时）上验证连续方案优于离散方案
2. **FSQ 语义瓶颈**：优雅地分离语义和声学，不需要离散 Token
3. **分层 AR 设计**：TSLM + RALM 的分工比单一模型更高效
4. **10 步 CFM**：极少的扩散步数，兼顾质量和速度

---

## 练习

### 练习 1：理解 FSQ 的量化级别（难度：★）

当 `scale=9` 时，FSQ 量化后每维有 19 个可能取值。
- 如果 `scale=4`，有多少个取值？
- 如果 `latent_dim=128, scale=9`，总共可以表示多少种不同的向量？
- 对比 EnCodec 一个 RVQ 层（1024 条目）的信息容量。

### 练习 2：AudioVAE 帧率计算（难度：★）

如果 encoder_rates 改为 `[2, 4, 4, 5]`：
- 总下采样率是多少？
- 16kHz 输入对应的 latent 帧率是多少？
- 1 秒音频产生多少个 latent 帧？

### 练习 3：CFM 插值路径（难度：★★）

CFM 使用线性插值 `y_t = (1-t)·x₁ + t·z`。
- 当 t=0 时，y_t = ？当 t=1 时，y_t = ？
- 目标速度 `v = z - x₁` 的几何意义是什么？
- 如果改为 `y_t = cos(πt/2)·x₁ + sin(πt/2)·z`，目标速度应该是什么？

### 练习 4：对比实验（难度：★★★）

修改代码实现以下对比：
1. 把 FSQ 换成 Identity（直通），观察 loss 变化。FSQ 是否真的有用？
2. 把 CFM 步数从 10 降到 5 和 2，观察生成质量（MSE with ground truth）。
3. 把 TSLM 从 8 层降到 4 层，把 RALM 从 4 层加到 8 层。哪个更重要？

---

## 参考资料

- **VoxCPM 论文**: [arXiv:2509.24650](https://arxiv.org/abs/2509.24650)
- **VoxCPM 代码**: [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- **Conditional Flow Matching**: [Lipman et al., 2022](https://arxiv.org/abs/2210.02747)
- **FSQ**: [Mentzer et al., 2023](https://arxiv.org/abs/2309.15505)
- **DiT**: [Peebles & Xie, 2022](https://arxiv.org/abs/2212.09748)
- **MiniCPM-4**: [OpenBMB/MiniCPM4](https://huggingface.co/openbmb/MiniCPM4-0.5B)
- **VALL-E** (Ch07): 离散 Token 方案的基线

---

> **Neko 的学习笔记：**
>
> "原来不一定要把声音切成积木块啊。连续空间就像是一条流动的河，声音在里面自然地流淌，不会被码本的格子卡住。FSQ 就像是在河上建了一座矮坝——水还是能流过去，但被稍微'整理'了一下。TSLM 负责决定河流的方向（语义），RALM 负责水面的波纹和涟漪（声学）。最后 CFM 就像是魔法——从一团迷雾（噪声）中，一步一步凝聚出清澈的水流（latent）。
>
> 猫娘觉得，连续空间才是声音的真正家园喵~"
