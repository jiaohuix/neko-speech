# Ch05: VITS — 端到端语音合成

> 前面几章我们学了"文本 → Mel → 波形"的两阶段方法。
>
> 这一章，Neko 要学一个更优雅的方案：**直接从文本生成波形**。

## 本章导学

### 为什么需要 VITS？

Ch02 的 Tacotron2 实现了 "文本 → Mel Spectrogram"，但要得到波形，还需要一个独立的声码器（如 Griffin-Lim 或 WaveNet）。这条两阶段路线有几个问题：

| 问题 | 影响 |
|------|------|
| **级联误差** | Mel 预测的小误差会在声码器中被放大 |
| **训练割裂** | 声码器和声学模型分开训练，无法联合优化 |
| **Mel 信息损失** | Mel 频谱丢弃了相位信息，声码器只能猜测 |

VITS (Kim et al., 2021) 的解决方案：**端到端** — 文本直接出波形，无需 Mel 中间表示。

### VITS 的核心创新

VITS 把三个强大的工具组合在一起：

```
                    ┌── VAE (变分自编码器): 学习隐空间
文本 ─→ VITS ──→ 波形 ├── Flow (归一化流): 增强表达力
                    └── GAN (对抗训练): 提高波形质量
```

1. **VAE**：编码器把文本/波形编码到隐空间，解码器从隐空间生成波形
2. **Flow**：可逆变换，让简单的高斯先验能表达复杂的后验分布
3. **GAN**：判别器迫使生成器产出更逼真的波形

### 学习路线

| 节 | 内容 | 目标 |
|---|------|------|
| 5.1 | VITS 架构总览 | 理解训练和推理的数据流 |
| 5.2 | VAE 原理 | 理解变分推断和 ELBO |
| 5.3 | 归一化流 | 理解可逆变换如何提高表达力 |
| 5.4 | GAN 对抗训练 | 理解判别器如何提升质量 |
| 5.5 | 核心模块 | TextEncoder, PosteriorEncoder, Flow, Generator |
| 5.6 | Monotonic Alignment Search | 理解无监督对齐 |
| 5.7 | 训练 | 多损失函数的平衡 |
| 5.8 | 推理 | 端到端文本到波形 |
| 5.9 | 端到端 vs 两阶段 | 定量对比 |

---

## 5.1 VITS 架构总览

### 训练时的数据流

```
Text ──→ TextEncoder ──→ (μ_p, σ_p) 先验分布
                              │
                              │ KL 散度
                              ↓
Wav ──→ Spectrogram ──→ PosteriorEncoder ──→ (μ_q, σ_q) 后验分布
                              │
                              ↓ 采样 z_q (重参数化)
                              │
                         Flow (可逆变换)
                              │
                              ↓ z_p (匹配先验)
                              │
                        Generator (HiFi-GAN)
                              │
                              ↓ 生成波形
                              │
                        Discriminator (对抗训练)
                              │
                              ↓ 对抗损失 + 特征匹配
```

### 推理时的数据流

```
Text ──→ TextEncoder ──→ (μ_p, σ_p)
                              │
                              ↓ 采样 z_p ~ N(μ_p, σ_p²)
                              │
                         Flow⁻¹ (逆变换)
                              │
                              ↓ z_q
                              │
                        Generator (HiFi-GAN)
                              │
                              ↓ Waveform!
```

关键区别：**推理时不需要 PosteriorEncoder 和 Flow 的前向变换**。
Flow 的逆变换用于把先验采样映射到解码器能理解的隐空间。

---

## 5.2 VAE：变分自编码器

### 5.2.1 直觉

VAE 的核心思想：**学习一个压缩的隐空间，能捕捉语音的本质特征**。

想象把一段语音压缩成几个数字（隐变量 z），再从这几个数字还原出整段语音。
如果压缩得好，z 就包含了"说了什么"的所有信息。

```
编码器: Wav → z (压缩)
解码器: z → Wav (解压)
```

### 5.2.2 变分推断

普通 AutoEncoder 的问题是：隐空间 z 可以是任何形状，难以采样。

VAE 的解决：**强制 z 服从高斯分布**。

编码器不再输出单个 z，而是输出分布参数：

$$q_\phi(z|x) = \mathcal{N}(z; \mu_q, \sigma_q^2)$$

- $\mu_q$：后验均值（波形的"中心"表示）
- $\sigma_q$：后验标准差（表示的不确定性）

### 5.2.3 重参数化技巧

问题：从分布中采样是随机操作，无法反向传播。

解决：**重参数化**

$$z = \mu_q + \sigma_q \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

把随机性转移到 $\epsilon$ 上，$\mu_q$ 和 $\sigma_q$ 是可微的。

### 5.2.4 KL 散度

VAE 要求后验分布 $q(z|x)$ 接近先验分布 $p(z)$：

$$\text{KL}(q \| p) = \mathbb{E}_q\left[\log \frac{q(z|x)}{p(z)}\right]$$

对两个对角高斯分布，KL 有解析解：

$$\text{KL} = \sum_i \left[\log\sigma_{p,i} - \log\sigma_{q,i} + \frac{\sigma_{q,i}^2 + (\mu_{q,i} - \mu_{p,i})^2}{2\sigma_{p,i}^2} - \frac{1}{2}\right]$$

直觉：KL 惩罚后验偏离先验。如果先验是 $\mathcal{N}(0, I)$，KL 鼓励后验的均值接近 0、方差接近 1。

### 5.2.5 ELBO

VAE 的目标函数是最大化 ELBO (Evidence Lower Bound)：

$$\text{ELBO} = \mathbb{E}_q[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))$$

- 第一项：**重建损失** — 从 z 还原的 x 应该接近原始 x
- 第二项：**KL 正则** — z 的分布应该接近先验

> **Neko 笔记**：ELBO 是一个下界——真实的对数似然 ≥ ELBO。最大化 ELBO 就是在逼近真实似然。

---

## 5.3 归一化流 (Normalizing Flow)

### 5.3.1 问题

VAE 假设先验 $p(z)$ 是简单高斯。但真实语音的隐空间可能非常复杂——多峰、弯曲、不规则。

简单高斯无法表达这种复杂性 → 生成质量受限。

### 5.3.2 解决思路

归一化流：**通过一系列可逆变换，把简单分布变成复杂分布**。

```
简单高斯 z₀
    ↓ f₁ (可逆)
z₁
    ↓ f₂ (可逆)
z₂
    ↓ ...
    ↓ f_K (可逆)
z_K (复杂分布)
```

每个 $f_i$ 都是可逆的，所以：
- 正向（训练）：$z_0 \to z_K$，把后验变换到先验空间
- 逆向（推理）：$z_K \to z_0$，从先验采样得到解码器输入

### 5.3.3 仿射耦合层

VITS 使用 **仿射耦合层** (Affine Coupling Layer)：

```
输入 x = [x₁, x₂]  (按通道分成两半)

x₁ 保持不变
z₂ = (x₂ - t(x₁)) × exp(-s(x₁))    ← 由 x₁ 决定如何变换 x₂

输出 z = [x₁, z₂]
```

其中 $s(x_1)$ 和 $t(x_1)$ 是神经网络输出的 scale 和 shift。

**为什么这很巧妙？**

1. **可逆**：逆变换只需 $x_2 = z_2 \times \exp(s(x_1)) + t(x_1)$
2. **雅可比行列式高效**：三角矩阵 → 行列式 = 对角元素之积
3. **交替反转**：堆叠多层时交替反转通道，让所有维度都被变换

### 5.3.4 对数行列式

Flow 变换会改变概率密度。为了正确计算 KL 散度，需要修正：

$$\log p(z_K) = \log p(z_0) + \sum_{i=1}^K \log \left|\det \frac{\partial f_i}{\partial z_{i-1}}\right|$$

仿射耦合层的行列式：$\log|\det J| = \sum s(x_1)$

### 5.3.5 VITS 中 Flow 的作用

```
训练: PosteriorEncoder → z_q → Flow → z_p (应该匹配先验)
推理: 先验采样 z_p → Flow⁻¹ → z_q → Generator → 波形
```

Flow 让简单的高斯先验能够表达复杂的后验分布，提升生成质量。

---

## 5.4 GAN 对抗训练

### 5.4.1 为什么需要 GAN？

VAE 的重建损失（L1/L2）倾向于产出**平均化**的结果：
- 波形在高频段被平滑
- 听起来"闷"、缺乏清晰度

GAN 的判别器提供另一种信号：**不比较像素级的差异，而是判断"听起来像不像真的"**。

### 5.4.2 多周期判别器 (MPD)

VITS 使用 HiFi-GAN 的多周期判别器：

```
波形 (1D) ──→ 重塑为 2D (period × subsequence) ──→ Conv2d ──→ 真/假
```

使用质数周期 [2, 3, 5, 7, 11]：
- period=2：捕获半周期模式
- period=3：捕获三周期模式
- 不同周期看到不同尺度的结构

为什么用质数？避免周期性重复，让每个判别器关注不同的频率范围。

### 5.4.3 对抗损失

使用 Least Squares GAN：

**判别器损失**：
$$L_D = \mathbb{E}[(D(x) - 1)^2] + \mathbb{E}[D(G(z))^2]$$

**生成器损失**：
$$L_G = \mathbb{E}[(D(G(z)) - 1)^2]$$

### 5.4.4 特征匹配

仅靠对抗损失训练不稳定。VITS 还加入**特征匹配损失**：

$$L_{feat} = \sum_l \| f_l^{real} - f_l^{fake} \|_1$$

要求生成样本在判别器中间层的特征接近真实样本。这比纯对抗损失更稳定。

---

## 5.5 核心模块详解

### 5.5.1 TextEncoder

```python
class TextEncoder:
    """文本 → 先验分布参数 (μ_p, log σ_p)"""

    架构:
    Token Embedding
        ↓
    相对位置编码 + N × Transformer Encoder Block
        ↓
    线性投影 → (μ_p, log σ_p)
```

设计选择：
- **Transformer 而非 BiLSTM**：并行计算，训练更快
- **相对位置编码**：对不同长度文本泛化更好
- **输出高斯参数**：为 VAE 提供先验分布

### 5.5.2 PosteriorEncoder

```python
class PosteriorEncoder:
    """线性频谱 → 后验分布参数 (μ_q, log σ_q) + 采样 z_q"""

    架构:
    Linear Spectrogram (B, 513, T_spec)
        ↓
    Conv1d 投影
        ↓
    N × WaveNet 门控卷积块 (膨胀卷积)
        ↓
    Conv1d → (μ_q, log σ_q)
        ↓
    重参数化采样 → z_q
```

关键点：
- **仅在训练时使用**，推理时不需要
- **输入线性频谱**（非 Mel），保留更多频率细节
- **膨胀卷积**：dilation 按 2^k 递增，感受野指数增长
- **跳跃连接**：汇总所有层输出，避免信息丢失

### 5.5.3 Flow

```python
class Flow:
    """归一化流：可逆变换 z_q ↔ z_p"""

    架构:
    K × (AffineCouplingLayer + Flip)
```

每个 AffineCouplingLayer：
1. 按通道分成两半 (x₁, x₂)
2. 神经网络预测 scale s 和 shift t
3. z₂ = (x₂ - t) × exp(-s)
4. 最后一层初始化为 0 → 初始变换接近恒等映射

### 5.5.4 Generator (HiFi-GAN)

```python
class Generator:
    """z → 波形 (HiFi-GAN Decoder)"""

    架构:
    z (B, 192, T_z)
        ↓  Conv1d 预处理
        ↓  4 × [ConvTranspose1d(上采样) + ResBlock(MRF)]
        ↓  Conv1d → tanh
    waveform (B, 1, T_wav)
```

上采样倍率：[8, 8, 2, 2] → 总倍率 256

每 1 帧隐变量 → 256 个音频采样点 (16kHz / 256 = 62.5 Hz 帧率)

### 5.5.5 Discriminator (MPD)

```python
class Discriminator:
    """多周期判别器"""

    5 × PeriodDiscriminator(period ∈ [2, 3, 5, 7, 11])
```

### 5.5.6 DurationPredictor

```python
class DurationPredictor:
    """预测每个文本 token 的帧数"""

    3 × Conv1d + LayerNorm + ReLU → 投影 → log(duration)
```

推理时用于把文本编码序列展开到音频时间轴。

---

## 5.6 Monotonic Alignment Search (MAS)

### 5.6.1 问题

训练时，文本有 T_text 个 token，频谱有 T_spec 帧。我们不知道哪个 token 对应哪些帧。

### 5.6.2 解决

MAS 是一种动态规划算法，找到**使总概率最大的单调对齐**：

```
文本:   [你]  [好]  [猫]  [娘]
         ↓     ↓     ↓     ↓
频谱: [1 2 3 4 5 6 7 8 9 10]

最优对齐: [你]→[1,2,3], [好]→[4,5], [猫]→[6,7,8], [娘]→[9,10]
```

约束：
1. **单调性**：对齐不能倒退
2. **完整性**：每个 token 至少对应 1 帧
3. **最优性**：总 log 似然最大

### 5.6.3 动态规划

```
dp[i][j] = 对齐到文本第 i 个 token、频谱第 j 帧的最大概率

转移:
dp[i][j] = max(dp[i-1][j-1], dp[i][j-1]) + log_p[i][j]
           （消耗一个 token）  （不消耗）

回溯路径得到硬对齐矩阵 (每列只有一个 1)
```

从对齐矩阵可以提取每个 token 的时长：`duration[i] = sum(attn[i, :])`

---

## 5.7 训练

### 5.7.1 多损失函数

VITS 的总损失是多个损失的加权和：

```
L_gen = c_mel × L_mel + c_kl × L_KL + c_dur × L_dur + c_adv × L_adv_G + c_feat × L_feat
L_disc = L_adv_D
```

| 损失 | 含义 | 默认权重 |
|------|------|----------|
| L_mel | Mel 频谱重建 (L1) | 45 |
| L_KL | KL 散度 (后验→先验) | 1 |
| L_dur | 时长预测 MSE | 1 |
| L_adv_G | 生成器对抗 | 1 |
| L_feat | 特征匹配 | 2 |

### 5.7.2 训练流程

```
每个 batch:
1. Generator forward:
   - TextEncoder → 先验参数
   - PosteriorEncoder → 后验参数 + z_q
   - Flow(z_q) → z_p
   - MAS → 对齐 + 时长
   - Generator(z_q) → 波形

2. Discriminator:
   - 判断真假波形

3. Generator backward:
   - L_gen = L_mel + L_KL + L_dur + L_adv_G + L_feat

4. Discriminator backward:
   - L_disc = L_adv_D
```

### 5.7.3 启动训练

```bash
cd chapters/ch05_vits/code

# 下载数据（如果还没有）
cd ../../../data
python download_neko_1k.py --output-dir processed --num-samples 1000

# 训练
cd ../chapters/ch05_vits/code
python train.py \
    --data-dir ../../../data/processed \
    --epochs 50 \
    --batch-size 2 \
    --max-spec-len 400 \
    --max-wav-sec 10
```

**显存需求**：batch_size=2 + max_spec_len=400 约需 12GB 显存。
如果显存不足，减小 `--max-spec-len` 和 `--batch-size`。

**预期训练曲线**：
- Epoch 1-5：loss_g 从 ~500 快速下降到 ~200
- Epoch 5-20：loss_mel 稳步下降，KL 趋于稳定
- Epoch 20+：对抗损失开始起作用，波形质量提升

---

## 5.8 推理

### 5.8.1 端到端推理

```bash
# 使用训练好的模型
python inference.py \
    --checkpoint ../checkpoints/vits_final.pt \
    --text "你好，我是猫娘。" \
    --output ../outputs/vits_neko.wav

# 调整语速（1.2 = 慢 20%）
python inference.py \
    --checkpoint ../checkpoints/vits_final.pt \
    --text "你好，我是猫娘。" \
    --length-scale 1.2 \
    --output ../outputs/vits_slow.wav

# 快速测试（随机模型，输出噪声）
python inference.py \
    --text "测试" \
    --output ../outputs/vits_test.wav
```

### 5.8.2 推理参数

- `noise_scale`：采样噪声缩放。越小 → 越确定性（但缺乏变化）。推荐 0.667。
- `length_scale`：语速控制。1.0 = 正常，>1.0 = 慢，<1.0 = 快。

### 5.8.3 推理速度

VITS 是非自回归的（不像 Tacotron2 逐帧生成），所以推理很快：

| 模型 | 生成方式 | 典型 RTF |
|------|----------|----------|
| Tacotron2 + WaveNet | 自回归 | 0.01-0.1x |
| FastSpeech2 + HiFi-GAN | 非自回归（两阶段）| 0.001-0.01x |
| **VITS** | **非自回归（端到端）** | **0.001-0.01x** |

---

## 5.9 端到端 vs 两阶段对比

### 5.9.1 定量对比

| 特性 | Tacotron2 (两阶段) | VITS (端到端) |
|------|-------------------|---------------|
| 输出 | Mel Spectrogram | Waveform |
| 声码器 | 需要独立声码器 | 内置 HiFi-GAN |
| 训练方式 | 分阶段训练 | 联合训练 |
| 推理步骤 | Text→Mel + Mel→Wav | Text→Wav |
| 自回归 | 是（慢） | 否（快） |
| 信息损失 | Mel 丢弃相位 | 无 |
| 训练复杂度 | 低 | 高 |

### 5.9.2 质量对比

端到端方法的优势：
1. **无级联误差**：不会在 Mel→Wav 阶段引入额外失真
2. **联合优化**：所有组件共同优化同一目标
3. **相位恢复**：Generator 直接输出波形，无需猜测相位

### 5.9.3 代价

端到端的代价：
1. **训练更复杂**：需要平衡多个损失函数
2. **显存需求大**：同时加载 Generator + Discriminator
3. **调试困难**：出问题时难以定位是哪个组件的问题

---

## 5.10 本章小结

### VITS 的核心贡献

| 问题 | VITS 的解决方案 |
|------|----------------|
| 两阶段级联误差 | 端到端：Text → Waveform |
| 隐空间过于简单 | 归一化流增强表达力 |
| VAE 波形质量差 | GAN 对抗训练提高质量 |
| 不知道对齐 | MAS 无监督对齐 |
| 说话速度固定 | SDP 随机时长预测 |

### 遗留问题

1. **音色固定**：VITS 只能合成训练时的音色，无法克隆 → **GPT-SoVITS (Ch06)**
2. **需要大量数据**：单说话人 VITS 需要数小时数据 → **Few-shot TTS (Ch07)**
3. **可控性有限**：难以精确控制音调、情感 → **Controllable TTS (Ch08)**

### 参考文献

- [1] Kim et al., 2021. *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech*.
- [2] Dinh et al., 2017. *Density estimation using Real-NVP*.
- [3] Kong et al., 2020. *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis*.
- [4] Kingma & Welling, 2014. *Auto-Encoding Variational Bayes*.

---

## 习题

1. **VAE vs AE**：普通 AutoEncoder 和 VAE 的编码器有什么区别？为什么 VAE 可以采样新数据而 AE 不行？

2. **Flow 的可逆性**：仿射耦合层为什么是可逆的？如果去掉 scale（只用 shift），它还是可逆的吗？行列式是什么？

3. **MAS 的必要性**：如果没有 MAS，直接用 DurationPredictor 的预测时长来训练，会出什么问题？

4. **GAN 训练稳定性**：如果判别器太强（loss_adv_d 趋近 0），生成器会怎样？实践中如何解决？

5. **端到端的代价**：VITS 相比 Tacotron2，训练时需要多加载哪些模块？显存占用大约是多少？

6. **对比实验**：用同样的文本分别用 Ch02 (Tacotron2) 和 Ch05 (VITS) 合成，对比音质和速度。

---

## 目录结构

```
ch05_vits/
├── README.md              # 本章教程（你在这里）
├── code/
│   ├── modules.py         # 子模块（TextEncoder, PosteriorEncoder, Flow, Generator, Discriminator）
│   ├── model.py           # VITS 主架构 + 损失函数 + MAS
│   ├── train.py           # 训练脚本（多损失函数，G/D 交替训练）
│   └── inference.py       # 端到端推理（Text → Waveform）
├── checkpoints/           # 模型保存
└── outputs/               # 生成音频
```
