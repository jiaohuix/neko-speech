# Ch09: GPT-SoVITS -- Few-Shot 声音克隆

> 前面几章，Neko 学会了从文本生成语音 (Ch02-Ch05)，
> 还学会了把音频编码成离散 Token (Ch06)，
> 甚至用 VALL-E 的思路"听一遍就会说" (Ch07)。
>
> 但 VALL-E 需要多层 codec tokens，AR 模型要预测 8 层 × 50Hz 的序列，很慢。
> 有没有更聪明的办法？
>
> 这一章，Neko 要学 **GPT-SoVITS** —— 只用 **1 层语义 Token** 就能克隆声音。

## 本章导学

### 为什么需要 GPT-SoVITS？

回顾 VALL-E 的架构：

| 问题 | VALL-E 的做法 | 代价 |
|------|-------------|------|
| 音频如何变成 Token？ | EnCodec 多层码本 (n_q=8) | 每帧 8 个 Token |
| AR 模型预测什么？ | 逐层自回归 (AR + NAR) | 序列长度 = 50Hz × 8层 |
| 需要多少参考音频？ | 3-10 秒 | 需要 codec 预训练 |

GPT-SoVITS 的核心洞察：**不是所有 Token 层都同等重要**。

HuBERT 的第一层量化码本已经捕获了绝大部分**语义信息**（说了什么），
而**音色信息**（谁在说）可以通过参考音频直接传递给声码器。

```
VALL-E:     文本 + 参考 → AR(8层tokens) → Codec Decoder → 波形
                          很慢，序列长

GPT-SoVITS: 文本 + 参考 → AR(1层tokens) → VITS Vocoder → 波形
                          快 8 倍，质量更好
```

### GPT-SoVITS 的核心创新

```
                  ┌── AR Transformer (GPT-style): 文本 → 语义 Token
文本 ─→ GPT-SoVITS ─┤
                  └── SoVITS Vocoder (VITS-based): Token + 参考音频 → 波形
```

1. **单层语义 Token (n_q=1)**：AR 模型只需预测 1 个 Token/帧，序列短 8 倍
2. **VITS 声码器**：比 codec decoder 质量更高，端到端生成波形
3. **两阶段分离**：AR 管"说什么"，声码器管"怎么说"

### 学习路线

| 节 | 内容 | 目标 |
|---|------|------|
| 9.1 | 系统架构总览 | 理解两阶段数据流 |
| 9.2 | 语义 Token：为什么 n_q=1 就够了？ | 理解语义 vs 声学 Token |
| 9.3 | AR 模型：GPT-style Transformer | 理解注意力掩码设计 |
| 9.4 | SoVITS 声码器：VITS 变体 | 理解 TextEncoder 和 VAE+Flow |
| 9.5 | 两阶段训练 | 理解分离训练的动机 |
| 9.6 | 推理流程 | 端到端：文本 + 参考 → 波形 |
| 9.7 | 代码走读 | 完整实现解析 |
| 9.8 | 与 VALL-E 的对比 | 理解设计权衡 |
| 9.9 | ONNX 导出 | 部署优化 |

---

## 9.1 系统架构总览

### 推理时的完整数据流

```
  参考音频 (3-10s, 16kHz)
        |
        v
  [HuBERT / SSL Model]  ──→ SSL features (768-dim, 50Hz)
        |                           |
        v                           v
  [RVQ Quantizer]             (提取 prompt tokens)
  n_q=1, bins=1024                  |
        |                           |
        v                           v
  prompt_semantic_ids        ┌──────────────┐
  (e.g. [42, 187, 3, ...])   │  AR Model    │ ← 文本 Phonemes
                             │  (GPT-style) │
                             └──────────────┘
                                     |
                                     v
                             predicted_semantic_ids
                             (e.g. [42, 187, 3, 55, 201, ...])
                                     |
                                     v
                        ┌────────────────────────┐
                        │   SoVITS Vocoder        │ ← 文本 + 参考 Mel
                        │   (VITS-based)          │
                        │                         │
                        │  TextEncoder → Prior    │
                        │  Flow⁻¹ → z             │
                        │  Generator → Waveform   │
                        └────────────────────────┘
                                     |
                                     v
                               输出波形 (32kHz)
```

### 关键数据维度

| 信号 | 采样率 | 维度 | 说明 |
|------|--------|------|------|
| 文本 Phonemes | ~10-20 Hz | (T_text,) | 离散 ID, vocab=512 |
| SSL Features | 50 Hz | (768, T_ssl) | HuBERT 输出 |
| Semantic Tokens | 50 Hz | (T_audio,) | 离散 ID, vocab=1025 (含 EOS) |
| Linear Spec | 50 Hz | (1025, T_spec) | n_fft=2048 |
| Mel Spectrogram | 50 Hz | (128, T_mel) | 用于参考编码器 |
| 波形 | 32 kHz | (T_wav,) | 最终输出 |

### 训练 vs 推理

```
训练时:
  Stage 1 (AR):  phoneme_ids + semantic_ids → 预测下一个 semantic_id
  Stage 2 (SoVITS): phoneme_ids + spec + ref_mel → 重建波形 (VAE+Flow+GAN)

推理时:
  1. AR: phoneme_ids + prompt_tokens → 自回归生成 semantic_ids
  2. SoVITS: phoneme_ids + speaker_emb → 采样 + 逆变换 + 生成波形
  (PosteriorEncoder 和 Discriminator 仅在训练时使用)
```

---

## 9.2 语义 Token：为什么 n_q=1 就够了？

### 9.2.1 多层 Codec Token 的问题

Ch06 中我们学了 EnCodec 等神经音频编解码器。它们用 **多层 RVQ** 将音频压缩成多层 Token：

```
音频波形
    ↓ Encoder
连续特征 (50Hz, 128-dim)
    ↓ RVQ (n_q=8)
8 层 Token:
  Layer 0: [42, 187, 3, 55, ...]     ← 语义层（说了什么）
  Layer 1: [12, 89, 201, 7, ...]     ← 声学细节层
  Layer 2: [45, 123, 8, 99, ...]     ← 更细的细节
  ...
  Layer 7: [1, 0, 3, 2, ...]         ← 最细的残余
```

层数越深，Token 承载的信息越"声学化"（音色、共振峰细节），越不"语义化"。

### 9.2.2 HuBERT + 单层量化

GPT-SoVITS 不做完整的音频编解码。它只做一件事：

> **用 HuBERT 提取语义特征，然后用单层 VQ 量化成离散 Token。**

```
音频 (16kHz)
    ↓ HuBERT (冻结的 SSL 模型)
SSL features (768-dim, 50Hz)    ← 语义-rich 的连续表示
    ↓ RVQ (n_q=1, bins=1024)
semantic_ids (50Hz)              ← 每帧 1 个整数 Token
```

**为什么 HuBERT 特征天然"语义化"？**

HuBERT 的训练目标是 **masked prediction of phoneme clusters**。
也就是说，HuBERT 被训练来理解"说了什么"，而不是"怎么说的"。

所以 HuBERT 特征的第一层 VQ 量化就已经捕获了主要语义信息。

### 9.2.3 音色怎么办？

语义 Token 丢失了音色信息。但 GPT-SoVITS 有一个巧妙的解决方案：

```
                   语义 Token (丢失了音色)
                        |
                        v
SoVITS 声码器 ← 参考音频的 Mel Spectrogram (包含音色)
```

声码器同时接收：
1. **语义 Token** → 知道"说了什么"
2. **参考音频** → 知道"谁在说"（通过 ReferenceEncoder 提取 speaker embedding）

音色信息通过参考音频直接注入声码器，**不需要 AR 模型预测**。

### 9.2.4 n_q=1 的效率优势

| 方案 | Token 率 | AR 序列长度 (1秒) | AR 预测目标 |
|------|----------|-------------------|-------------|
| VALL-E (EnCodec) | 50Hz × 8层 = 400 tokens/s | 400 | 8 层码本 |
| **GPT-SoVITS** | **50Hz × 1层 = 50 tokens/s** | **50** | **1 层码本** |

AR 序列长度缩短 **8 倍**，推理速度快 **8 倍**。

> **Neko 笔记**：这不是"免费午餐"。单层 Token 的代价是声码器需要更"聪明"，
> 要从粗糙的语义 Token + 参考音频中恢复出高质量波形。
> 这就是为什么 GPT-SoVITS 选择 VITS 而不是简单的 codec decoder。

---

## 9.3 AR 模型：GPT-style Transformer

### 9.3.1 核心思想

AR 模型的任务极其简单：

> **给定文本 phonemes 和已有的 semantic tokens，预测下一个 semantic token。**

这和大语言模型完全一样的范式：

```
GPT:    "今天天气" → 预测 → "很好"
AR:     phonemes + [42, 187, 3] → 预测 → 55
```

### 9.3.2 架构设计

```
  AR Model Architecture
  ======================

  phoneme_ids (B, T_text)
      ↓
  Text Embedding (512 → 384)
      ↓
  + Sine Positional Encoding
      |
      |  concat
      v
  [text_emb | audio_emb]  (B, T_text + T_audio, 384)
      ↓
  8 × CausalTransformerBlock
  (causal self-attention, 8 heads, 4× FFN)
      ↓
  LayerNorm
      ↓
  取 audio 部分 (positions T_text:)
      ↓
  Linear(384, 1025, bias=False)  ← 预测层
      ↓
  logits (B, T_audio, 1025)
```

### 9.3.3 注意力掩码设计

GPT-SoVITS 的注意力掩码是其最精妙的设计之一。

在原始实现中，拼接的 [text | audio] 序列使用混合掩码：

```
  Attention Mask Layout (original):

             text    audio
  text   [ bidirectional  |   masked   ]   ← text 可以互相看
         [                |            ]      text 不能看 audio
  audio  [ all-to-text    |   causal   ]   ← audio 看所有 text
         [                |  (upper    ]      audio 只能看过去的 audio
                            triangular)
```

**为什么这样设计？**

1. **Text 双向注意力**：phonemes 之间没有因果关系，互相看能得到更好的文本表示
2. **Audio 因果注意力**：每个 semantic token 只能依赖过去的 token（自回归约束）
3. **Audio→Text 全注意力**：每个 audio token 可以看到完整的文本（知道要说什么）
4. **Text 不看 Audio**：文本表示不需要知道音频（训练时避免泄露）

> 我们的简化实现使用纯因果掩码 (causal attention for all)，
> 效果类似但实现更简单。text 部分虽然也是因果的，但因为 text 很短，
> 双向 vs 因果的差异不大。

### 9.3.4 超参数

| 参数 | 原始 GPT-SoVITS | 简化版 (本章) | 变化 |
|------|-----------------|--------------|------|
| hidden_dim | 512 | 384 | -25% |
| num_layers | 12-24 | 8 | -33% |
| num_heads | 16 | 8 | -50% |
| vocab_size | 1025 | 1025 | 不变 |
| phoneme_vocab | 512-732 | 512 | 不变 |
| BERT features | 1024-dim | 不使用 | 简化 |
| 参数量 | ~40M | ~15M | -62% |

### 9.3.5 推理：自回归生成 + Top-k 采样

```python
# 伪代码：AR 推理
def generate(phoneme_ids, prompt_tokens, max_tokens=500):
    current = prompt_tokens.clone()

    for step in range(max_tokens):
        logits = ar_model(phoneme_ids, current)   # (1, T, 1025)
        next_logits = logits[:, -1, :]             # 最后一个位置

        # Top-k 过滤
        top_values = topk(next_logits, k=5)
        next_logits[next_logits < top_values[-1]] = -inf

        # 采样
        probs = softmax(next_logits / temperature)
        next_token = multinomial(probs)

        current = concat(current, next_token)

        if next_token == EOS:  # 1024
            break

    return current[len(prompt_tokens):]  # 去掉 prompt
```

### 9.3.6 KV-Cache 加速（原始实现）

在原始 GPT-SoVITS 中，推理使用了 **KV-Cache** 优化：

```
不用 KV-Cache:
  step 1: process [text, token_0]                    → predict token_1
  step 2: process [text, token_0, token_1]            → predict token_2
  step 3: process [text, token_0, token_1, token_2]   → predict token_3
  ...
  复杂度: O(T^2) 每步，O(T^3) 总计

用 KV-Cache:
  step 0: process [text, prompt_tokens]  → 缓存 K,V
  step 1: process [token_new] + 缓存 K,V → predict, 更新缓存
  step 2: process [token_new] + 缓存 K,V → predict, 更新缓存
  ...
  复杂度: O(T) 每步，O(T^2) 总计
```

我们的简化实现不使用 KV-Cache（更易于理解），但代价是推理较慢。

---

## 9.4 SoVITS 声码器：VITS 变体

SoVITS 是一个修改版的 VITS，核心区别是：
- **没有 DurationPredictor**（AR 模型隐式处理时长）
- **语义 Token 作为输入**（而不是纯 phonemes）
- **ReferenceEncoder 提取 speaker embedding**（用于零样本克隆）

### 9.4.1 整体架构

```
  SoVITS Vocoder Architecture (Training)
  ========================================

  phoneme_ids ──→ TextEncoder ──→ (μ_p, logσ_p)  先验分布
       ↑                              |
  speaker_emb ────────────────────────┘  (speaker conditioning)
       ↑                                | KL 散度
  ref_mel ──→ ReferenceEncoder ──→ speaker_emb
                                        ↓
  spec ──→ PosteriorEncoder ──→ (μ_q, logσ_q)  后验分布
                                        |
                                  z_q = μ_q + ε·σ_q  (重参数化)
                                        |
                                  Flow(z_q) → z_p  (归一化流)
                                        |
                                  Generator(z_q) → ŷ  (HiFi-GAN)
                                        |
                                  Discriminator(y, ŷ)  (对抗训练)
```

### 9.4.2 TextEncoder（简化版，无 MRTE）

原始 GPT-SoVITS 的 TextEncoder 有三个子编码器：

```
原始:
  SSL features → SSL Encoder (3层) → content
  phonemes → Text Encoder (6层) → text
  ref_audio → Reference Encoder → speaker_emb (ge)

  MRTE: content × text (cross-attn) + ge → fused
  Final Encoder (3层) → (μ_p, logσ_p)
```

MRTE (Multi-Reference Timbre Encoder) 是原始系统最复杂的组件，
通过 cross-attention 将语义内容、文本信息和说话人特征融合在一起。

我们的简化版本将这三个子编码器合并为一个：

```
简化:
  phonemes → Embedding → + speaker_proj(speaker_emb)
                        ↓
                  6层 Transformer Encoder
                        ↓
                  Linear → (μ_p, logσ_p)
```

### 9.4.3 PosteriorEncoder（WaveNet 膨胀卷积）

```
  Linear Spectrogram (B, 1025, T)
      ↓ Conv1d(1025, 192)
      ↓ 8× WaveNet Block (dilation: 1,2,4,8,1,2,4,8)
      ↓   每层: Conv1d(192, 384) → split → tanh × sigmoid
      ↓ Conv1d(192, 384) → (μ_q, logσ_q)
      ↓ 重参数化: z_q = μ_q + ε·σ_q
  z_q (B, 192, T)
```

仅在训练时使用。推理时从先验分布采样。

### 9.4.4 Flow（归一化流）

```
  2 × (AffineCouplingLayer + Flip)

  每个 AffineCouplingLayer:
    x = [x1, x2]  (通道对半切)
    s, t = WaveNet(x1)
    z2 = (x2 - t) × exp(-s)
    z = [x1, z2]
```

训练：z_q → z_p (后验 → 先验空间)
推理：z_p → z_q (先验采样 → 解码器空间)

### 9.4.5 Generator（HiFi-GAN）

```
  z (B, 192, T)
      ↓ Conv1d(192, 256, k=7)
      ↓ 5 × [ConvTranspose1d (上采样) + 3× ResBlock1]
      |   upsample_rates: [10, 8, 2, 2, 2]
      |   总上采样: 10×8×2×2×2 = 640
      ↓ Conv1d → tanh
  waveform (B, 1, T×640)
```

### 9.4.6 ReferenceEncoder

```
  ref_mel (B, 128, T)
      ↓ 4× Conv2d (stride=2, 逐步下采样)
      ↓ Reshape (B, T', C×F)
      ↓ Linear → GRU → Linear
  speaker_emb (B, 256)
```

### 9.4.7 Discriminator（MPD）

```
  Multi-Period Discriminator
  periods = [2, 3, 5, 7, 11]

  每个子判别器:
    waveform (1D) → reshape 为 2D (period × subseq)
    → 4× Conv2d → LeakyReLU
    → Conv2d → 真/假
```

---

## 9.5 两阶段训练

### 9.5.1 为什么要分两阶段？

GPT-SoVITS 的两个模型解决完全不同的问题：

| | AR Model | SoVITS Vocoder |
|---|---------|---------------|
| **输入** | 文本 + 历史 Token | 文本 + 参考音频 |
| **输出** | 语义 Token | 波形 |
| **损失函数** | CrossEntropy | GAN (mel + KL + adv + feat) |
| **训练方式** | 标准自回归 | GAN 对抗训练 |
| **帧率** | 50 Hz | 50 Hz → 32 kHz |

混合训练不仅复杂，而且两个阶段的梯度会互相干扰。

### 9.5.2 Stage 1: AR 模型训练

```
  Stage 1 训练流程
  =================

  数据准备:
    1. HuBERT 提取所有音频的 SSL features
    2. RVQ 量化 → semantic_ids (每段音频的"语义标签")
    3. G2P 转换文本 → phoneme_ids

  训练循环:
    Input:  phoneme_ids + semantic_ids[:-1]  (teacher forcing)
    Target: semantic_ids[1:]                  (右移 1 位)

    Loss = CrossEntropy(logits, targets)

  优化器: AdamW (简化版, 原系统用 ScaledAdam)
    - betas: (0.9, 0.95)
    - warmup: 200 steps
    - cosine decay
    - gradient clipping: 1.0
```

**训练目标**：给定文本和已知的语义 Token 前缀，正确预测下一个 Token。

### 9.5.3 Stage 2: SoVITS 声码器训练

```
  Stage 2 训练流程 (GAN 训练)
  =============================

  每个 batch:
    1. Generator forward:
       ref_mel → ReferenceEncoder → speaker_emb
       phonemes + speaker_emb → TextEncoder → (μ_p, logσ_p)
       spec → PosteriorEncoder → z_q, (μ_q, logσ_q)
       Flow(z_q) → z_p  (后验→先验)
       Generator(z_q) → ŷ  (生成波形)

    2. Discriminator step:
       L_D = MSE(D(real), 1) + MSE(D(fake.detach()), 0)

    3. Generator step:
       L_G = gen_loss + feat_loss + 45×mel_loss + 1×kl_loss
       gen_loss  = MSE(D(fake), 1)
       feat_loss = Σ |f_real - f_fake|  (特征匹配)
       mel_loss  = L1(mel(fake), mel(real))
       kl_loss   = KL(q || p)

  优化器: AdamW × 2 (G 和 D 各一个)
    - betas: (0.8, 0.99)
    - lr: 1e-4
```

### 9.5.4 损失函数权重

```
L_gen = L_adv_G + L_feat + 45 × L_mel + 1 × L_KL
```

| 损失 | 权重 | 含义 |
|------|------|------|
| L_mel | 45 | Mel 频谱重建（最重要） |
| L_KL | 1 | KL 散度正则化 |
| L_adv_G | 1 | 对抗损失（提升自然度） |
| L_feat | 1 | 特征匹配（稳定训练） |

---

## 9.6 推理流程

### 9.6.1 完整推理步骤

```python
# 伪代码：GPT-SoVITS 推理

def synthesize(text, ref_audio_path):
    # 1. 预处理
    phoneme_ids = g2p(text)               # 文本 → phoneme IDs
    ref_wav = load_audio(ref_audio_path)   # 加载参考音频

    # 2. 提取参考特征
    ref_mel = mel_spectrogram(ref_wav)
    speaker_emb = ref_encoder(ref_mel)     # 说话人嵌入

    # 3. 提取 prompt tokens
    ssl_feat = hubert(ref_wav)             # HuBERT 特征
    prompt_tokens = rvq.encode(ssl_feat)   # 量化为 semantic IDs

    # 4. AR 生成
    generated_tokens = ar_model.generate(
        phoneme_ids, prompt_tokens,
        top_k=5, temperature=1.0,
    )

    # 5. SoVITS 声码器
    all_tokens = concat(prompt_tokens, generated_tokens)
    m_p, logs_p = text_encoder(phoneme_ids, speaker_emb)
    z_p = m_p + randn × exp(logs_p) × 0.667   # 从先验采样
    z = flow.inverse(z_p)                       # 逆变换
    waveform = generator(z)                     # 生成波形

    return waveform
```

### 9.6.2 关键推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| top_k | 5 | AR 采样宽度。越小越确定，越大越多样 |
| temperature | 1.0 | 采样温度。>1 更随机，<1 更保守 |
| noise_scale | 0.667 | VAE 采样噪声。控制生成波形的变化度 |
| max_tokens | 500 | 最大生成 Token 数 (50Hz × 10s) |

### 9.6.3 RTF (Real-Time Factor)

```
RTF = 生成时间 / 音频时长

RTF < 1:  比实时快（理想目标）
RTF = 1:  实时
RTF > 1:  比实时慢
```

GPT-SoVITS 的 RTF 主要取决于 AR 模型（自回归是瓶颈）：

| 组件 | 时间占比 | 优化方向 |
|------|----------|----------|
| AR 自回归 | ~80% | KV-Cache, 并行解码 |
| SoVITS 声码器 | ~20% | 单次前向传播，已经很快 |

---

## 9.7 代码走读

### 9.7.1 文件结构

```
ch09_gpt_sovits/
├── README.md              # 本章教程（你在这里）
├── code/
│   ├── model.py           # 所有模型组件 + 损失函数
│   ├── train.py           # 两阶段训练脚本
│   ├── inference.py       # 端到端推理
│   └── export_onnx.py     # ONNX 导出
├── checkpoints/           # 训练保存的模型权重
├── outputs/               # 生成的音频
└── onnx_models/           # 导出的 ONNX 模型
```

### 9.7.2 model.py 核心类

```python
# --- AR 模型 ---
class SimpleAR:
    """384-dim, 8-layer, 8-head Transformer decoder"""
    def forward(phoneme_ids, semantic_ids) -> logits
    def generate(phoneme_ids, prompt, top_k, temperature) -> tokens

# --- RVQ 量化器 ---
class SimpleRVQ:
    """单码本量化器 (n_q=1, bins=1024, dim=768)"""
    def encode(ssl_features) -> token_ids
    def decode(token_ids) -> quantised_features

# --- SoVITS 声码器组件 ---
class SoVITSTextEncoder:       # 文本 → (μ_p, logσ_p)
class PosteriorEncoder:        # 频谱 → z_q (仅训练)
class Flow:                    # 可逆变换 z_q ↔ z_p
class Generator:               # z → 波形 (HiFi-GAN)
class ReferenceEncoder:        # 参考 Mel → speaker_emb
class Discriminator:           # MPD 判别器 (仅训练)

# --- 完整模型 ---
class GPTSoVITS:
    """包装器: AR + RVQ + SoVITS"""
    def forward_stage1(phoneme_ids, semantic_ids) -> logits
    def forward_stage2(phoneme_ids, spec, ref_mel) -> waveform, stats
```

### 9.7.3 训练脚本

```bash
# Stage 1: AR 模型
python train.py --stage 1 --epochs 10 --batch-size 4 --lr 1e-4

# Stage 2: SoVITS 声码器
python train.py --stage 2 --epochs 10 --batch-size 4 --lr 1e-4
```

### 9.7.4 推理脚本

```bash
# 基本推理
python inference.py --text "hello world" --output output.wav

# 使用训练好的模型 + 参考音频
python inference.py \
    --ar-checkpoint ../checkpoints/ar_model.pt \
    --sovits-checkpoint ../checkpoints/sovits_model.pt \
    --ref-audio reference.wav \
    --text "你好世界" \
    --output clone.wav

# 性能基准测试
python inference.py --text "hello world" --output output.wav --benchmark
```

### 9.7.5 参数量统计

运行 `python model.py` 查看：

```
SimpleAR:           15.2M
SimpleRVQ:           0.8M  (frozen)
SoVITSTextEncoder:   2.9M
PosteriorEncoder:    3.2M
Flow:                1.3M
Generator:           3.8M
ReferenceEncoder:    0.3M
Discriminator:       4.2M  (仅训练)
─────────────────────────
Total:              31.6M
Trainable:          30.8M
```

---

## 9.8 与 VALL-E 的对比

### 9.8.1 架构对比

| 特性 | VALL-E (Ch07) | GPT-SoVITS (本章) |
|------|---------------|-------------------|
| 音频 Token 来源 | EnCodec (训练好的 codec) | HuBERT + RVQ (SSL 特征) |
| Token 层数 | 8 层 | 1 层 |
| Token 率 | 400 tokens/s | 50 tokens/s |
| AR 模型 | 预测 Layer-0 | 预测唯一层 |
| 第二模型 | NAR (并行预测 Layer 1-7) | SoVITS 声码器 (VITS) |
| 最终波形生成 | Codec Decoder | HiFi-GAN Generator |
| 训练数据需求 | 60,000+ 小时 | 3-10 分钟 fine-tune |
| 推理速度 | 较慢 (8× 序列) | 较快 (1× 序列) |

### 9.8.2 设计哲学对比

```
VALL-E 哲学:
  "让一个强大的 codec 处理一切，AR 模型学习 codec 的语言"
  → 通用但慢，codec 是黑箱

GPT-SoVITS 哲学:
  "用 SSL 特征捕获语义，用 VITS 声码器恢复波形"
  → 更快，声码器能利用参考音频的音色信息
```

### 9.8.3 质量对比

在零样本场景下（3 秒参考音频）：

| 指标 | VALL-E | GPT-SoVITS | 说明 |
|------|--------|------------|------|
| MOS (自然度) | ~3.5-4.0 | ~4.0-4.5 | GPT-SoVITS 略优 |
| 相似度 | ~70-80% | ~80-90% | VITS 声码器音色还原更好 |
| 推理速度 | ~2-5x RTF | ~1-3x RTF | 序列短 8 倍 |
| 训练成本 | 极高 (60k小时) | 低 (预训练+微调) | 不同的训练范式 |

> **Neko 笔记**：GPT-SoVITS 在实际使用中更受欢迎，
> 因为它可以用很少的数据（几分钟）微调出高质量的声音克隆。
> 而 VALL-E 需要海量数据预训练才能实现零样本。

---

## 9.9 ONNX 导出

### 9.9.1 导出策略

GPT-SoVITS 将系统拆分为多个 ONNX 模型用于部署：

```
  ONNX Export Architecture
  =========================

  1. AR Model: ar_model.onnx
     Inputs:  phoneme_ids, semantic_ids
     Output:  logits
     Opset:   16

  2. SoVITS Vocoder: sovits_model.onnx
     Inputs:  phoneme_ids, speaker_emb
     Output:  waveform
     Opset:   17
     (包含 TextEncoder + Flow⁻¹ + Generator)
     (不含 PosteriorEncoder/Discriminator -- 仅训练需要)
```

### 9.9.2 动态轴

所有 ONNX 模型支持变长输入：

```python
dynamic_axes = {
    'phoneme_ids':  {1: 'text_len'},   # 文本长度可变
    'semantic_ids': {1: 'audio_len'},  # 音频长度可变
    'waveform':     {2: 'wav_len'},    # 输出波形长度可变
}
```

### 9.9.3 导出与测试

```bash
# 导出 ONNX
python export_onnx.py --output-dir ../onnx_models

# 带性能基准测试
python export_onnx.py --output-dir ../onnx_models --benchmark
```

### 9.9.4 ONNX Runtime 推理

```python
import onnxruntime as ort

# 加载 ONNX 模型
ar_session = ort.InferenceSession('ar_model.onnx')
vits_session = ort.InferenceSession('sovits_model.onnx')

# AR 推理 (单次前向传播)
logits = ar_session.run(None, {
    'phoneme_ids': phoneme_ids,
    'semantic_ids': current_tokens,
})

# 声码器推理
waveform = vits_session.run(None, {
    'phoneme_ids': phoneme_ids,
    'speaker_emb': speaker_emb,
})
```

ONNX Runtime 通常比 PyTorch 快 **1.5-2x**（通过算子融合和硬件优化）。

---

## 9.10 本章小结

### GPT-SoVITS 的核心贡献

| 问题 | GPT-SoVITS 的解决方案 |
|------|----------------------|
| 多层 Token 导致 AR 慢 | 单层语义 Token (n_q=1) |
| Codec decoder 质量有限 | VITS 声码器 (VAE+Flow+GAN) |
| 需要海量预训练数据 | 预训练 + 少样本微调 (3-10分钟) |
| 音色和语义耦合 | 语义 Token + 参考音频分离 |
| 推理延迟高 | 序列短 8 倍 + KV-Cache |

### 遗留问题

1. **自回归瓶颈**：即使只有 1 层 Token，AR 生成仍然是串行的 → **非自回归方案**
2. **HuBERT 依赖**：需要预训练的 HuBERT 模型 (~95M) → **轻量 SSL 模型**
3. **长文本质量下降**：AR 生成越长，累积误差越大 → **分段合成 + 拼接**
4. **情感控制有限**：主要靠参考音频传递情感 → **情感条件生成 (Ch08)**

### 参考文献

- [1] Kim et al., 2021. *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech* (VITS).
- [2] Wang et al., 2023. *Neural Codec Language Models for Zero-Shot TTS* (VALL-E).
- [3] Hsu et al., 2021. *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*.
- [4] Defossez et al., 2022. *High Fidelity Neural Audio Compression* (EnCodec).
- [5] Kong et al., 2020. *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis*.
- [6] GPT-SoVITS open-source project. https://github.com/RVC-Boss/GPT-SoVITS

---

## 习题

1. **n_q=1 的信息论分析**：如果 EnCodec 用 n_q=8 层码本 (每层 1024 entries)，总信息容量是 $1024^8$。GPT-SoVITS 只用 n_q=1 (1024 entries)，信息容量只有 1024。这意味着什么？GPT-SoVITS 靠什么弥补信息损失？

2. **注意力掩码设计**：在 AR 模型中，为什么 text 部分使用双向注意力而 audio 部分使用因果注意力？如果全部使用因果注意力会怎样？如果全部使用双向注意力会怎样？

3. **KL 散度的角色**：Stage 2 训练中，KL loss 的权重是 1.0，而 mel loss 的权重是 45.0。如果去掉 KL loss，会发生什么？如果把 KL loss 权重提高到 45，又会怎样？

4. **Flow 的必要性**：如果去掉 Flow（直接从先验采样送给 Generator），生成质量会如何变化？Flow 解决了什么根本问题？

5. **对比实验**：用相同文本和参考音频，分别用 Ch07 (VALL-E) 和 Ch09 (GPT-SoVITS) 合成。对比以下维度：
   - 推理速度 (RTF)
   - 音色相似度
   - 自然度 (MOS)
   - 生成 Token 序列长度

---

## 目录结构

```
ch09_gpt_sovits/
├── README.md              # 本章教程（你在这里）
├── code/
│   ├── model.py           # 所有模型组件 (~400行)
│   ├── train.py           # 两阶段训练脚本 (~300行)
│   ├── inference.py       # 端到端推理 (~200行)
│   └── export_onnx.py     # ONNX 导出 (~150行)
├── checkpoints/           # 训练保存的模型权重
├── outputs/               # 生成的音频
└── onnx_models/           # 导出的 ONNX 模型
```
