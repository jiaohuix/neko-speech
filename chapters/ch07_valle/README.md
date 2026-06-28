# Ch07: VALL-E — Neko 学会了克隆声音

> 前六章，Neko 从理解声音到学会说话。但她有一个遗憾：
> 她只能用自己训练时的音色。如果有人给她一段陌生的声音，她无法模仿。
>
> 这一章，我们用 VALL-E 的思路，让 Neko 学会"听一遍就会说"。

## 本章导学

### 为什么需要 VALL-E？

回顾前面的模型：

| 模型 | 能力 | 限制 |
|------|------|------|
| **Tacotron2** (Ch02) | 端到端文本→Mel | 单音色，需要大量配对数据 |
| **FastSpeech2** (Ch04) | 并行生成，稳定对齐 | 仍然单音色 |
| **VITS** (Ch05) | 端到端，音质最好 | 单音色，或需要 speaker embedding |
| **GPT-SoVITS** (Ch06) | Few-shot 音色克隆 | 需要 finetune 或 speaker adapter |

所有这些模型都有一个共同的限制：**音色是训练时固定的**。要换一个人的声音，需要重新训练或微调。

VALL-E 的核心突破：**把 TTS 变成语言建模问题**。只要有 3 秒参考音频，就能用那个人的声音说话，无需任何微调。

这就是**零样本声音克隆 (Zero-Shot Voice Cloning)**。

### 核心直觉：音频 Token 就是一种"语言"

回想 GPT 是怎么工作的：

```
"今天天气" → GPT → "很好"
```

GPT 把文本当作 Token 序列，用 Transformer 预测下一个 Token。

VALL-E 的核心洞察：**音频也可以被编码为离散 Token 序列**。

```
文本 Token:    [你, 好, 世, 界]
              +
参考音频 Token: [♪, ♪, ♪, ...]     ← 3秒参考音频的 codec tokens
              ↓
         VALL-E (AR Transformer)
              ↓
生成音频 Token: [♫, ♫, ♫, ♫, ...]  ← 用参考音频的声音说"你好世界"
              ↓
         Codec Decoder
              ↓
           波形 (wav)
```

一旦音频变成了 Token，TTS 就变成了和 GPT 完全相同的问题：

> **给定前面的 Token，预测下一个 Token。**

### 学习路线

| 节 | 内容 | 目标 |
|---|------|------|
| 7.1 | 音频 Codec：从连续到离散 | 理解 EnCodec/VQ-VAE 如何将音频压缩为 Token |
| 7.2 | AR 模型：预测第一个 Token 层级 | 理解"把 TTS 当语言建模"的核心思想 |
| 7.3 | NAR 模型：并行填充细节 | 理解双阶段生成的设计动机 |
| 7.4 | 零样本声音克隆 | 理解为什么 prompt tokens 能传递音色信息 |
| 7.5 | 从零实现 VALL-E | 完整代码走读 |
| 7.6 | 训练与推理 | 跑通完整流程 |
| 7.7 | 与工业界的关系 | CosyVoice、FishSpeech 等现代模型 |

---

## 7.1 音频 Codec：从连续到离散

### 7.1.1 问题：音频是连续的，语言模型要离散的

GPT 能预测文本 Token，因为文本天然是离散的（每个字/词对应一个整数 ID）。

但音频是**连续的**：一秒 16000 个采样点，每个点是浮点数。直接预测浮点数序列，GPT 无能为力。

解决方案：**用 Neural Codec 把连续音频压缩成离散 Token 序列**。

### 7.1.2 EnCodec 的核心思想

EnCodec (Defossez et al., 2022) 是 Meta 提出的神经音频压缩模型：

```
音频波形 (16kHz, 1秒 = 16000 个采样点)
    ↓ Encoder (卷积下采样)
连续向量序列 (50Hz, 1秒 = 50 个向量)
    ↓ Vector Quantizer (码本查表)
离散 Token 序列 (50Hz, 1秒 = 50 × num_levels 个整数)
```

关键步骤：

1. **Encoder**：把波形/Mel 通过卷积下采样，得到低帧率的连续表示
2. **Vector Quantizer (VQ)**：对每个帧，在码本 (codebook) 中找到最近的向量，用它的索引作为 Token
3. **Decoder**：从离散 Token 重建波形

### 7.1.3 多层码本：从粗到细

真实 EnCodec 使用 **Residual VQ**（残差向量量化），将音频表示为多层 Token：

```
Level 0: 粗粒度 — 节奏、音高轮廓、时长
Level 1: 中等   — 音色、频谱包络
Level 2: 细粒度 — 频谱细节
Level 3: 极细   — 高频噪声、气息声
```

在我们的简化实现中，使用独立的 **Multi-level VQ**：每一层有独立的码本，分别量化同一个连续向量。虽然不如 Residual VQ 优雅，但核心思想相同——**多层离散表示**。

### 7.1.4 代码位置

`code/codec.py` 中的 `NeuralCodec` 类。

```python
codec = NeuralCodec(
    mel_bins=80,        # 输入 Mel 频谱维度
    hidden_dim=128,     # 编码器隐藏维度
    codebook_size=256,  # 每层码本大小
    num_levels=4,       # 4 层 Token
)

# 编码：mel → 多层 Token
codes = codec.encode(mel)        # (B, 4, T_latent)

# 解码：多层 Token → mel
mel_hat = codec.decode(codes)    # (B, 80, T_mel)
```

时间压缩比：**8×**（3层 stride-2 卷积）。
- 1 秒音频 (100 帧 Mel) → ~13 个 codec 帧
- 每个 codec 帧 = 4 个 Token (4 层)
- 1 秒音频 = 13 × 4 = 52 个 Token

---

## 7.2 AR 模型：用 GPT 的方式预测音频 Token

### 7.2.1 核心思想

一旦音频变成了 Token 序列，TTS 就变成了：

```
输入: [文本Token₁, ..., 文本Tokenₙ, 音频Token₁, ..., 音频Tokenₜ₋₁]
预测: 音频Tokenₜ
```

这和 GPT 的训练目标完全一致：

```
GPT:   [word₁, word₂, ..., wordₜ₋₁] → wordₜ
VALL-E: [text₁, ..., textₙ, audio₁, ..., audioₜ₋₁] → audioₜ
```

### 7.2.2 AR Transformer 结构

我们使用 **decoder-only Transformer**（和 GPT-2 相同）：

```
Text embeddings ──────┐
                       │
Audio embeddings ─────┤  → Causal Transformer → Predict next audio token
(BOS + shifted)        │
```

关键设计：

1. **Causal Masking**：每个音频 Token 只能看到它之前的 Token（包括所有文本 Token）
2. **Text as Memory**：文本 Token 作为"条件"，音频 Token 作为"被预测的序列"
3. **BOS Token**：音频序列以一个特殊的学习 token 开始

### 7.2.3 为什么只预测 Level-0？

VALL-E 的 AR 模型只预测 **Level-0（粗粒度）Token**。

原因：Level-0 包含了最重要的时序信息（节奏、时长、音高轮廓）。这些是"说什么"的核心。

而 Level 1-3 是细节信息（音色纹理、频谱细节），可以用更简单的方式（NAR）并行生成。

### 7.2.4 推理：自回归采样

```python
# 从参考音频获取 prompt tokens
prompt_codes = codec.encode(reference_audio)  # (1, 4, T_prompt)

# AR 模型逐个生成 Level-0 tokens
generated_l0 = ar_model.generate(
    text_emb, prompt_codes[:, 0, :],  # 条件: 文本 + prompt 的 level-0
    max_new_tokens=200,
    temperature=1.0,
    top_k=50,
)
```

每一步：
1. 将已有 Token 序列输入 Transformer
2. 取最后一个位置的 logits
3. 温度缩放 + top-k 过滤 → 采样下一个 Token
4. 将新 Token 追加到序列末尾，重复

### 7.2.5 代码位置

`code/valle.py` 中的 `ARTransformer` 类。

---

## 7.3 NAR 模型：并行填充细节

### 7.3.1 为什么需要 NAR？

如果只用 AR 模型逐帧生成所有层级：
- 1 秒音频 ≈ 13 帧 × 4 层 = 52 步推理 → **太慢**
- 每一层的细节其实可以并行预测

NAR (Non-Autoregressive) 模型的设计动机：

```
AR: 负责"什么时候说什么" (时序结构)
NAR: 负责"怎么说" (声学细节)
```

### 7.3.2 NAR Transformer 结构

NAR 使用 **encoder-only Transformer**（双向注意力）：

```
Text embeddings ──────────┐
                           │
Level-0 audio embeddings ──┤  → Bidirectional Transformer → Predict level-l tokens
                           │
Level embedding (目标层)  ──┘
```

关键设计：

1. **双向注意力**：每个位置可以看到所有其他位置（因为细节不依赖时序）
2. **Level Embedding**：告诉模型"你要预测的是第几层"
3. **条件输入**：Level-0 Token + 文本 → 预测 Level-l Token

### 7.3.3 推理流程

```python
for level in [1, 2, 3]:
    tokens_level_l = nar_model.generate(
        text_emb, generated_l0, target_level=level,
    )
    all_codes[:, level] = tokens_level_l
```

每个层级一次前向传播，所有帧的 Token 并行生成。

### 7.3.4 代码位置

`code/valle.py` 中的 `NARTransformer` 类。

---

## 7.4 零样本声音克隆：为什么 3 秒就够？

### 7.4.1 核心机制

零样本克隆的关键在于 **prompt tokens**：

```
[参考音频的 codec tokens] + [目标文本] → 生成的音频 tokens
        ↑ 声音信息在这里
```

Codec 在训练时学到了一个重要的能力：**把声音的音色信息编码到离散 Token 中**。

所以，当我们把参考音频的 Token 放在文本 Token 之前，AR 模型就会"理解"：
> "哦，前面这段声音的音色是 XXX，接下来的话我要用同样的音色来说。"

### 7.4.2 和 GPT 的类比

这和 GPT 的 in-context learning 本质相同：

| GPT | VALL-E |
|-----|--------|
| "请用正式语气回复" | [正式语气的音频 tokens] |
| 上下文 (prompt) | 参考音频 tokens |
| 生成文本 | 生成音频 tokens |
| 风格迁移 | 声音克隆 |

两者都是：**prompt 提供"怎么做"的信息，模型提供"做得好"的能力**。

### 7.4.3 Prompt 长度与质量

- **3 秒**：基本够用，能捕捉音色的大致特征
- **5-10 秒**：效果更好，能捕捉更丰富的声学特征
- **30 秒+**：边际效果递减

---

## 7.5 从零实现 VALL-E：代码走读

### 7.5.1 文件结构

```
ch07_valle/
├── README.md              # 本章教程
├── code/
│   ├── codec.py           # 神经音频 Codec (VQ encoder-decoder)
│   ├── valle.py           # VALL-E 模型 (AR + NAR Transformers)
│   ├── generate.py        # 零样本推理流水线
│   └── train.py           # 两阶段训练脚本
├── checkpoints/           # 模型权重
└── outputs/               # 生成音频
```

### 7.5.2 Codec (`codec.py`)

```
NeuralCodec:
├── encoder: Conv1d × 3 blocks (stride-2) → 8× 下采样
├── quantizer: VectorQuantizer (4 层码本 × 256 entries)
└── decoder: ConvTranspose1d × 3 blocks (stride-2) → 8× 上采样
```

核心类：
- `ResBlock`: 残差卷积块
- `VectorQuantizer`: 多层向量量化器（straight-through estimator）
- `NeuralCodec`: 完整的 encode/decode 管道

### 7.5.3 VALL-E (`valle.py`)

```
VALLE:
├── text_encoder: Embedding + Conv (字符级文本编码)
├── ar_model: ARTransformer
│   ├── text_proj, audio_embed, bos_embed, pos_embed
│   ├── TransformerDecoder (causal, 6 layers)
│   └── out_proj → codebook_size logits
└── nar_model: NARTransformer
    ├── text_proj, audio_embed, level_embed, pos_embed
    ├── TransformerEncoder (bidirectional, 6 layers)
    └── out_proj → codebook_size logits
```

核心方法：
- `forward_ar()`: AR 模型训练前向传播
- `forward_nar()`: NAR 模型训练前向传播
- `generate()`: 完整零样本生成流程

### 7.5.4 生成流水线 (`generate.py`)

```
generate_speech():
1. 加载参考音频 → mel spectrogram
2. mel → codec.encode() → prompt tokens
3. 文本 → tokenizer.encode() → text tokens
4. AR.generate(text + prompt_l0) → generated level-0 tokens
5. NAR.generate(level=1,2,3) → generated detail tokens
6. codec.decode(all_tokens) → reconstructed mel
7. Griffin-Lim vocoder → waveform
```

### 7.5.5 参数量

| 组件 | 参数量 | 说明 |
|------|--------|------|
| Codec | ~1.3M | Encoder + VQ + Decoder |
| AR Transformer | ~4.9M | 6 layers, dim=256 |
| NAR Transformer | ~3.9M | 6 layers, dim=256 |
| **VALL-E Total** | **~9.5M** | 不含 Codec |

（真实 VALL-E 使用 dim=1024, 16 layers，约 3 亿参数）

---

## 7.6 训练与推理

### 7.6.1 形状验证（无需数据）

```bash
cd chapters/ch07_valle/code
python train.py --phase test
```

预期输出：所有模型形状测试通过，参数统计正确。

### 7.6.2 端到端流水线演示（随机权重）

```bash
python generate.py --demo
```

这会创建一个合成参考音频，跑通完整的生成流水线。输出是噪音（因为模型未训练），但验证了整个链路。

### 7.6.3 阶段一：训练 Codec

```bash
python train.py \
    --phase codec \
    --data-dir ../../data/processed \
    --epochs 30 \
    --batch-size 4 \
    --codec-dim 128 \
    --codebook-size 256 \
    --num-levels 4 \
    --lr 1e-4
```

**目标**：让 Codec 学会压缩和重建 mel spectrogram。

**损失函数**：

$$\mathcal{L}_{codec} = \mathcal{L}_{recon} + \mathcal{L}_{vq}$$

其中：
- $\mathcal{L}_{recon} = ||\text{mel} - \hat{\text{mel}}||_1$（重建损失）
- $\mathcal{L}_{vq}$ = VQ commitment + codebook 损失

**预期现象**：
- 前 10 epoch：recon loss 快速下降
- 20-30 epoch：趋于稳定
- 重建的 mel 应该保留原始 mel 的整体结构

### 7.6.4 阶段二：训练 VALL-E

```bash
python train.py \
    --phase valle \
    --data-dir ../../data/processed \
    --codec-checkpoint ../checkpoints/codec_final.pt \
    --epochs 30 \
    --batch-size 4 \
    --valle-dim 256 \
    --ar-layers 6 \
    --nar-layers 6 \
    --lr 1e-4
```

**损失函数**：

AR（交叉熵）：
$$\mathcal{L}_{AR} = -\sum_{t} \log P(\text{token}_t | \text{text}, \text{token}_{<t})$$

NAR（交叉熵）：
$$\mathcal{L}_{NAR} = -\sum_{l=1}^{3} \sum_{t} \log P(\text{token}_{t}^{(l)} | \text{text}, \text{token}^{(0)})$$

**预期现象**：
- AR loss：从 ~5.5 (≈ log(256), 随机猜测) 逐步下降
- NAR loss：类似趋势
- 充分训练后，生成音频开始有语调轮廓

### 7.6.5 零样本推理

```bash
python generate.py \
    --codec-checkpoint ../checkpoints/codec_final.pt \
    --valle-checkpoint ../checkpoints/valle_final.pt \
    --reference-audio path/to/voice_sample.wav \
    --text "你好，我是猫娘。" \
    --output ../outputs/valle_output.wav \
    --max-new-tokens 200 \
    --temperature 1.0 \
    --top-k 50
```

**参数说明**：
- `--temperature`: 1.0 = 默认，>1 更随机，<1 更确定
- `--top-k`: 只从概率最高的 k 个 token 中采样
- `--max-new-tokens`: 最多生成多少 codec 帧

---

## 7.7 VALL-E 与工业界的演进

### 7.7.1 VALL-E 论文

**VALL-E** (Wang et al., 2023) 的核心贡献：

1. **首次证明**：TTS 可以被建模为语言模型任务
2. **AR + NAR 双阶段**：AR 负责时序，NAR 负责细节
3. **EnCodec tokens**：用预训练的神经音频 Codec 产生离散表示
4. **3 秒零样本**：只需 3 秒参考音频即可克隆声音
5. **大规模训练**：60,000 小时英文语音（LibriLight）

### 7.7.2 后续演进

| 模型 | 年份 | 改进 |
|------|------|------|
| **VALL-E X** | 2023 | 跨语言零样本 TTS |
| **SoundStorm** | 2023 | 改进 NAR，MaskGIT 风格并行解码 |
| **CosyVoice** | 2024 | 阿里，LLM + Flow Matching，工业级质量 |
| **FishSpeech** | 2024 | 开源，VQGAN + LLaMA 架构 |
| **MaskGCT** | 2024 | 全 Masked 模型，不用 AR |
| **F5-TTS** | 2024 | Flow Matching + DiT，非自回归 |

### 7.7.3 "TTS as Language Modeling" 范式

VALL-E 开创了一个新范式。所有后续模型都遵循类似的结构：

```
                    ┌── VALL-E:     EnCodec + GPT
音频 Codec ────────┤
(连续→离散Token)   ├── CosyVoice:  S2Tokenizer + LLM + Flow
                    │
                    ├── FishSpeech: VQGAN + LLaMA
                    │
                    └── F5-TTS:     (不用Codec, 直接用Flow)
```

核心区别在于：
- **Codec 的选择**：EnCodec / DAC / 自研 Codec
- **语言模型的选择**：GPT / LLaMA / 自定义
- **是否有 NAR 阶段**：有的用 Flow Matching 替代

### 7.7.4 与前面章节的关系

```
Ch01 音频基础 → 理解波形、Mel、STFT
Ch02 Tacotron → 端到端 TTS（连续 Mel 预测）
Ch04 FastSpeech → 并行 TTS（非自回归）
Ch05 VITS → Flow + VAE（连续潜空间）
Ch06 GPT-SoVITS → Few-shot（speaker embedding）
Ch07 VALL-E → 语言建模范式（离散 Token 预测）← 你在这一章
```

从 Ch02 到 Ch07 的演进线索：

```
连续 Mel (Tacotron)
    → 连续潜空间 (VITS)
        → 离散 Token (VALL-E) ← 范式转换
```

**VALL-E 的核心贡献不是某个具体技术，而是一个思想**：

> 一旦你有了好的音频 Codec，TTS 就变成了语言模型问题。
> 而语言模型问题，我们已经知道怎么解决了。

---

## 7.8 本章小结

### VALL-E 解决了什么

| 问题 | 解决方案 |
|------|----------|
| 音色训练时固定 | Prompt tokens 传递音色信息 |
| 需要大量配对数据 | 语言模型的 in-context learning |
| TTS 是回归问题 | TTS 是 Token 预测问题 |
| 连续输出不稳定 | 离散 Token + 交叉熵损失 |

### 遗留问题

1. **Codec 质量决定上限**：如果 Codec 重建质量差，生成的音频也差
2. **AR 推理仍然慢**：逐 Token 生成，一秒音频需要几十步推理 → MaskGCT 等全并行方案
3. **幻觉问题**：AR 模型可能生成不匹配的文本/音频 → 需要更好的对齐
4. **Codec 的码本利用率**：训练不充分时，很多码本条目从未被使用（codebook collapse）

### 下一步

- **Ch08**: F5-TTS / CosyVoice — Flow Matching 替代 NAR，更高质量
- **Ch09**: 部署优化 — ONNX 导出、CPU 推理

---

## 习题

1. **Codec 的多层码本**：为什么 Level-0 Token 包含节奏/音高信息，而 Level-3 包含高频细节？这和 Encoder 的感受野有什么关系？

2. **AR vs NAR 的权衡**：如果只用 AR 模型预测所有层级的 Token（不使用 NAR），会发生什么？生成质量和速度分别受到什么影响？

3. **Prompt 长度**：参考音频的 prompt 太短（< 1 秒）会怎样？太长（> 30 秒）呢？从 Transformer 的注意力机制角度解释。

4. **Top-k 采样**：AR 生成时使用 top-k=1（贪心解码）和 top-k=100（高度随机）分别会产生什么效果？这和 GPT 文本生成中的 temperature 有什么相似之处？

5. **范式对比**：对比 Tacotron2 (Ch02) 和 VALL-E (Ch07) 的生成过程。两者的"自回归"有什么本质区别？（提示：一个预测连续向量，一个预测离散 Token）

---

## 参考文献

- [1] Wang et al., 2023. *Language Models Are General-Purpose Interfaces* (VALL-E).
- [2] Defossez et al., 2022. *High Fidelity Neural Audio Compression* (EnCodec).
- [3] van den Oord et al., 2017. *Neural Discrete Representation Learning* (VQ-VAE).
- [4] Zhang et al., 2023. *VALL-E X: Speak In-Context Learning for Cross-Lingual TTS*.
- [5] Du et al., 2024. *CosyVoice: A Scalable Multilingual Expressive TTS*.
- [6] Leng et al., 2024. *MaskGCT: Zero-Shot TTS with Masked Generative Codec Transformer*.
- [7] Chen et al., 2024. *F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech*.

---

## 目录结构

```
ch07_valle/
├── README.md              # 本章教程
├── code/
│   ├── codec.py           # 神经音频 Codec (VQ encoder-decoder, ~1.3M params)
│   ├── valle.py           # VALL-E 模型 (AR + NAR, ~9.5M params)
│   ├── generate.py        # 零样本推理流水线
│   └── train.py           # 两阶段训练 (codec → VALL-E)
├── checkpoints/           # 模型权重 (gitignore)
└── outputs/               # 生成音频 (gitignore)
```
