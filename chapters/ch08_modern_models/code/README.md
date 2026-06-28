# Ch08: Modern TTS Models — Flow Matching, Zero-Shot, and Controllability

> 从 2023 到 2026，TTS 经历了范式转移：从 Seq2Seq 到 Flow Matching，
> 从单说话人到零样本克隆，从"能说话"到"可控地说话"。
>
> 这一章，我们从零实现三个代表性现代 TTS 系统的简化版。

## 0. 本章导学

### 为什么学现代 TTS？

Ch02 的 Tacotron2 解决了"端到端"问题，但留下了三个难题：

| 问题 | Tacotron2 的局限 | 现代方案 |
|------|-----------------|---------|
| **音质** | 自回归 + 线性 attention → 模糊的 mel | Flow Matching → 清晰的频谱 |
| **新说话人** | 必须微调 | 3 秒参考音频 → 零样本克隆 |
| **可控性** | 无法控制发音/情感 | 显式拼音/音调/时长控制 |

### 技术演进时间线

```
2017  Tacotron          首个端到端 TTS (Google)
2018  Tacotron2         Location-Sensitive Attention (Google)
2018  FastSpeech        Non-autoregressive + Duration Predictor (MSRA)
2020  HiFi-GAN          Neural vocoder, 实时推理 (KAIST)
2021  VITS              VAE + Normalizing Flow + GAN (JAIST)
2023  VALL-E            TTS = language modeling over audio codes (Microsoft)
2023  NaturalSpeech 3   Factorized codecs + diffusion (Microsoft)
2024  CosyVoice         Scalable zero-shot TTS (Alibaba / FunAudioLLM)
2024  F5-TTS            Flow Matching + DiT (SJTU / 陈耘辉)
2024  E2-TTS            Non-autoregressive Flow Matching (Microsoft)
2025  IndexTTS          Industrial controllable TTS (Xiaomi)
2025  CosyVoice 2       Streaming speech with LLM (Alibaba)
2025  F5-TTS            ACL 2025 正式发表
2026  ???               (你在读这一章，下一个突破可能来自你)
```

### 本章模型选择逻辑

| 模型 | 代表方向 | 核心创新 |
|------|---------|---------|
| **F5-TTS** | 生成范式 | Flow Matching + Diffusion Transformer |
| **CosyVoice** | 零样本 | Speaker Encoder + AR Language Model over Speech |
| **IndexTTS** | 可控性 | Pinyin/Tone/Duration 显式控制 |

三个模型共同覆盖了现代 TTS 的三个核心维度：**音质、泛化、控制**。

---

## 1. F5-TTS: Flow Matching + DiT

### 1.1 从 Diffusion 到 Flow Matching

**Diffusion 的问题**：
- 需要预定义 noise schedule (β₁, β₂, ..., β_T)
- 生成时需要反向马尔可夫链，步数多 (50-1000)
- 轨迹弯曲 → 每步都有累积误差

**Flow Matching 的直觉**：

想象你要把一团云（噪声）变成一只猫（数据）。
- Diffusion：让云慢慢凝结成猫，每一步都去一点雾 → 很多步
- Flow Matching：直接告诉每个雾滴"你的终点是猫的哪个位置"，走直线 → 少步数

数学上，Flow Matching 学习一个**速度场** `v_t(x)`，满足：
- 在 `t=0`，`x ~ N(0, I)` （噪声）
- 在 `t=1`，`x ~ data distribution` （真实 mel）
- 中间用线性插值：`x_t = (1-t)·x₀ + t·x₁`

速度场的目标就是常数向量：`v_target = x₁ - x₀`

### 1.2 训练目标

```python
# 采样
x0 ~ N(0, I)          # 噪声
x1 = real_mel          # 真实数据
t  ~ Uniform(0, 1)     # 时间

# 构造中间状态
x_t = (1 - t) * x0 + t * x1

# 速度场目标
v_target = x1 - x0

# 损失：预测速度 vs 真实速度
loss = MSE(v_pred(x_t, t, cond), v_target)
```

就这么简单！没有 noise schedule，没有 β_t，没有马尔可夫链。

### 1.3 推理：Euler ODE 积分

```python
x = noise               # t = 0
dt = 1 / n_steps
for i in range(n_steps):
    t = i / n_steps
    v = velocity_net(x, t, text, ref_mel)
    x = x + dt * v      # Euler step
# x ≈ data at t = 1
```

10-20 步就够了，因为轨迹是直线。

### 1.4 代码结构

```
f5_tts.py
├── SinusoidalTimeEmbedding    # 时间 → 向量
├── TransformerBlock           # 标准 Pre-LN Transformer
├── F5VelocityNet              # 速度场网络（Transformer + cross-attn to text）
├── F5TTS                      # 完整的 Flow Matching 训练 + 采样
└── SimpleTextEncoder          # 文本编码器（共用）
```

### 1.5 与原版 F5-TTS 的差异

| 组件 | 原版 | 我们的简化版 |
|------|------|-------------|
| Backbone | DiT (Diffusion Transformer, 2D patch) | 1D Transformer |
| 时长处理 | Duration-aware masking | 固定长度输入 |
| Vocoder | BigVGAN / Vocos | 无 (只生成 mel) |
| 数据 | LibriTTS 960h | 合成数据 demo |

---

## 2. CosyVoice: Zero-Shot Voice Cloning

### 2.1 核心问题：3 秒克隆一个从未见过的声音

传统 TTS：每个说话人需要 10+ 小时录音 + 微调。
CosyVoice（和 VALL-E）：给 3 秒参考音频，生成该说话人说的任意内容。

### 2.2 怎么做到的？

**关键洞察**：把 TTS 变成"语音语言模型"。

```
Text → "你好世界"
Ref  → [3秒音频] → Speaker Encoder → spk_emb (256维向量)

AR Model:  P(speech_t | speech_<t, text, spk_emb)
           ↑ 条件概率：给定上文 + 文本 + 音色，预测下一帧
```

训练时，模型见过上千个说话人，每个说话人的声音都被压缩成一个 `spk_emb`。
推理时，新说话人的 3 秒音频也被压缩成 `spk_emb` → 模型虽然没见过这个人，
但见过"这种类型的向量"，所以能生成对应的声音。

### 2.3 Speaker Encoder 的作用

Speaker Encoder 是一个**说话人验证模型**（speaker verification）：
- 输入：一段语音 mel
- 输出：一个向量，捕捉"说话人身份"（音色、音高、口音）
- 关键属性：同一个人的不同语音 → 相似向量；不同人 → 不同向量

```python
class SpeakerEncoder(nn.Module):
    # Conv1d stack → temporal mean pooling → projection
    def forward(self, ref_mel):
        x = self.convs(ref_mel)   # 提取局部声学特征
        x = x.mean(dim=1)         # 时间维度池化 → 长度不变
        return self.proj(x)       # 256维 speaker embedding
```

**时间池化是关键**：让 embedding 与参考音频长度无关。

### 2.4 VALL-E 范式 vs CosyVoice

| | VALL-E (2023) | CosyVoice (2024) |
|---|---|---|
| 音频表示 | EnCodec 离散 token | 连续 speech tokens |
| AR 模型 | GPT-style, 预测 token | Decoder-only, 预测帧 |
| 非 AR 部分 | 独立模型细化 | Flow Matching decoder |
| 数据 | 60K hours | 多语言大规模 |
| 流式 | 不支持 | CosyVoice 2 支持 |

### 2.5 代码结构

```
cosyvoice.py
├── SpeakerEncoder         # ref_mel → spk_emb (256维)
├── AutoregressiveDecoder  # Decoder-only Transformer (causal)
└── CosyVoice              # 完整 pipeline
```

---

## 3. IndexTTS: Controllable TTS

### 3.1 工业界的真问题

消费级 TTS 只需要"好听"。工业级 TTS 还需要：

1. **多音字**："银行" vs "行走" — 同一个字，不同读音
2. **韵律控制**：某个词要读重音、拉长
3. **情感控制**：客服场景需要温和，导航需要清晰
4. **鲁棒性**：不能念错字（哪怕 G2P 出错）

IndexTTS 的解法：**显式的拼音中间表示**。

### 3.2 Pinyin Embedding

中文拼音由三部分组成：

```
"ma3" (马) = 声母(initial) "m" + 韵母(final) "a" + 声调(tone) 3
```

我们分别 embedding 这三个组件，然后合并：

```python
pinyin_emb = proj(concat(
    initial_emb[initial_id],   # 21 种声母
    final_emb[final_id],       # 35+ 种韵母
    tone_emb[tone_id],         # 1-5 声调
))
```

这样：
- 改 tone_id → 同一音节不同声调
- 改 initial_id → 替换声母（修音）
- 完全显式，不依赖 G2P 黑箱

### 3.3 Duration Predictor + Length Regulator

```
Syllable-level:    [ma3]   [ni3]   [hao3]
                    ↓ Dur   ↓ Dur   ↓ Dur
Frames per syl:    [15]    [20]    [25]
                    ↓ LR    ↓ LR    ↓ LR
Frame-level:     [ma3]×15 [ni3]×20 [hao3]×25  → 输入 acoustic decoder
```

**控制时长**：直接把 duration 乘以一个系数：
- `dur_scale = 0.8` → 说快点
- `dur_scale = 1.5` → 说慢点
- 某个 syllable 的 duration × 2 → 只拉长那个字

### 3.4 代码结构

```
indextts.py
├── PinyinEmbedding      # (声母, 韵母, 声调) → vector
├── DurationPredictor    # syllable → 帧数
├── length_regulate()    # syllable-level → frame-level
└── IndexTTS             # 完整可控 TTS 模型
```

---

## 4. 模型对比

### 4.1 架构对比

| | F5-TTS | CosyVoice | IndexTTS |
|---|---|---|---|
| **生成范式** | Flow Matching (非自回归) | Autoregressive + Flow | Non-AR + Duration |
| **文本编码** | BERT-style | BERT-style | Pinyin embedding |
| **音色条件** | 参考 mel 直接输入 | Speaker embedding | Speaker embedding |
| **推理步数** | 10-20 (Euler) | T 帧 × AR + Flow | 1 pass (非 AR) |
| **可控性** | 低 (black-box) | 中 (ref audio) | 高 (拼音/音调/时长) |

### 4.2 参数量与性能（工业版本）

| 模型 | 参数量 | 推理速度 (RTF) | MOS (音质) | 零样本 | 论文 |
|------|--------|---------------|-----------|--------|------|
| F5-TTS | ~300M | ~0.2 (GPU) | 4.5/5 | 需要参考音频 | Chen et al., ACL 2025 |
| CosyVoice | ~500M | ~0.3 (流式) | 4.6/5 | 3秒克隆 | Du et al., 2024 |
| CosyVoice 2 | ~1B | ~0.15 (流式) | 4.7/5 | 3秒+流式 | Du et al., 2025 |
| IndexTTS | ~300M | ~0.25 | 4.5/5 | 支持 | Xiaomi, 2025 |
| VALL-E | ~1B | 较慢 | 4.3/5 | 开创性 | Wang et al., 2023 |

> RTF (Real-Time Factor) < 1 表示比实时快。例如 RTF=0.2 表示生成 1 秒音频只需 0.2 秒。
> MOS (Mean Opinion Score) 是人工主观评分，1-5 分，5 分最好。

### 4.3 我们的简化版 vs 工业版

| | 简化版 | 工业版 |
|---|---|---|
| 参数量 | ~1-5M | 300M-1B |
| 训练数据 | 合成/小数据集 | 数千小时 |
| 声码器 | 无 (输出 mel) | BigVGAN / Vocos / HIFIGAN |
| 推理 | CPU 可跑 | 需要 GPU |
| 音质 | 噪声/demo | 接近真人 |

**关键原则**：我们关注的是**架构思想**而非音质。
理解了 Flow Matching、Zero-shot、Controllability 的原理，
就能读懂任何 2025+ 的 TTS 论文。

---

## 5. 工业界应用现状 (2024-2026)

### 5.1 主流产品

| 产品 | 底层技术 | 特点 |
|------|---------|------|
| **Azure Neural TTS** | VALL-E 系列 | 多语言、低延迟 |
| **Google Cloud TTS** | 内部 Flow/Diffusion | Studio 级音质 |
| **ElevenLabs** | 闭源 zero-shot | 最强克隆、情感丰富 |
| **阿里通义 (CosyVoice)** | CosyVoice 1/2 | 中文最强、开源 |
| **字节 Seed-TTS** | 内部 AR + diffusion | 抖音/TikTok 内部 |
| **小米 IndexTTS** | IndexTTS | 手机助手、可控性强 |
| **MiniMax Speech-02** | 闭源 | 情感控制、多语言 |

### 5.2 关键趋势

1. **Flow Matching 取代 Diffusion**：更少步数、更好音质 (F5-TTS, E2-TTS)
2. **LLM + TTS 融合**：用 LLM 做文本理解，TTS 做语音输出 (CosyVoice 2, GLM-4-Voice)
3. **流式推理**：边生成边播放，延迟 < 300ms (CosyVoice 2)
4. **细粒度控制**：情感、语气、节奏都可调 (IndexTTS, Seed-TTS)
5. **开源生态爆发**：F5-TTS、CosyVoice、MeloTTS、ChatTTS 全部开源

---

## 6. 运行代码

### 6.1 快速验证

```bash
# 验证每个模型的 sanity check
python f5_tts.py
python cosyvoice.py
python indextts.py

# 训练（合成数据，验证 pipeline）
python train.py --model f5_tts --epochs 20
python train.py --model cosyvoice --epochs 20
python train.py --model indextts --epochs 20

# 推理 demo
python inference.py --model all
```

### 6.2 真实训练 checklist

要用真实数据训练，你需要：

1. **数据准备**：
   - 下载 LibriTTS 或 AISHELL-3
   - 用 ch01 的 mel 提取脚本预处理
   - 计算 CMVN (均值方差归一化)

2. **声码器**：
   - 下载预训练 HiFi-GAN (mel → waveform)
   - 或者用 Vocos (更现代的选择)

3. **训练配置**：
   - 替换 `SyntheticTTSDataset` 为真实 `DataLoader`
   - 添加 cosine annealing LR scheduler
   - 添加 gradient accumulation (如果 GPU 小)
   - 每 10K steps 保存 checkpoint + 生成 sample

4. **评估**：
   - 客观：Mel Cepstral Distortion (MCD), F0 RMSE
   - 主观：MOS (人工评分), WER (用 ASR 评估可懂度)
   - 相似度：Speaker Verification (cosine similarity)

---

## 7. 延伸阅读

### 必读论文

| 论文 | 年份 | 关键贡献 |
|------|------|---------|
| [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://arxiv.org/abs/2410.06885) | 2024 (ACL 2025) | Flow Matching 用于 TTS 的 SOTA |
| [CosyVoice: A Scalable Multilingual Zero-Shop Text-to-Speech Synthesizer Based on Supervised Semantic Tokens](https://arxiv.org/abs/2407.05407) | 2024 | 大规模零样本 TTS |
| [CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models](https://arxiv.org/abs/2412.10117) | 2025 | LLM + 流式 TTS |
| [IndexTTS](https://arxiv.org/abs/2504.19683) | 2025 | 工业级可控 TTS |
| [VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/abs/2301.02111) | 2023 | 开创性：TTS = 语言模型 |
| [E2-TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS](https://arxiv.org/abs/2406.11427) | 2024 | 非自回归 Flow Matching |
| [NaturalSpeech 3](https://arxiv.org/abs/2403.03100) | 2024 | Factorized codecs + diffusion |

### 前置知识

- Flow Matching 数学基础：[Lipman et al., Flow Matching for Generative Modeling, ICLR 2023](https://arxiv.org/abs/2210.02747)
- Optimal Transport 条件流：[Tong et al., 2024](https://arxiv.org/abs/2302.00482)
- DiT (Diffusion Transformer)：[Peebles & Xie, ICCV 2023](https://arxiv.org/abs/2212.09748)

---

## 8. 思考题

1. **Flow Matching vs Diffusion**：为什么 Flow Matching 的推理步数可以更少？
   从轨迹几何角度解释。

2. **零样本的本质**：为什么 3 秒音频就能"克隆"一个声音？
   Speaker Embedding 编码了什么信息？丢失了什么信息？

3. **可控性的代价**：IndexTTS 需要显式拼音标注。
   这在工业部署中的成本是什么？有没有替代方案？

4. **流式推理**：CosyVoice 2 如何实现"边生成边播放"？
   F5-TTS 的 Flow Matching 能做流式吗？为什么？

5. **未来方向**：2026+ 的 TTS 会是什么样子？
   （提示：想想多模态、实时对话、情感迁移）

---

## 9. 文件清单

```
chapters/ch08_modern_models/code/
├── f5_tts.py          # F5-TTS: Flow Matching + Transformer
├── cosyvoice.py       # CosyVoice: 零样本 TTS
├── indextts.py        # IndexTTS: 拼音/音调控制
├── train.py           # 训练脚本 (三个模型)
├── inference.py       # 推理 demo (三个模型)
├── outputs/           # 推理生成的 mel 图
└── README.md          # 本文档
```

---

> Neko 说：现代 TTS 的三个关键词是 **Flow**（流匹配，不是扩散）、
> **Zero-shot**（3 秒克隆）、**Control**（拼音级控制）。
>
> 理解了这三个，你就掌握了 2024-2026 TTS 的全部精华。
>
> 现在，去读论文吧。🐱
