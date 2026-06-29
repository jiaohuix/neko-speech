# Ch12: Fish Speech S2 — 工业级 Dual-AR TTS

> 从零实现 4B 参数 Dual-AR 架构，理解工业级 TTS 的核心设计

---

## 0. 本章导学

### 为什么学 Fish Speech？

前面的章节（ch02-ch11）我们学习了 TTS 的基础模型。但工业级 TTS 系统需要解决更多问题：

| 需求 | 基础模型的局限 | Fish Speech 的方案 |
|------|---------------|-------------------|
| **多语言** | 单语言训练 | 80+ 语言统一模型 |
| **零样本克隆** | 需要微调 | 3 秒音频即可克隆 |
| **情感控制** | 无法控制 | 15,000+ 内联情感标签 |
| **实时性** | RTF > 1 | RTF 0.195 (H200) |
| **工业规模** | 百万参数 | 4B 参数，10M+ 小时数据 |

### Fish Speech 的核心创新

1. **Dual-AR 架构**: Slow AR (4B) + Fast AR (400M)
2. **RL 对齐**: GRPO (Group Relative Policy Optimization)
3. **LLM 优化**: 复用 LLaMA/Qwen 的推理优化技术
4. **工业级规模**: 10M+ 小时训练数据，80+ 语言

### 本章目标

通过实现简化版 Fish Speech（6+2 层，~50M 参数），理解：
- Dual-AR 的设计思想
- 主从 Transformer 的协作机制
- 工业级 TTS 的工程实践

---

## 1. Fish Speech 架构详解

### 1.1 整体架构

```
Fish Speech S2 Pro Architecture
================================

Input: Text (80+ languages) + Reference Audio (3s for zero-shot)
   │
   ├─→ [Text Encoder] ─→ Text Embeddings
   │
   ├─→ [Speaker Encoder] ─→ Speaker Embedding (from ref audio)
   │
   └─→ [Dual-AR TTS]
        │
        ├─→ Slow AR (4B, 32 layers)
        │    │
        │    │ Predicts semantic tokens (codebook 0)
        │    │ Autoregressive along time axis
        │    │
        │    └─→ Semantic Tokens: [t₁, t₂, ..., tₙ]
        │
        ├─→ Fast AR (400M, 4 layers)
        │    │
        │    │ For each time step:
        │    │   Input: semantic token + speaker embedding
        │    │   Predicts acoustic tokens (codebooks 1-9)
        │    │   Parallel along codebook axis
        │    │
        │    └─→ Acoustic Tokens: [c₁, c₂, ..., c₉] per time step
        │
        └─→ Codec Decoder (10 codebooks → waveform)
             │
             └─→ Output: Waveform (16kHz, 24-bit)

Key Design:
  - Slow AR: Large, slow, semantic planning
  - Fast AR: Small, fast, acoustic rendering
  - Separation of concerns: linguistic vs acoustic
```

### 1.2 Dual-AR 的直觉理解

想象你在写一篇文章：

**Slow AR (Master)**: 负责构思文章内容
- 思考：这段要讲什么？
- 决策：生成大纲（semantic tokens）
- 特点：深度思考，慢但准确

**Fast AR (Slave)**: 负责润色文字表达
- 输入：大纲（semantic tokens）
- 决策：选择具体用词（acoustic tokens）
- 特点：快速执行，快但依赖大纲

**为什么这样设计？**

1. **语义 vs 声学分离**
   - Semantic tokens: 语言内容（"你好"）
   - Acoustic tokens: 声学细节（音高、音色、语速）
   - 分离后更容易建模

2. **计算效率**
   - Slow AR: 只生成 1 个 codebook（10% 的计算）
   - Fast AR: 生成 9 个 codebooks（90% 的计算）
   - 但 Fast AR 小 10x，总体更快

3. **复用 LLM 技术**
   - Slow AR 结构类似 LLaMA/Qwen
   - 可以直接用 LLM 的推理优化（KV cache, PagedAttention）

### 1.3 与基础模型的对比

| 特性 | VALL-E (ch07) | GPT-SoVITS (ch09) | Fish Speech |
|------|--------------|-------------------|-------------|
| **架构** | Single AR | AR + VITS | Dual-AR |
| **参数量** | 9.5M | 30M | 4B |
| **Codebooks** | 1 (semantic) | 1 (semantic) | 10 (1 semantic + 9 acoustic) |
| **零样本** | ✓ (3s) | ✓ (5s) | ✓ (3s) |
| **多语言** | ✗ | ✗ | ✓ (80+) |
| **情感控制** | ✗ | ✗ | ✓ (15k tags) |
| **RTF** | 0.5 | 0.8 | 0.195 |

**关键差异**:
- VALL-E: 单 AR，只建模 semantic tokens
- GPT-SoVITS: AR + VITS，两阶段
- Fish Speech: Dual-AR，统一建模所有 codebooks

---

## 2. 核心组件实现

### 2.1 Rotary Position Embedding (RoPE)

**为什么需要 RoPE？**

传统位置编码（absolute, learned）的问题：
- 无法外推到更长序列
- 不编码相对位置信息

RoPE 的解决方案：
- 通过旋转编码相对位置
- 可以外推到训练时未见过的长度

**数学原理**:

```python
# 位置编码
q_m = R(θ, m) · W_q · x_m  # query at position m
k_n = R(θ, n) · W_k · x_n  # key at position n

# 注意力分数
q_m · k_n = x_m^T · W_q^T · R(θ, m-n) · W_k · x_n
                     ↑
              只依赖 (m-n)，相对位置！
```

**代码实现**:

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base

        # 旋转频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # 计算旋转角度
        t = torch.arange(seq_len, device=x.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # 应用旋转
        cos = freqs.cos()
        sin = freqs.sin()

        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        rotated = torch.cat([-x2, x1], dim=-1)

        return x * cos + rotated * sin
```

### 2.2 Transformer Block (LLaMA-style)

**与标准 Transformer 的区别**:

| 特性 | Standard Transformer | LLaMA-style (Fish Speech) |
|------|---------------------|--------------------------|
| **Norm** | Post-norm | Pre-norm |
| **Position** | Absolute/Learned | RoPE |
| **Activation** | ReLU/GELU | SwiGLU |
| **Attention** | MHA | GQA (Grouped Query) |
| **Bias** | With bias | No bias |

**Pre-norm vs Post-norm**:

```python
# Post-norm (Standard)
h = x + Attention(x)
h = LayerNorm(h)
h = h + FFN(h)
output = LayerNorm(h)

# Pre-norm (LLaMA, Fish Speech)
h = LayerNorm(x)
h = x + Attention(h)
h = LayerNorm(h)
output = h + FFN(h)
```

**Pre-norm 的优势**:
- 训练更稳定
- 梯度流动更好
- 适合深层网络

**SwiGLU Activation**:

```python
# Standard FFN
h = ReLU(W1 · x)
output = W2 · h

# SwiGLU (LLaMA, Fish Speech)
h1 = SiLU(W1 · x)  # Swish activation
h2 = W3 · x
output = W2 · (h1 ⊗ h2)  # Element-wise multiplication
```

**SwiGLU 的优势**:
- 更好的性能
- 计算量略大（3 个矩阵 vs 2 个）
- 工业界标准选择

**Grouped Query Attention (GQA)**:

```python
# MHA (Multi-Head Attention)
n_heads = 32
n_kv_heads = 32  # 每个 head 独立的 K, V

# GQA (Grouped Query Attention)
n_heads = 32
n_kv_heads = 8  # 8 个 KV heads，每个被 4 个 query heads 共享

# 节省内存：KV cache 减少 4x
# 性能损失：几乎无损
```

### 2.3 Slow AR: Semantic Token Prediction

**Slow AR 的任务**:

```python
Input: Text embeddings + Speaker embedding + Previous tokens
Output: Next semantic token (codebook 0)

Training:
  tokens_in = [BOS, t₁, t₂, ..., tₙ₋₁]
  tokens_out = [t₁, t₂, ..., tₙ]
  loss = CrossEntropy(SlowAR(tokens_in), tokens_out)

Inference:
  for _ in range(max_len):
    logits = SlowAR(tokens_in)
    next_token = sample(logits[:, -1, :])
    tokens_in = concat(tokens_in, next_token)
```

**Slow AR 的架构**:

```python
class SlowAR(nn.Module):
    def __init__(self, vocab_size, dim=4096, n_layers=32, n_heads=32):
        self.tok_emb = Embedding(vocab_size, dim)
        self.layers = [TransformerBlock(dim, n_heads) for _ in range(n_layers)]
        self.out_proj = Linear(dim, vocab_size)

    def forward(self, tokens):
        h = self.tok_emb(tokens)
        for layer in self.layers:
            h = layer(h, causal_mask)
        logits = self.out_proj(h)
        return logits
```

**简化版 vs 原版**:

| 参数 | 原版 (4B) | 简化版 (50M) |
|------|----------|-------------|
| dim | 4096 | 1024 |
| n_layers | 32 | 6 |
| n_heads | 32 | 16 |
| n_kv_heads | 8 | 4 |
| ffn_dim | 16384 | 4096 |

### 2.4 Fast AR: Acoustic Token Prediction

**Fast AR 的任务**:

```python
Input: Semantic token + Previous acoustic tokens
Output: Next acoustic tokens (codebooks 1-9)

Training:
  for each time step t:
    acoustic_in = [c₁, c₂, ..., c₉]  # previous
    acoustic_out = [c₁', c₂', ..., c₉']  # next
    loss += CrossEntropy(FastAR(acoustic_in), acoustic_out)

Inference:
  for each time step t:
    acoustic_tokens = FastAR(semantic_token[t])
    # Parallel generation along codebook axis
```

**Fast AR 的架构**:

```python
class FastAR(nn.Module):
    def __init__(self, vocab_size, n_codebooks=10, dim=2048, n_layers=4):
        self.codebook_embs = [Embedding(vocab_size, dim) for _ in range(n_codebooks)]
        self.layers = [TransformerBlock(dim) for _ in range(n_layers)]
        self.out_projs = [Linear(dim, vocab_size) for _ in range(n_codebooks)]

    def forward(self, codebook_tokens):
        # Sum codebook embeddings
        h = sum(self.codebook_embs[k](codebook_tokens[:, k]) for k in range(K))

        # Transformer
        for layer in self.layers:
            h = layer(h)

        # Output projections
        logits = [self.out_projs[k](h) for k in range(K)]
        return logits
```

**关键设计**:

1. **Codebook Embeddings**: 每个 codebook 独立的 embedding
2. **Parallel Generation**: 一次生成所有 codebooks
3. **Smaller Model**: 4x 更少的层数，8x 更小的维度

---

## 3. 训练流程

### 3.1 数据准备

**Fish Speech S2 Pro**:
- 10M+ 小时音频
- 80+ 语言
- 采样率: 24kHz
- 格式: WAV (24-bit)

**简化版**:
- 使用 ch04 的合成数据
- 中文 + 英文
- 采样率: 16kHz
- 格式: WAV (16-bit)

**Codec 训练**:

```python
# Step 1: Train neural codec (EnCodec/DAC)
codec = EnCodec(n_codebooks=10, sample_rate=24000)
codec.train(audio_data)

# Step 2: Encode audio to tokens
for audio in dataset:
    tokens = codec.encode(audio)  # (10, T)
    semantic_tokens = tokens[0]  # codebook 0
    acoustic_tokens = tokens[1:]  # codebooks 1-9
```

### 3.2 Dual-AR 训练

**Stage 1: Train Slow AR**

```python
# Objective: Predict semantic tokens
for batch in dataloader:
    semantic_tokens = batch["semantic"]  # (B, T)

    # Input: [BOS, t₁, t₂, ..., tₙ₋₁]
    input_tokens = F.pad(semantic_tokens[:, :-1], (1, 0), value=BOS_TOKEN)

    # Forward
    logits = slow_ar(input_tokens)

    # Loss
    loss = CrossEntropy(logits.view(-1, V), semantic_tokens.view(-1))

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Stage 2: Train Fast AR**

```python
# Objective: Predict acoustic tokens conditioned on semantic
for batch in dataloader:
    semantic_tokens = batch["semantic"]  # (B, T)
    acoustic_tokens = batch["acoustic"]  # (B, 9, T)

    # Concatenate semantic + acoustic
    all_tokens = cat([semantic_tokens.unsqueeze(1), acoustic_tokens], dim=1)  # (B, 10, T)

    # Forward
    logits_list = fast_ar(all_tokens)

    # Loss: sum over all codebooks
    loss = 0
    for k in range(9):
        loss += CrossEntropy(logits_list[k], acoustic_tokens[:, k])
    loss /= 9

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 3.3 RL 对齐 (GRPO)

**为什么需要 RL？**

监督学习的问题：
- 只优化 next-token prediction
- 不考虑整体质量
- 可能有 distribution shift

RL 的优势：
- 优化整体序列质量
- 可以加入人类偏好
- 更好的鲁棒性

**GRPO (Group Relative Policy Optimization)**:

```python
# 1. Generate multiple candidates
for _ in range(G):  # G = group size (e.g., 8)
    candidate = model.generate(text)
    score = reward_model(candidate)  # Human preference score

# 2. Compute relative advantages
scores = [score₁, score₂, ..., score_G]
mean_score = mean(scores)
advantages = [s - mean_score for s in scores]

# 3. Update policy
for candidate, advantage in zip(candidates, advantages):
    log_prob = model.log_prob(candidate)
    loss = -advantage * log_prob  # Policy gradient
    loss.backward()

optimizer.step()
```

**Reward Model**:
- 语义准确性: ASR WER
- 音质: MOS predictor
- 说话人相似度: Speaker verification cosine
- 情感匹配: Emotion classifier accuracy

---

## 4. 推理优化

### 4.1 KV Cache

**问题**: 自回归推理重复计算

```python
# 朴素推理
for step in range(max_len):
    logits = model(tokens[:step])  # 每次都重新计算所有 tokens
    next_token = sample(logits[:, -1])
    tokens.append(next_token)

# 复杂度: O(T²)
```

**解决方案: KV Cache**

```python
# KV Cache 推理
kv_cache = {}
for step in range(max_len):
    logits, kv_cache = model(tokens[step:step+1], kv_cache)  # 只计算新 token
    next_token = sample(logits[:, -1])
    tokens.append(next_token)

# 复杂度: O(T)
# 内存: O(T) for KV cache
```

**Fish Speech 的 KV Cache**:

```python
class SlowAR(nn.Module):
    def forward(self, tokens, kv_cache=None):
        h = self.tok_emb(tokens)

        for i, layer in enumerate(self.layers):
            h, new_kv = layer(h, kv_cache[i] if kv_cache else None)
            if kv_cache is not None:
                kv_cache[i] = new_kv

        logits = self.out_proj(h)
        return logits, kv_cache
```

### 4.2 PagedAttention (vLLM)

**问题**: KV cache 内存碎片化

```python
# 传统实现
kv_cache = torch.zeros(max_len, n_layers, 2, batch, n_heads, head_dim)
# 问题：预分配固定大小，内存浪费
```

**PagedAttention**:

```python
# 分页管理 KV cache
kv_cache = PagedKVCache(
    block_size=16,  # 16 tokens per block
    num_blocks=1000,
)

# 动态分配
for token in generated_tokens:
    block = kv_cache.allocate_block()
    kv_cache[block] = compute_kv(token)
```

**优势**:
- 内存利用率: 90%+ (vs 50% 传统)
- 支持更大 batch size
- 支持动态 batching

### 4.3 Continuous Batching (SGLang)

**问题**: 不同请求长度不同，batch 效率低

```python
# 传统 batching
batch = [req₁, req₂, req₃]  # 所有请求必须同时开始、同时结束
# 问题：短请求完成后，GPU 空闲
```

**Continuous Batching**:

```python
# 动态 batching
active_requests = {req₁, req₂, req₃}

while active_requests:
    batch = sample(active_requests, batch_size=8)
    outputs = model(batch)

    for req, output in zip(batch, outputs):
        if req.is_done():
            active_requests.remove(req)
        else:
            req.append(output)

    # 新请求可以随时加入
    if new_request:
        active_requests.add(new_request)
```

**优势**:
- GPU 利用率: 95%+ (vs 60% 传统)
- 吞吐量: 3-5x 提升
- 延迟: 更稳定

---

## 5. 实战指南

### 5.1 训练简化版 Fish Speech

```bash
# Step 1: Prepare data
cd chapters/ch12_fish_speech/code
python prepare_data.py --data-dir ../../data/processed

# Step 2: Train (synthetic data for demo)
python train.py --epochs 20 --batch-size 4

# Step 3: Inference
python inference.py --checkpoint ../checkpoints/fish_speech_final.pt \
                    --text "你好，世界" \
                    --output output_mel.npy
```

### 5.2 使用真实数据训练

**数据要求**:
- 采样率: 16kHz 或 24kHz
- 格式: WAV (16-bit or 24-bit)
- 时长: 3-30 秒
- 标注: 文本 + 语言 + 说话人 ID

**训练配置**:

```yaml
# config.yaml
model:
  vocab_size: 1024
  n_codebooks: 4
  slow_dim: 1024
  fast_dim: 512

training:
  batch_size: 32
  epochs: 100
  lr: 1e-4
  warmup_steps: 1000

data:
  train: data/train.list
  val: data/val.list
  sample_rate: 16000
```

### 5.3 推理优化

```python
# 使用 KV cache
model.eval()
kv_cache = None

tokens = prompt_tokens
for _ in range(max_len):
    logits, kv_cache = model(tokens[-1:], kv_cache)
    next_token = sample(logits[:, -1])
    tokens = cat([tokens, next_token])

# 使用 PagedAttention (需要 vLLM)
from vllm import LLM

llm = LLM(model="fish-speech-s2")
output = llm.generate("你好，世界")
```

---

## 6. 性能对比

### 6.1 简化版 vs 原版

| 指标 | 简化版 (50M) | 原版 (4B) | 差距 |
|------|-------------|----------|------|
| **参数量** | 50M | 4B | 80x |
| **训练数据** | 193 samples | 10M hours | - |
| **语言数** | 2 | 80+ | 40x |
| **RTF (A100)** | ~5.0 | 0.195 (H200) | 25x |
| **音质 (MOS)** | ~2.5 | ~4.2 | - |

### 6.2 与其他模型对比

| 模型 | 参数量 | RTF | 零样本 | 多语言 |
|------|--------|-----|--------|--------|
| Tacotron2 (ch02) | 7.8M | 0.3 | ✗ | ✗ |
| FastSpeech2 (ch04) | 2.3M | 0.05 | ✗ | ✗ |
| VITS (ch05) | 55M | 3.9 | ✗ | ✗ |
| VALL-E (ch07) | 9.5M | 0.5 | ✓ | ✗ |
| GPT-SoVITS (ch09) | 30M | 0.8 | ✓ | ✗ |
| **Fish Speech** | **4B** | **0.195** | **✓** | **✓** |

---

## 7. 延伸阅读

### 必读论文

1. **Fish Speech S2**
   - Paper: arXiv:2603.08823 (2026)
   - Code: https://github.com/fishaudio/fish-speech
   - Demo: https://fish.audio

2. **Dual-AR 相关**
   - SoundStorm: Efficient Decoding for Audio Language Models (Google, 2023)
   - VALL-E X: A Generalist Zero-Shot TTS (Microsoft, 2023)

3. **RL 对齐**
   - GRPO: Group Relative Policy Optimization (DeepSeek, 2024)
   - RLHF: Training language models to follow instructions (OpenAI, 2022)

4. **推理优化**
   - vLLM: Easy, Fast, and Cheap LLM Serving (2023)
   - SGLang: Efficient Execution of Structured Language Model Programs (2023)

### 前置知识

- Transformer: ch04 FastSpeech2
- Codec: ch06 Neural Audio Codec
- AR 建模: ch07 VALL-E
- RL 基础: ch11 MiniMind-O

---

## 8. 思考题

1. **Dual-AR 的设计动机**
   为什么要把语义和声学分开建模？能否用单个 AR 建模所有 codebooks？
   分析 pros and cons。

2. **RL 对齐的必要性**
   监督学习 + 大量数据是否足够？为什么还需要 RL？
   RL 带来了哪些额外的好处？

3. **工业级部署的挑战**
   4B 参数的模型如何部署到移动端？
   有哪些压缩和加速技术？

4. **未来方向**
   Fish Speech 还有哪些改进空间？
   下一个突破口可能在哪里？

---

## 9. 文件清单

```
chapters/ch12_fish_speech/code/
├── model.py          # Dual-AR 架构实现 (436 lines)
├── train.py          # 训练脚本 (235 lines)
├── inference.py      # 推理脚本 (210 lines)
└── README.md         # 本文档

Checkpoints:
└── checkpoints/
    ├── fish_speech_epoch_5.pt
    ├── fish_speech_epoch_10.pt
    └── fish_speech_final.pt
```

---

> Neko 说：Fish Speech 代表了工业级 TTS 的最高水平。
>
> 4B 参数、80+ 语言、RTF 0.195 —— 这不是学术玩具，而是真正能用的产品。
>
> 理解了 Dual-AR 的设计，你就理解了现代 TTS 的工程实践。🐱
