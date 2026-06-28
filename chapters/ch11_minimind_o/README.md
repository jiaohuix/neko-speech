# Ch11: MiniMind-O -- Neko 学会听、看、说：全模态模型的诞生

> 前十章，Neko 学会了理解声音（Ch01-05）、压缩声音（Ch06）、用语言模型生成声音（Ch07-10）。
> 但她一直在做一件事：**只听不说，或只说不看**。
>
> 这一章，Neko 要成为真正的**全模态猫娘** -- 能听、能看、能说、能思考。
> 她不再是一个"管道"，而是一个**完整的对话者**。

---

## 本章导学

### 为什么需要全模态模型？

回顾前面所有模型：

| 模型 | 输入 | 输出 | 限制 |
|------|------|------|------|
| **Tacotron2** (Ch02) | 文本 | Mel频谱 | 只能说，不能听 |
| **VALL-E** (Ch07) | 文本+参考音频 | 音频Token | 只能克隆，不能对话 |
| **GPT-SoVITS** (Ch09) | 文本+参考音频 | 波形 | 仍然是TTS，不是对话 |
| **VoxCPM** (Ch10) | 文本 | 连续潜变量 | 还是单向生成 |

所有模型都是**管道**：输入文本，输出音频。但人类的交流不是管道——

> "你说我听，我说你听" 是**轮替**的，不是**单向**的。

GPT-4o 的突破在于：一个模型**同时**处理文本、语音、图像输入，**同时**产生文本和流式语音输出。
不是 ASR -> LLM -> TTS 的级联，而是**一个统一的序列**包含所有模态。这就是 **Omni Model**。

### MiniMind-O：1000倍更小的 GPT-4o

| 对比 | GPT-4o | Gemini Live | Qwen2.5-Omni | **MiniMind-O** |
|------|--------|-------------|-------------|---------------|
| 参数 | ~1.8T (估计) | ~10B+ | ~7B | **~0.1B** |
| 开源 | 否 | 否 | 部分 | **完全开源** |
| 训练成本 | 数百万美元 | 未知 | 多卡集群 | **单卡 RTX 3090, 2小时** |
| 架构 | 未知 | 未知 | Thinker-Talker | **Thinker-Talker** |
| 消费级GPU可训 | 否 | 否 | 否 | **是 (RTX 3060 12GB)** |

MiniMind-O 的意义：**你可以在家训练一个能听、能看、能说的全模态模型**。
它虽然小，但架构和 Qwen2.5-Omni 相同——都是 Thinker-Talker。

### 核心直觉

```
你说话 --> 耳朵 (SenseVoice) --> 大脑理解 (Thinker) --> 嘴巴 (Talker) --> 对方听到
             冻结的编码器          8层Transformer        4层Transformer       Mimi解码器
             提取特征              语义推理               声学渲染              24kHz波形
```

人的大脑也不是一个"管道"。听觉皮层处理声音，视觉皮层处理图像，前额叶做推理，运动皮层控制说话——它们是**分工合作**的。MiniMind-O 就是这种分工：

- **Thinker**（思考者）：理解文本、语音、图像，产生语义表示
- **Talker**（说话者）：把语义表示变成音频编码，产生流式语音

### 学习路线

| 节 | 内容 | 核心问题 |
|---|------|---------|
| 11.1 | 从 TTS 到 Omni | 为什么级联 ASR+LLM+TTS 不够好？ |
| 11.2 | Thinker-Talker 架构 | 语义路径与声学路径如何分离？ |
| 11.3 | 音频输入：Speech Encoder | 如何把声音变成可理解的特征？ |
| 11.4 | 图像输入：Image Encoder | 图像如何融入语言序列？ |
| 11.5 | 桥接层 | 为什么中间层比最终层更适合做条件？ |
| 11.6 | 多 Token 预测 (MTP) | 如何并行预测所有码本？ |
| 11.7 | 序列格式与延迟模式 | 文本和多路音频如何共存？ |
| 11.8 | 流式生成 | 模型如何在生成未完成时就开始播放？ |
| 11.9 | 训练流水线 | 增量式能力引入策略 |
| 11.10 | 与工业模型对比 | GPT-4o/Gemini/Qwen-Omni 的异同 |
| 11.11 | 代码走读 | ~50M params 教学版完整实现 |

---

## 11.1 从 TTS 到 Omni

### 级联方案的三大问题

传统的语音对话系统是 ASR + LLM + TTS 的级联：

```
用户说话 --> Whisper(ASR) --> 文本 --> GPT(LLM) --> 回复文本 --> CosyVoice(TTS) --> 回复语音
             ~300ms              ~500ms                   ~400ms
                                          总延迟: ~1200ms
```

**问题 1: 延迟叠加**。三个模型串行推理，延迟相加。即使每个模型 300ms，总计也要 1 秒以上。

**问题 2: 信息丢失**。ASR 把声音变成文本，丢失了语调、情感、口音等副语言信息。
你对猫娘说"我很难过"和用哭腔说"我很难过"——ASR 输出相同的文本，但含义完全不同。

**问题 3: 无法端到端优化**。三个模型分别训练，无法联合优化对话质量。

### Omni 方案的突破

```
用户说话 --> Omni Model --> 回复语音
               ~400ms (端到端)
```

Omni 模型把三个步骤统一在一个 Transformer 里。声音不经过文本中间表示，直接作为序列中的"token"参与推理。

Neko 的理解："级联就像翻译三次——日语翻译成中文，中文思考，中文翻译成日语。
Omni 就像直接听懂日语并直接用日语回答。"

---

## 11.2 Thinker-Talker 架构

### 为什么分两个模块？

一个直觉性的问题：为什么不用一个 Transformer 同时处理理解和说话？

答案是**认知负荷冲突**。理解需要你关注"对方说了什么"（语义），说话需要你关注"我该怎么发音"（声学）。
这两个关注点在 Transformer 的注意力机制中是**竞争关系**——同一个注意力矩阵不可能同时优化两种截然不同的目标。

Thinker-Talker 的分离解决了这个问题：

| | Thinker | Talker |
|---|---------|--------|
| **角色** | 语义理解 + 文本生成 | 声学渲染 + 音频生成 |
| **层数** | 6层 (教学版) | 2层 |
| **Hidden** | 512 | 512 |
| **关注** | 文本逻辑、多模态融合 | 码本分布、声学连贯性 |
| **输出** | 文本 logits | 音频码本 logits (x4) |

### 参数分布

```
SimpleOmni (48.62M params):
  Speech Encoder:   4.59M  (9%)   -- 把声音变成特征
  Image Encoder:    4.46M  (9%)   -- 把图像变成特征
  Audio Projector:  0.39M  (1%)   -- 音频特征投影
  Image Projector:  0.39M  (1%)   -- 图像特征投影
  Thinker:         20.19M  (41%)  -- 核心推理引擎
  Talker:          11.58M  (24%)  -- 声学渲染
  Text Head:        3.28M  (7%)   -- 文本输出
  Mel Decoder:      7.01M  (14%)  -- 音频解码
```

Thinker 占 41%，因为它需要最强的语义理解能力。Talker 占 24%，因为声学渲染相对简单。

---

## 11.3 音频输入：Speech Encoder

### 在 MiniMind-O 原版中

原版使用冻结的 **SenseVoice-Small**（234M 参数），由 FunASR 团队开发。
SenseVoice 把 16kHz 的语音波形转换成 512 维的特征序列，然后通过一个 2 层 MLP（Audio Projector）投影到 Thinker 的隐藏维度。

**为什么冻结？** 因为 SenseVoice 已经在海量语音数据上预训练好了，它的特征提取能力比从头训练的编码器强得多。冻结它可以：
1. 保护预训练知识不被覆盖
2. 节省显存和计算（不参与梯度计算）
3. 让 Thinker 专注于学习如何利用这些特征

### 在教学版中

我们实现了 **SimpleSpeechEncoder**（4.59M 参数），一个简化版的 Whisper 风格编码器：

```python
class SimpleSpeechEncoder(nn.Module):
    # Conv1d 前端: 2层, stride 2 = 4x 下采样
    # 4层 Transformer (双向注意力)
    # RMSNorm
    # 输入: (B, 80, T_mel) log-mel 频谱
    # 输出: (B, T_mel//4, 256) 特征序列
```

关键设计：
- **Conv 下采样**：把 mel 帧数缩减 4 倍，降低后续 Transformer 的计算量
- **双向注意力**：语音理解需要上下文（不像文本生成是单向的）
- **可训练**：教学版全部可训练，方便观察端到端学习过程

---

## 11.4 图像输入：Image Encoder

### 在 MiniMind-O 原版中

原版使用冻结的 **SigLIP2 base-p32-256**（94.5M 参数），Google 的视觉语言模型。
把 256x256 的图像分成 32x32 的 patch，产生 64 个 768 维的 patch token，然后通过 Vision Projector 投影到 Thinker 空间。

### 在教学版中

我们实现了 **SimpleImageEncoder**（4.46M 参数），一个简化的 Vision Transformer：

```python
class SimpleImageEncoder(nn.Module):
    # Conv2d patch embedding: 16x16 patch -> 256-d
    # 可学习位置编码
    # 4层 Transformer (双向注意力)
    # 输入: (B, 3, 256, 256) 图像
    # 输出: (B, 256, 256) -- 256个patch token
```

图像特征被投影到 Thinker 空间后，**直接拼接到文本序列前面**。Thinker 的因果注意力自然地将图像信息融入文本理解。

Neko 的理解："就像猫娘看到一条鱼——图像信息和'鱼'这个概念在大脑里融合。
不需要先把图像翻译成文字再理解。"

---

## 11.5 桥接层：为什么不用最后一层？

这是 MiniMind-O 架构中**最反直觉**的设计：Talker 不是从 Thinker 的最后一层获取信息，而是从**中间层**。

```
Thinker 层:    0    1    2    3    4    5
                    ^
                    |
              bridge_layer = 2  (6//2 - 1)
                    |
                    v
               Talker 输入
```

### 为什么？

| 层 | 特征 | 适合做条件？ |
|----|------|------------|
| Embedding层 (0) | 几乎没有语义信息 | 不适合 |
| 中间层 (2-3) | 丰富的上下文+跨模态融合 | **最适合** |
| 最终层 (5) | 过度特化为next-token预测 | 不太适合 |

**直觉解释**：Thinker 的最终层已经被"训练"成专门预测下一个文本 token。它丢弃了很多声学相关的信息（因为文本预测不需要这些）。
中间层保留了更"通用"的多模态表示，既有足够的语义理解，又保留了声学渲染所需的信息。

这个发现来自 Qwen2.5-Omni 团队的消融实验，MiniMind-O 沿用了相同的策略。

### 代码实现

```python
# 在 Thinker 的 forward() 中:
for i, layer in enumerate(self.layers):
    h = layer(h)
    if i == self.config.bridge_layer:  # 默认 = num_layers // 2 - 1
        bridge = h                      # 捕获中间层输出

# 在 Talker 中:
text_cond = self.embed_proj(bridge) * self.text_scale
```

`text_scale` 是一个可学习的标量（初始值 3.0），控制语义条件的强度。

---

## 11.6 多 Token 预测 (MTP)

### 问题：4 个码本，4 套 logits

Talker 需要同时预测 4 个码本（原版是 8 个），每个码本有 2082 个可能的值。
如果为每个码本单独一个输出头，参数量会乘以 4 倍。

### 解决方案：共享基础 + 低秩适配器

```
                Talker 隐藏状态 (B, T, 512)
                     |
         +-----------+-----------+
         |                       |
    base_linear(512->2082)    adapter_i: Linear(512->128) -> GELU -> Linear(128->2082)
         |                       |
    base_logits            adapter_i_output
         |                       |
         +---------- + ----------+
                     |
              logits_i = base + adapter_i
```

- **base_linear**: 捕获所有码本的共性分布（比如，某些码本值在任何码本中都很少出现）
- **adapter_i**: 捕获第 i 个码本的特殊性（比如，码本 0 侧重粗糙特征，码本 3 侧重细节）

**参数量对比**：
- 4 个独立头：4 x (512 x 2082) = 4.26M
- MTP (base + 4 adapters)：512 x 2082 + 4 x (512 x 128 + 128 x 2082) = 1.07M + 1.33M = 2.40M
- **节省 44% 参数**

### TalkerEmbedding：镜像设计

输入端也有对称设计——把多码本的 audio IDs 融合成单个 embedding：

```python
# 每个码本有自己的 embedding adapter
output = mean(base_embed(ids_i) + adapter_i(ids_i))  for i in range(num_codebooks)
```

---

## 11.7 序列格式与延迟模式

### 多流序列布局

SimpleOmni 的序列包含文本流和多路音频流：

```
Text stream:   [BOS][user_text...][assistant_text...][EOS][pad...]
Audio CB-0:    [pad][pad...][code_0_t0][code_0_t1][code_0_t2]...[stop][pad]
Audio CB-1:    [pad][pad...][pad][code_1_t0][code_1_t1][code_1_t2]...[stop][pad]
Audio CB-2:    [pad][pad...][pad][pad][code_2_t0][code_2_t1]...[stop][pad]
Audio CB-3:    [pad][pad...][pad][pad][pad][code_3_t0][code_3_t1]...[stop][pad]
```

注意音频流是**交错延迟**的：
- CB-0 在 step 0 开始生成
- CB-1 在 step 1 开始（延迟 1 步）
- CB-2 在 step 2 开始（延迟 2 步）
- CB-3 在 step 3 开始（延迟 3 步）

### 为什么需要延迟？

这是流式生成的关键。如果所有码本同时开始生成，你需要等所有码本都完成后才能解码第一帧音频。
延迟模式让你可以在 step 3（= num_codebooks - 1）时就拥有所有 4 个码本的第一帧代码，可以立即开始解码播放。

```
延迟模式时间线 (4 码本):
Step 0: CB-0 产生 code    -> 等待
Step 1: CB-1 产生 code    -> 等待
Step 2: CB-2 产生 code    -> 等待
Step 3: CB-3 产生 code    -> FRAME 0 完整! 解码播放
Step 4: CB-0 产生 code    -> 等待
Step 5: CB-1 产生 code    -> 等待
Step 6: CB-2 产生 code    -> 等待
Step 7: CB-3 产生 code    -> FRAME 1 完整! 解码播放
...
```

**首帧延迟** = num_codebooks / 文本生成速度
对于 4 码本、20 tokens/秒：4/20 = **0.2 秒**。

---

## 11.8 流式生成

### 生成循环

```python
for step in range(max_new_tokens):
    # 1. Thinker 生成文本 token (总是比 Talker 领先一步)
    text_token = sample(text_logits)

    # 2. Talker 按延迟模式生成音频码
    for cb in range(num_codebooks):
        if step >= cb:
            code_i = sample(audio_logits[cb])

    # 3. 当所有码本都产生了至少 1 帧，开始播放
    if step >= num_codebooks - 1:
        frame = read_frame(audio_codes, step)
        mel_chunk = mel_decoder.decode(frame)
        play(mel_chunk)  # 流式播放!
```

### Neko 的实时对话

想象猫娘在和你实时对话：
1. 你说完一句话
2. Neko 的 Speech Encoder 把你的声音编码成特征
3. Thinker 理解你的话，开始生成回复文本
4. Thinker 生成第一个文本 token 的同时，Talker 开始生成音频码
5. 4 步之后（~0.2 秒），第一帧音频开始播放
6. Neko 一边"思考"后面的话，一边"说"前面的话

这就是 Omni 模型的魔力——**思考和说话是并行的**。

---

## 11.9 训练流水线

### 原版：四阶段训练

MiniMind-O 的训练分四个阶段，逐步引入新能力：

| 阶段 | 名称 | 数据 | 目标 | LR |
|------|------|------|------|-----|
| 1 | T2A | 1636h 文本->语音 | 对齐文本和语音输出 | 5e-4 |
| 2 | A2A-P1 | 1712h 语音->语音 | 只训练音频投影层 | 5e-4 |
| 3 | A2A-P2 | 同上 | 全模型微调 | 2e-5 |
| 4 | I2T | 图像指令数据 | 添加视觉理解 | 5e-4 |

**为什么分阶段？** 如果一次性训练所有能力，新能力（比如语音输入）会覆盖旧能力（比如文本生成）。
分阶段引入让模型在掌握一个能力后再学下一个。

### 教学版：两阶段训练

```bash
# Stage 1: T2A - "教 Neko 说话"
python train.py --stage t2a --learning_rate 5e-4 --max_seq_len 64

# Stage 2: A2A - "教 Neko 听并回答"
python train.py --stage a2a --learning_rate 2e-5 --max_seq_len 96
```

### 损失函数

```python
# 文本损失: Thinker 的标准交叉熵
text_loss = CrossEntropy(thinker_logits, text_labels)  # ignore_index=-100

# 音频损失: 每个码本的交叉熵，停止 token 权重 10x
audio_loss = mean(CrossEntropy(audio_logits_i, audio_labels_i) * stop_weight)

# 总损失
total_loss = text_loss + audio_loss
```

**停止 token 加权**：停止 token 非常稀少（每段话只有 1 个），但至关重要。
如果不加权，模型学不会何时停止说话——它会一直生成无意义的音频码。10 倍权重是实验确定的经验值。

---

## 11.10 与工业模型对比

### 架构对比

| 特性 | GPT-4o | Gemini Live | Qwen2.5-Omni | Moshi | MiniMind-O |
|------|--------|-------------|-------------|-------|------------|
| 规模 | ~1.8T | ~10B+ | ~7B | ~7B | ~0.1B |
| 语音输入 | 原生 | 原生 | SenseVoice | 自定义ASR | SenseVoice |
| 语音输出 | 流式 | 流式 | 流式 | 流式 | 流式 |
| 视觉 | 有 | 有 | 有 | 无 | 有 (SigLIP2) |
| 音频编解码 | 未知 | 未知 | CosyVoice | 自定义 | Mimi |
| 核心架构 | 未知 | 未知 | Thinker-Talker | Inner Monologue | Thinker-Talker |
| 开源 | 否 | 否 | 部分 | 是 | 完全 |
| 可在家训练 | 否 | 否 | 否 | 需多卡 | **是 (单卡)** |

### Omni 模型的设计空间

```
              理解能力
                ^
                |
    GPT-4o  *   |   *  Gemini Live
                |
                |      *  Qwen2.5-Omni
                |
                |   *  Moshi
                |
                |           *  Mini-Omni2
                |
                |              *  MiniMind-O  <-- 我们的焦点
                |
                +---------------------------------> 参数量
                0.1B    1B     7B     70B    1T+
```

MiniMind-O 的宣言："我不能像 GPT-4o 那样推理，但你可以**在自己的笔记本上从头训练我**，
并且理解我的**每一行代码**。"

### Thinker-Talker 在不同模型中的变体

| 模型 | Thinker 实现 | Talker 实现 | Bridge 方式 |
|------|-------------|-------------|-------------|
| Qwen2.5-Omni | Qwen2.5 7B | 独立小 Transformer | 中间层 hidden states |
| MiniMind-O | MiniMind 0.1B | 独立小 Transformer | 中间层 hidden states |
| Moshi | 共享 Transformer | Inner Monologue (同一序列) | 无分离 |

Qwen2.5-Omni 和 MiniMind-O 使用了几乎相同的 Thinker-Talker 范式——区别只在规模。
这意味着你在这里学到的架构知识可以**直接迁移**到理解工业级模型。

---

## 11.11 代码走读

### 项目结构

```
ch11_minimind_o/
  code/
    model.py        -- SimpleOmni 完整实现 (~48.6M params)
    train.py        -- 两阶段训练脚本
    inference.py    -- 推理演示 (T2A, A2T, I2A, Streaming)
    export_onnx.py  -- ONNX 导出
```

### model.py 核心类

```
SimpleOmniConfig         -- 所有超参数
SimpleSpeechEncoder      -- Whisper-like 语音编码器 (4层, hidden=256)
SimpleImageEncoder       -- 简化 ViT 图像编码器 (4层, patch=16)
SimpleThinker            -- 6层因果 Transformer (hidden=512)
SimpleTalker             -- 2层 Transformer + MTP Head
SimpleTalkerHead         -- 共享base + 4个低秩adapter
SimpleTalkerEmbed        -- 多码本输入融合
SimpleMelDecoder         -- 码本 -> mel 频谱解码器
SimpleOmni               -- 顶层模型：forward() + generate() + stream_generate()
```

### 一次完整的前向传播

```python
# 1. 编码语音输入
audio_feat = model.encode_speech(mel_input)    # (B, T_a, hidden)

# 2. 编码图像输入
image_feat = model.encode_image(image_input)   # (B, 256, hidden)

# 3. 拼接多模态特征 + 文本 embedding
h = cat([audio_feat, image_feat, text_emb])    # (B, T_full, hidden)

# 4. Thinker 处理 (因果注意力)
h_final, bridge, _ = model.thinker(input_ids, h)

# 5. Talker 结合 bridge + 音频码历史
text_cond = talker.embed_proj(bridge) * talker.text_scale
audio_cond = talker.codec_proj(audio_emb) * talker.audio_scale
h_talker = text_cond + audio_cond

# 6. 输出
text_logits = text_head(h_final)               # (B, T, vocab_size)
audio_logits = talker.lm_head(h_talker)        # list of 4 x (B, T, 2082)
```

### 训练命令

```bash
# 快速验证 (模拟数据)
python train.py --stage t2a --steps_per_epoch 100 --batch_size 4

# 检查推理
python inference.py --mode t2a --weight checkpoints/t2a_epoch1.pth --text "Hello Neko"
python inference.py --mode stream --weight checkpoints/t2a_epoch1.pth
python inference.py --mode delay_pattern
```

---

## 练习

### 概念练习

1. **桥接层消融**：把 bridge_layer 从默认值（中间层）改为第 0 层和最后一层。
   比较生成质量。你能观察到什么差异？为什么中间层效果最好？

2. **码本数量实验**：分别用 1、2、4 个码本训练模型。
   比较音频质量（可以用 mel loss 衡量）和训练时间。码本数量的边际收益在哪里？

3. **延迟模式可视化**：运行 `python inference.py --mode delay_pattern`，
   画出每个码本开始生成的时间线。如果去掉延迟（所有码本从 step 0 开始），
   首帧延迟会变成多少？

### 编程练习

4. **添加 KV-Cache**：当前 `generate()` 每一步都重新计算整个序列。
   实现 KV-Cache 让推理只计算新 token，加速生成。
   提示：`Attention` 类已经有 `past_kv` 和 `use_cache` 参数。

5. **实现 Voice Cloning**：在 `forward()` 中注入 `spk_emb`（speaker embedding），
   让 Talker 根据说话人特征控制输出音色。
   提示：`talker.spk_proj` 已经实现了投影层。

6. **替换 Mel Decoder 为 Mimi**：用 HuggingFace 的 Mimi 模型替换 `SimpleMelDecoder`，
   实现真正的 24kHz 波形输出。参考：`transformers.MimiModel`。

### 研究练习

7. **对比 Qwen2.5-Omni**：阅读 Qwen2.5-Omni 技术报告 (arXiv:2503.20215)。
   列出它和 MiniMind-O 在 Thinker-Talker 设计上的 3 个相同点和 3 个不同点。

8. **Scaling Law**：用 hidden_size=256、384、512、768 分别训练 SimpleOmni。
   画出验证集 loss vs. 参数量的曲线。在哪个规模开始出现收益递减？

---

## 前置知识

- **Ch01 (Audio Fundamentals)**：采样率、STFT、Mel 频谱 —— Speech Encoder 的输入
- **Ch06 (Neural Codec)**：RVQ、码本 —— Mimi 编解码器的核心思想
- **Ch07 (VALL-E)**：把语音当语言建模 —— Omni 模型的基石
- **Ch08 (Modern Models)**：Transformer、流式生成 —— 工业模型概览
- **Ch09 (GPT-SoVITS)**：声音克隆 —— In-context voice cloning 的基础

---

## 参考资料

1. Gong, J. (2026). "MiniMind-O Technical Report: An Open Small-Scale Speech-Native Omni Model."
   [arXiv:2605.03937](http://arxiv.org/abs/2605.03937)
2. [MiniMind-O GitHub](https://github.com/jingyaogong/minimind-o) (Apache 2.0)
3. [MiniMind LLM](https://github.com/jingyaogong/minimind) -- 语言模型基础
4. [Mimi Neural Codec](https://huggingface.co/docs/transformers/model_doc/mimi) -- Kyutai 音频编解码器
5. Qwen2.5-Omni Technical Report. [arXiv:2503.20215](https://arxiv.org/abs/2503.20215)
6. Moshi. [GitHub](https://github.com/kyutai-labs/moshi) -- Inner Monologue omni model
7. Vaswani et al. (2017). "Attention Is All You Need." [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
8. Wang et al. (2023). "VALL-E: Neural Codec Language Models are Zero-Shot TTS."
   [arXiv:2301.02111](https://arxiv.org/abs/2301.02111)
