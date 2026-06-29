# TTS 模型演进图谱

> 从 WaveNet 到 Omni 模型的完整技术演进路线

---

## 技术演进时间线

```
2016 ─── WaveNet (ch03)
         │ 神经声码器，自回归波形生成
         │ 创新：因果卷积 + 扩张卷积
         │ 缺点：极慢（RTF > 1000）
         │
         ├──→ 2017 Tacotron (ch02)
         │     │ Seq2Seq + Attention
         │     │ 创新：端到端 TTS，文本→Mel→波形
         │     │ 缺点：自回归慢，attention 不稳定
         │     │
         │     └──→ 2020 FastSpeech2 (ch04)
         │           │ 非自回归，并行生成
         │           │ 创新：Length Regulator + Duration/Pitch/Energy Predictor
         │           │ 优点：63x 加速（RTF 0.05）
         │           │ 缺点：需要 duration label，音质受限
         │           │
         │           └──→ 2021 VITS (ch05)
         │                 │ VAE + Flow + GAN 端到端
         │                 │ 创新：MAS 自动对齐，对抗训练
         │                 │ 优点：端到端，音质好
         │                 │ 缺点：训练复杂，不稳定
         │
         └──→ 2019 Neural Codec (ch06)
               │ RVQ-VAE 音频 tokenize
               │ 创新：将连续波形离散化为 token 序列
               │ 意义：为语言模型方法奠定基础
               │
               └──→ 2023 VALL-E (ch07)
                     │ Codec Language Model
                     │ 创新：TTS as Language Modeling
                     │ 优点：零样本克隆，3s prompt
                     │ 缺点：需要 codec 训练，复杂
                     │
                     ├──→ 2024 GPT-SoVITS (ch09)
                     │     │ AR Transformer + VITS
                     │     │ 创新：单码本语义 token + 少样本克隆
                     │     │ 优点：5s 音频克隆，效果好
                     │     │ 缺点：两阶段训练
                     │
                     ├──→ 2024 IndexTTS (ch08)
                     │     │ 拼音/音调/时长解耦控制
                     │     │ 创新：细粒度韵律控制
                     │     │ 优点：可控性强
                     │     │ 缺点：需要语言学知识
                     │
                     ├──→ 2024 F5-TTS (ch08)
                     │     │ Flow Matching + Transformer
                     │     │ 创新：ODE 采样，10-20 步
                     │     │ 优点：快速采样
                     │     │ 缺点：训练目标复杂
                     │
                     ├──→ 2024 CosyVoice (ch08)
                     │     │ Speaker Encoder + 零样本
                     │     │ 创新：时间池化 speaker embedding
                     │     │ 优点：3s 零样本克隆
                     │     │ 缺点：需要 speaker encoder
                     │
                     ├──→ 2024 VoxCPM (ch10)
                     │     │ Tokenizer-free, 连续隐空间
                     │     │ 创新：CFM/DiT 直接建模连续空间
                     │     │ 优点：无量化误差
                     │     │ 缺点：计算量大
                     │
                     └──→ 2025 MiniMind-O (ch11)
                           │ Omni Thinker-Talker
                           │ 创新：思考-对话架构
                           │ 优点：多模态统一
                           │ 缺点：复杂度高

2026 ─── Fish Speech S2 (ch12) [新增]
         │ Dual-AR (4B slow + 400M fast)
         │ 创新：主从 Transformer + RL 对齐
         │ 优点：80+ 语言，工业级
         │ 缺点：4B 参数，资源需求大
         │
         ├──→ FireRedTTS2 (ch13) [新增]
         │   │ 双流 Transformer，流式对话
         │   │ 创新：12.5Hz 流式，多说话人对话
         │   │ 优点：实时对话
         │   │ 缺点：复杂度高
         │
         └──→ Bert-VITS2 (ch14) [新增]
               │ 多语言 BERT + VITS2
               │ 创新：语言学特征增强
               │ 优点：多语言支持
               │ 缺点：需要 BERT
               │
               └──→ Speculative Decoding for Speech (ch15) [开创性工作]
                     │ 并行生成 + 自回归优化
                     │ 创新：将 DeepSeek 投机解码引入语音
                     │ 优点：理论 3-5x 加速
                     │ 状态：待研究和验证
```

---

## 模型对比矩阵

| 模型 | 章节 | 年份 | 推理方式 | RTF | 参数量 | 零样本 | 核心创新 | 优缺点 |
|------|------|------|----------|-----|--------|--------|----------|--------|
| **WaveNet** | ch03 | 2016 | 自回归 | >1000 | 1.9M | ✗ | 因果卷积 | 音质好，极慢 |
| **Tacotron2** | ch02 | 2017 | 自回归 | 0.3 | 7.8M | ✗ | Seq2Seq | 端到端，attention 不稳定 |
| **FastSpeech2** | ch04 | 2020 | 并行 | 0.05 | 2.3M | ✗ | Length Regulator | 快，需 duration label |
| **VITS** | ch05 | 2021 | 端到端 | 3.9 | 55M | ✗ | VAE+Flow+GAN | 音质好，训练复杂 |
| **Neural Codec** | ch06 | 2022 | - | - | 1.3M | - | RVQ-VAE | 音频 tokenize |
| **VALL-E** | ch07 | 2023 | LM | 0.5 | 9.5M | ✓ | Codec LM | 零样本，需 codec |
| **GPT-SoVITS** | ch09 | 2024 | AR+VITS | 0.8 | 30M | ✓ | 单码本 AR | 少样本克隆 |
| **IndexTTS** | ch08 | 2024 | 并行 | 0.2 | 25M | ✗ | 拼音控制 | 可控性强 |
| **F5-TTS** | ch08 | 2024 | Flow | 1.5 | 40M | ✗ | Flow Matching | 快速采样 |
| **CosyVoice** | ch08 | 2024 | 端到端 | 2.0 | 35M | ✓ | Speaker Enc | 零样本克隆 |
| **VoxCPM** | ch10 | 2024 | CFM | 5.0 | 80M | ✗ | Tokenizer-free | 无量化误差 |
| **MiniMind-O** | ch11 | 2025 | Omni | 3.0 | 100M | ✓ | Thinker-Talker | 多模态统一 |
| **Fish Speech** | ch12 | 2026 | Dual-AR | 0.2 | 4B | ✓ | RL 对齐 | 工业级，资源大 |
| **FireRedTTS** | ch13 | 2026 | 双流 | 0.3 | 2B | ✓ | 流式对话 | 实时对话 |
| **Bert-VITS2** | ch14 | 2026 | 端到端 | 2.5 | 60M | ✗ | BERT+VITS2 | 多语言 |
| **SpecDecode Speech** | ch15 | 2026 | 投机 | **0.1** | 待定 | - | 并行+AR | **开创性工作** |

---

## 技术路线分类

### 1. 自回归 (Autoregressive)
- **代表**: WaveNet, Tacotron2, VALL-E, GPT-SoVITS
- **优点**: 音质好，建模能力强
- **缺点**: 慢（串行生成）
- **演进**: WaveNet → Tacotron2 → VALL-E → GPT-SoVITS

### 2. 非自回归 (Non-Autoregressive)
- **代表**: FastSpeech2, IndexTTS
- **优点**: 快（并行生成）
- **缺点**: 音质受限，需要额外信息（duration）
- **演进**: FastSpeech2 → IndexTTS

### 3. 端到端 (End-to-End)
- **代表**: VITS, CosyVoice, Bert-VITS2
- **优点**: 简洁，统一优化
- **缺点**: 训练复杂，不稳定
- **演进**: VITS → CosyVoice → Bert-VITS2

### 4. 语言模型 (Language Model)
- **代表**: VALL-E, GPT-SoVITS, Fish Speech
- **优点**: 零样本，可复用 LLM 技术
- **缺点**: 需要大量数据，复杂
- **演进**: VALL-E → GPT-SoVITS → Fish Speech

### 5. 流式/对话 (Streaming/Dialogue)
- **代表**: FireRedTTS2
- **优点**: 实时交互
- **缺点**: 复杂度高
- **演进**: FireRedTTS2

### 6. 连续空间 (Continuous Latent)
- **代表**: VoxCPM
- **优点**: 无量化误差
- **缺点**: 计算量大
- **演进**: VoxCPM

### 7. 多模态 (Omni)
- **代表**: MiniMind-O
- **优点**: 统一架构
- **缺点**: 复杂度高
- **演进**: MiniMind-O

---

## 关键技术创新点

### 声学建模
1. **因果卷积** (WaveNet): 保证时序因果性
2. **Attention** (Tacotron2): 自动对齐文本-音频
3. **Flow** (VITS): 可逆变换，提高表达力
4. **Flow Matching** (F5-TTS): ODE 采样，快速

### 韵律控制
1. **Duration Predictor** (FastSpeech2): 时长预测
2. **Pitch/Energy Predictor** (FastSpeech2): 音高/能量
3. **Pinyin Embedding** (IndexTTS): 拼音解耦

### 声音克隆
1. **Speaker Embedding** (CosyVoice): speaker encoder
2. **Prompt Tokens** (VALL-E): 音频 prompt
3. **Few-shot AR** (GPT-SoVITS): 少样本适应

### 加速技术
1. **并行生成** (FastSpeech2): 非自回归
2. **KV Cache** (Fish Speech): LLM 优化
3. **投机解码** (ch15): **开创性工作**

---

## 章节递进关系

```
ch01 音频基础
  │ 提供音频处理基础知识
  ↓
ch02 Tacotron2 (自回归基线)
  │ 建立 Seq2Seq + Attention 基线
  ↓
ch03 WaveNet (声码器改进)
  │ 用神经网络替代 Griffin-Lim
  ↓
ch04 FastSpeech2 (非自回归加速)
  │ 解决自回归慢的问题
  ↓
ch05 VITS (端到端统一)
  │ 统一声学模型和声码器
  ↓
ch06 Neural Codec (音频 tokenize)
  │ 将波形离散化，为 LM 方法铺路
  ↓
ch07 VALL-E (Codec LM)
  │ 用语言模型方法做 TTS
  ↓
ch08 现代模型 (F5-TTS, IndexTTS, CosyVoice)
  │ 展示多种技术路线
  ↓
ch09 GPT-SoVITS (少样本克隆)
  │ 实际应用：声音克隆
  ↓
ch10 VoxCPM (连续空间)
  │ 探索无 tokenize 路线
  ↓
ch11 MiniMind-O (全能模型)
  │ 多模态统一
  ↓
ch12 Fish Speech (工业级)
  │ 大规模应用
  ↓
ch13 FireRedTTS (流式对话)
  │ 实时交互
  ↓
ch14 Bert-VITS2 (多语言)
  │ 跨语言支持
  ↓
ch15 投机解码 (开创性工作)
    │ 将 DeepSeek 技术引入语音
    └→ 未来方向
```

---

## 下一步工作

### 立即完成
1. ✅ ch08 README（IndexTTS, F5-TTS, CosyVoice）
2. ✅ ch12 Fish Speech 章节
3. ✅ ch13 FireRedTTS 章节
4. ✅ ch14 Bert-VITS2 章节

### 开创性工作
1. **研究 DeepSeek 投机解码**
2. **设计语音版投机解码算法**
3. **实现并训练**
4. **实验验证效果**
5. **撰写 ch15 章节**

---

*最后更新: 2026-06-29*
