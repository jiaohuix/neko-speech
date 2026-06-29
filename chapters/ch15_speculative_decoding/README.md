# Speculative Decoding for Speech TTS

> 开创性工作：将 DeepSeek 的投机解码技术引入语音合成领域

---

## 研究动机

### 问题：自回归 TTS 的瓶颈

自回归 TTS 模型（Tacotron2, VALL-E, GPT-SoVITS, Fish Speech）的推理速度慢，因为需要串行生成每个 token：

```
Token 1 → Token 2 → Token 3 → ... → Token N
   │         │         │
   └─────┘─────┘─────┘
      串行，无法并行
```

即使使用 GPU，自回归推理也难以充分利用并行计算能力。

### 灵感：DeepSeek 的投机解码

DeepSeek 最近开源了一种类似投机解码（Speculative Decoding）的技术：

**核心思想**：
1. **Draft Model**（草稿模型）：小模型快速生成 K 个候选 token
2. **Target Model**（目标模型）：大模型并行验证这 K 个 token
3. **Accept/Reject**：接受正确的，拒绝错误的，从错误位置重新生成

**数学原理**：

设 draft model 生成的序列为 $y_1, y_2, ..., y_K$，target model 的条件概率为 $p(y_i | y_{<i})$。

对于每个位置 $i$，以概率 $\min\left(1, \frac{p_{target}(y_i)}{p_{draft}(y_i)}\right)$ 接受 $y_i$。

如果接受，继续验证下一个；如果拒绝，从该位置重新采样。

**理论加速比**：

如果 draft model 的准确率为 $\alpha$，则每次验证可以接受 $\frac{1 - \alpha^K}{1 - \alpha}$ 个 token。

当 $K=5, \alpha=0.8$ 时，每次验证接受 3.4 个 token，理论加速 3.4x。

### 应用到语音 TTS

**关键洞察**：

语音 token 序列有很强的局部相关性（相邻帧相似），这使得投机解码非常适合：

1. **Draft Model**：轻量级非自回归模型（如 FastSpeech2）
   - 并行生成 K 个 mel 帧或 codec tokens
   - 速度快（RTF < 0.1）
   - 准确率中等（α ≈ 0.7-0.8）

2. **Target Model**：高质量自回归模型（如 VALL-E, GPT-SoVITS）
   - 自回归生成，但一次验证 K 个 token
   - 音质好
   - 速度慢（RTF ≈ 0.5-1.0）

3. **验证机制**：
   - 比较 draft 和 target 的概率分布
   - 接受高概率的 token，拒绝低概率的
   - 从拒绝位置重新生成

**预期效果**：

- 理论加速：2-5x
- 音质保持：与 target model 相同（因为最终输出由 target 决定）
- 延迟降低：从 RTF 0.5 降低到 RTF 0.1-0.2

---

## 技术方案

### 方案 1：Codec Token 投机解码

**适用模型**：VALL-E, GPT-SoVITS, Fish Speech

**流程**：
```
Text → Draft Model (FastSpeech2) → K 个候选 codec tokens
                                    ↓
                              Target Model (VALL-E) 验证
                                    ↓
                              接受/拒绝 → 输出 tokens → Codec → Waveform
```

**实现细节**：
- Draft: FastSpeech2，并行生成 K=10 帧
- Target: VALL-E，自回归验证 K 帧
- 验证：比较 codec token 的概率分布
- 加速：每次验证 10 帧，接受 7 帧，加速 7x

### 方案 2：Mel 频谱投机解码

**适用模型**：Tacotron2, VITS

**流程**：
```
Text → Draft Model (FastSpeech2) → K 个候选 mel 帧
                                    ↓
                              Target Model (VITS) 验证
                                    ↓
                              接受/拒绝 → 输出 mel → Vocoder → Waveform
```

**实现细节**：
- Draft: FastSpeech2，并行生成 K=20 帧 mel
- Target: VITS，自回归验证 K 帧
- 验证：比较 mel 频谱的 L1 距离
- 加速：每次验证 20 帧，接受 15 帧，加速 15x

### 方案 3：双流投机解码

**适用模型**：Fish Speech (Dual-AR)

**流程**：
```
Text → Draft (Fast AR, 400M) → K 个 acoustic tokens
                                  ↓
                            Target (Slow AR, 4B) 验证
                                  ↓
                            接受/拒绝 → 输出 tokens → Codec → Waveform
```

**实现细节**：
- Draft: Fast AR（400M 参数），快速生成
- Target: Slow AR（4B 参数），高质量验证
- 验证：比较 semantic token 的概率
- 加速：Fish Speech 本身已经是 Dual-AR，可以进一步优化

---

## 实验设计

### 实验 1：FastSpeech2 + VALL-E

**目标**：验证 Codec Token 投机解码的有效性

**配置**：
- Draft: FastSpeech2 (2.3M params, RTF 0.05)
- Target: VALL-E (9.5M params, RTF 0.5)
- K: [5, 10, 20, 50]
- 数据集: LibriTTS (test-clean)

**指标**：
- RTF (Real-Time Factor)
- 接受率 (Acceptance Rate)
- 音质 (MOS, MCD, F0 RMSE)

**预期结果**：
- K=10: 接受率 70%, RTF 0.15, 加速 3.3x
- K=20: 接受率 60%, RTF 0.12, 加速 4.2x
- K=50: 接受率 50%, RTF 0.10, 加速 5.0x

### 实验 2：Draft Model 质量影响

**目标**：研究 draft model 准确率对加速比的影响

**配置**：
- Draft: FastSpeech2，训练不同 epochs (5, 10, 20, 50)
- Target: VALL-E (固定)
- K=20

**预期结果**：
- Draft 质量越高，接受率越高，加速比越大
- Draft 训练 20 epochs 即可达到最佳性价比

### 实验 3：不同语音模型的适配性

**目标**：验证投机解码在不同 TTS 模型上的效果

**模型**：
- Tacotron2 + FastSpeech2
- VITS + FastSpeech2
- GPT-SoVITS + FastSpeech2

**预期结果**：
- 自回归模型都能受益于投机解码
- 加速比与模型的自回归程度正相关

---

## 创新点

### 1. 首次将投机解码引入语音领域

- DeepSeek 的投机解码用于 LLM 文本生成
- 我们首次将其应用到语音 TTS
- 解决了自回归 TTS 的推理瓶颈

### 2. 语音 token 的特殊性

- 语音 token 有很强的局部相关性
- 相邻帧相似，适合投机解码
- 比文本 token 更容易预测

### 3. 双流架构的天然适配

- Fish Speech 的 Dual-AR 架构天然适合投机解码
- Fast AR 可以作为 Draft，Slow AR 可以作为 Target
- 无需额外训练 draft model

### 4. 理论分析与实验验证

- 推导语音投机解码的理论加速比
- 系统实验验证不同配置的效果
- 提供开源实现和预训练模型

---

## 实现计划

### Phase 1: 基础实现 (Week 1)

1. **实现 Draft Model**
   - 基于 FastSpeech2 的并行生成器
   - 输出 codec tokens 或 mel 帧

2. **实现验证机制**
   - 概率比较（codec tokens）
   - L1 距离（mel 频谱）
   - 接受/拒绝逻辑

3. **实现 Target Model 接口**
   - 支持批量验证 K 个 token
   - 从拒绝位置重新生成

### Phase 2: 训练与优化 (Week 2)

1. **训练 Draft Model**
   - 使用与 target model 相同的数据
   - 训练 20 epochs

2. **优化验证逻辑**
   - 调整接受阈值
   - 优化 K 值

3. **性能测试**
   - RTF 测量
   - 接受率统计

### Phase 3: 实验与分析 (Week 3)

1. **消融实验**
   - 不同 K 值
   - 不同 draft model 质量
   - 不同 target model

2. **对比实验**
   - vs 纯自回归
   - vs 纯并行

3. **音质评估**
   - MOS 主观评分
   - MCD, F0 RMSE 客观指标

### Phase 4: 论文撰写 (Week 4)

1. **撰写章节**
   - ch15_speculative_decoding
   - 包含理论、方法、实验

2. **开源发布**
   - 代码开源
   - 预训练模型
   - 详细文档

---

## 预期贡献

### 学术贡献

1. **开创性研究**：首次将投机解码引入语音 TTS
2. **理论分析**：推导语音投机解码的加速比公式
3. **实验验证**：系统实验证明有效性
4. **开源工具**：提供完整的实现和预训练模型

### 工业价值

1. **推理加速**：2-5x 加速，降低部署成本
2. **实时应用**：使高质量 TTS 达到实时要求
3. **广泛适用**：可应用于多种自回归 TTS 模型

### 教育意义

1. **教学案例**：展示如何将 LLM 技术迁移到语音领域
2. **实践指南**：提供完整的实现和训练代码
3. **研究启发**：启发更多语音+LLM 的交叉研究

---

## 参考文献

1. **DeepSeek Speculative Decoding**
   - DeepSeek-V3 Technical Report, 2024
   - https://github.com/deepseek-ai/DeepSeek-V3

2. **Speculative Decoding (Original)**
   - Fast Inference from Transformers with Speculative Decoding (Google, 2022)
   - arXiv:2211.17192

3. **TTS Models**
   - VALL-E (Microsoft, 2023)
   - GPT-SoVITS (RVC-Boss, 2024)
   - Fish Speech (Fish Audio, 2026)

4. **Draft Models**
   - FastSpeech2 (MSRA, 2020)
   - E2-TTS (Microsoft, 2024)

---

*这是 neko-speech 教材的开创性工作，将为 TTS 推理加速提供新的思路。*
