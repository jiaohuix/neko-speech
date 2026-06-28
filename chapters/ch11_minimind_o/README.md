# Ch11: MiniMind-O — Neko Learns to Listen, See, and Speak

> 前十章，Neko 学会了理解声音（Ch01-05）、压缩声音（Ch06）、用语言模型生成声音（Ch07-10）。
> 但她一直在做一件事：**只听不说，或只说不看**。
>
> 这一章，Neko 要成为真正的**全模态猫娘**——能听、能看、能说、能思考。

## 本章导学

### 为什么需要全模态模型？

回顾前面的模型：

| 模型 | 输入 | 输出 | 限制 |
|------|------|------|------|
| **Tacotron2** (Ch02) | 文本 | Mel频谱 | 单向：只能说，不能听 |
| **VALL-E** (Ch07) | 文本+参考音频 | 音频Token | 单向：只能克隆，不能对话 |
| **GPT-SoVITS** (Ch09) | 文本+参考音频 | 波形 | 仍然是TTS，不是对话 |
| **VoxCPM** (Ch10) | 文本 | 连续潜变量→波形 | 还是单向生成 |

所有这些模型都是**管道**：输入文本，输出音频。但人类的交流不是这样的——

> "你说我听，我说你听" 是**轮替**的，不是**管道**的。

GPT-4o 的突破：一个模型**同时**处理文本、语音、图像输入，**同时**产生文本和流式语音输出。
不是 ASR → LLM → TTS 的级联，而是**一个统一的序列**包含所有模态。

这就是 **Omni Model**（全模态模型）。

### MiniMind-O：1000倍更小的GPT-4o

| 对比 | GPT-4o | MiniMind-O |
|------|--------|-----------|
| 参数 | ~1.8T (估计) | ~0.1B |
| 开源 | 否 | 完全开源 |
| 训练成本 | 数百万美元 | 单卡 RTX 3090, 2小时 |
| 架构 | 未知 | Thinker-Talker |

MiniMind-O 的意义：**你可以在家训练一个能听、能看、能说的全模态模型**。

### 核心直觉：Thinker-Talker

```
你说话 ──► 耳朵 (SenseVoice) ──► 大脑理解 (Thinker) ──► 嘴巴 (Talker) ──► 对方听到
              冻结的编码器           8层Transformer        4层Transformer       Mimi解码器
              提取特征               语义推理              声学渲染              24kHz波形
```

人的大脑也不是一个"管道"。你的听觉皮层处理声音，视觉皮层处理图像，
前额叶做推理，运动皮层控制说话——它们是**分工合作**的。

MiniMind-O 的 Thinker-Talker 就是这个思路：
- **Thinker**（思考者）：理解文本、语音、图像，产生语义表示
- **Talker**（说话者）：把语义表示变成音频编码，产生流式语音

### 学习路线

| 节 | 内容 | 目标 |
|---|------|------|
| 11.1 | 从 TTS 到 Omni | 理解为什么级联 ASR+LLM+TTS 不够好 |
| 11.2 | Thinker-Talker 架构 | 理解语义路径与声学路径的分离 |
| 11.3 | 音频输入：冻结编码器 | 理解 SenseVoice 如何把声音变成特征 |
| 11.4 | 桥接层 | 理解为什么中间层比最终层更适合做条件 |
| 11.5 | 音频输出：Mimi Codec | 理解 8 层码本如何表示声音 |
| 11.6 | 多 Token 预测 (MTP) | 理解如何并行预测所有码本 |
| 11.7 | 序列格式 | 理解文本和 8 路音频如何共存于同一序列 |
| 11.8 | 流式生成 | 理解模型如何在生成未完成时就开始播放 |
| 11.9 | 声音克隆 | 理解参考音频如何控制输出音色 |
| 11.10 | 训练流水线 | 理解增量式能力引入策略 |
| 11.11 | VAD 与打断 | 理解实时交互的工程实现 |
| 11.12 | 从零实现 SimpleOmni | 完整代码走读 |
| 11.13 | 实验与对比 | 训练、评估、与工业模型对比 |

---

<!-- TODO: 以下各节待实现 -->

## 11.1 从 TTS 到 Omni

<!-- TODO: 级联 vs 端到端 omni 的延迟、韵律、情感对比 -->

## 11.2 Thinker-Talker 架构

<!-- TODO: 架构图，参数表，前向传播详解 -->

## 11.3 音频输入：冻结编码器

<!-- TODO: SenseVoice 原理，audio projector，特征注入 -->

## 11.4 桥接层：为什么不用最后一层？

<!-- TODO: 中间层 vs 最终层的 ablation，直觉解释 -->

## 11.5 音频输出：Mimi Codec

<!-- TODO: 连接 Ch06，8层码本，12.5Hz，24kHz -->

## 11.6 多 Token 预测 (MTP)

<!-- TODO: TalkerHead 架构，adapter 设计，参数量分析 -->

## 11.7 序列格式：9 条流的故事

<!-- TODO: 序列布局图，text stream + 8 audio streams，delay pattern -->

## 11.8 流式生成

<!-- TODO: 生成循环，增量解码，streaming playback -->

## 11.9 声音克隆

<!-- TODO: ref_codes + spk_emb，seen vs unseen voices -->

## 11.10 训练流水线

<!-- TODO: T2A → A2A → I2T 三阶段，loss 函数，训练曲线 -->

## 11.11 VAD 与打断 (Barge-In)

<!-- TODO: SileroVAD，RealtimeSession，近双工交互 -->

## 11.12 从零实现 SimpleOmni

<!-- TODO: 完整代码走读，~50M params 教学版 -->

## 11.13 实验与对比

<!-- TODO: CER/WER评估，Talker ablation，与其他 omni 模型对比 -->

---

## 代码

```
ch11_minimind_o/
├── README.md          ← 你在这里
└── code/
    ├── model.py       ← SimpleOmni 教学实现
    ├── train.py       ← 训练脚本 (T2A + A2A)
    ├── inference.py   ← 推理演示 (文本/语音→语音)
    └── export_onnx.py ← ONNX 导出
```

## 参考资料

1. Gong, J. (2026). "MiniMind-O Technical Report: An Open Small-Scale Speech-Native Omni Model." [arXiv:2605.03937](http://arxiv.org/abs/2605.03937)
2. [MiniMind-O GitHub](https://github.com/jingyaogong/minimind-o) (Apache 2.0)
3. [MiniMind LLM](https://github.com/jingyaogong/minimind) — 语言模型基础
4. [Mimi Neural Codec](https://huggingface.co/docs/transformers/model_doc/mimi) — Kyutai 音频编解码器
5. Qwen2.5-Omni Technical Report. [arXiv:2503.20215](https://arxiv.org/abs/2503.20215)

## 前置知识

- **Ch06 (Neural Codec)**: Mimi 就是一种神经编解码器
- **Ch07 (VALL-E)**: 把语音当语言建模的核心思想
- **Ch08 (Modern Models)**: 工业级模型的架构概览
