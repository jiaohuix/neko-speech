# Neko Speech 教科书开发总体规划

## 当前状态

**已完成 (2/8 章)**
- ✅ Ch01 音频基础：波形、FFT、STFT、Mel、Griffin-Lim（5个脚本 + 5张插画）
- 🔄 Ch02 Tacotron2：训练中（epoch 15/20，loss 1.07），预计 15 分钟完成

**待开发 (6 章)**
- Ch03 WaveNet 神经声码器
- Ch04 FastSpeech2 非自回归 TTS
- Ch05 VITS 端到端 TTS
- Ch06 GPT-SoVITS 少样本声音克隆
- Ch07 现代音频模型（F5-TTS、CosyVoice）
- Ch08 部署与本地助手

---

## 章节开发规划

### Phase 1: 基础声码器 (Week 1)

#### Ch03 WaveNet 神经声码器
**目标**: 用神经网络替代 Griffin-Lim，提升音质

**核心概念**:
- 扩张因果卷积 (Dilated Causal Convolution)
- 门控激活单元 (Gated Activation)
- 残差连接 + 跳跃连接
- 条件机制 (Mel 频谱条件)

**交付物**:
```python
chapters/ch03_wavenet/
├── README.md              # 原理 + 公式 + 实验
├── code/
│   ├── model.py           # WaveNet 架构 (<300 行)
│   ├── train.py           # 训练脚本
│   ├── inference.py       # 推理脚本
│   └── utils.py           # 音频工具函数
├── checkpoints/           # .gitignore
└── outputs/               # .gitignore
```

**质量标准**:
- [ ] `model.py` 底部形状验证通过
- [ ] 训练 1 epoch 能跑通
- [ ] README 包含：因果卷积可视化、感受野计算、训练 loss 曲线
- [ ] 提供 2-3 个生成样本对比（Griffin-Lim vs WaveNet）

**论文引用**:
- van den Oord et al. (2016) "WaveNet: A Generative Model for Raw Audio"

**参考实现**:
- [r9y9/wavenet_vocoder](https://r9y9.github.io/wavenet_vocoder/)
- [kan-bayashi/PytorchWaveNetVocoder](https://github.com/kan-bayashi/PytorchWaveNetVocoder)

**预估工作量**: 3-4 天

---

### Phase 2: 并行生成 (Week 2)

#### Ch04 FastSpeech2 非自回归 TTS
**目标**: 用并行解码替代自回归，加速 10-100x

**核心概念**:
- 长度调节器 (Length Regulator)
- 时长预测器 (Duration Predictor)
- 音高/能量预测器 (Variance Adaptors)
- 教师强制 (Teacher Forcing) vs 预测时长

**交付物**:
```python
chapters/ch04_fastspeech/
├── README.md              # 非自回归原理 + 时长预测
├── code/
│   ├── model.py           # FastSpeech2 (<400 行)
│   ├── train.py           # 训练脚本（需要时长标签）
│   ├── inference.py       # 并行推理
│   └── dataset.py         # 时长提取工具
├── checkpoints/
└── outputs/
```

**依赖**: Ch02 Tacotron2（复用 encoder，但 decoder 不同）

**质量标准**:
- [ ] 时长预测器输出与真实时长对齐
- [ ] 推理速度比 Tacotron2 快 10x+
- [ ] README 对比：自回归 vs 非自回归的优缺点

**论文引用**:
- Ren et al. (2020) "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"

**参考实现**:
- [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)（最清晰的教学实现）
- [NVIDIA NeMo FastSpeech2](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/tts_en_fastspeech_2)

**预估工作量**: 4-5 天

---

### Phase 3: 端到端模型 (Week 3-4)

#### Ch05 VITS 端到端 TTS
**目标**: 结合 VAE + Flow + GAN，实现最佳音质

**核心概念**:
- 条件变分自编码器 (Conditional VAE)
- 归一化流 (Normalizing Flow)
- 对抗训练 (Adversarial Training)
- 单调对齐搜索 (Monotonic Alignment Search)

**交付物**:
```python
chapters/ch05_vits/
├── README.md              # VAE + Flow + GAN 三合一原理
├── code/
│   ├── model.py           # VITS 架构 (<500 行，分段实现)
│   ├── modules.py         # Flow/Encoder/Decoder 模块
│   ├── train.py           # 多损失函数训练
│   └── inference.py       # 端到端推理
├── checkpoints/
└── outputs/
```

**质量标准**:
- [ ] Flow 模块可逆变换验证
- [ ] 生成器 + 判别器联合训练稳定
- [ ] README 详细解释 KL 散度 + Flow 损失 + 对抗损失

**论文引用**:
- Kim et al. (2021) "VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech"

**参考实现**:
- [jaywalnut310/vits](https://github.com/jaywalnut310/vits)（官方实现）

**风险**: 本章最复杂，可能需要拆分 week3/week4 两部分

**预估工作量**: 7-10 天

---

### Phase 4: 声音克隆 (Week 5)

#### Ch06 GPT-SoVITS 少样本声音克隆
**目标**: 用 1 分钟数据克隆音色

**核心概念**:
- GPT 风格的自回归模型
- SoVITS 解码器
- 少样本微调 (Few-Shot Fine-Tuning)
- 零样本推理 (Zero-Shot Inference)

**交付物**:
```python
chapters/ch06_gpt_sovits/
├── README.md              # 少样本克隆原理 + 实操指南
├── code/
│   ├── model.py           # GPT-SoVITS 架构
│   ├── fine_tune.py       # 1 分钟数据微调
│   ├── zero_shot.py       # 零样本推理
│   └── extract_features.py # 音色特征提取
├── checkpoints/
└── outputs/
```

**质量标准**:
- [ ] 微调后生成语音与参考音色相似
- [ ] README 提供完整的克隆流程（录音→微调→推理）

**论文/项目**:
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)（官方项目）

**预估工作量**: 5-6 天

---

### Phase 5: 前沿综述 (Week 6)

#### Ch07 现代音频模型
**目标**: 综述 2024-2025 最新进展，提供概念理解 + 最小 demo

**覆盖模型**:
- F5-TTS（Flow Matching + DiT）
- CosyVoice（零样本多语言）
- 其他：VoiceBox, AudioLM 等

**交付物**:
```python
chapters/ch07_modern_models/
├── README.md              # 综述文章（对比表格 + 技术演进图）
├── code/
│   ├── f5_tts_demo.py     # F5-TTS 最小 demo
│   ├── cosyvoice_demo.py  # CosyVoice 零样本 demo
│   └── comparison.py      # 模型对比实验
└── figures/               # 架构图 + 对比图
```

**质量标准**:
- [ ] README 包含技术演进时间线
- [ ] 每个模型提供可运行的 demo
- [ ] 对比表格：参数量、推理速度、音质、零样本能力

**核心论文**:
- F5-TTS (ACL 2025): [arxiv.org/html/2410.06885v3](https://arxiv.org/html/2410.06885v3)
- CosyVoice 2 (2024): [arxiv.org/html/2412.10117v1](https://arxiv.org/html/2412.10117v1)

**参考项目**:
- [swivid/f5-tts](https://github.com/swivid/f5-tts)
- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)

**预估工作量**: 4-5 天

---

### Phase 6: 部署优化 (Week 7)

#### Ch08 部署与本地助手
**目标**: 让猫娘助手在消费级硬件上实时运行

**核心任务**:
- ONNX 导出
- 量化（FP16 → INT8）
- CPU 推理优化
- 简单 Web UI（Gradio/Streamlit）

**交付物**:
```python
chapters/ch08_deployment/
├── README.md              # 部署指南 + 性能基准
├── code/
│   ├── export_onnx.py     # 模型导出
│   ├── quantize.py        # 量化脚本
│   ├── inference_optimized.py # 优化推理
│   └── web_ui.py          # Gradio 界面
├── benchmarks/            # 性能对比数据
└── outputs/
```

**质量标准**:
- [ ] ONNX 模型推理速度 < 实时
- [ ] 量化后音质损失 < 5%（MOS 评估）
- [ ] README 提供完整的部署流程（训练→导出→优化→部署）

**预估工作量**: 4-5 天

---

## Agent 分派策略

### 并行化原则

**独立章节可并行**:
- Ch03 WaveNet 和 Ch04 FastSpeech2 可以并行（无依赖）
- Ch07 现代模型综述可以提前准备（调研性质）

**依赖关系必须串行**:
- Ch05 VITS 依赖 Ch03 + Ch04
- Ch06 GPT-SoVITS 依赖 Ch02 + Ch03
- Ch08 部署依赖所有模型章节

### Agent 分配方案

```
Week 1: 
  Agent A: Ch03 WaveNet（基础声码器）
  Agent B: Ch07 调研（开始收集论文 + 写综述草稿）

Week 2:
  Agent A: Ch04 FastSpeech2（并行生成）
  Agent B: Ch03 Review + 修复

Week 3-4:
  Agent A: Ch05 VITS Part 1（VAE + Flow）
  Agent B: Ch05 VITS Part 2（GAN + 训练）
  Agent C: Ch06 GPT-SoVITS（少样本克隆）

Week 5:
  Agent A: Ch06 Review + 集成测试
  Agent B: Ch07 现代模型（写 demo + 完成综述）

Week 6:
  Agent A: Ch08 部署（ONNX + 优化）
  Agent B: 全局 Review + 文档整合

Week 7:
  全体: 最终测试 + 猫娘助手集成
```

---

## 质量控制标准

### 代码规范（来自 CONTRIBUTING.md）

每章必须：
- [ ] `model.py` 底部形状验证通过
- [ ] 训练脚本跑通至少 1 epoch
- [ ] 无硬编码路径，支持 `--data-dir` 参数
- [ ] checkpoint 包含 `tokenizer_chars`
- [ ] `python -m py_compile` 语法检查通过

### 文档规范

README 必须包含：
- [ ] 导学（为什么学这个？）
- [ ] 原理（核心公式 + 架构图）
- [ ] 动手实验（可复制的命令 + 预期输出）
- [ ] 本章小结（核心贡献表格）
- [ ] 遗留问题（引出下一章）
- [ ] 习题（3-5 道思考题）
- [ ] 参考文献（作者 + 年份 + 标题）

### 猫娘世界观

每章必须：
- [ ] 至少 1 张 Neko Teacher 插画
- [ ] 叙事连贯（"猫娘学会听" → "猫娘学说话" → "猫娘声音更像人"）
- [ ] 最终服务于"本地猫娘助手"目标

---

## 资源汇总

### 核心论文

| 章节 | 论文 | 年份 | 链接 |
|------|------|------|------|
| Ch03 | WaveNet | 2016 | https://arxiv.org/abs/1609.03499 |
| Ch04 | FastSpeech 2 | 2020 | https://arxiv.org/abs/2006.04558 |
| Ch05 | VITS | 2021 | https://arxiv.org/abs/2106.06103 |
| Ch06 | GPT-SoVITS | 2024 | https://github.com/RVC-Boss/GPT-SoVITS |
| Ch07 | F5-TTS | 2024 | https://arxiv.org/abs/2410.06885 |
| Ch07 | CosyVoice 2 | 2024 | https://arxiv.org/abs/2412.10117 |

### 参考实现

| 章节 | 项目 | 链接 |
|------|------|------|
| Ch03 | r9y9/wavenet_vocoder | https://r9y9.github.io/wavenet_vocoder/ |
| Ch04 | ming024/FastSpeech2 | https://github.com/ming024/FastSpeech2 |
| Ch05 | jaywalnut310/vits | https://github.com/jaywalnut310/vits |
| Ch06 | RVC-Boss/GPT-SoVITS | https://github.com/RVC-Boss/GPT-SoVITS |
| Ch07 | swivid/f5-tts | https://github.com/swivid/f5-tts |
| Ch07 | FunAudioLLM/CosyVoice | https://github.com/FunAudioLLM/CosyVoice |

---

## 时间线估算

```
Week 1:  Ch03 WaveNet (4d) + Ch07 调研 (1d)
Week 2:  Ch04 FastSpeech2 (5d) + Ch03 Review (1d)
Week 3:  Ch05 VITS Part 1 (5d)
Week 4:  Ch05 VITS Part 2 (5d) + Ch06 GPT-SoVITS (2d)
Week 5:  Ch06 完成 (4d) + Ch07 现代模型 (5d)
Week 6:  Ch08 部署 (5d) + 全局 Review (2d)
Week 7:  集成测试 + 文档完善 (5d)

总计: ~7 周（1 人全职）/ ~4 周（3 Agent 并行）
```

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Ch05 VITS 过于复杂 | 延期 1-2 周 | 拆分为 2 部分，先实现简化版 |
| 数据量不足 | 音质差 | 优先扩充到 3000+ 条，使用数据增强 |
| 训练不稳定 | 无法收敛 | 每章提供预训练 checkpoint 作为 fallback |
| Agent 协调成本 | 接口不一致 | 每周同步会议，统一代码风格 |

---

## 下一步行动

**立即执行**:
1. ✅ Ch02 训练完成（等待 15 分钟）
2. 🔄 创建 `chapters/ch03_wavenet/` 目录结构
3. 🔄 分派 Agent A 开始 Ch03 WaveNet 实现
4. 🔄 分派 Agent B 开始 Ch07 论文调研

**本周目标**:
- 完成 Ch02 训练 + 评估
- Ch03 WaveNet 跑通第一个 epoch
- Ch07 综述草稿完成 50%

---

*计划制定时间: 2026-06-28*
*下次更新: Week 1 结束时*
