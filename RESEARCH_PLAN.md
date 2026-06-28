# 深度调研计划：GPT-SoVITS & VoxCPM

## 🎯 目标

基于你本地的 GPT-SoVITS 和 VoxCPM 项目，深入理解核心架构，然后：
1. 从零实现简化版（教学目的）
2. 设计完整训练流程（20 epochs）
3. 实现 ONNX 导出（不依赖 PyTorch）
4. 设计对比实验（Tacotron2 vs FastSpeech2 vs VITS vs GPT-SoVITS vs VoxCPM）
5. 制作小白友好的教程

---

## 📚 第一阶段：深度调研（2-3 小时）

### 1.1 GPT-SoVITS 核心架构调研

#### 需要理解的关键模块：

**A. AR 模型（GPT 部分）**
- 📍 文件: `GPT_SoVITS/AR/models/t2s_model.py`
- 🔍 核心问题:
  - 如何使用 Transformer 做自回归？
  - KV cache 如何工作？
  - 8 个 codebook 如何预测？
  - 与 VALL-E 的关系？
- ✅ 已有发现:
  - embedding_dim=512, hidden_dim=512
  - num_head=8, num_layers=12
  - vocab_size=1025（1024 + EOS）
  - 使用 scaled_dot_product_attention
  - 有 ONNX 版本: `t2s_model_onnx.py`

**B. SoVITS 模型（VITS 改进版）**
- 📍 文件: `GPT_SoVITS/module/models.py`
- 🔍 核心问题:
  - StochasticDurationPredictor 如何实现？
  - Flow 如何用于时长预测？
  - 声音克隆的 embedding 如何注入？
  - 与原版 VITS 的区别？
- ✅ 已有发现:
  - 使用 Flow 做时长预测（不是简单的 CNN）
  - 有 speaker embedding 注入
  - 支持 ONNX 导出: `models_onnx.py`

**C. 声音克隆机制**
- 📍 文件: `GPT_SoVITS/module/quantize.py`
- 🔍 核心问题:
  - ResidualVectorQuantizer 如何工作？
  - 如何用少量数据微调？
  - 零样本 vs 少样本的区别？
- ✅ 待调研

**D. ONNX 导出流程**
- 📍 文件: `GPT_SoVITS/onnx_export.py`
- 🔍 核心问题:
  - 哪些部分可以导出？
  - 如何处理动态长度？
  - 推理速度提升多少？
- ✅ 待调研

---

### 1.2 VoxCPM 核心架构调研

#### 需要理解的关键模块：

**A. Tokenizer-free 设计**
- 📍 文件: `src/voxcpm/model/voxcpm2.py`
- 🔍 核心问题:
  - 不使用离散 token，如何表示音频？
  - 连续表示的维度是多少？
  - 如何与文本对齐？
- ✅ 已有发现:
  - 使用 AudioVAE V2 做编解码
  - feat_dim=64, patch_size=4
  - 使用 ScalarQuantizationLayer（标量量化，不是 RVQ）

**B. 扩散自回归架构**
- 📍 文件: `src/voxcpm/modules/locdit/unified_cfm.py`
- 🔍 核心问题:
  - 条件流匹配（CFM）如何工作？
  - 如何在自回归中做扩散？
  - 推理时需要多少步？
- ✅ 待调研

**C. AudioVAE V2**
- 📍 文件: `src/voxcpm/modules/audiovae/audio_vae_v2.py`
- 🔍 核心问题:
  - 非对称编码/解码设计是什么？
  - 如何实现 16kHz → 48kHz？
  - 与 EnCodec 的区别？
- ✅ 待调研

**D. MiniCPM-4 骨干**
- 📍 文件: `src/voxcpm/modules/minicpm4/`
- 🔍 核心问题:
  - 如何复用 LLM 架构？
  - 参数量多大？
  - 如何支持 30 种语言？
- ✅ 待调研

---

### 1.3 调研输出

完成后生成：
1. **架构对比表**:
   | 特性 | GPT-SoVITS | VoxCPM |
   |------|------------|--------|
   | Tokenizer | RVQ (8层) | Tokenizer-free |
   | 骨干 | Transformer | MiniCPM-4 |
   | 参数量 | ~100M | ~2B |
   | 推理方式 | 自回归 | 扩散自回归 |
   | 声音克隆 | 少样本 | 零样本 |
   | 语言支持 | 5种 | 30种 |

2. **核心代码片段**（每个关键模块）
3. **训练流程图**
4. **ONNX 导出方案**

---

## 🛠️ 第二阶段：从零实现（4-6 小时）

### 2.1 实现简化版 GPT-SoVITS

**目标**: 实现一个教学版，参数量 ~50M（原版的 1/2）

**模块拆分**:
1. `simple_gpt.py`: 简化版 AR 模型
   - num_layers: 6（原版 12）
   - hidden_dim: 256（原版 512）
   - num_codebook: 4（原版 8）
   - 保留 KV cache 机制

2. `simple_sovits.py`: 简化版 SoVITS
   - 去掉 StochasticDurationPredictor（用简单 CNN）
   - 保留 Flow 用于声音克隆
   - 参数量 ~30M

3. `simple_quantizer.py`: 简化版 RVQ
   - 4 层码本（原版 8 层）
   - 每层 512 维（原版 1024）

4. `train_gpt_sovits.py`: 训练脚本
   - 两阶段训练：
     - Stage 1: 训练 AR 模型（GPT）
     - Stage 2: 训练 SoVITS（微调）

5. `export_onnx.py`: ONNX 导出
   - 参考 GPT-SoVITS 的 `onnx_export.py`
   - 导出 AR + SoVITS

---

### 2.2 实现简化版 VoxCPM

**目标**: 实现一个教学版，参数量 ~100M（原版的 1/20）

**模块拆分**:
1. `simple_audio_vae.py`: 简化版 AudioVAE
   - 去掉超分辨率部分
   - 直接输出 16kHz（原版 48kHz）
   - feat_dim: 32（原版 64）

2. `simple_cfm.py`: 简化版条件流匹配
   - 10 步推理（原版 50 步）
   - hidden_dim: 256（原版 1024）
   - num_layers: 4（原版 8）

3. `simple_voxcpm.py`: 简化版主模型
   - 使用小型 Transformer（不是 MiniCPM-4）
   - hidden_dim: 256
   - num_layers: 6
   - 只支持中文 + 英文

4. `train_voxcpm.py`: 训练脚本
   - 端到端训练
   - 使用 flow matching loss

5. `export_onnx.py`: ONNX 导出
   - 导出 AudioVAE + CFM

---

## 🧪 第三阶段：实验设计（2-3 小时）

### 3.1 统一实验设置

**数据**:
- 使用 Neko Audio 数据集（1000 条）
- 统一采样率: 16kHz
- 统一格式: wav + manifest

**训练**:
- 每个模型训练 20 epochs
- 统一 batch_size: 8
- 统一 learning_rate: 1e-4
- 统一 optimizer: AdamW

**评估**:
- 每 5 个 epoch 保存 checkpoint
- 生成音频样本（epoch 5, 10, 15, 20）
- 计算指标:
  - Mel MSE
  - MCD (dB)
  - RTF（推理速度）
  - 训练时间
  - 参数量

---

### 3.2 对比实验

**实验 1: 模型对比**

| 模型 | 参数量 | 数据量 | Epochs | Mel MSE | MCD | RTF | 训练时间 |
|------|--------|--------|--------|---------|-----|-----|----------|
| Tacotron2 | 26.7M | 1000 | 20 | ? | ? | ? | ? |
| FastSpeech2 | 7.55M | 1000 | 20 | ? | ? | ? | ? |
| VITS | ~45M | 1000 | 20 | ? | ? | ? | ? |
| GPT-SoVITS | ~50M | 1000 | 20 | ? | ? | ? | ? |
| VoxCPM | ~100M | 1000 | 20 | ? | ? | ? | ? |

**实验 2: Scaling Law**

固定模型（FastSpeech2），改变数据量:
- 100 条 → 20 epochs
- 500 条 → 20 epochs
- 1000 条 → 20 epochs
- 5000 条 → 20 epochs

记录: Mel MSE vs 数据量曲线

**实验 3: 推理速度对比**

使用 ONNX 模型，在 CPU 上测试:
- 生成 10 秒音频
- 测量推理时间
- 计算 RTF

| 模型 | PyTorch RTF | ONNX RTF | 加速比 |
|------|-------------|----------|--------|
| Tacotron2 | ? | ? | ? |
| FastSpeech2 | ? | ? | ? |
| VITS | ? | ? | ? |
| GPT-SoVITS | ? | ? | ? |
| VoxCPM | ? | ? | ? |

---

### 3.3 可视化

**Loss 曲线**:
- 5 个模型的 loss 曲线对比

**音频样本**:
- 同一个文本，5 个模型的生成结果
-  epoch 5, 10, 15, 20 的音频质量变化

**注意力图**:
- Tacotron2: Attention 矩阵
- FastSpeech2: Duration 预测
- VITS: Alignment 结果
- GPT-SoVITS: AR 的 KV cache
- VoxCPM: CFM 的扩散过程

---

## 📖 第四阶段：教程制作（3-4 小时）

### 4.1 参考 hello-agents 的风格

**关键原则**:
1. **小白友好**: 假设读者只会 Python，不懂深度学习
2. **循序渐进**: 从最简单的开始，逐步增加复杂度
3. **图文并茂**: 每个概念都有图示
4. **代码可运行**: 每个章节都有完整的代码
5. **猫娘世界观**: 贯穿始终

**章节结构**:
```
第 X 章: [模型名称]
├── X.1 为什么需要这个模型？（痛点）
├── X.2 核心思想（直觉解释）
├── X.3 数学原理（公式推导）
├── X.4 代码实现（逐行解释）
├── X.5 训练与评估（实验）
├── X.6 ONNX 导出（部署）
└── X.7 小结与展望
```

---

### 4.2 教程内容规划

**Ch03: Tacotron2（已完成，需要补充实验）**
- 补充 20 epochs 训练结果
- 补充 ONNX 导出
- 补充与 FastSpeech2 的对比

**Ch04: FastSpeech2（已完成，需要补充实验）**
- 补充 20 epochs 训练结果
- 补充 ONNX 导出
- 补充 Scaling Law 实验

**Ch05: VITS（已完成，需要补充实验）**
- 补充 20 epochs 训练结果
- 补充 ONNX 导出
- 补充与 VALL-E 的对比

**Ch06: GPT-SoVITS（新章节）**
- 6.1 为什么需要声音克隆？
- 6.2 GPT-SoVITS 的核心思想
- 6.3 AR 模型（GPT 部分）
  - Transformer 自回归
  - KV cache 加速
  - 8 个 codebook 预测
- 6.4 SoVITS 模型（VITS 改进）
  - StochasticDurationPredictor
  - Flow 用于声音克隆
- 6.5 声音克隆机制
  - 零样本 vs 少样本
  - 微调流程
- 6.6 代码实现
  - 简化版 AR（6 层，256 维）
  - 简化版 SoVITS（去掉随机时长）
  - 简化版 RVQ（4 层）
- 6.7 训练与评估
  - 两阶段训练
  - 声音克隆实验
  - 与 VITS 的对比
- 6.8 ONNX 导出
  - 导出 AR + SoVITS
  - CPU 推理测试
- 6.9 小结与展望

**Ch07: VoxCPM（新章节）**
- 7.1 为什么需要 Tokenizer-free？
- 7.2 VoxCPM 的核心思想
- 7.3 AudioVAE V2
  - 非对称编码/解码
  - 连续表示 vs 离散 token
- 7.4 条件流匹配（CFM）
  - 扩散自回归
  - 10 步推理
- 7.5 MiniCPM-4 骨干
  - 复用 LLM 架构
  - 多语言支持
- 7.6 代码实现
  - 简化版 AudioVAE（32 维）
  - 简化版 CFM（4 层，256 维）
  - 简化版主模型（6 层）
- 7.7 训练与评估
  - 端到端训练
  - 零样本声音克隆
  - 与 GPT-SoVITS 的对比
- 7.8 ONNX 导出
  - 导出 AudioVAE + CFM
  - CPU 推理测试
- 7.9 小结与展望

**Ch08: 综合对比（重构）**
- 8.1 模型演进时间线
- 8.2 架构对比表
- 8.3 实验结果对比
  - 质量对比（Mel MSE, MCD）
  - 速度对比（RTF）
  - 参数量对比
- 8.4 Scaling Law
- 8.5 部署方案
  - PyTorch vs ONNX
  - CPU vs GPU
- 8.6 选择指南
  - 什么场景用什么模型？
- 8.7 未来展望

---

## 🎯 执行计划

### 时间安排（总计 12-16 小时）

**Day 1: 调研 + 设计（4 小时）**
- 1h: GPT-SoVITS 代码阅读
- 1h: VoxCPM 代码阅读
- 1h: 架构对比表
- 1h: 实验设计

**Day 2: 实现 GPT-SoVITS（4 小时）**
- 1h: simple_gpt.py
- 1h: simple_sovits.py
- 1h: train_gpt_sovits.py
- 1h: export_onnx.py

**Day 3: 实现 VoxCPM（4 小时）**
- 1h: simple_audio_vae.py
- 1h: simple_cfm.py
- 1h: simple_voxcpm.py
- 1h: train_voxcpm.py

**Day 4: 训练 + 评估（4 小时）**
- 2h: 训练所有模型（20 epochs）
- 1h: 生成评估报告
- 1h: 可视化

**Day 5: 教程制作（4 小时）**
- 2h: Ch06 GPT-SoVITS 教程
- 2h: Ch07 VoxCPM 教程

---

## ✅ 交付物

1. **代码**:
   - `chapters/ch06_gpt_sovits/code/` - 简化版 GPT-SoVITS
   - `chapters/ch07_voxcpm/code/` - 简化版 VoxCPM
   - 每个模型都有 ONNX 导出

2. **实验**:
   - 5 个模型的完整训练（20 epochs）
   - 对比报告（质量 + 速度 + 参数量）
   - Scaling Law 实验

3. **教程**:
   - Ch06 GPT-SoVITS 完整教程
   - Ch07 VoxCPM 完整教程
   - Ch08 综合对比（重构）

4. **文档**:
   - 架构对比表
   - 训练日志
   - 评估报告

---

## 🔍 关键问题（需要你确认）

1. **模型大小**: 
   - GPT-SoVITS 简化版 ~50M 参数，是否合适？
   - VoxCPM 简化版 ~100M 参数，是否合适？

2. **训练数据**:
   - 使用 1000 条数据，是否足够？
   - 是否需要扩充到 5000 条？

3. **训练时间**:
   - 每个模型 20 epochs，是否足够？
   - 是否需要增加到 50 epochs？

4. **ONNX 导出**:
   - 是否所有模型都需要导出？
   - 还是只导出最常用的 3 个？

5. **教程风格**:
   - 是否需要更多的图示？
   - 是否需要视频教程？

---

*调研计划制定时间: 2026-06-29*  
*预计完成时间: 5 天*
