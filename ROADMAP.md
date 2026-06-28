# Neko Speech — 全局架构与开发路线图

> **开发者必读顺序**：`AGENTS.md`（治理原则）→ `ROADMAP.md`（本文档，架构与分工）→ `CONTRIBUTING.md`（章节开发 SOP）

## 1. 项目定位

一本**开源的 Audio AI 教科书**，从零开始用 PyTorch 实现经典语音合成模型。

最终产品：一个可在本地 CPU 运行的猫娘语音助手。

> 原则：每章必须产生**可运行的代码 + 教程文档 + 可视化资产**，不是纯理论。

---

## 2. 章节总览（12 章）

| 章节 | 内容 | 状态 | 参数量 | 关键特性 |
|------|------|------|--------|----------|
| **Ch01** | 音频基础：波形、采样、FFT、STFT、Mel、Griffin-Lim | ✅ 完成 | — | 5个演示脚本 + 5张Neko插画 |
| **Ch02** | Tacotron2：Seq2Seq + Attention | ✅ 完成 | 26.7M | 自回归基线 |
| **Ch03** | WaveNet：因果卷积声码器 | ✅ 完成 | 1.9M | 感受野 6139 |
| **Ch04** | FastSpeech2：非自回归并行 | ✅ 完成 | 7.55M | 19× 实时 |
| **Ch05** | VITS：VAE+Flow+GAN 端到端 | ✅ 完成 | ~45M | 最复杂章节 |
| **Ch06** | Neural Audio Codec：RVQ-VAE | ✅ 完成 | 1.3M | 8× 时间压缩 |
| **Ch07** | VALL-E：Codec Language Model | ✅ 完成 | 10.8M | 零样本克隆 |
| **Ch08** | Modern Models: F5-TTS / CosyVoice | ✅ 完成 | ~5M | Flow Matching |
| **Ch09** | GPT-SoVITS: 少样本声音克隆 | 🔄 代码完成，README 编写中 | ~50M | AR+VITS 两阶段 |
| **Ch10** | VoxCPM: Tokenizer-free TTS | 🔄 代码完成，README 编写中 | ~100M | 扩散自回归 |
| **Ch11** | MiniMind-O: 全模态语音模型 | 🔄 代码完成，README 编写中 | ~115M | Thinker-Talker |
| **Ch12** | 端侧部署: ONNX / MNN / sherpa-onnx | 🔄 进行中 | — | Android/iOS 推理 |

**状态说明**：✅ 完成 = 代码+文档均完成 | 🔄 进行中 = 代码或文档尚有一部分未完成 | 📋 计划中 = 尚未开始

---

## 3. 项目产出状态

### 教材

| 产出 | 状态 | 说明 |
|------|------|------|
| PDF 教材 | ✅ 已生成 | 114 页，覆盖 Ch01–Ch08 |
| 章节 README | 🔄 编写中 | Ch01–Ch08 完成，Ch09–Ch11 编写中 |

### 部署与推理

| 项目 | 状态 | 说明 |
|------|------|------|
| ONNX 模型转换 | ✅ 完成 | 主要模型已导出 ONNX |
| MNN 模型转换 | ✅ 完成 | 适配移动端推理框架 |
| 推理 Benchmark | ✅ 完成 | 延迟/内存/音质对比测试完成 |
| sherpa-onnx 集成 | 🔄 进行中 | 端侧推理引擎对接 |

### 实验与训练

| 项目 | 状态 | 说明 |
|------|------|------|
| Ch02 Tacotron2 训练 | 🔄 进行中 | 26.7M 参数，AMP FP16 |
| Ch09 GPT-SoVITS 训练 | 🔄 进行中 | AR+VITS 两阶段训练 |
| Ch10 VoxCPM 训练 | 🔄 进行中 | 扩散自回归训练 |
| Ch11 MiniMind-O 训练 | 🔄 进行中 | Thinker-Talker 训练 |
| 数据集 Neko Audio 80K | 🔄 下载中 | ~858 条 / 目标 3000+ |

---

## 4. 全局架构

```
neko-speech/
|
|-- README.md              # 项目入口：Quick Start + 目录结构
|-- AGENTS.md              # 9条治理原则（不可改）
|-- ROADMAP.md             # 本文档：全局架构 + 分工
|-- .gitignore             # 排除大文件（checkpoints/ outputs/ data/）
|-- requirements.txt       # Python依赖
|
|-- data/
|   |-- download_neko_1k.py    # 数据集下载器（ModelScope API）
|   |-- processed/             # 生成的数据（gitignore）
|   |   |-- wavs/
|   |   |-- train.list         # 格式：wav_path|speaker|lang|text
|   |   |-- metadata.csv
|   |   |-- dataset_info.json
|
|-- chapters/
|   |-- ch01_audio_fundamentals/
|   |   |-- README.md          # 章节教程
|   |   |-- code/              # 可运行代码
|   |   |   |-- 01_waveform.py
|   |   |   |-- 02_fft.py
|   |   |   |-- 03_stft.py
|   |   |   |-- 04_mel.py
|   |   |   |-- 05_reconstruct.py
|   |   |-- figures/           # 生成的图（gitignore）
|   |
|   |-- ch02_tacotron/
|   |   |-- README.md
|   |   |-- code/
|   |   |   |-- model.py           # 模型定义
|   |   |   |-- train.py           # 训练脚本
|   |   |   |-- inference.py       # 推理（自回归）
|   |   |   |-- test_tts.py        # 端到端测试
|   |   |   |-- eval_reconstruct.py # 训练集复现评估
|   |   |-- checkpoints/           # 模型保存（gitignore）
|   |   |-- outputs/               # 生成音频（gitignore）
|   |
|   |-- ch07_valle/
|   |   |-- README.md
|   |   |-- code/
|   |   |   |-- codec.py           # 神经音频Codec（VQ encoder-decoder）
|   |   |   |-- valle.py           # VALL-E模型（AR + NAR Transformers）
|   |   |   |-- generate.py        # 零样本推理流水线
|   |   |   |-- train.py           # 两阶段训练脚本
|   |   |-- checkpoints/           # 模型保存（gitignore）
|   |   |-- outputs/               # 生成音频（gitignore）
|   |
|   |-- ch12_deployment/
|       |-- README.md
|       |-- code/
|       |   |-- export_onnx.py     # ONNX 导出
|       |   |-- export_mnn.py      # MNN 转换
|       |   |-- benchmark.py       # 推理性能测试
|       |-- onnx/                  # ONNX 模型（gitignore）
|       |-- mnn/                   # MNN 模型（gitignore）
|
|-- skills/
|   |-- image-gen/             # Neko插画生成
|       |-- gen_ch01_figures.py
|       |-- PROMPTS.md
|       |-- ROLE.md
|       |-- STYLE_GUIDE.md
```

### 每章标准结构

```
chXX_name/
|-- README.md              # 教程文档（必含：原理+公式+代码位置+习题）
|-- code/                  # 可独立运行的代码
|   |-- model.py           # 模型定义（如有）
|   |-- train.py           # 训练脚本（如有）
|   |-- inference.py       # 推理脚本（如有）
|   |-- *.py               # 演示脚本
|-- checkpoints/           # .gitignore
|-- outputs/               # .gitignore
|-- figures/               # .gitignore（如有可视化）
```

---

## 5. 开发规范

### 代码规范
1. **每章独立**：code/ 目录下的脚本可以独立运行，不依赖其他章
2. **最小依赖**：只使用 torch/numpy/soundfile/librosa/scipy/tqdm
3. **自文档化**：变量名清晰，不写废话注释
4. **中文兼容**：tokenizer必须支持中文，字符表从数据动态构建

### 数据规范
- 音频：16kHz, mono, wav格式
- 标注：`wav_path|speaker|language|text`（GPT-SoVITS兼容）
- 数据集目录：`data/processed/`（gitignore）

### 模型规范
- PyTorch原生实现，不依赖外部TTS框架
- 每个模型文件底部包含形状验证测试（`if __name__ == "__main__"`）
- checkpoint 必须包含 tokenizer_chars，确保推理可复现

### 文档规范
- README.md 必须包含：原理讲解、启动命令、预期现象、习题
- 引用论文必须给出作者+年份+标题
- 每章必须有"本章小结"和"遗留问题"

---

## 6. 当前进行中工作

### Ch09–Ch11 README 编写
- [ ] Ch09 GPT-SoVITS README 完成
- [ ] Ch10 VoxCPM README 完成
- [ ] Ch11 MiniMind-O README 完成

### Ch12 部署收尾
- [ ] sherpa-onnx 集成测试通过
- [ ] 端侧推理 demo 完成（Android 或 iOS）

### 训练与数据
- [ ] 数据集扩充至 3000+ 条
- [ ] Ch09/Ch10/Ch11 模型训练收敛
- [ ] PDF 教材更新覆盖 Ch09–Ch11

---

## 7. 评审流程

1. **局部开发**：负责人 fork/branch，按 checklist 开发
2. **提交 PR**：包含 `README.md` + `code/` + 训练/推理结果截图
3. **全局 review**：检查：
   - 是否遵循 AGENTS.md 原则
   - 是否与前后章接口兼容
   - 代码是否自包含、可运行
   - 是否引入不必要的外部依赖
4. **合并后**：更新 ROADMAP.md，标记完成状态

---

## 8. 关键决策记录

| 决策 | 原因 | 时间 |
|------|------|------|
| 使用 PyTorch 而非框架（如 espnet） | 教学目的，理解原理 | 初始 |
| 数据来源用 ModelScope 而非 HuggingFace | 国内网络可达性 | 2025-06-28 |
| 音频统一 16kHz | 兼容 Tacotron2 论文参数 | 2025-06-28 |
| tokenizer 动态构建字符表 | 中文文本覆盖不可预测 | 2025-06-28 |
| AMP FP16 训练 | 加速 ~2x | 2025-06-28 |
| cudnn 禁用（`cudnn.enabled=False`） | RTX 3060 + FP16 存在 cuDNN 兼容性 bug | 2025-06-28 |
| inference 用 mean pooling 简化 attention | 教学简化，但限制 inference 质量 | 2025-06-28 |
