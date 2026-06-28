# Neko Speech — 全局架构与开发路线图

> **开发者必读顺序**：`AGENTS.md`（治理原则）→ `ROADMAP.md`（本文档，架构与分工）→ `CONTRIBUTING.md`（章节开发 SOP）

## 1. 项目定位

一本**开源的 Audio AI 教科书**，从零开始用 PyTorch 实现经典语音合成模型。

最终产品：一个可在本地 CPU 运行的猫娘语音助手。

> 原则：每章必须产生**可运行的代码 + 教程文档 + 可视化资产**，不是纯理论。

---

## 2. 已完成内容

| 章节 | 内容 | 状态 | 资产 |
|------|------|------|------|
| **Ch01** | 音频基础：波形、采样、FFT、STFT、Mel、Griffin-Lim | 完成 | 5个演示脚本 + 5张Neko插画 |
| **Ch02** | Tacotron2：端到端TTS | 训练中 | model.py / train.py / inference.py / test_tts.py / eval_reconstruct.py |
| **数据集** | Neko Audio 80K（ModelScope） | 下载中 | ~858条/目标3000+ |

### Ch02 当前状态

- **模型**：26.7M参数，Encoder-Decoder-PostNet结构
- **训练**：20 epoch，batch=8，AMP混合精度，cudnn禁用
- **数据**：16kHz重采样，25秒截断，动态字符表
- **已修复**：24kHz→16kHz重采样、中文tokenizer、loss mask shape、Griffin-Lim vocoder
- **已知限制**：
  - inference decoder 使用 mean pooling 而非真正的 Location-Sensitive Attention（简化版）
  - 数据量仍偏少（~500条有效），TTS音质有限
  - 自回归推理慢，Griffin-Lim音质差

---

## 3. 全局架构

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

## 4. 开发规范

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

## 5. 后续章节分工

| 章节 | 负责人 | 依赖 | 交付物 | 预估难度 |
|------|--------|------|--------|----------|
| **Ch03 WaveNet** | TBD | Ch01音频基础 | 神经声码器，替代Griffin-Lim | 中 |
| **Ch04 FastSpeech2** | TBD | Ch02 Tacotron2 | 非自回归TTS，并行生成 | 中 |
| **Ch05 VITS** | TBD | Ch03+Ch04 | 端到端+flow，音质最佳 | 高 |
| **Ch06 GPT-SoVITS** | TBD | Ch02+Ch03 | Few-shot音色克隆，实用路线 | 高 |
| **Ch07 SoVITS / RVC** | TBD | Ch06 | 歌声转换/实时变声 | 高 |
| **Ch08 现代模型** | TBD | 前面全部 | F5-TTS, CosyVoice 等最新工作 | 高 |
| **Ch09 部署** | TBD | 前面全部 | ONNX导出, CPU推理优化, 本地GUI | 中 |

### 每章开发 Checklist

开发者在开始某一章之前，必须阅读：
1. `AGENTS.md` — 理解项目治理原则
2. `ROADMAP.md` — 理解全局架构和前置依赖
3. 前一章的 `README.md` 和 `code/` — 理解接口约定

每章交付前必须满足：
- [ ] `README.md` 教程完整（原理+代码+习题）
- [ ] `code/` 可独立运行，通过形状验证
- [ ] 如有训练，提供训练日志和 loss 曲线
- [ ] 如有推理，提供示例输出
- [ ] 不提交大文件（checkpoints/ outputs/ 在 .gitignore）
- [ ] 代码通过 `python -m py_compile` 检查

---

## 6. 当前未解决问题

### Ch02 Tacotron2（训练进行中）
- [ ] 20 epoch 训练完成（当前 epoch 13/20）
- [ ] inference decoder 缺少真正的 Location-Sensitive Attention
- [ ] 数据量需扩充到 3000+ 条
- [ ] 需要评估：reconstruction MSE 降到多少算"可听"？

### 基础设施
- [ ] 统一的音频预处理工具（resample + VAD + 截断）
- [ ] 统一的 checkpoint 管理（自动保存 best + 最新）
- [ ] 数据集质量检查脚本（时长分布、采样率一致性）

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
