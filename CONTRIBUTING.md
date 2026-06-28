# Neko Speech — 章节开发 SOP

> 本文档是分配给具体章节的开发者的操作手册。
> 开始开发前，确保已阅读：`AGENTS.md`（治理原则）→ `ROADMAP.md`（架构与分工）→ 本文档。

---

## 1. 开发前准备

### 1.1 认领章节

在 `ROADMAP.md` 中找到待开发章节，确认：
- [ ] 前置依赖章节已完成
- [ ] 无其他人正在开发同一章
- [ ] 理解该章在最终产品中的角色（ROADMAP.md "每章标准结构"）

### 1.2 环境准备

```bash
git clone <repo>
cd neko-speech
pip install -r requirements.txt
```

### 1.3 创建分支

```bash
git checkout -b ch03-wavenet   # 示例：ch{编号}-{英文名}
```

---

## 2. 每章标准结构（强制）

```
chXX_name/
|-- README.md              # 教程文档（见第3节规范）
|-- code/                  # 可独立运行的代码
|   |-- model.py           # 模型定义（如有）
|   |-- train.py           # 训练脚本（如有）
|   |-- inference.py       # 推理脚本（如有）
|   |-- *.py               # 演示/工具脚本
|-- checkpoints/           # 模型保存（.gitignore，不入库）
|-- outputs/               # 生成物（.gitignore，不入库）
|-- figures/               # 生成的图（.gitignore，不入库）
|   |-- small/             # 压缩图（~60KB，可入库用于README）
```

**禁止在仓库根目录或跨章节目录放代码。**

---

## 3. README.md 写作规范

README 是本章的核心资产。必须包含以下章节：

### 3.1 必含内容

| 章节 | 要求 |
|------|------|
| **导学** | 为什么学这个？解决什么问题？前置知识是什么？ |
| **原理** | 核心公式（LaTeX）、关键概念解释 |
| **架构/流程** | 模型结构图或数据流图（ASCII 或 Mermaid） |
| **代码位置** | 每个类/函数对应哪个文件 |
| **动手实验** | 可复制的命令行 + 预期输出 |
| **本章小结** | 核心贡献表格 |
| **遗留问题** | 故意留下的"坑"，引出下一章 |
| **习题** | 3-5 道思考题 |
| **参考文献** | 作者+年份+标题 |

### 3.2 写作风格

- 用**第二人称**（"你"），像老师在和学生对话
- 每个代码块后必须有**观察重点**或**预期现象**
- 复杂概念用 **Neko 笔记** 框标注
- 所有图片用相对路径引用 `figures/small/*.jpg`

### 3.3 示例结构

参考现有章节：
- `chapters/ch01_audio_fundamentals/README.md` — 基础科普风格
- `chapters/ch02_tacotron/README.md` — 模型架构风格

---

## 4. 代码规范

### 4.1 基本要求

1. **自包含**：`code/` 目录下的脚本可以独立运行，不依赖其他章
2. **最小依赖**：仅使用 `requirements.txt` 中已列出的包
3. **形状验证**：每个 `model.py` 底部必须有：

```python
if __name__ == "__main__":
    model = MyModel(...)
    x = torch.randn(...)
    y = model(x)
    print(f"input: {x.shape}, output: {y.shape}")
    assert y.shape == expected_shape
```

4. **中文兼容**：tokenizer 必须从数据动态构建字符表，不能硬编码
5. **设备无关**：支持 CPU 和 CUDA 自动切换

### 4.2 checkpoint 格式

所有 checkpoint 必须包含 `tokenizer_chars`，确保推理可复现：

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "tokenizer_chars": tokenizer.chars,
}, path)
```

### 4.3 训练脚本规范

- 必须支持 `--data-dir` 参数指向 `data/processed/`
- 必须记录 loss 到 `checkpoints/loss_log.csv`
- 必须支持从 checkpoint 恢复训练
- 默认 batch_size 不超过 8（考虑低显存用户）

---

## 5. 数据规范

### 5.1 音频格式

- 采样率：16kHz（除非模型要求其他）
- 通道：mono
- 格式：wav
- 时长：训练数据建议 5-20 秒/条

### 5.2 标注格式

GPT-SoVITS 兼容：
```
wav_path|speaker|language|text
```

示例：
```
wavs/000001.wav|neko|zh|你好，我是猫娘。
```

### 5.3 数据存放

- 原始数据：`data/raw/`（.gitignore）
- 处理后：`data/processed/`（.gitignore）
- 下载脚本：`data/download_*.py`（入库）

---

## 6. 开发流程（SOP）

```
Day 1-2: 调研
  ├── 读3-5篇核心论文
  ├── 找到2-3个开源参考实现
  └── 写调研笔记（放 PR 描述里）

Day 3-5: 实现
  ├── 搭模型骨架（model.py + 形状验证）
  ├── 写训练/推理脚本
  └── 在本章 outputs/ 目录做初步实验

Day 6-7: 验证
  ├── 训练至少跑通1个epoch
  ├── 推理脚本能生成有效输出
  └── 写 README.md

Day 8: 提交
  ├── 按下方 Checklist 自查
  ├── 创建 PR
  └── 等待 Review
```

---

## 7. 提交前自查清单

### 7.1 代码检查

```bash
# 1. 语法检查
python -m py_compile chapters/chXX_name/code/*.py

# 2. 形状验证
python chapters/chXX_name/code/model.py

# 3. 训练脚本帮助信息
python chapters/chXX_name/code/train.py --help

# 4. 检查无大文件
find chapters/chXX_name -type f -size +1M | grep -v ".gitignore"
# 预期：无输出（outputs/ checkpoints/ 已在 .gitignore）
```

### 7.2 文档检查

- [ ] README.md 包含"导学、原理、动手实验、小结、习题"
- [ ] 所有代码命令可复制执行
- [ ] 图片引用路径正确（`figures/small/*.jpg`）
- [ ] 公式用 LaTeX 渲染正确
- [ ] 无错别字（至少通读一遍）

### 7.3 Git 检查

- [ ] 分支名格式：`ch{编号}-{英文名}`
- [ ] 无 checkpoint/outputs/data 等大文件混入
- [ ] 提交信息遵循格式：`feat(chXX): 简短描述`

---

## 8. PR 模板

创建 PR 时填写：

```markdown
## 章节

Ch{编号}: {中文名} / {英文名}

## 目标

- 主要教什么？
- 最终能生成/演示什么？

## 已完成

- [ ] model.py + 形状验证
- [ ] train.py（如有训练）
- [ ] inference.py（如有推理）
- [ ] README.md 完整教程
- [ ] 自查清单全部通过

## 实验结果

```bash
# 贴训练日志或推理输出
```

## 截图

（贴 loss 曲线、生成样本、架构图等）

## 依赖变更

（如有新增 pip 包，列出并更新 requirements.txt）

## Review 关注点

（希望 reviewer 重点检查什么）
```

---

## 9. Review 标准

Reviewer（由项目维护者担任）检查：

### 9.1 架构层面
- [ ] 是否与 AGENTS.md 原则冲突？
- [ ] 是否与前后章接口兼容？
- [ ] 是否引入不必要的外部依赖？

### 9.2 代码层面
- [ ] `python -m py_compile` 通过
- [ ] model.py 形状验证通过
- [ ] 训练脚本能跑通至少1个 epoch
- [ ] 无硬编码路径、无魔法数字

### 9.3 文档层面
- [ ] README 结构完整（导学→原理→实验→小结→习题）
- [ ] 所有命令可复制执行
- [ ] 公式和引用正确

### 9.4 合并后
- [ ] 更新 ROADMAP.md 标记完成
- [ ] 更新全局 README.md 进度表

---

## 10. 常见问题

**Q: 我的模型需要新的 pip 包怎么办？**
A: 先在 `requirements.txt` 添加，PR 中说明为什么必需。优先用标准库替代。

**Q: 训练需要大量数据，怎么处理？**
A: 写下载脚本放 `data/` 目录，数据本身不入库。参考 `data/download_neko_1k.py`。

**Q: 我的章需要引用前一章的代码？**
A: 不允许 import 其他章。把需要的工具函数复制到本章 `code/` 中，保持自包含。

**Q: 生成图片太大怎么办？**
A: 原始图放 `figures/`（gitignored），压缩版（~60KB JPEG, 500px宽）放 `figures/small/`（入库）。README 引用 `figures/small/*.jpg`。

---

## 11. 参考示例

| 任务 | 参考文件 |
|------|----------|
| 基础科普章节 | `chapters/ch01_audio_fundamentals/` |
| 模型章节 | `chapters/ch02_tacotron/` |
| 数据下载器 | `data/download_neko_1k.py` |
| 图片生成 | `skills/image-gen/gen_ch01_figures.py` |
| PR 模板 | 见本文档第8节 |

---

*最后更新: 2025-06-28*
