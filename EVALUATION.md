# Neko Speech - 评估与 Scaling Law 实验计划

## 评估指标体系

### 1. 重建质量指标

| 指标 | 说明 | 目标值 | 计算方式 |
|------|------|--------|----------|
| **Waveform MSE** | 波形均方误差 | < 0.01 | `mean((y - ŷ)²)` |
| **Mel MSE** | Mel 频谱均方误差 | < 0.1 | `mean((mel - mel̂)²)` |
| **MCD (dB)** | 梅尔倒谱失真 | < 3.0 dB | 标准 MCD 公式 |

### 2. 效率指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **RTF** | 实时因子 | < 1.0（实时） |
| **Inference FPS** | 每秒推理次数 | > 10 |
| **Epoch Time** | 每 epoch 训练时间 | < 5 分钟 |
| **Params (M)** | 参数量（百万） | 记录即可 |

### 3. 训练指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **Final Loss** | 最终损失 | 记录收敛趋势 |
| **Loss Curve** | 损失曲线 | 单调下降 |
| **Convergence Epoch** | 收敛所需 epoch | < 50 |

---

## Scaling Law 实验设计

### 实验 1: 数据量 vs 音质

**目的**：探索数据量与重建质量的关系

**变量**：
- 自变量：训练数据量 [1000, 5000, 10000, 30000] 条
- 因变量：Mel MSE, MCD

**实验流程**：
```python
for data_size in [1000, 5000, 10000, 30000]:
    1. 从 NekoDataset 采样 data_size 条数据
    2. 训练模型（固定超参数，训练 20 epochs）
    3. 在测试集上评估 Mel MSE, MCD
    4. 记录训练时间、参数量
```

**预期结论**：
- 数据量翻倍，MSE 下降 ~X%
- 找到"性价比拐点"（如 5000 条后收益递减）

### 实验 2: 模型参数量 vs 效果

**目的**：探索模型规模与性能的关系

**变量**：
- 自变量：模型参数量（通过调整 hidden_size 控制）
- 因变量：Mel MSE, 推理速度

**实验流程**：
```python
for hidden_size in [128, 256, 512, 1024]:
    1. 创建模型（参数量 = f(hidden_size)）
    2. 用固定数据（5000 条）训练
    3. 评估 Mel MSE + 推理速度
```

**预期结论**：
- 参数量翻倍，MSE 下降 ~Y%
- 推理速度与参数量的关系

### 实验 3: 不同模型对比

**目的**：横向对比各章节模型

**对比维度**：
| 模型 | 参数量 | 训练数据 | Mel MSE | RTF | 特点 |
|------|--------|----------|---------|-----|------|
| Tacotron2 | ?M | 5000 | ? | 0.3x | 自回归基线 |
| WaveNet | ?M | 5000 | ? | 0.25x | 神经声码器 |
| FastSpeech2 | ?M | 5000 | ? | 10x | 并行生成 |
| VITS | ?M | 5000 | ? | 1.0x | 端到端 |

---

## 评估脚本使用

### 单模型评估

```python
from neko.evaluation import ModelEvaluator

evaluator = ModelEvaluator(device="cuda")

# 重建质量
metrics = evaluator.evaluate_reconstruction(original_wav, reconstructed_wav)
print(f"Mel MSE: {metrics['mel_mse']:.4f}")
print(f"MCD: {metrics['mcd_db']:.2f} dB")

# 推理速度
speed = evaluator.evaluate_inference_speed(model, input_data, n_trials=100)
print(f"Inference: {speed['inference_time_mean']*1000:.1f} ms")

# 参数量
params = evaluator.count_parameters(model)
print(f"Params: {params['total_params_millions']:.2f}M")
```

### Scaling Law 实验

```python
from neko.evaluation import run_scaling_experiment

def train_fn(model, data_size):
    # 训练逻辑
    train_loader = create_dataloader(manifest, batch_size=8)
    for epoch in range(20):
        loss = train_one_epoch(model, train_loader)
    return {"final_loss": loss, "epochs": 20}

def eval_fn(model):
    # 评估逻辑
    metrics = evaluator.evaluate_reconstruction(...)
    speed = evaluator.evaluate_inference_speed(model, ...)
    return {**metrics, **speed}

results = run_scaling_experiment(
    model_class=Tacotron2,
    model_kwargs={"n_chars": 100, "n_mels": 80},
    data_sizes=[1000, 5000, 10000],
    train_fn=train_fn,
    eval_fn=eval_fn,
    output_dir="experiments/tacotron2_scaling",
)
```

---

## GPU 时间管理

### 6-8 小时分配

| 阶段 | 时间 | GPU 占用 | 任务 |
|------|------|----------|------|
| 0. 基础设施 | 30 min | 0% | 写工具代码 |
| 1. Ch03 WaveNet | 60 min | 30% | 训练 1 epoch（验证可跑通） |
| 2. Ch04 FastSpeech | 60 min | 30% | 训练 1 epoch |
| 3. Ch05 VITS | 90 min | 40% | 训练 1 epoch |
| 4. Ch06 Codec | 60 min | 30% | 训练 1 epoch |
| 5. Ch07 VALL-E | 60 min | 30% | 训练 1 epoch |
| 6. Ch08 Modern | 60 min | 40% | 训练 1 epoch |
| 7. 评估对比 | 30 min | 20% | 运行评估脚本 |

**总计**：~7 小时，GPU 平均占用 30%（防止过热）

### 训练策略

- **每章只训练 1 epoch**（验证可跑通）
- **不追求收敛**（留给读者自己训练）
- **重点在于**：代码可运行 + 教程完整 + 评估框架

---

## 标准评估流程

每章完成后必须执行：

```bash
# 1. 语法检查
python -m py_compile chapters/chXX_*/code/*.py

# 2. 形状验证
python chapters/chXX_*/code/model.py

# 3. 训练 1 epoch
python chapters/chXX_*/code/train.py --data-dir data/processed --epochs 1

# 4. 推理测试
python chapters/chXX_*/code/inference.py --text "测试" --output test.wav

# 5. 评估
python -c "
from neko.evaluation import ModelEvaluator
import torch
# ... 加载模型和音频
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_reconstruction(...)
print(metrics)
"
```

---

## 预期产出

### 评估报告模板

```markdown
# ChXX 模型评估报告

## 模型信息
- 参数量：X.XXM
- 训练数据：N 条
- 训练时间：T 分钟/epoch

## 重建质量
- Waveform MSE: X.XXXX
- Mel MSE: X.XXXX
- MCD: X.XX dB

## 效率
- 推理时间: X ms/样本
- RTF: X.XX
- 训练速度: X 分钟/epoch

## 对比
| 模型 | Mel MSE | RTF | 参数量 |
|------|---------|-----|--------|
| ChXX | X.XX | X.X | X.XM |
| 基线 | X.XX | X.X | X.XM |

## 结论
- 相比前一章，Mel MSE 下降 X%
- 推理速度提升 X 倍
- 适合场景：...
```

---

*评估框架设计时间: 2026-06-28*
*下一步: 在每个章节中集成评估脚本*
