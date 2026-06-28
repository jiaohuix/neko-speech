"""
Neko Speech - 评估框架
统一的模型评估指标，支持 Scaling Law 研究
"""

import torch
import time
import numpy as np
from pathlib import Path
from typing import Dict, List
from .utils import load_audio, mel_spectrogram


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.metrics = {}

    def evaluate_reconstruction(
        self,
        original_wav: torch.Tensor,
        reconstructed_wav: torch.Tensor,
        sample_rate: int = 16000,
    ) -> Dict[str, float]:
        """评估重建质量

        Args:
            original_wav: 原始波形 [T]
            reconstructed_wav: 重建波形 [T]
            sample_rate: 采样率

        Returns:
            指标字典：MSE, Mel Cepstral Distortion, etc.
        """
        # 确保长度一致
        min_len = min(len(original_wav), len(reconstructed_wav))
        orig = original_wav[:min_len]
        recon = reconstructed_wav[:min_len]

        # 1. 波形 MSE
        mse = torch.mean((orig - recon) ** 2).item()

        # 2. Mel 频谱 MSE
        orig_mel = mel_spectrogram(orig, sample_rate=sample_rate)
        recon_mel = mel_spectrogram(recon, sample_rate=sample_rate)

        # 对齐长度
        min_mel_len = min(orig_mel.shape[-1], recon_mel.shape[-1])
        mel_mse = torch.mean(
            (orig_mel[..., :min_mel_len] - recon_mel[..., :min_mel_len]) ** 2
        ).item()

        # 3. Mel Cepstral Distortion (MCD)
        mcd = self._compute_mcd(orig_mel, recon_mel)

        return {
            "waveform_mse": mse,
            "mel_mse": mel_mse,
            "mcd_db": mcd,
        }

    def _compute_mcd(self, mel1: torch.Tensor, mel2: torch.Tensor) -> float:
        """计算梅尔倒谱失真（MCD）"""
        # 简化版 MCD
        diff = mel1 - mel2
        mcd = torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=0))).item()
        return mcd * 10 / np.log(10)  # 转换为 dB

    def evaluate_inference_speed(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        n_trials: int = 10,
    ) -> Dict[str, float]:
        """评估推理速度

        Args:
            model: 模型
            input_data: 输入数据
            n_trials: 测试次数

        Returns:
            速度指标
        """
        model.eval()
        times = []

        with torch.no_grad():
            for _ in range(n_trials):
                start = time.time()
                _ = model(input_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.time()
                times.append(end - start)

        return {
            "inference_time_mean": np.mean(times),
            "inference_time_std": np.std(times),
            "inference_fps": 1.0 / np.mean(times),  # 每秒推理次数
        }

    def evaluate_training_speed(
        self,
        train_fn,
        n_epochs: int = 3,
    ) -> Dict[str, float]:
        """评估训练速度

        Args:
            train_fn: 训练函数（一个 epoch）
            n_epochs: 测试 epoch 数

        Returns:
            速度指标
        """
        times = []

        for epoch in range(n_epochs):
            start = time.time()
            loss = train_fn()
            end = time.time()
            times.append(end - start)

        return {
            "epoch_time_mean": np.mean(times),
            "epoch_time_std": np.std(times),
            "final_loss": loss,
        }

    def count_parameters(self, model: torch.nn.Module) -> Dict[str, int]:
        """统计模型参数"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "total_params": total,
            "trainable_params": trainable,
            "total_params_millions": total / 1e6,
        }


def run_scaling_experiment(
    model_class,
    model_kwargs: Dict,
    data_sizes: List[int],
    train_fn,
    eval_fn,
    output_dir: str = "experiments/scaling",
):
    """运行 Scaling Law 实验

    Args:
        model_class: 模型类
        model_kwargs: 模型参数
        data_sizes: 数据量列表（如 [1000, 5000, 10000]）
        train_fn: 训练函数（接受 data_size 参数）
        eval_fn: 评估函数
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for data_size in data_sizes:
        print(f"\n{'='*60}")
        print(f"Training with {data_size} samples")
        print(f"{'='*60}")

        # 训练
        model = model_class(**model_kwargs)
        train_metrics = train_fn(model, data_size)

        # 评估
        eval_metrics = eval_fn(model)

        result = {
            "data_size": data_size,
            "params": sum(p.numel() for p in model.parameters()) / 1e6,
            **train_metrics,
            **eval_metrics,
        }
        results.append(result)

        # 保存中间结果
        import json
        with open(output_dir / "scaling_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults for {data_size} samples:")
        for k, v in result.items():
            print(f"  {k}: {v}")

    return results


def print_model_comparison(results: List[Dict], metric_keys: List[str] = None):
    """打印模型对比表

    Args:
        results: 多个模型的评估结果
        metric_keys: 要显示的指标
    """
    if not results:
        return

    if metric_keys is None:
        metric_keys = ["model", "data_size", "mel_mse", "inference_time_mean"]

    # 表头
    header = " | ".join(metric_keys)
    print(header)
    print("-" * len(header))

    # 表格内容
    for result in results:
        row = " | ".join(str(result.get(k, "N/A")) for k in metric_keys)
        print(row)
