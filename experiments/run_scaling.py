#!/usr/bin/env python3
"""
Scaling Law 实验脚本
探索数据量、参数量与模型效果的关系
"""

import sys
import json
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neko import create_dataloader, ModelEvaluator, set_seed


def run_scaling_experiment(
    model_class,
    model_kwargs: dict,
    manifest_path: str,
    data_sizes: list,
    output_dir: str,
    n_epochs: int = 5,
    batch_size: int = 4,
):
    """运行 Scaling Law 实验

    Args:
        model_class: 模型类
        model_kwargs: 模型参数
        manifest_path: 数据 manifest 路径
        data_sizes: 数据量列表 [1000, 5000, 10000]
        output_dir: 输出目录
        n_epochs: 训练 epoch 数
        batch_size: 批次大小
    """
    set_seed(42)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ModelEvaluator(device="cuda" if torch.cuda.is_available() else "cpu")
    device = evaluator.device

    results = []

    for data_size in data_sizes:
        print(f"\n{'='*60}")
        print(f"实验: data_size = {data_size}")
        print(f"{'='*60}\n")

        # 创建数据加载器（采样 data_size 条）
        # TODO: 实现数据采样逻辑
        train_loader = create_dataloader(
            manifest_path,
            batch_size=batch_size,
            shuffle=True,
        )

        # 创建模型
        model = model_class(**model_kwargs).to(device)
        params = evaluator.count_parameters(model)

        print(f"模型参数量: {params['total_params_millions']:.2f}M")
        print(f"训练数据: {data_size} 条")
        print(f"批次大小: {batch_size}")

        # 训练（简化版，每章需要自己实现训练逻辑）
        # 这里只是一个框架
        print(f"\n训练 {n_epochs} epochs...")
        train_times = []

        # TODO: 每章需要实现具体的训练循环
        # for epoch in range(n_epochs):
        #     start = time.time()
        #     loss = train_one_epoch(model, train_loader, device)
        #     train_times.append(time.time() - start)
        #     print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}")

        # 评估（需要测试样本）
        # TODO: 实现评估逻辑
        # metrics = evaluator.evaluate_reconstruction(...)

        result = {
            "data_size": data_size,
            "params_millions": params["total_params_millions"],
            # "final_loss": loss,
            # "train_time_per_epoch": np.mean(train_times),
            # **metrics,
        }
        results.append(result)

        # 保存中间结果
        with open(output_dir / "scaling_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n结果: {result}")

    print(f"\n{'='*60}")
    print("Scaling Law 实验完成!")
    print(f"结果保存在: {output_dir / 'scaling_results.json'}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scaling Law 实验")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--manifest", type=str, required=True, help="数据 manifest")
    parser.add_argument(
        "--data-sizes",
        type=int,
        nargs="+",
        default=[1000, 5000, 10000],
        help="数据量列表",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--epochs", type=int, default=5, help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")

    args = parser.parse_args()

    # TODO: 根据 args.model 加载对应的模型类
    # 示例：
    # if args.model == "tacotron2":
    #     from chapters.ch02_tacotron.code.model import Tacotron2
    #     model_class = Tacotron2
    #     model_kwargs = {"n_chars": 100, "n_mels": 80}

    print("Scaling Law 实验框架已就绪")
    print("每章需要实现具体的训练和评估逻辑")
    print(f"模型: {args.model}")
    print(f"数据量: {args.data_sizes}")
