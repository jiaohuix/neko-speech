#!/usr/bin/env python3
"""
模型对比实验脚本
横向对比不同章节的模型性能
"""

import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neko import ModelEvaluator, load_audio, save_audio, set_seed


def compare_models(
    model_configs: list,
    test_text: str,
    output_dir: str,
):
    """对比多个模型

    Args:
        model_configs: 模型配置列表
            [{"name": "tacotron2", "class": Tacotron2, "checkpoint": "..."}, ...]
        test_text: 测试文本
        output_dir: 输出目录
    """
    set_seed(42)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ModelEvaluator(device="cuda" if torch.cuda.is_available() else "cpu")
    device = evaluator.device

    results = []

    for config in model_configs:
        print(f"\n{'='*60}")
        print(f"评估模型: {config['name']}")
        print(f"{'='*60}\n")

        # 加载模型
        model = config["class"](**config.get("kwargs", {})).to(device)

        if "checkpoint" in config:
            ckpt = torch.load(config["checkpoint"], map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"加载 checkpoint: {config['checkpoint']}")

        # 统计参数
        params = evaluator.count_parameters(model)
        print(f"参数量: {params['total_params_millions']:.2f}M")

        # 推理测试
        # TODO: 每章需要实现统一的推理接口
        # wav = model.inference(test_text)
        # save_audio(wav, output_dir / f"{config['name']}_output.wav")

        # 评估推理速度
        # speed = evaluator.evaluate_inference_speed(model, ...)

        result = {
            "name": config["name"],
            "params_millions": params["total_params_millions"],
            # "inference_time_ms": speed["inference_time_mean"] * 1000,
            # "rtf": speed["rtf"],
        }
        results.append(result)

        print(f"结果: {result}")

    # 保存对比结果
    with open(output_dir / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # 打印对比表
    print(f"\n{'='*60}")
    print("模型对比表")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Params (M)':<15} {'Inference (ms)':<20}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<20} {r['params_millions']:<15.2f} {r.get('inference_time_ms', 'N/A')}")

    print(f"\n结果保存在: {output_dir / 'model_comparison.json'}")

    return results


if __name__ == "__main__":
    print("模型对比实验框架已就绪")
    print("每章需要实现统一的推理接口")
    print("\n示例用法:")
    print("""
    from experiments.compare_models import compare_models

    # 加载各章模型
    from chapters.ch02_tacotron.code.model import Tacotron2
    from chapters.ch04_fastspeech.code.model import FastSpeech2

    configs = [
        {"name": "Tacotron2", "class": Tacotron2, "checkpoint": "..."},
        {"name": "FastSpeech2", "class": FastSpeech2, "checkpoint": "..."},
    ]

    compare_models(configs, test_text="你好，我是猫娘。", output_dir="experiments/comparison")
    """)
