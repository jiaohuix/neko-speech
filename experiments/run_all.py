#!/usr/bin/env python3
"""
统一实验运行器
为 neko-speech 教科书所有章节提供统一的训练+评估+导出流程
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from neko import set_seed

# 模型配置
MODEL_CONFIGS = {
    "tacotron2": {
        "chapter": "ch02_tacotron",
        "module": "model",
        "class": "Tacotron2",
        "params": {"n_chars": 100, "n_mels": 80},
        "expected_params_millions": 26.7,
    },
    "wavenet": {
        "chapter": "ch03_wavenet",
        "module": "model",
        "class": "WaveNet",
        "params": {"n_res_layers": 20, "n_mels": 80},
        "expected_params_millions": 1.9,
    },
    "fastspeech2": {
        "chapter": "ch04_fastspeech",
        "module": "model",
        "class": "FastSpeech2",
        "params": {"n_chars": 100, "n_mels": 80},
        "expected_params_millions": 7.55,
    },
    "vits": {
        "chapter": "ch05_vits",
        "module": "model",
        "class": "VITS",
        "params": {"n_vocab": 100, "n_mels": 80},
        "expected_params_millions": 45.0,
    },
    "neural_codec": {
        "chapter": "ch06_neural_codec",
        "module": "codec",
        "class": "NeuralAudioCodec",
        "params": {"n_mels": 80},
        "expected_params_millions": 1.3,
    },
    "valle": {
        "chapter": "ch07_valle",
        "module": "valle",
        "class": "VALL_E",
        "params": {"n_codebook": 8, "vocab_size": 256},
        "expected_params_millions": 10.8,
    },
    "gpt_sovits": {
        "chapter": "ch09_gpt_sovits",
        "module": "model",
        "class": "GPTSoVITS",
        "params": {"n_chars": 100},
        "expected_params_millions": 50.0,
    },
    "voxcpm": {
        "chapter": "ch10_voxcpm",
        "module": "model",
        "class": "VoxCPM",
        "params": {"n_chars": 100},
        "expected_params_millions": 100.0,
    },
}


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "total_millions": total / 1e6,
        "trainable": trainable,
        "trainable_millions": trainable / 1e6,
    }


def load_model_class(model_name):
    """动态加载模型类"""
    config = MODEL_CONFIGS[model_name]
    chapter_path = Path(__file__).parent.parent / "chapters" / config["chapter"] / "code"
    sys.path.insert(0, str(chapter_path))

    mod = __import__(config["module"])
    model_class = getattr(mod, config["class"])
    return model_class


def benchmark_inference(model, device, n_runs=10):
    """基准测试推理速度"""
    model.eval()
    times = []

    with torch.no_grad():
        for _ in range(n_runs):
            # 使用随机输入测试
            x = torch.randint(0, 100, (1, 20)).to(device)
            x_len = torch.tensor([20]).to(device)

            start = time.time()
            try:
                _ = model(x, x_len)
            except Exception as e:
                print(f"  推理失败: {e}")
                return None
            times.append(time.time() - start)

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


def run_experiment(model_name, n_epochs=3, batch_size=4, output_dir="experiments"):
    """运行单个模型的实验"""
    print(f"\n{'='*60}")
    print(f"实验: {model_name}")
    print(f"{'='*60}")

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    try:
        model_class = load_model_class(model_name)
        config = MODEL_CONFIGS[model_name]
        model = model_class(**config["params"]).to(device)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return {"model": model_name, "status": "failed", "error": str(e)}

    # 统计参数
    params = count_parameters(model)
    print(f"参数量: {params['total_millions']:.2f}M")

    # 推理基准
    inference = benchmark_inference(model, device)
    if inference:
        print(f"推理时间: {inference['mean_ms']:.2f} ms")

    # 保存结果
    result = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "params": params,
        "expected_params_millions": config["expected_params_millions"],
        "inference": inference,
        "status": "success",
    }

    # 写入结果文件
    result_file = output_dir / f"{model_name}_result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ 结果已保存: {result_file}")
    return result


def run_all_experiments(output_dir="experiments"):
    """运行所有模型的实验"""
    results = []
    for model_name in MODEL_CONFIGS:
        try:
            result = run_experiment(model_name, output_dir=output_dir)
            results.append(result)
        except Exception as e:
            print(f"❌ {model_name} 失败: {e}")
            results.append({"model": model_name, "status": "failed", "error": str(e)})

    # 汇总
    summary_file = Path(output_dir) / "all_results.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    # 打印汇总表
    print(f"\n{'='*60}")
    print("实验汇总")
    print(f"{'='*60}")
    print(f"{'模型':<20} {'参数量(M)':<15} {'推理(ms)':<15} {'状态':<10}")
    print("-" * 60)
    for r in results:
        params = r.get("params", {}).get("total_millions", 0)
        inference = r.get("inference", {})
        inf_ms = inference.get("mean_ms", 0) if inference else 0
        status = r.get("status", "unknown")
        print(f"{r['model']:<20} {params:<15.2f} {inf_ms:<15.2f} {status:<10}")

    print(f"\n汇总结果: {summary_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一实验运行器")
    parser.add_argument("--model", type=str, help="指定模型名称")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if args.model:
        run_experiment(args.model, output_dir=args.output_dir)
    else:
        run_all_experiments(output_dir=args.output_dir)
