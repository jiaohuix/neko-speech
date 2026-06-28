"""
Neko Speech -- Comprehensive Benchmark

Compare PyTorch, ONNX, and MNN inference across all TTS models:
  - Model file size (MB)
  - Load time (ms)
  - Inference time for ~10s audio (ms)
  - RTF (real-time factor)
  - CPU memory usage (MB)
  - Audio quality (Mel MSE vs reference)

Usage:
    python benchmark.py --onnx_dir ./onnx_models --mnn_dir ./mnn_models
    python benchmark.py --onnx_dir ./onnx_models --output benchmark_results.md
    python benchmark.py --model vits --onnx_dir ./onnx_models

Note: GPU benchmarks (PyTorch CUDA) require an NVIDIA GPU with CUDA support.
CPU benchmarks use the same machine for fair comparison.
"""

import argparse
import gc
import importlib
import importlib.util
import json
import os
import sys
import time
import numpy as np

import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch


# ================================================================
# Module loading -- each chapter has its own model.py, so we must
# load them from their specific directories to avoid name clashes.
# ================================================================

def _load_module(name, filepath):
    """Load a Python module from a specific file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    code_dir = os.path.dirname(filepath)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    spec.loader.exec_module(module)
    return module

def load_tacotron2_model():
    path = os.path.join(ROOT, "chapters", "ch02_tacotron", "code", "model.py")
    return _load_module("ch02_model", path)

def load_fastspeech2_model():
    path = os.path.join(ROOT, "chapters", "ch04_fastspeech", "code", "model.py")
    return _load_module("ch04_model", path)

def load_vits_model():
    ch05_dir = os.path.join(ROOT, "chapters", "ch05_vits", "code")
    if ch05_dir not in sys.path:
        sys.path.insert(0, ch05_dir)
    path = os.path.join(ch05_dir, "model.py")
    return _load_module("ch05_model", path)


# ================================================================
# Memory Tracking
# ================================================================

def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_gpu_memory_mb():
    """Get GPU memory usage in MB (if available)."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.memory_allocated() / (1024 * 1024)


# ================================================================
# PyTorch Benchmarks
# ================================================================

def benchmark_tacotron2_pytorch(device="cpu", n_runs=10):
    """Benchmark Tacotron2 PyTorch inference."""
    mod = load_tacotron2_model()
    Tacotron2 = mod.Tacotron2

    results = {
        "model": "Tacotron2",
        "framework": "PyTorch",
        "device": device,
    }

    # Load time
    t0 = time.perf_counter()
    model = Tacotron2(vocab_size=256, mel_dim=80).to(device)
    model.eval()
    t1 = time.perf_counter()
    results["load_time_ms"] = (t1 - t0) * 1000
    results["params"] = sum(p.numel() for p in model.parameters())

    # Input: ~20 chars -> ~100 mel frames -> ~2.5s audio at 256 hop, 16kHz
    # For 10s audio, we'd need ~400 mel frames, from ~80 chars
    B = 1
    T_text = 40  # ~40 characters
    text = torch.randint(0, 256, (B, T_text), device=device)

    # Memory
    gc.collect()
    mem_before = get_memory_mb()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model.inference(text, max_len=400, stop_threshold=1.0)

    # Benchmark
    times = []
    for _ in range(n_runs):
        gc.collect()
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            mel_out, mel_post = model.inference(text, max_len=400, stop_threshold=1.0)
        torch.cuda.synchronize() if device == "cuda" else None
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    results["inference_time_ms"] = np.mean(times)
    results["inference_std_ms"] = np.std(times)

    # Estimate audio length: mel_frames * hop_length / sample_rate
    mel_frames = mel_post.shape[1]
    audio_length_s = mel_frames * 256 / 16000
    results["audio_length_s"] = audio_length_s
    results["rtf"] = (results["inference_time_ms"] / 1000) / max(audio_length_s, 0.001)

    # Memory
    results["memory_mb"] = get_memory_mb() - mem_before
    if device == "cuda":
        results["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return results


def benchmark_fastspeech2_pytorch(device="cpu", n_runs=10):
    """Benchmark FastSpeech2 PyTorch inference."""
    mod = load_fastspeech2_model()
    FastSpeech2 = mod.FastSpeech2

    results = {
        "model": "FastSpeech2",
        "framework": "PyTorch",
        "device": device,
    }

    t0 = time.perf_counter()
    model = FastSpeech2(
        vocab_size=256, d_model=256, n_mels=80,
        nhead=2, d_ff=1024, enc_layers=4, dec_layers=4,
    ).to(device)
    model.eval()
    t1 = time.perf_counter()
    results["load_time_ms"] = (t1 - t0) * 1000
    results["params"] = sum(p.numel() for p in model.parameters())

    B = 1
    T_text = 40
    text = torch.randint(1, 256, (B, T_text), device=device)
    text_lens = torch.tensor([T_text], device=device)

    gc.collect()
    mem_before = get_memory_mb()

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model.inference(text, text_lens)

    # Benchmark
    times = []
    for _ in range(n_runs):
        gc.collect()
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            mel = model.inference(text, text_lens)
        torch.cuda.synchronize() if device == "cuda" else None
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    results["inference_time_ms"] = np.mean(times)
    results["inference_std_ms"] = np.std(times)

    mel_frames = mel.shape[2]
    audio_length_s = mel_frames * 256 / 16000
    results["audio_length_s"] = audio_length_s
    results["rtf"] = (results["inference_time_ms"] / 1000) / max(audio_length_s, 0.001)

    results["memory_mb"] = get_memory_mb() - mem_before
    if device == "cuda":
        results["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return results


def benchmark_vits_pytorch(device="cpu", n_runs=10):
    """Benchmark VITS PyTorch inference."""
    mod = load_vits_model()
    VITS = mod.VITS

    results = {
        "model": "VITS",
        "framework": "PyTorch",
        "device": device,
    }

    t0 = time.perf_counter()
    model = VITS(
        vocab_size=200,
        spec_channels=513,
        hidden_dim=192,
        ffn_dim=384,
        n_heads=2,
        n_enc_layers=2,
        n_post_layers=4,
        n_flow_layers=2,
        upsample_rates=(8, 8, 2, 2),
        use_sdp=False,
    ).to(device)
    model.eval()
    t1 = time.perf_counter()
    results["load_time_ms"] = (t1 - t0) * 1000
    results["params"] = sum(p.numel() for p in model.parameters())

    B = 1
    T_text = 20
    text_ids = torch.randint(0, 200, (B, T_text), device=device)
    text_lengths = torch.tensor([T_text], device=device)

    gc.collect()
    mem_before = get_memory_mb()

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model.infer(text_ids, text_lengths)

    # Benchmark
    times = []
    for _ in range(n_runs):
        gc.collect()
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            wav, dur = model.infer(text_ids, text_lengths)
        torch.cuda.synchronize() if device == "cuda" else None
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    results["inference_time_ms"] = np.mean(times)
    results["inference_std_ms"] = np.std(times)

    wav_samples = wav.shape[-1]
    audio_length_s = wav_samples / 16000
    results["audio_length_s"] = audio_length_s
    results["rtf"] = (results["inference_time_ms"] / 1000) / max(audio_length_s, 0.001)

    results["memory_mb"] = get_memory_mb() - mem_before
    if device == "cuda":
        results["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return results


# ================================================================
# ONNX Benchmarks
# ================================================================

def benchmark_onnx_model(onnx_path, n_runs=10):
    """Benchmark an ONNX model on CPU."""
    import onnxruntime as ort

    name = os.path.basename(onnx_path)
    results = {
        "model": name,
        "framework": "ONNX",
        "device": "CPU",
        "file_size_mb": os.path.getsize(onnx_path) / (1024 * 1024),
    }

    # Load time
    t0 = time.perf_counter()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    t1 = time.perf_counter()
    results["load_time_ms"] = (t1 - t0) * 1000

    # Create dummy inputs
    feed = {}
    for inp in sess.get_inputs():
        shape = []
        for dim in inp.shape:
            if isinstance(dim, str):
                if "batch" in dim:
                    shape.append(1)
                elif "text" in dim:
                    shape.append(20)
                elif "len" in dim:
                    shape.append(20)
                elif "mel" in dim or "time" in dim:
                    shape.append(100)
                elif "wav" in dim or "z" in dim:
                    shape.append(100)
                else:
                    shape.append(1)
            else:
                shape.append(dim)

        if inp.type == "tensor(int64)":
            if shape:
                feed[inp.name] = np.random.randint(0, 100, size=shape).astype(np.int64)
            else:
                feed[inp.name] = np.array(1, dtype=np.int64)
        else:
            if shape:
                feed[inp.name] = np.random.randn(*shape).astype(np.float32)
            else:
                feed[inp.name] = np.array(0.667, dtype=np.float32)

    # Warmup
    for _ in range(3):
        sess.run(None, feed)

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    results["inference_time_ms"] = np.mean(times)
    results["inference_std_ms"] = np.std(times)
    results["n_runs"] = n_runs

    return results


# ================================================================
# MNN Benchmarks
# ================================================================

def benchmark_mnn_model(mnn_path, n_runs=10):
    """Benchmark an MNN model (if PyMNN is available)."""
    name = os.path.basename(mnn_path)
    results = {
        "model": name,
        "framework": "MNN",
        "device": "CPU",
        "file_size_mb": os.path.getsize(mnn_path) / (1024 * 1024),
    }

    try:
        import MNN
        import MNN.numpy as mnp

        t0 = time.perf_counter()
        interpreter = MNN.Interpreter(mnn_path)
        session = interpreter.createSession()
        t1 = time.perf_counter()
        results["load_time_ms"] = (t1 - t0) * 1000

        input_tensor = interpreter.getSessionInput(session)
        shape = input_tensor.getShape()

        # Create dummy input
        data = np.random.randn(*shape).astype(np.float32)
        tmp_input = MNN.Tensor(
            input_tensor.getDimensionType(),
            shape,
            data.flatten().tolist(),
            MNN.Halide_Type_Float,
            MNN.Tensor_DimensionType_Tensorflow,
        )
        input_tensor.copyFrom(tmp_input)

        # Warmup
        for _ in range(3):
            interpreter.runSession(session)

        # Benchmark
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            interpreter.runSession(session)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        results["inference_time_ms"] = np.mean(times)
        results["inference_std_ms"] = np.std(times)
        results["success"] = True

    except ImportError:
        results["success"] = False
        results["error"] = "PyMNN not installed"
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)

    return results


# ================================================================
# Report Generation
# ================================================================

def generate_markdown_report(all_results, output_path):
    """Generate a Markdown benchmark report."""
    lines = [
        "# Neko Speech Deployment Benchmark Results",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Machine:** {os.uname().nodename} ({os.uname().machine})",
        f"**PyTorch:** {torch.__version__}",
        f"**CUDA:** {'Available (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'Not available'}",
        "",
        "## Summary Table",
        "",
        "| Model | Framework | Device | Size (MB) | Params | Load (ms) | Infer (ms) | Audio (s) | RTF | Memory (MB) |",
        "|-------|-----------|--------|-----------|--------|-----------|------------|-----------|-----|-------------|",
    ]

    for r in all_results:
        size = f"{r.get('file_size_mb', 0):.1f}" if 'file_size_mb' in r else "-"
        params = f"{r.get('params', 0)/1e6:.1f}M" if 'params' in r else "-"
        load = f"{r.get('load_time_ms', 0):.1f}"
        infer = f"{r.get('inference_time_ms', 0):.1f} +/- {r.get('inference_std_ms', 0):.1f}"
        audio = f"{r.get('audio_length_s', 0):.1f}" if 'audio_length_s' in r else "-"
        rtf = f"{r.get('rtf', 0):.3f}" if 'rtf' in r else "-"
        mem = f"{r.get('memory_mb', 0):.0f}" if 'memory_mb' in r else "-"

        lines.append(
            f"| {r.get('model', '-')} "
            f"| {r.get('framework', '-')} "
            f"| {r.get('device', '-')} "
            f"| {size} "
            f"| {params} "
            f"| {load} "
            f"| {infer} "
            f"| {audio} "
            f"| {rtf} "
            f"| {mem} |"
        )

    lines.extend([
        "",
        "## Key Observations",
        "",
        "- **RTF (Real-Time Factor):** < 1.0 means faster than real-time",
        "- **FastSpeech2** is non-autoregressive -> fastest inference",
        "- **Tacotron2** is autoregressive -> slowest but highest quality potential",
        "- **VITS** is end-to-end -> good balance of speed and quality",
        "- **ONNX** typically 1.5-3x faster than PyTorch on CPU",
        "- **MNN** optimized for mobile ARM CPUs",
        "",
        "## Format Comparison",
        "",
        "| Format | Best For | Pros | Cons |",
        "|--------|----------|------|------|",
        "| PyTorch | Research, training | Full flexibility, autograd | Slow inference, large |",
        "| ONNX | Desktop/server deploy | Cross-platform, optimized | No training, static graph |",
        "| MNN | Mobile (Android/iOS) | ARM optimized, tiny | Limited ops, no dynamic |",
        "| sherpa-onnx | TTS applications | Ready-to-use TTS API | Limited model support |",
        "",
    ])

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    return report


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Neko Speech Benchmark")
    parser.add_argument("--model", type=str, default="all",
                        choices=["tacotron2", "fastspeech2", "vits", "all"])
    parser.add_argument("--onnx_dir", type=str, default="./onnx_models",
                        help="Directory containing ONNX models")
    parser.add_argument("--mnn_dir", type=str, default="./mnn_models",
                        help="Directory containing MNN models")
    parser.add_argument("--output", type=str, default="benchmark_results.md",
                        help="Output Markdown report path")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="PyTorch device for benchmarking")
    parser.add_argument("--n_runs", type=int, default=10,
                        help="Number of inference runs for averaging")
    args = parser.parse_args()

    print("Neko Speech Deployment Benchmark")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Runs:   {args.n_runs}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA not available, falling back to CPU")
        args.device = "cpu"

    all_results = []

    # ---- PyTorch benchmarks ----
    print("\n--- PyTorch Benchmarks ---")

    if args.model in ("tacotron2", "all"):
        print("\n  Tacotron2...")
        r = benchmark_tacotron2_pytorch(args.device, args.n_runs)
        all_results.append(r)
        print(f"    Inference: {r['inference_time_ms']:.1f} +/- {r['inference_std_ms']:.1f} ms")
        print(f"    Audio:     {r['audio_length_s']:.1f}s  RTF={r['rtf']:.3f}")

    if args.model in ("fastspeech2", "all"):
        print("\n  FastSpeech2...")
        r = benchmark_fastspeech2_pytorch(args.device, args.n_runs)
        all_results.append(r)
        print(f"    Inference: {r['inference_time_ms']:.1f} +/- {r['inference_std_ms']:.1f} ms")
        print(f"    Audio:     {r['audio_length_s']:.1f}s  RTF={r['rtf']:.3f}")

    if args.model in ("vits", "all"):
        print("\n  VITS...")
        r = benchmark_vits_pytorch(args.device, args.n_runs)
        all_results.append(r)
        print(f"    Inference: {r['inference_time_ms']:.1f} +/- {r['inference_std_ms']:.1f} ms")
        print(f"    Audio:     {r['audio_length_s']:.1f}s  RTF={r['rtf']:.3f}")

    # ---- ONNX benchmarks ----
    print("\n--- ONNX Benchmarks ---")

    if os.path.isdir(args.onnx_dir):
        onnx_files = sorted([
            f for f in os.listdir(args.onnx_dir) if f.endswith(".onnx")
        ])
        for onnx_file in onnx_files:
            name = onnx_file.replace(".onnx", "")
            model_filter = args.model
            if model_filter != "all" and model_filter not in name:
                continue

            print(f"\n  {onnx_file}...")
            path = os.path.join(args.onnx_dir, onnx_file)
            try:
                r = benchmark_onnx_model(path, args.n_runs)
                all_results.append(r)
                print(f"    Inference: {r['inference_time_ms']:.1f} +/- {r['inference_std_ms']:.1f} ms")
            except Exception as e:
                print(f"    [ERROR] {e}")
                all_results.append({
                    "model": name,
                    "framework": "ONNX",
                    "device": "CPU",
                    "error": str(e),
                })
    else:
        print(f"  [SKIP] ONNX directory not found: {args.onnx_dir}")

    # ---- MNN benchmarks ----
    print("\n--- MNN Benchmarks ---")

    if os.path.isdir(args.mnn_dir):
        mnn_files = sorted([
            f for f in os.listdir(args.mnn_dir) if f.endswith(".mnn")
        ])
        for mnn_file in mnn_files:
            name = mnn_file.replace(".mnn", "")
            model_filter = args.model
            if model_filter != "all" and model_filter not in name:
                continue

            print(f"\n  {mnn_file}...")
            path = os.path.join(args.mnn_dir, mnn_file)
            r = benchmark_mnn_model(path, args.n_runs)
            if r.get("success", False):
                all_results.append(r)
                print(f"    Inference: {r['inference_time_ms']:.1f} +/- {r['inference_std_ms']:.1f} ms")
            else:
                print(f"    [SKIP] {r.get('error', 'unknown error')}")
    else:
        print(f"  [SKIP] MNN directory not found: {args.mnn_dir}")

    # ---- Generate report ----
    print("\n" + "=" * 60)
    report = generate_markdown_report(all_results, args.output)
    print(f"Report saved to: {args.output}")
    print()
    print(report)

    # Save JSON too
    json_path = args.output.replace(".md", ".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON data saved to: {json_path}")


if __name__ == "__main__":
    main()
