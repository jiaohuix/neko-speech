#!/usr/bin/env python3
"""
Comprehensive experiment runner for neko-speech textbook models.

Trains each model for 3 epochs with small batch size, runs inference,
and collects metrics (loss, time, GPU memory, parameter count).

Models:
  - ch03_wavenet (real data)
  - ch04_fastspeech (real data)
  - ch05_vits (real data)
  - ch09_gpt_sovits (simulated data, 2-stage)
  - ch10_voxcpm (simulated data)

Usage:
    python run_experiments.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

BASE_DIR = Path(__file__).parent.parent
CHAPTERS_DIR = BASE_DIR / "chapters"
DATA_DIR = BASE_DIR / "data" / "processed"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
OUTPUTS_DIR = EXPERIMENTS_DIR / "outputs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

# Ensure dirs exist
for d in [LOGS_DIR, OUTPUTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_gpu_memory_mb():
    """Return current GPU memory usage in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_gpu_memory_reserved_mb():
    """Return GPU memory reserved by caching allocator in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024 / 1024
    return 0.0


def reset_gpu_memory():
    """Clear GPU cache between models."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_command(cmd, log_path, timeout=600):
    """Run a shell command, capture output to log file, return (success, output)."""
    print(f"  CMD: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            cwd=str(log_path.parent),
        )
        with open(log_path, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n")
            f.write(result.stdout)
        if result.returncode != 0:
            print(f"  FAILED (rc={result.returncode})")
            print(f"  Last 10 lines:")
            for line in result.stdout.strip().split("\n")[-10:]:
                print(f"    {line}")
            return False, result.stdout
        return True, result.stdout
    except subprocess.TimeoutExpired:
        with open(log_path, "w") as f:
            f.write(f"TIMEOUT after {timeout}s\n")
        print(f"  TIMEOUT after {timeout}s")
        return False, "TIMEOUT"
    except Exception as e:
        with open(log_path, "w") as f:
            f.write(f"ERROR: {e}\n")
        print(f"  ERROR: {e}")
        return False, str(e)


def parse_training_output(output):
    """Extract final loss and epoch times from training output."""
    lines = output.strip().split("\n")
    final_loss = None
    epoch_times = []

    for line in lines:
        line = line.strip()
        # Try to parse various formats
        # ch03_wavenet: [epoch 1/3] loss: 5.5189
        if "loss:" in line.lower() and ("epoch" in line.lower() or "[epoch" in line.lower()):
            parts = line.split("loss:")
            if len(parts) > 1:
                try:
                    val = parts[-1].strip().split()[0]
                    final_loss = float(val)
                except (ValueError, IndexError):
                    pass

        # ch09: Epoch 1/3 | loss: 7.0521 | lr: ... | time: 4.2s
        if "time:" in line.lower():
            for part in line.split("|"):
                part = part.strip()
                if "time:" in part.lower():
                    try:
                        t = float(part.split(":")[-1].strip().rstrip("s"))
                        epoch_times.append(t)
                    except (ValueError, IndexError):
                        pass

        # ch10: Epoch 1/3  train_loss=...  val_loss=...  time=12.3s
        if "time=" in line:
            try:
                t = float(line.split("time=")[-1].rstrip("s"))
                epoch_times.append(t)
            except (ValueError, IndexError):
                pass

    return final_loss, epoch_times


# ===================================================================
# Model-specific experiment functions
# ===================================================================

def experiment_wavenet():
    """Run ch03_wavenet experiment: 3 epochs training + inference."""
    model_name = "ch03_wavenet"
    print(f"\n{'='*60}")
    print(f"Experiment: {model_name}")
    print(f"{'='*60}")

    code_dir = CHAPTERS_DIR / model_name / "code"
    save_dir = EXPERIMENTS_DIR / "checkpoints" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
    }

    # Training
    print("  [1/3] Training (3 epochs)...")
    reset_gpu_memory()
    t0 = time.time()

    train_cmd = [
        sys.executable, str(code_dir / "train.py"),
        "--data-dir", str(DATA_DIR),
        "--epochs", "3",
        "--batch-size", "2",
        "--segment-length", "4096",
        "--res-channels", "32",
        "--skip-channels", "64",
        "--n-blocks", "10",
        "--n-cycles", "2",
        "--save-dir", str(save_dir),
    ]

    log_path = LOGS_DIR / f"{model_name}_train.log"
    success, output = run_command(train_cmd, log_path, timeout=300)
    train_time = time.time() - t0

    if not success:
        result["status"] = "train_failed"
        result["train_error"] = output[-500:] if output else "Unknown error"
        save_result(result, model_name)
        return result

    final_loss, epoch_times = parse_training_output(output)
    gpu_mem = get_gpu_memory_reserved_mb()

    result.update({
        "final_loss": final_loss,
        "train_time_total": train_time,
        "epoch_times": epoch_times,
        "gpu_memory_mb": gpu_mem,
        "status": "trained",
    })

    # Count parameters
    print("  [2/3] Counting parameters...")
    try:
        sys.path.insert(0, str(code_dir))
        from model import WaveNet
        model = WaveNet(n_mels=80, res_channels=32, skip_channels=64,
                        n_blocks=10, n_cycles=2)
        params = sum(p.numel() for p in model.parameters())
        result["params"] = params
        result["params_millions"] = params / 1e6
        del model
        reset_gpu_memory()
    except Exception as e:
        print(f"  Param count failed: {e}")

    # Inference
    print("  [3/3] Running inference...")
    ckpt_path = save_dir / "wavenet_final.pt"
    if ckpt_path.exists():
        inf_cmd = [
            sys.executable, str(code_dir / "inference.py"),
            "--checkpoint", str(ckpt_path),
            "--output-dir", str(OUTPUTS_DIR),
            "--max-samples", "8000",
        ]
        inf_log = LOGS_DIR / f"{model_name}_inference.log"
        inf_success, inf_output = run_command(inf_cmd, inf_log, timeout=120)

        if inf_success:
            result["inference_success"] = True
            # Parse RTF from output
            for line in inf_output.split("\n"):
                if "RTF" in line and "real-time" in line.lower():
                    try:
                        rtf_val = float(line.split("RTF:")[-1].split("x")[0].strip())
                        result["rtf"] = rtf_val
                    except (ValueError, IndexError):
                        pass
            # Rename output
            src = OUTPUTS_DIR / "wavenet_ground_truth.wav"
            # The output files depend on source naming, just check what exists
            for f in OUTPUTS_DIR.glob("wavenet_*"):
                dst = OUTPUTS_DIR / f"{model_name}_test.wav"
                if not f.name.startswith("ground_truth") and not f.name.startswith("griffinlim"):
                    os.rename(str(f), str(dst))
                    break
        else:
            result["inference_success"] = False
            result["inference_error"] = inf_output[-300:] if inf_output else "Unknown"
    else:
        result["inference_success"] = False
        result["inference_error"] = "No checkpoint found"

    result["status"] = "complete"
    save_result(result, model_name)
    return result


def experiment_fastspeech():
    """Run ch04_fastspeech experiment: 3 epochs training + inference."""
    model_name = "ch04_fastspeech"
    print(f"\n{'='*60}")
    print(f"Experiment: {model_name}")
    print(f"{'='*60}")

    code_dir = CHAPTERS_DIR / model_name / "code"
    save_dir = EXPERIMENTS_DIR / "checkpoints" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
    }

    # Training
    print("  [1/3] Training (3 epochs)...")
    reset_gpu_memory()
    t0 = time.time()

    train_cmd = [
        sys.executable, str(code_dir / "train.py"),
        "--data-dir", str(DATA_DIR),
        "--epochs", "3",
        "--batch-size", "4",
        "--d-model", "128",
        "--enc-layers", "2",
        "--dec-layers", "2",
        "--save-dir", str(save_dir),
    ]

    log_path = LOGS_DIR / f"{model_name}_train.log"
    success, output = run_command(train_cmd, log_path, timeout=300)
    train_time = time.time() - t0

    if not success:
        result["status"] = "train_failed"
        result["train_error"] = output[-500:] if output else "Unknown error"
        save_result(result, model_name)
        return result

    final_loss, epoch_times = parse_training_output(output)
    gpu_mem = get_gpu_memory_reserved_mb()

    result.update({
        "final_loss": final_loss,
        "train_time_total": train_time,
        "epoch_times": epoch_times,
        "gpu_memory_mb": gpu_mem,
        "status": "trained",
    })

    # Count parameters
    print("  [2/3] Counting parameters...")
    try:
        sys.path.insert(0, str(code_dir))
        # Need fresh import
        import importlib
        if "model" in sys.modules:
            del sys.modules["model"]
        mod = importlib.import_module("model")
        FS2 = getattr(mod, "FastSpeech2")
        m = FS2(vocab_size=200, d_model=128, enc_layers=2, dec_layers=2)
        params = sum(p.numel() for p in m.parameters())
        result["params"] = params
        result["params_millions"] = params / 1e6
        del m
        reset_gpu_memory()
    except Exception as e:
        print(f"  Param count failed: {e}")

    # Inference
    print("  [3/3] Running inference...")
    ckpt_path = save_dir / "fs2_final.pt"
    if ckpt_path.exists():
        out_wav = OUTPUTS_DIR / f"{model_name}_test.wav"
        inf_cmd = [
            sys.executable, str(code_dir / "inference.py"),
            "--checkpoint", str(ckpt_path),
            "--text", "hello neko world",
            "--output", str(out_wav),
        ]
        inf_log = LOGS_DIR / f"{model_name}_inference.log"
        inf_success, inf_output = run_command(inf_cmd, inf_log, timeout=120)

        if inf_success:
            result["inference_success"] = True
            for line in inf_output.split("\n"):
                if "RTF" in line:
                    try:
                        rtf_val = float(line.split("=")[-1].split()[0])
                        result["rtf"] = rtf_val
                    except (ValueError, IndexError):
                        pass
        else:
            result["inference_success"] = False
            result["inference_error"] = inf_output[-300:] if inf_output else "Unknown"
    else:
        result["inference_success"] = False
        result["inference_error"] = "No checkpoint found"

    result["status"] = "complete"
    save_result(result, model_name)
    return result


def experiment_vits():
    """Run ch05_vits experiment: 3 epochs training + inference."""
    model_name = "ch05_vits"
    print(f"\n{'='*60}")
    print(f"Experiment: {model_name}")
    print(f"{'='*60}")

    code_dir = CHAPTERS_DIR / model_name / "code"
    save_dir = EXPERIMENTS_DIR / "checkpoints" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
    }

    # Training
    print("  [1/3] Training (3 epochs)...")
    reset_gpu_memory()
    t0 = time.time()

    train_cmd = [
        sys.executable, str(code_dir / "train.py"),
        "--data-dir", str(DATA_DIR),
        "--epochs", "3",
        "--batch-size", "2",
        "--max-spec-len", "200",
        "--max-wav-sec", "5",
        "--hidden-dim", "96",
        "--save-dir", str(save_dir),
    ]

    log_path = LOGS_DIR / f"{model_name}_train.log"
    success, output = run_command(train_cmd, log_path, timeout=600)
    train_time = time.time() - t0

    if not success:
        result["status"] = "train_failed"
        result["train_error"] = output[-500:] if output else "Unknown error"
        save_result(result, model_name)
        return result

    final_loss, epoch_times = parse_training_output(output)
    gpu_mem = get_gpu_memory_reserved_mb()

    result.update({
        "final_loss": final_loss,
        "train_time_total": train_time,
        "epoch_times": epoch_times,
        "gpu_memory_mb": gpu_mem,
        "status": "trained",
    })

    # Count parameters
    print("  [2/3] Counting parameters...")
    try:
        sys.path.insert(0, str(code_dir))
        import importlib
        for mod_name in ["model", "modules"]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        mod = importlib.import_module("model")
        VITS_cls = getattr(mod, "VITS")
        m = VITS_cls(vocab_size=200, hidden_dim=96, ffn_dim=384, n_heads=2,
                     n_enc_layers=4, n_post_layers=6, n_flow_layers=4,
                     upsample_rates=(8, 8, 2, 2))
        params = sum(p.numel() for p in m.parameters())
        result["params"] = params
        result["params_millions"] = params / 1e6
        del m
        reset_gpu_memory()
    except Exception as e:
        print(f"  Param count failed: {e}")

    # Inference
    print("  [3/3] Running inference...")
    ckpt_path = save_dir / "vits_final.pt"
    if ckpt_path.exists():
        out_wav = OUTPUTS_DIR / f"{model_name}_test.wav"
        inf_cmd = [
            sys.executable, str(code_dir / "inference.py"),
            "--checkpoint", str(ckpt_path),
            "--text", "hello neko world",
            "--output", str(out_wav),
        ]
        inf_log = LOGS_DIR / f"{model_name}_inference.log"
        inf_success, inf_output = run_command(inf_cmd, inf_log, timeout=120)

        if inf_success:
            result["inference_success"] = True
            for line in inf_output.split("\n"):
                if "RTF" in line and "real-time" in line.lower():
                    try:
                        rtf_val = float(line.split(":")[-1].strip().rstrip("x"))
                        result["rtf"] = rtf_val
                    except (ValueError, IndexError):
                        pass
        else:
            result["inference_success"] = False
            result["inference_error"] = inf_output[-300:] if inf_output else "Unknown"
    else:
        result["inference_success"] = False
        result["inference_error"] = "No checkpoint found"

    result["status"] = "complete"
    save_result(result, model_name)
    return result


def experiment_gpt_sovits():
    """Run ch09_gpt_sovits experiment: 3 epochs per stage + inference."""
    model_name = "ch09_gpt_sovits"
    print(f"\n{'='*60}")
    print(f"Experiment: {model_name}")
    print(f"{'='*60}")

    code_dir = CHAPTERS_DIR / model_name / "code"
    save_dir = EXPERIMENTS_DIR / "checkpoints" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
    }

    # Stage 1: AR model training
    print("  [1/4] Stage 1: AR model training (3 epochs)...")
    reset_gpu_memory()
    t0 = time.time()

    stage1_cmd = [
        sys.executable, str(code_dir / "train.py"),
        "--stage", "1",
        "--epochs", "3",
        "--batch-size", "4",
        "--num-samples", "100",
        "--save-dir", str(save_dir),
    ]

    log_path = LOGS_DIR / f"{model_name}_stage1_train.log"
    success, output = run_command(stage1_cmd, log_path, timeout=300)
    stage1_time = time.time() - t0

    if not success:
        result["status"] = "stage1_failed"
        result["train_error"] = output[-500:] if output else "Unknown error"
        save_result(result, model_name)
        return result

    final_loss_s1, epoch_times_s1 = parse_training_output(output)
    gpu_mem = get_gpu_memory_reserved_mb()

    result.update({
        "stage1_loss": final_loss_s1,
        "stage1_time": stage1_time,
        "stage1_epoch_times": epoch_times_s1,
        "gpu_memory_stage1_mb": gpu_mem,
    })

    # Stage 2: SoVITS vocoder training
    print("  [2/4] Stage 2: SoVITS vocoder training (3 epochs)...")
    reset_gpu_memory()
    t0 = time.time()

    stage2_cmd = [
        sys.executable, str(code_dir / "train.py"),
        "--stage", "2",
        "--epochs", "3",
        "--batch-size", "2",
        "--num-samples", "100",
        "--save-dir", str(save_dir),
    ]

    log_path = LOGS_DIR / f"{model_name}_stage2_train.log"
    success, output = run_command(stage2_cmd, log_path, timeout=300)
    stage2_time = time.time() - t0

    if not success:
        result["status"] = "stage2_failed"
        result["stage2_error"] = output[-500:] if output else "Unknown error"
        # Still have stage 1 results
    else:
        final_loss_s2, epoch_times_s2 = parse_training_output(output)
        result.update({
            "stage2_loss": final_loss_s2,
            "stage2_time": stage2_time,
            "stage2_epoch_times": epoch_times_s2,
        })

    result["train_time_total"] = stage1_time + stage2_time
    result["final_loss"] = result.get("stage1_loss")  # Use AR loss as primary

    # Count parameters
    print("  [3/4] Counting parameters...")
    try:
        sys.path.insert(0, str(code_dir))
        import importlib
        for mod_name in ["model", "modules"]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        mod = importlib.import_module("model")
        GPTSoVITS_cls = getattr(mod, "GPTSoVITS")
        m = GPTSoVITS_cls()
        params = sum(p.numel() for p in m.parameters())
        result["params"] = params
        result["params_millions"] = params / 1e6
        del m
        reset_gpu_memory()
    except Exception as e:
        print(f"  Param count failed: {e}")

    # Inference
    print("  [4/4] Running inference...")
    out_wav = OUTPUTS_DIR / f"{model_name}_test.wav"
    ar_ckpt = save_dir / "ar_model.pt"
    sovits_ckpt = save_dir / "sovits_model.pt"

    inf_cmd = [
        sys.executable, str(code_dir / "inference.py"),
        "--text", "hello neko world",
        "--output", str(out_wav),
        "--max-tokens", "50",
        "--benchmark",
    ]
    if ar_ckpt.exists():
        inf_cmd.extend(["--ar-checkpoint", str(ar_ckpt)])
    if sovits_ckpt.exists():
        inf_cmd.extend(["--sovits-checkpoint", str(sovits_ckpt)])

    inf_log = LOGS_DIR / f"{model_name}_inference.log"
    inf_success, inf_output = run_command(inf_cmd, inf_log, timeout=120)

    if inf_success:
        result["inference_success"] = True
        for line in inf_output.split("\n"):
            if "RTF:" in line:
                try:
                    rtf_val = float(line.split("RTF:")[-1].strip())
                    result["rtf"] = rtf_val
                except (ValueError, IndexError):
                    pass
    else:
        result["inference_success"] = False
        result["inference_error"] = inf_output[-300:] if inf_output else "Unknown"

    result["status"] = "complete"
    save_result(result, model_name)
    return result


def experiment_voxcpm():
    """Run ch10_voxcpm experiment: 3 epochs training + inference."""
    model_name = "ch10_voxcpm"
    print(f"\n{'='*60}")
    print(f"Experiment: {model_name}")
    print(f"{'='*60}")

    code_dir = CHAPTERS_DIR / model_name / "code"
    save_dir = EXPERIMENTS_DIR / "checkpoints" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
    }

    # Training
    print("  [1/3] Training (3 epochs)...")
    reset_gpu_memory()
    t0 = time.time()

    ckpt_path = save_dir / "voxcpm.pt"
    train_cmd = [
        sys.executable, str(code_dir / "train.py"),
        "--epochs", "3",
        "--batch-size", "2",
        "--n-train", "32",
        "--n-val", "4",
        "--audio-len", "3200",
        "--save-path", str(ckpt_path),
    ]

    log_path = LOGS_DIR / f"{model_name}_train.log"
    success, output = run_command(train_cmd, log_path, timeout=300)
    train_time = time.time() - t0

    if not success:
        result["status"] = "train_failed"
        result["train_error"] = output[-500:] if output else "Unknown error"
        save_result(result, model_name)
        return result

    final_loss, epoch_times = parse_training_output(output)
    gpu_mem = get_gpu_memory_reserved_mb()

    result.update({
        "final_loss": final_loss,
        "train_time_total": train_time,
        "epoch_times": epoch_times,
        "gpu_memory_mb": gpu_mem,
        "status": "trained",
    })

    # Count parameters
    print("  [2/3] Counting parameters...")
    try:
        sys.path.insert(0, str(code_dir))
        import importlib
        for mod_name in ["model", "modules"]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        mod = importlib.import_module("model")
        VoxCPM_cls = getattr(mod, "SimpleVoxCPM")
        m = VoxCPM_cls(
            vocab_size=256, encoder_dim=64, latent_dim=32, decoder_dim=256,
            patch_size=1, loc_enc_hidden=512, loc_enc_layers=2,
            tslm_hidden=512, tslm_layers=8, tslm_heads=8, tslm_ffn=2048,
            fsq_latent=128, fsq_scale=9,
            ralm_hidden=512, ralm_layers=4, ralm_heads=8, ralm_ffn=2048,
            dit_hidden=256, dit_layers=4, dit_heads=4, dit_ffn=1024,
            cfm_steps=10,
        )
        params = sum(p.numel() for p in m.parameters())
        result["params"] = params
        result["params_millions"] = params / 1e6
        del m
        reset_gpu_memory()
    except Exception as e:
        print(f"  Param count failed: {e}")

    # Inference
    print("  [3/3] Running inference...")
    out_wav = OUTPUTS_DIR / f"{model_name}_test.wav"
    inf_cmd = [
        sys.executable, str(code_dir / "inference.py"),
        "--text", "hello neko world",
        "--n-steps", "5",
        "--output", str(out_wav),
    ]
    if ckpt_path.exists():
        inf_cmd.extend(["--checkpoint", str(ckpt_path)])

    inf_log = LOGS_DIR / f"{model_name}_inference.log"
    inf_success, inf_output = run_command(inf_cmd, inf_log, timeout=120)

    if inf_success:
        result["inference_success"] = True
        for line in inf_output.split("\n"):
            if "RTF:" in line:
                try:
                    rtf_val = float(line.split("RTF:")[-1].strip().split()[0])
                    result["rtf"] = rtf_val
                except (ValueError, IndexError):
                    pass
    else:
        result["inference_success"] = False
        result["inference_error"] = inf_output[-300:] if inf_output else "Unknown"

    result["status"] = "complete"
    save_result(result, model_name)
    return result


def save_result(result, model_name):
    """Save result JSON."""
    result_path = RESULTS_DIR / f"{model_name}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Result saved: {result_path}")


def generate_comparison_report(results):
    """Generate comparison_report.md."""
    report_path = EXPERIMENTS_DIR / "comparison_report.md"

    with open(report_path, "w") as f:
        f.write("# Neko-Speech Model Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Hardware: NVIDIA GeForce RTX 3060 (12GB VRAM)\n\n")
        f.write("## Summary Table\n\n")
        f.write("| Model | Params (M) | Final Loss | Epoch Time (s) | GPU Mem (MB) | RTF | Status |\n")
        f.write("|-------|-----------|-----------|---------------|-------------|-----|--------|\n")

        for r in results:
            name = r.get("model", "?")
            params_m = r.get("params_millions", 0)
            params_str = f"{params_m:.1f}" if params_m else "?"

            loss = r.get("final_loss")
            loss_str = f"{loss:.4f}" if loss is not None else "?"

            epoch_times = r.get("epoch_times") or r.get("stage1_epoch_times", [])
            if epoch_times:
                avg_epoch = sum(epoch_times) / len(epoch_times)
                epoch_str = f"{avg_epoch:.1f}"
            else:
                total_time = r.get("train_time_total")
                epoch_str = f"{total_time/3:.1f}" if total_time else "?"

            gpu_mem = r.get("gpu_memory_mb", 0)
            gpu_str = f"{gpu_mem:.0f}" if gpu_mem else "?"

            rtf = r.get("rtf")
            rtf_str = f"{rtf:.3f}" if rtf is not None else "?"

            status = r.get("status", "?")

            f.write(f"| {name} | {params_str} | {loss_str} | {epoch_str} | {gpu_str} | {rtf_str} | {status} |\n")

        f.write("\n## Detailed Results\n\n")

        for r in results:
            name = r.get("model", "?")
            f.write(f"### {name}\n\n")
            f.write(f"- **Status**: {r.get('status', '?')}\n")
            f.write(f"- **Parameters**: {r.get('params_millions', '?')}M\n")

            loss = r.get("final_loss")
            if loss is not None:
                f.write(f"- **Final Loss**: {loss:.4f}\n")

            total_time = r.get("train_time_total")
            if total_time:
                f.write(f"- **Total Training Time**: {total_time:.1f}s\n")

            rtf = r.get("rtf")
            if rtf is not None:
                f.write(f"- **RTF**: {rtf:.3f}\n")

            inf_ok = r.get("inference_success", False)
            f.write(f"- **Inference**: {'OK' if inf_ok else 'Failed'}\n")

            if "train_error" in r:
                f.write(f"- **Error**: {r['train_error'][:200]}\n")
            if "inference_error" in r:
                f.write(f"- **Inference Error**: {r['inference_error'][:200]}\n")

            f.write("\n")

        f.write("---\n")
        f.write("*Report generated by run_experiments.py*\n")

    print(f"\nComparison report saved: {report_path}")


def main():
    print("=" * 60)
    print("Neko-Speech Comprehensive Experiment Runner")
    print("=" * 60)
    print(f"Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device:   {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU:      {torch.cuda.get_device_name(0)}")
        print(f"VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Data:     {DATA_DIR}")
    print(f"Output:   {EXPERIMENTS_DIR}")
    print("=" * 60)

    # Run experiments sequentially
    experiments = [
        ("ch03_wavenet", experiment_wavenet),
        ("ch04_fastspeech", experiment_fastspeech),
        ("ch05_vits", experiment_vits),
        ("ch09_gpt_sovits", experiment_gpt_sovits),
        ("ch10_voxcpm", experiment_voxcpm),
    ]

    all_results = []

    for name, fn in experiments:
        try:
            result = fn()
            all_results.append(result)
        except Exception as e:
            print(f"\n  FATAL ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            err_result = {
                "model": name,
                "status": "fatal_error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(err_result)
            save_result(err_result, name)

        # Clean up between models
        reset_gpu_memory()

    # Generate comparison report
    generate_comparison_report(all_results)

    # Save all results
    all_path = RESULTS_DIR / "all_results.json"
    with open(all_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved: {all_path}")

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Status':<15} {'Params(M)':<12} {'Loss':<10} {'RTF':<10}")
    print("-" * 67)
    for r in all_results:
        name = r.get("model", "?")
        status = r.get("status", "?")
        params = f"{r.get('params_millions', 0):.1f}" if r.get("params_millions") else "?"
        loss = f"{r.get('final_loss', 0):.4f}" if r.get("final_loss") else "?"
        rtf = f"{r.get('rtf', 0):.3f}" if r.get("rtf") else "?"
        print(f"{name:<20} {status:<15} {params:<12} {loss:<10} {rtf:<10}")


if __name__ == "__main__":
    main()
