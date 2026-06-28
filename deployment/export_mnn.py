"""
Neko Speech -- MNN Conversion Toolkit

Convert ONNX models to MNN format for mobile deployment (Android/iOS).

Pipeline:
    PyTorch -> ONNX -> MNN

Prerequisites:
    1. ONNX models from export_onnx.py
    2. MNNConvert tool (built from MNN source)

Usage:
    python export_mnn.py --onnx_dir ./onnx_models --output_dir ./mnn_models
    python export_mnn.py --onnx_dir ./onnx_models --output_dir ./mnn_models --fp16
    python export_mnn.py --onnx_dir ./onnx_models --output_dir ./mnn_models --quant 8

MNN Tool Locations (typical):
    /path/to/MNN/build/MNNConvert

Note: If MNNConvert is not found, this script documents the commands
that would be run and exits gracefully.
"""

import argparse
import json
import os
import subprocess
import sys
import time


# ================================================================
# MNNConvert Discovery
# ================================================================

# Common locations for MNNConvert
MNN_CONVERT_PATHS = [
    os.path.expanduser("~/Projects/nlp/MNN/build/MNNConvert"),
    os.path.expanduser("~/MNN/build/MNNConvert"),
    "/usr/local/bin/MNNConvert",
    "/usr/bin/MNNConvert",
]


def find_mnn_convert():
    """Find MNNConvert binary."""
    # Check PATH
    result = subprocess.run(["which", "MNNConvert"],
                            capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()

    # Check known paths
    for p in MNN_CONVERT_PATHS:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p

    return None


# ================================================================
# Conversion Functions
# ================================================================

def convert_onnx_to_mnn(onnx_path, mnn_path, mnn_convert,
                         fp16=False, quant_bits=0, optimize_level=1):
    """
    Convert a single ONNX model to MNN format.

    Args:
        onnx_path: Path to input ONNX model
        mnn_path: Path to output MNN model
        mnn_convert: Path to MNNConvert binary
        fp16: Use FP16 weights (reduces size ~2x)
        quant_bits: Weight quantization bits (0=none, 2-8)
        optimize_level: Graph optimization level (0-2)

    Returns:
        dict with conversion results
    """
    cmd = [
        mnn_convert,
        "--framework", "ONNX",
        "--modelFile", onnx_path,
        "--MNNModel", mnn_path,
        "--optimizeLevel", str(optimize_level),
    ]

    if fp16:
        cmd.append("--fp16")

    if quant_bits > 0:
        cmd.extend(["--weightQuantBits", str(quant_bits)])

    result = {
        "onnx_path": onnx_path,
        "mnn_path": mnn_path,
        "command": " ".join(cmd),
        "success": False,
        "error": None,
    }

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )

        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["returncode"] = proc.returncode

        if proc.returncode == 0 and os.path.exists(mnn_path):
            result["success"] = True
            result["onnx_size_mb"] = os.path.getsize(onnx_path) / (1024 * 1024)
            result["mnn_size_mb"] = os.path.getsize(mnn_path) / (1024 * 1024)
            result["compression_ratio"] = result["onnx_size_mb"] / max(result["mnn_size_mb"], 0.001)
            print(f"  [OK] {os.path.basename(onnx_path)} -> "
                  f"{os.path.basename(mnn_path)}  "
                  f"({result['onnx_size_mb']:.1f} MB -> {result['mnn_size_mb']:.1f} MB, "
                  f"{result['compression_ratio']:.1f}x)")
        else:
            result["error"] = proc.stderr or f"Exit code {proc.returncode}"
            print(f"  [FAIL] {os.path.basename(onnx_path)}: {result['error'][:200]}")

    except subprocess.TimeoutExpired:
        result["error"] = "Conversion timed out (>300s)"
        print(f"  [TIMEOUT] {os.path.basename(onnx_path)}")
    except Exception as e:
        result["error"] = str(e)
        print(f"  [ERROR] {os.path.basename(onnx_path)}: {e}")

    return result


def convert_all_models(onnx_dir, output_dir, mnn_convert,
                        fp16=False, quant_bits=0):
    """Convert all ONNX models in a directory to MNN."""
    os.makedirs(output_dir, exist_ok=True)

    onnx_files = sorted([
        f for f in os.listdir(onnx_dir)
        if f.endswith(".onnx")
    ])

    if not onnx_files:
        print(f"No ONNX files found in {onnx_dir}")
        return []

    print(f"Found {len(onnx_files)} ONNX models to convert")
    print(f"Options: fp16={fp16}, quant={quant_bits}bit")
    print("-" * 60)

    results = []
    for onnx_file in onnx_files:
        onnx_path = os.path.join(onnx_dir, onnx_file)
        mnn_file = onnx_file.replace(".onnx", ".mnn")
        mnn_path = os.path.join(output_dir, mnn_file)

        result = convert_onnx_to_mnn(
            onnx_path, mnn_path, mnn_convert,
            fp16=fp16, quant_bits=quant_bits,
        )
        results.append(result)

    return results


# ================================================================
# MNN Inference Test (if PyMNN is available)
# ================================================================

def test_mnn_inference(mnn_path):
    """
    Test MNN inference using PyMNN (if available).

    This is a basic sanity check that loads the model and runs
    a dummy forward pass.
    """
    try:
        import MNN
        import MNN.numpy as np_mnn

        print(f"  Testing MNN inference: {os.path.basename(mnn_path)}")

        # Load model
        interpreter = MNN.Interpreter(mnn_path)
        session = interpreter.createSession()
        input_tensor = interpreter.getSessionInput(session)

        # Get input shape
        shape = input_tensor.getShape()
        print(f"    Input shape: {shape}")

        # Create dummy data
        tmp_input = MNN.Tensor(
            input_tensor.getDimensionType(),
            shape,
            MNN.Halide_Type_Float,
            MNN.Tensor_DimensionType_Tensorflow,
        )

        # Fill with random data
        import numpy as np
        data = np.random.randn(*shape).astype(np.float32).flatten().tolist()
        tmp_input.copyFromHostTensor(MNN.Tensor(shape, MNN.Halide_Type_Float, data))
        input_tensor.copyFrom(tmp_input)

        # Run inference
        t0 = time.perf_counter()
        interpreter.runSession(session)
        t1 = time.perf_counter()

        print(f"    MNN inference time: {(t1-t0)*1000:.1f} ms")
        return True

    except ImportError:
        print(f"  [SKIP] PyMNN not installed -- cannot test MNN inference")
        print(f"         Install: pip install MNN")
        return None
    except Exception as e:
        print(f"  [ERROR] MNN inference failed: {e}")
        return False


# ================================================================
# Generate MNN Commands Documentation
# ================================================================

def generate_commands_doc(onnx_dir, output_dir, fp16=False, quant_bits=0):
    """Generate a document with MNNConvert commands for manual execution."""
    onnx_files = sorted([
        f for f in os.listdir(onnx_dir)
        if f.endswith(".onnx")
    ])

    lines = [
        "# MNN Conversion Commands",
        "# Run these manually if MNNConvert is not auto-detected.",
        "",
        f"# ONNX directory: {onnx_dir}",
        f"# MNN directory:  {output_dir}",
        "",
    ]

    for onnx_file in onnx_files:
        mnn_file = onnx_file.replace(".onnx", ".mnn")
        onnx_path = os.path.join(onnx_dir, onnx_file)
        mnn_path = os.path.join(output_dir, mnn_file)

        cmd = f"MNNConvert --framework ONNX --modelFile {onnx_path} --MNNModel {mnn_path}"
        if fp16:
            cmd += " --fp16"
        if quant_bits:
            cmd += f" --weightQuantBits {quant_bits}"

        lines.append(f"# {onnx_file}")
        lines.append(cmd)
        lines.append("")

    return "\n".join(lines)


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Neko Speech MNN Conversion")
    parser.add_argument("--onnx_dir", type=str, default="./onnx_models",
                        help="Directory containing ONNX models")
    parser.add_argument("--output_dir", type=str, default="./mnn_models",
                        help="Output directory for MNN models")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 weight storage")
    parser.add_argument("--quant", type=int, default=0,
                        help="Weight quantization bits (0=none, 2-8)")
    parser.add_argument("--mnn_convert", type=str, default=None,
                        help="Path to MNNConvert binary")
    parser.add_argument("--test_inference", action="store_true",
                        help="Test MNN inference after conversion")
    args = parser.parse_args()

    print("Neko Speech MNN Conversion")
    print("=" * 60)

    # Find MNNConvert
    mnn_convert = args.mnn_convert or find_mnn_convert()

    if mnn_convert is None:
        print("\n[!] MNNConvert not found!")
        print("    Build it from: https://github.com/alibaba/MNN")
        print("    cd MNN && mkdir build && cd build")
        print("    cmake .. -DMNN_BUILD_CONVERTER=ON && make -j$(nproc)")
        print()
        print("    Or pass --mnn_convert /path/to/MNNConvert")
        print()

        # Generate commands doc
        if os.path.isdir(args.onnx_dir):
            doc = generate_commands_doc(
                args.onnx_dir, args.output_dir,
                fp16=args.fp16, quant_bits=args.quant,
            )
            doc_path = os.path.join(args.output_dir or ".", "mnn_commands.sh")
            os.makedirs(os.path.dirname(doc_path), exist_ok=True)
            with open(doc_path, "w") as f:
                f.write(doc)
            print(f"    Commands saved to: {doc_path}")

        return

    print(f"MNNConvert: {mnn_convert}")
    print(f"ONNX dir:   {args.onnx_dir}")
    print(f"Output dir: {args.output_dir}")

    if not os.path.isdir(args.onnx_dir):
        print(f"\n[!] ONNX directory not found: {args.onnx_dir}")
        print(f"    Run export_onnx.py first to generate ONNX models.")
        return

    # Convert all models
    print()
    results = convert_all_models(
        args.onnx_dir, args.output_dir, mnn_convert,
        fp16=args.fp16, quant_bits=args.quant,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)

    n_success = sum(1 for r in results if r["success"])
    n_total = len(results)

    for r in results:
        status = "OK" if r["success"] else "FAIL"
        name = os.path.basename(r["onnx_path"])
        if r["success"]:
            print(f"  [{status}] {name:35s}  "
                  f"ONNX={r['onnx_size_mb']:.1f}MB  "
                  f"MNN={r['mnn_size_mb']:.1f}MB  "
                  f"ratio={r['compression_ratio']:.1f}x")
        else:
            print(f"  [{status}] {name:35s}  error={r['error'][:60]}")

    print(f"\n{n_success}/{n_total} models converted successfully")

    # Test inference
    if args.test_inference:
        print("\n" + "=" * 60)
        print("MNN Inference Test")
        print("=" * 60)
        for r in results:
            if r["success"]:
                test_mnn_inference(r["mnn_path"])

    # Save results
    results_path = os.path.join(args.output_dir, "conversion_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
