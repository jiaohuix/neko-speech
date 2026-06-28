"""
Ch09: GPT-SoVITS -- ONNX Export

Export the AR model and SoVITS vocoder components to ONNX format
for deployment with ONNX Runtime.

The export follows the GPT-SoVITS convention of splitting the system
into separate ONNX models:
    1. AR model (text -> semantic tokens)
    2. SoVITS vocoder (semantic tokens -> waveform)

Usage:
    python export_onnx.py --output-dir ../onnx_models
    python export_onnx.py --checkpoint ../checkpoints/ar_model.pt --output-dir ../onnx_models

Note:
    The AR model uses dynamic axes for variable-length text/audio.
    The SoVITS model exports TextEncoder + Flow + Generator as a single graph.
"""

import argparse
import os
import sys
import time

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from model import SimpleAR, GPTSoVITS


# ===================================================================
# AR Model ONNX Wrapper
# ===================================================================

class ARModelONNXWrapper(torch.nn.Module):
    """
    Wrapper to export the AR model as a single ONNX graph.

    The full autoregressive loop is NOT exported -- instead we export
    a single forward pass.  The autoregressive loop is implemented
    in the inference runtime (ONNX Runtime or Python).

    Inputs:
        phoneme_ids:  (1, T_text)   -- text phoneme IDs
        semantic_ids: (1, T_audio)  -- current semantic token sequence

    Output:
        logits: (1, T_audio, vocab_size) -- predicted token logits
    """

    def __init__(self, ar_model):
        super().__init__()
        self.ar = ar_model

    def forward(self, phoneme_ids, semantic_ids):
        return self.ar(phoneme_ids, semantic_ids)


# ===================================================================
# SoVITS ONNX Wrapper
# ===================================================================

class SoVITSModelONNXWrapper(torch.nn.Module):
    """
    Wrapper to export the SoVITS vocoder (inference path only).

    Exports: TextEncoder + Flow(inverse) + Generator
    Skips: PosteriorEncoder (training only), Discriminator (training only)

    Inputs:
        phoneme_ids:  (1, T_text)    -- text phoneme IDs
        speaker_emb:  (1, speaker_dim) -- speaker embedding
        noise_scale:  scalar

    Output:
        waveform: (1, 1, T_wav) -- generated audio
    """

    def __init__(self, model: GPTSoVITS):
        super().__init__()
        self.text_encoder = model.text_encoder
        self.flow = model.flow
        self.generator = model.generator

    def forward(self, phoneme_ids, speaker_emb):
        # Prior from text
        m_p, logs_p = self.text_encoder(phoneme_ids, speaker_emb)

        # Sample from prior (deterministic for export -- use zeros as noise)
        z_p = m_p  # mean only, no sampling noise in exported model

        # Inverse flow
        z, _ = self.flow(z_p, reverse=True)

        # Generate waveform
        waveform = self.generator(z)
        return waveform


# ===================================================================
# Export functions
# ===================================================================

def export_ar_model(model: SimpleAR, output_dir: str, opset: int = 16):
    """
    Export the AR model to ONNX.

    Args:
        model: SimpleAR model
        output_dir: output directory
        opset: ONNX opset version

    Returns:
        path to exported .onnx file
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'ar_model.onnx')

    wrapper = ARModelONNXWrapper(model)
    wrapper.eval()

    # Dummy inputs
    phoneme_ids = torch.randint(0, 512, (1, 10))
    semantic_ids = torch.randint(0, 1024, (1, 20))

    print(f"  Exporting AR model to {path}...")
    try:
        torch.onnx.export(
            wrapper,
            (phoneme_ids, semantic_ids),
            path,
            input_names=['phoneme_ids', 'semantic_ids'],
            output_names=['logits'],
            dynamic_axes={
                'phoneme_ids': {1: 'text_len'},
                'semantic_ids': {1: 'audio_len'},
                'logits': {1: 'audio_len'},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
        file_size = os.path.getsize(path) / (1024 * 1024)
        print(f"  AR model exported: {path} ({file_size:.1f} MB)")
        return path
    except Exception as e:
        print(f"  AR export failed: {e}")
        return None


def export_sovits_model(model: GPTSoVITS, output_dir: str, opset: int = 17):
    """
    Export the SoVITS vocoder to ONNX (inference path only).

    Args:
        model: GPTSoVITS model
        output_dir: output directory
        opset: ONNX opset version (17 for newer ops)

    Returns:
        path to exported .onnx file
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'sovits_model.onnx')

    wrapper = SoVITSModelONNXWrapper(model)
    wrapper.eval()

    # Dummy inputs
    phoneme_ids = torch.randint(0, 512, (1, 10))
    speaker_emb = torch.randn(1, 256)

    print(f"  Exporting SoVITS model to {path}...")
    try:
        torch.onnx.export(
            wrapper,
            (phoneme_ids, speaker_emb),
            path,
            input_names=['phoneme_ids', 'speaker_emb'],
            output_names=['waveform'],
            dynamic_axes={
                'phoneme_ids': {1: 'text_len'},
                'waveform': {2: 'wav_len'},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
        file_size = os.path.getsize(path) / (1024 * 1024)
        print(f"  SoVITS model exported: {path} ({file_size:.1f} MB)")
        return path
    except Exception as e:
        print(f"  SoVITS export failed: {e}")
        return None


# ===================================================================
# ONNX Runtime benchmark
# ===================================================================

def benchmark_onnx_vs_pytorch(
    ar_model: SimpleAR,
    ar_onnx_path: str,
    n_runs: int = 10,
):
    """
    Compare ONNX Runtime inference speed vs PyTorch.

    Only benchmarks the AR model single forward pass (not the full
    autoregressive loop).

    Args:
        ar_model: PyTorch AR model
        ar_onnx_path: path to exported ONNX model
        n_runs: number of benchmark iterations
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed, skipping ONNX benchmark")
        print("  Install with: pip install onnxruntime")
        return

    device = 'cpu'
    ar_model.eval()

    phoneme_ids = torch.randint(0, 512, (1, 10))
    semantic_ids = torch.randint(0, 1024, (1, 50))

    # --- PyTorch benchmark ---
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = ar_model(phoneme_ids, semantic_ids)

        t_start = time.time()
        for _ in range(n_runs):
            _ = ar_model(phoneme_ids, semantic_ids)
        pytorch_time = (time.time() - t_start) / n_runs

    # --- ONNX Runtime benchmark ---
    sess = ort.InferenceSession(ar_onnx_path)
    inputs = {
        'phoneme_ids': phoneme_ids.numpy(),
        'semantic_ids': semantic_ids.numpy(),
    }

    # Warmup
    for _ in range(3):
        _ = sess.run(None, inputs)

    t_start = time.time()
    for _ in range(n_runs):
        _ = sess.run(None, inputs)
    onnx_time = (time.time() - t_start) / n_runs

    print(f"\n  AR Model Forward Pass Benchmark ({n_runs} runs):")
    print(f"    PyTorch:      {pytorch_time*1000:.2f} ms")
    print(f"    ONNX Runtime: {onnx_time*1000:.2f} ms")
    if onnx_time > 0:
        speedup = pytorch_time / onnx_time
        print(f"    Speedup:      {speedup:.2f}x")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description='GPT-SoVITS ONNX Export')
    parser.add_argument('--ar-checkpoint', type=str, default=None,
                        help='Path to AR model checkpoint')
    parser.add_argument('--sovits-checkpoint', type=str, default=None,
                        help='Path to SoVITS model checkpoint')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'onnx_models'),
                        help='Output directory for ONNX models')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run ONNX vs PyTorch speed benchmark')
    args = parser.parse_args()

    print("=" * 60)
    print("GPT-SoVITS ONNX Export")
    print("=" * 60)

    # Build models
    print("\nBuilding models...")
    ar_model = SimpleAR(dim=384, n_heads=8, n_layers=8)

    sovits_model = GPTSoVITS()

    # Load checkpoints if available
    if args.ar_checkpoint and os.path.exists(args.ar_checkpoint):
        print(f"  Loading AR checkpoint: {args.ar_checkpoint}")
        ckpt = torch.load(args.ar_checkpoint, map_location='cpu', weights_only=True)
        ar_model.load_state_dict(ckpt['model_state_dict'])

    if args.sovits_checkpoint and os.path.exists(args.sovits_checkpoint):
        print(f"  Loading SoVITS checkpoint: {args.sovits_checkpoint}")
        ckpt = torch.load(args.sovits_checkpoint, map_location='cpu', weights_only=True)
        sovits_model.load_state_dict(ckpt['model_state_dict'])

    # Export
    print("\n--- Exporting AR model ---")
    ar_onnx_path = export_ar_model(ar_model, args.output_dir, opset=16)

    print("\n--- Exporting SoVITS model ---")
    sovits_onnx_path = export_sovits_model(sovits_model, args.output_dir, opset=17)

    # Verify exported models
    if ar_onnx_path:
        try:
            import onnx
            model = onnx.load(ar_onnx_path)
            onnx.checker.check_model(model)
            print(f"\n  AR ONNX model verification: PASSED")
        except ImportError:
            print("\n  onnx package not installed, skipping verification")
        except Exception as e:
            print(f"\n  AR ONNX verification failed: {e}")

    if sovits_onnx_path:
        try:
            import onnx
            model = onnx.load(sovits_onnx_path)
            onnx.checker.check_model(model)
            print(f"  SoVITS ONNX model verification: PASSED")
        except ImportError:
            print("  onnx package not installed, skipping verification")
        except Exception as e:
            print(f"  SoVITS ONNX verification failed: {e}")

    # Benchmark
    if args.benchmark and ar_onnx_path:
        print("\n--- Speed Benchmark ---")
        benchmark_onnx_vs_pytorch(ar_model, ar_onnx_path, n_runs=20)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
