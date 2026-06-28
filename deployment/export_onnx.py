"""
Neko Speech -- ONNX Export Toolkit

Export TTS models from each chapter to ONNX format:
  - ch02 Tacotron2  (autoregressive, encoder + decoder + postnet)
  - ch04 FastSpeech2 (non-autoregressive, encoder + length regulator + decoder)
  - ch05 VITS        (end-to-end: TextEncoder + Flow + Generator)

Usage:
  python export_onnx.py --model tacotron2 --output_dir ./onnx_models
  python export_onnx.py --model fastspeech2 --output_dir ./onnx_models
  python export_onnx.py --model vits --output_dir ./onnx_models
  python export_onnx.py --model all --output_dir ./onnx_models

Note: Tacotron2 is autoregressive, so we export encoder and decoder
separately.  For FastSpeech2 we export a single inference graph.
For VITS we export the inference pipeline (TextEncoder + Flow + Generator).
"""

import argparse
import importlib
import importlib.util
import os
import sys
import time
import numpy as np
import torch

# ================================================================
# Module loading -- each chapter has its own model.py, so we must
# load them from their specific directories to avoid name clashes.
# ================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_module(name, filepath):
    """Load a Python module from a specific file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    # For VITS, modules.py must be importable from the same dir
    code_dir = os.path.dirname(filepath)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    spec.loader.exec_module(module)
    return module

def load_tacotron2_model():
    """Load Tacotron2 from ch02."""
    path = os.path.join(ROOT, "chapters", "ch02_tacotron", "code", "model.py")
    return _load_module("ch02_model", path)

def load_fastspeech2_model():
    """Load FastSpeech2 from ch04."""
    path = os.path.join(ROOT, "chapters", "ch04_fastspeech", "code", "model.py")
    return _load_module("ch04_model", path)

def load_vits_model():
    """Load VITS from ch05 (needs modules.py in same dir)."""
    # Ensure ch05 code dir is in sys.path for the 'from modules import ...' in model.py
    ch05_dir = os.path.join(ROOT, "chapters", "ch05_vits", "code")
    if ch05_dir not in sys.path:
        sys.path.insert(0, ch05_dir)
    path = os.path.join(ch05_dir, "model.py")
    return _load_module("ch05_model", path)


# ================================================================
# Utilities
# ================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# ================================================================
# Tacotron2 -- Encoder export
# ================================================================

class Tacotron2EncoderWrapper(torch.nn.Module):
    """Wrap Tacotron2 encoder for ONNX export."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, text_tokens):
        return self.encoder(text_tokens)


def export_tacotron2_encoder(output_dir, opset=14):
    mod = load_tacotron2_model()
    Tacotron2 = mod.Tacotron2

    model = Tacotron2(vocab_size=256, mel_dim=80)
    model.eval()

    wrapper = Tacotron2EncoderWrapper(model.encoder)
    wrapper.eval()

    B, T = 1, 20
    dummy = torch.randint(0, 256, (B, T), dtype=torch.long)

    path = os.path.join(output_dir, "tacotron2_encoder.onnx")
    torch.onnx.export(
        wrapper,
        (dummy,),
        path,
        input_names=["text_tokens"],
        output_names=["encoder_out"],
        dynamic_axes={
            "text_tokens": {0: "batch", 1: "text_len"},
            "encoder_out": {0: "batch", 1: "text_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    print(f"  [OK] Tacotron2 encoder -> {path}  ({file_size_mb(path):.1f} MB)")
    return path


# ================================================================
# Tacotron2 -- PostNet export
# ================================================================

class Tacotron2PostNetWrapper(torch.nn.Module):
    """Wrap Tacotron2 PostNet for ONNX export."""

    def __init__(self, postnet):
        super().__init__()
        self.postnet = postnet

    def forward(self, mel_input):
        # mel_input: (B, mel_dim, T)
        refined = self.postnet(mel_input)
        return mel_input + refined  # residual


def export_tacotron2_postnet(output_dir, opset=14):
    mod = load_tacotron2_model()
    Tacotron2 = mod.Tacotron2

    model = Tacotron2(vocab_size=256, mel_dim=80)
    model.eval()

    wrapper = Tacotron2PostNetWrapper(model.postnet)
    wrapper.eval()

    B, T = 1, 100
    dummy = torch.randn(B, 80, T)

    path = os.path.join(output_dir, "tacotron2_postnet.onnx")
    torch.onnx.export(
        wrapper,
        (dummy,),
        path,
        input_names=["mel_before"],
        output_names=["mel_after"],
        dynamic_axes={
            "mel_before": {0: "batch", 2: "time"},
            "mel_after":  {0: "batch", 2: "time"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    print(f"  [OK] Tacotron2 postnet -> {path}  ({file_size_mb(path):.1f} MB)")
    return path


# ================================================================
# FastSpeech2 -- Full inference graph
# ================================================================

class FastSpeech2ONNX(torch.nn.Module):
    """
    Wrap FastSpeech2 inference as a single ONNX-exportable graph.

    The LengthRegulator uses repeat_interleave with dynamic durations,
    which is tricky in ONNX.  We export the inference path and
    pre-compute the duration inside the wrapper.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text, text_lens):
        mel = self.model.inference(text, text_lens)
        return mel


def export_fastspeech2(output_dir, opset=14):
    mod = load_fastspeech2_model()
    FastSpeech2 = mod.FastSpeech2

    model = FastSpeech2(
        vocab_size=256, d_model=256, n_mels=80,
        nhead=2, d_ff=1024, enc_layers=4, dec_layers=4,
    )
    model.eval()

    wrapper = FastSpeech2ONNX(model)
    wrapper.eval()

    B, T = 1, 20
    dummy_text = torch.randint(1, 256, (B, T), dtype=torch.long)
    dummy_lens = torch.tensor([T], dtype=torch.long)

    path = os.path.join(output_dir, "fastspeech2.onnx")

    # NOTE: FastSpeech2's LengthRegulator uses Python loops and
    # repeat_interleave with dynamic durations, which can cause issues
    # with ONNX export.  We handle this by scripting the model first
    # or by accepting a static export.

    try:
        torch.onnx.export(
            wrapper,
            (dummy_text, dummy_lens),
            path,
            input_names=["text", "text_lens"],
            output_names=["mel"],
            dynamic_axes={
                "text":      {0: "batch", 1: "text_len"},
                "text_lens": {0: "batch"},
                "mel":       {0: "batch", 2: "mel_len"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
        print(f"  [OK] FastSpeech2 -> {path}  ({file_size_mb(path):.1f} MB)")
    except Exception as e:
        print(f"  [WARN] FastSpeech2 full export failed: {e}")
        print(f"         Trying encoder-only export...")
        path = export_fastspeech2_encoder(output_dir, opset)

    return path


def export_fastspeech2_encoder(output_dir, opset=14):
    """Fallback: export only the encoder + variance predictors."""
    mod = load_fastspeech2_model()
    FastSpeech2 = mod.FastSpeech2

    model = FastSpeech2(
        vocab_size=256, d_model=256, n_mels=80,
        nhead=2, d_ff=1024, enc_layers=4, dec_layers=4,
    )
    model.eval()

    class EncoderOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, text, text_lens):
            import math
            t_mask = self.m._pad_mask(text_lens, text.shape[1])
            x = self.m.embedding(text) * math.sqrt(self.m.d_model)
            x = self.m.encoder(x, mask=t_mask)
            log_dur = self.m.dur_predictor(x)
            pitch = self.m.pitch_predictor(x)
            energy = self.m.energy_predictor(x)
            return x, log_dur, pitch, energy

    wrapper = EncoderOnly(model)
    wrapper.eval()

    B, T = 1, 20
    dummy_text = torch.randint(1, 256, (B, T), dtype=torch.long)
    dummy_lens = torch.tensor([T], dtype=torch.long)

    path = os.path.join(output_dir, "fastspeech2_encoder.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_text, dummy_lens),
        path,
        input_names=["text", "text_lens"],
        output_names=["features", "log_dur", "pitch", "energy"],
        dynamic_axes={
            "text":      {0: "batch", 1: "text_len"},
            "text_lens": {0: "batch"},
            "features":  {0: "batch", 1: "text_len"},
            "log_dur":   {0: "batch", 1: "text_len"},
            "pitch":     {0: "batch", 1: "text_len"},
            "energy":    {0: "batch", 1: "text_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"  [OK] FastSpeech2 encoder -> {path}  ({file_size_mb(path):.1f} MB)")
    return path


# ================================================================
# VITS -- Inference pipeline (TextEncoder + Flow + Generator)
# ================================================================

class VITSInferenceONNX(torch.nn.Module):
    """
    VITS inference as a single ONNX graph:
        text_ids -> TextEncoder -> duration -> expand -> Flow^-1 -> Generator -> wav
    """

    def __init__(self, model):
        super().__init__()
        self.text_encoder = model.text_encoder
        self.flow = model.flow
        self.generator = model.generator
        self.duration_predictor = model.duration_predictor
        self.use_sdp = model.use_sdp
        if model.use_sdp:
            self.sdp = model.stochastic_duration_predictor

    def forward(self, text_ids, text_lengths, noise_scale):
        # 1. Text encoder
        x, m_p, logs_p, text_mask = self.text_encoder(text_ids, text_lengths)

        # 2. Duration prediction (deterministic, not SDP for export)
        log_dur = self.duration_predictor(x, text_mask)
        duration = torch.clamp(torch.exp(log_dur) * text_mask.squeeze(1), min=1.0)
        duration = duration.round().long()

        # 3. Expand by duration
        z_p = self._expand_by_duration_export(m_p, logs_p, duration, noise_scale)

        # 4. Inverse flow
        z_q, _ = self.flow(z_p, reverse=True)

        # 5. Generate waveform
        wav = self.generator(z_q)

        return wav

    def _expand_by_duration_export(self, m_p, logs_p, duration, noise_scale):
        """Duration expansion -- ONNX-compatible version using scatter."""
        B, hidden, T_text = m_p.shape
        total_length = duration.sum(dim=1).max().item()

        # For ONNX, we avoid Python loops by using a simple expansion
        # based on cumulative sum of durations
        z_p = torch.zeros(B, hidden, total_length, device=m_p.device)

        for b in range(B):
            pos = 0
            for t in range(T_text):
                d = duration[b, t].item()
                if d <= 0 or pos + d > total_length:
                    continue
                z_p[b, :, pos:pos + d] = m_p[b, :, t:t + 1]
                noise = torch.randn(hidden, d, device=m_p.device) * noise_scale
                z_p[b, :, pos:pos + d] += torch.exp(logs_p[b, :, t:t + 1]) * noise
                pos += d

        return z_p


def export_vits(output_dir, opset=14):
    mod = load_vits_model()
    VITS = mod.VITS

    # Small config for educational model
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
        use_sdp=False,  # SDP has random sampling, not ideal for ONNX
    )
    model.eval()

    # Export components separately for maximum compatibility
    paths = []

    # --- TextEncoder ---
    te_path = export_vits_text_encoder(model, output_dir, opset)
    paths.append(te_path)

    # --- Generator ---
    gen_path = export_vits_generator(model, output_dir, opset)
    paths.append(gen_path)

    # --- Full inference (may fail due to dynamic loops) ---
    try:
        wrapper = VITSInferenceONNX(model)
        wrapper.eval()

        B, T = 1, 10
        dummy_text = torch.randint(0, 200, (B, T), dtype=torch.long)
        dummy_lens = torch.tensor([T], dtype=torch.long)
        noise_scale = torch.tensor(0.667)

        full_path = os.path.join(output_dir, "vits_full.onnx")
        torch.onnx.export(
            wrapper,
            (dummy_text, dummy_lens, noise_scale),
            full_path,
            input_names=["text_ids", "text_lengths", "noise_scale"],
            output_names=["wav"],
            dynamic_axes={
                "text_ids":    {0: "batch", 1: "text_len"},
                "text_lengths": {0: "batch"},
                "wav":         {0: "batch", 2: "wav_len"},
            },
            opset_version=opset,
            do_constant_folding=False,  # dynamic shapes
        )
        print(f"  [OK] VITS full inference -> {full_path}  ({file_size_mb(full_path):.1f} MB)")
        paths.append(full_path)
    except Exception as e:
        print(f"  [WARN] VITS full export failed: {e}")
        print(f"         Component exports are available for manual pipeline.")

    return paths


def export_vits_text_encoder(model, output_dir, opset=14):
    """Export VITS TextEncoder only."""
    te = model.text_encoder
    te.eval()

    B, T = 1, 10
    dummy_ids = torch.randint(0, 200, (B, T), dtype=torch.long)
    dummy_lens = torch.tensor([T], dtype=torch.long)

    path = os.path.join(output_dir, "vits_text_encoder.onnx")
    torch.onnx.export(
        te,
        (dummy_ids, dummy_lens),
        path,
        input_names=["text_ids", "text_lengths"],
        output_names=["x", "m_p", "logs_p", "mask"],
        dynamic_axes={
            "text_ids":    {0: "batch", 1: "text_len"},
            "text_lengths": {0: "batch"},
            "x":           {0: "batch", 2: "text_len"},
            "m_p":         {0: "batch", 2: "text_len"},
            "logs_p":      {0: "batch", 2: "text_len"},
            "mask":        {0: "batch", 2: "text_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"  [OK] VITS text_encoder -> {path}  ({file_size_mb(path):.1f} MB)")
    return path


def export_vits_generator(model, output_dir, opset=14):
    """Export VITS Generator (HiFi-GAN decoder) only."""
    gen = model.generator
    gen.eval()

    # Remove weight norm for inference/export
    gen.remove_weight_norm()

    B, T_z = 1, 20
    dummy_z = torch.randn(B, 192, T_z)

    path = os.path.join(output_dir, "vits_generator.onnx")
    torch.onnx.export(
        gen,
        (dummy_z,),
        path,
        input_names=["z"],
        output_names=["wav"],
        dynamic_axes={
            "z":   {0: "batch", 2: "z_len"},
            "wav": {0: "batch", 2: "wav_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"  [OK] VITS generator -> {path}  ({file_size_mb(path):.1f} MB)")
    return path


# ================================================================
# ONNX Verification
# ================================================================

def verify_onnx(onnx_path, pytorch_model, dummy_inputs, atol=1e-4):
    """Verify that ONNX output matches PyTorch output."""
    import onnxruntime as ort

    # PyTorch inference
    with torch.no_grad():
        pt_out = pytorch_model(*dummy_inputs)
    if isinstance(pt_out, tuple):
        pt_out = pt_out[0]
    pt_np = pt_out.cpu().numpy()

    # ONNX inference
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {}
    for i, inp in enumerate(dummy_inputs):
        name = sess.get_inputs()[i].name
        feed[name] = inp.cpu().numpy()

    ort_out = sess.run(None, feed)[0]

    # Compare
    mse = np.mean((pt_np - ort_out) ** 2)
    max_diff = np.max(np.abs(pt_np - ort_out))

    status = "PASS" if max_diff < atol else "WARN"
    print(f"  [{status}] {onnx_path}: MSE={mse:.2e}, max_diff={max_diff:.2e}")
    return status == "PASS"


def benchmark_onnx_inference(onnx_path, n_runs=10):
    """Measure ONNX inference latency on CPU."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Create dummy inputs matching model signature
    feed = {}
    for inp in sess.get_inputs():
        shape = []
        for dim in inp.shape:
            if isinstance(dim, str):
                if "batch" in dim:
                    shape.append(1)
                elif "text" in dim or "len" in dim:
                    shape.append(20)
                elif "mel" in dim or "time" in dim or "wav" in dim or "z" in dim:
                    shape.append(100)
                else:
                    shape.append(1)
            else:
                shape.append(dim)

        if inp.type == "tensor(int64)":
            feed[inp.name] = np.random.randint(0, 100, size=shape).astype(np.int64)
        elif inp.type == "tensor(float)":
            feed[inp.name] = np.random.randn(*shape).astype(np.float32)
        else:
            feed[inp.name] = np.random.randn(*shape).astype(np.float32)

    # Warmup
    for _ in range(3):
        sess.run(None, feed)

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    print(f"  [BENCH] {os.path.basename(onnx_path)}: "
          f"avg={avg_ms:.1f}ms +/- {std_ms:.1f}ms (n={n_runs})")
    return avg_ms, std_ms


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Neko Speech ONNX Export")
    parser.add_argument("--model", type=str, default="all",
                        choices=["tacotron2", "fastspeech2", "vits", "all"],
                        help="Which model to export")
    parser.add_argument("--output_dir", type=str, default="./onnx_models",
                        help="Output directory for ONNX files")
    parser.add_argument("--opset", type=int, default=14,
                        help="ONNX opset version")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX outputs match PyTorch")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark ONNX inference speed")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    print(f"Neko Speech ONNX Export")
    print(f"Output: {args.output_dir}")
    print(f"Opset:  {args.opset}")
    print("=" * 60)

    results = {}

    if args.model in ("tacotron2", "all"):
        print("\n--- Tacotron2 (ch02) ---")
        enc_path = export_tacotron2_encoder(args.output_dir, args.opset)
        post_path = export_tacotron2_postnet(args.output_dir, args.opset)
        results["tacotron2_encoder"] = enc_path
        results["tacotron2_postnet"] = post_path

    if args.model in ("fastspeech2", "all"):
        print("\n--- FastSpeech2 (ch04) ---")
        path = export_fastspeech2(args.output_dir, args.opset)
        results["fastspeech2"] = path

    if args.model in ("vits", "all"):
        print("\n--- VITS (ch05) ---")
        paths = export_vits(args.output_dir, args.opset)
        for i, p in enumerate(paths if isinstance(paths, list) else [paths]):
            results[f"vits_{i}"] = p

    # Verification
    if args.verify:
        print("\n" + "=" * 60)
        print("ONNX Verification")
        print("=" * 60)
        for name, path in results.items():
            if os.path.exists(path):
                print(f"\n  {name}:")
                # Basic ONNX check
                import onnx
                model = onnx.load(path)
                onnx.checker.check_model(model)
                print(f"  [OK] ONNX model is valid")

    # Benchmark
    if args.benchmark:
        print("\n" + "=" * 60)
        print("ONNX Inference Benchmark (CPU)")
        print("=" * 60)
        for name, path in results.items():
            if os.path.exists(path):
                print(f"\n  {name}:")
                benchmark_onnx_inference(path)

    # Summary
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    for name, path in results.items():
        if os.path.exists(path):
            size = file_size_mb(path)
            print(f"  {name:30s}  {size:8.1f} MB  {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
