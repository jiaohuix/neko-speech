"""
SimpleVoxCPM ONNX Export
========================

Export AudioVAE encoder/decoder and TSLM+RALM+CFM to ONNX for deployment.

Usage:
    python export_onnx.py --checkpoint checkpoints/voxcpm.pt
    python export_onnx.py  # uses random weights

Exports three models:
  1. audio_vae_encoder.onnx  — waveform → continuous latent
  2. audio_vae_decoder.onnx  — continuous latent → waveform
  3. tslm_ralm_dit.onnx      — text_emb + prev_audio → next latent patch (one AR step)

Then benchmarks ONNX inference speed vs PyTorch.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SimpleVoxCPM


# ---------------------------------------------------------------------------
# Wrapper modules for clean ONNX export
# ---------------------------------------------------------------------------

class AudioVAEEncoderWrapper(nn.Module):
    """Wraps AudioVAE.encode for ONNX export."""

    def __init__(self, vae):
        super().__init__()
        self.encoder = vae.encoder
        self.enc_to_latent = vae.enc_to_latent

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        h = self.encoder(waveform)
        return self.enc_to_latent(h)


class AudioVAEDecoderWrapper(nn.Module):
    """Wraps AudioVAE.decode for ONNX export."""

    def __init__(self, vae):
        super().__init__()
        self.latent_to_dec = vae.latent_to_dec
        self.decoder = vae.decoder
        self.dec_to_out = vae.dec_to_out

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.latent_to_dec(z)
        h = self.decoder(h)
        out = self.dec_to_out(h)
        return out.squeeze(1)


class OneStepARWrapper(nn.Module):
    """Wraps one autoregressive step: text_emb + prev_audio_emb → next latent patch.

    In deployment, this would be called once per AR step with KV-cache.
    For ONNX export, we simplify to a single-step forward.
    """

    def __init__(self, model: SimpleVoxCPM):
        super().__init__()
        self.loc_enc = model.loc_enc
        self.enc_to_tslm = model.enc_to_tslm
        self.tslm = model.tslm
        self.fsq = model.fsq
        self.ralm = model.ralm
        self.lm_to_dit = model.lm_to_dit
        self.res_to_dit = model.res_to_dit

    def forward(self, text_emb: torch.Tensor, prev_audio_emb: torch.Tensor,
                position: torch.Tensor) -> torch.Tensor:
        """
        text_emb:      (1, L, H)  pre-computed text embeddings
        prev_audio_emb: (1, K, H)  previous audio embeddings (including BOS)
        position:      (1,)       current generation position index

        Returns: conditioning vector (1, dit_hidden)
        """
        combined = torch.cat([text_emb, prev_audio_emb], dim=1)
        tslm_out = self.tslm(combined)
        tslm_step = tslm_out[:, -1:, :]

        fsq_out = self.fsq(tslm_step)
        ralm_out = self.ralm.forward_step(fsq_out.squeeze(1), position.item())
        ralm_step = ralm_out.unsqueeze(1)

        dit_cond = self.lm_to_dit(tslm_step) + self.res_to_dit(ralm_step)
        return dit_cond.squeeze(1)


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_audio_vae(model: SimpleVoxCPM, out_dir: str, audio_len: int = 6400):
    """Export AudioVAE encoder and decoder."""
    latent_dim = model.audio_vae.latent_dim
    T_latent = audio_len // model.audio_vae.chunk_size

    # Encoder
    enc = AudioVAEEncoderWrapper(model.audio_vae).eval()
    dummy_wav = torch.randn(1, audio_len)
    enc_path = os.path.join(out_dir, "audio_vae_encoder.onnx")
    torch.onnx.export(
        enc, (dummy_wav,),
        enc_path,
        input_names=["waveform"],
        output_names=["latent"],
        dynamic_axes={"waveform": {1: "audio_len"}, "latent": {2: "t_latent"}},
        opset_version=14,
    )
    print(f"Exported encoder → {enc_path}")

    # Decoder
    dec = AudioVAEDecoderWrapper(model.audio_vae).eval()
    dummy_z = torch.randn(1, latent_dim, T_latent)
    dec_path = os.path.join(out_dir, "audio_vae_decoder.onnx")
    torch.onnx.export(
        dec, (dummy_z,),
        dec_path,
        input_names=["latent"],
        output_names=["waveform"],
        dynamic_axes={"latent": {2: "t_latent"}, "waveform": {1: "audio_len"}},
        opset_version=14,
    )
    print(f"Exported decoder → {dec_path}")

    return enc_path, dec_path


def export_ar_step(model: SimpleVoxCPM, out_dir: str, text_len: int = 16):
    """Export one AR step (TSLM + FSQ + RALM → conditioning vector)."""
    wrapper = OneStepARWrapper(model).eval()
    H = model.tslm_hidden
    dummy_text = torch.randn(1, text_len, H)
    dummy_audio = torch.randn(1, 1, H)
    dummy_pos = torch.tensor([0])

    ar_path = os.path.join(out_dir, "tslm_ralm_step.onnx")
    torch.onnx.export(
        wrapper, (dummy_text, dummy_audio, dummy_pos),
        ar_path,
        input_names=["text_emb", "prev_audio_emb", "position"],
        output_names=["dit_conditioning"],
        dynamic_axes={
            "text_emb": {1: "text_len"},
            "prev_audio_emb": {1: "audio_history_len"},
        },
        opset_version=14,
    )
    print(f"Exported AR step → {ar_path}")
    return ar_path


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_onnx(onnx_path: str, dummy_inputs: list, n_warmup: int = 5,
                   n_runs: int = 50) -> float:
    """Benchmark an ONNX model and return average latency in ms."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed — skipping ONNX benchmark")
        return -1.0

    sess = ort.InferenceSession(onnx_path)
    input_names = [inp.name for inp in sess.get_inputs()]
    feed = {name: inp.numpy() for name, inp in zip(input_names, dummy_inputs)}

    # Warmup
    for _ in range(n_warmup):
        sess.run(None, feed)

    # Timed runs
    t0 = time.time()
    for _ in range(n_runs):
        sess.run(None, feed)
    elapsed = time.time() - t0
    return elapsed / n_runs * 1000  # ms


def benchmark_pytorch(model: nn.Module, dummy_inputs: tuple, n_warmup: int = 5,
                      n_runs: int = 50) -> float:
    """Benchmark a PyTorch model and return average latency in ms."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(*dummy_inputs)
        t0 = time.time()
        for _ in range(n_runs):
            model(*dummy_inputs)
        elapsed = time.time() - t0
    return elapsed / n_runs * 1000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export SimpleVoxCPM to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to checkpoint (default: random weights)")
    parser.add_argument("--output-dir", type=str, default="onnx_models",
                        help="output directory for ONNX files")
    parser.add_argument("--audio-len", type=int, default=6400,
                        help="audio length for encoder export (default: 6400 = 0.4s)")
    parser.add_argument("--text-len", type=int, default=16,
                        help="text length for AR step export (default: 16)")
    parser.add_argument("--benchmark", action="store_true", default=True,
                        help="run ONNX vs PyTorch benchmark (default: True)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Build model ---
    model = SimpleVoxCPM()
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}\n")
    else:
        print("No checkpoint — using random weights\n")

    model.eval()

    # --- Export ---
    print("=" * 60)
    print("Exporting ONNX models")
    print("=" * 60)

    enc_path, dec_path = export_audio_vae(model, args.output_dir, args.audio_len)
    ar_path = export_ar_step(model, args.output_dir, args.text_len)

    # --- Benchmark ---
    if args.benchmark:
        print(f"\n{'='*60}")
        print("Benchmarking (CPU)")
        print("=" * 60)

        # AudioVAE Encoder
        dummy_wav = torch.randn(1, args.audio_len)
        pt_lat = benchmark_pytorch(AudioVAEEncoderWrapper(model.audio_vae), (dummy_wav,))
        onnx_lat = benchmark_onnx(enc_path, [dummy_wav])
        print(f"\nAudioVAE Encoder:")
        print(f"  PyTorch:    {pt_lat:.2f} ms")
        if onnx_lat > 0:
            print(f"  ONNX:       {onnx_lat:.2f} ms")
            print(f"  Speedup:    {pt_lat / onnx_lat:.2f}x")

        # AudioVAE Decoder
        T_lat = args.audio_len // model.audio_vae.chunk_size
        dummy_z = torch.randn(1, model.audio_vae.latent_dim, T_lat)
        pt_lat = benchmark_pytorch(AudioVAEDecoderWrapper(model.audio_vae), (dummy_z,))
        onnx_lat = benchmark_onnx(dec_path, [dummy_z])
        print(f"\nAudioVAE Decoder:")
        print(f"  PyTorch:    {pt_lat:.2f} ms")
        if onnx_lat > 0:
            print(f"  ONNX:       {onnx_lat:.2f} ms")
            print(f"  Speedup:    {pt_lat / onnx_lat:.2f}x")

    print(f"\n{'='*60}")
    print(f"Export complete. Files in: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
