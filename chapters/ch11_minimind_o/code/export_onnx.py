"""
SimpleOmni ONNX Export
======================

Export the Thinker and Talker components of SimpleOmni to ONNX format.

Why export separately?
- Thinker runs once per text token (causal LM)
- Talker runs once per audio frame (acoustic generation)
- Different optimization strategies for each

Usage:
    python export_onnx.py --weight ./checkpoints/t2a_epoch1.pth --output_dir ./onnx/
"""

import os
import argparse
import torch
import numpy as np

from model import SimpleOmni, SimpleOmniConfig, count_parameters


# ===========================================================================
# Export Wrappers
# ===========================================================================

class ThinkerWrapper(torch.nn.Module):
    """Wrapper to export Thinker as standalone ONNX module.

    Outputs: text_logits + bridge states for Talker.
    """

    def __init__(self, model: SimpleOmni):
        super().__init__()
        self.thinker = model.thinker
        self.text_head = model.text_head

    def forward(self, input_ids):
        h_final, bridge, _ = self.thinker(input_ids, use_cache=False)
        text_logits = self.text_head(h_final)
        return text_logits, bridge


class TalkerWrapper(torch.nn.Module):
    """Wrapper to export Talker as standalone ONNX module.

    Inputs: bridge states from Thinker + audio codebook IDs.
    Outputs: stacked audio logits (num_codebooks, T, audio_vocab_size).
    """

    def __init__(self, model: SimpleOmni):
        super().__init__()
        self.talker = model.talker

    def forward(self, bridge, audio_ids):
        B, T = bridge.shape[:2]
        talker_emb = self.talker.embed_tokens(audio_ids)
        text_cond = self.talker.embed_proj(bridge) * self.talker.text_scale
        audio_cond = self.talker.codec_proj(talker_emb) * self.talker.audio_scale
        h = text_cond + audio_cond

        cos = self.talker.freqs_cos[:T]
        sin = self.talker.freqs_sin[:T]

        for layer in self.talker.layers:
            h, _ = layer(h, cos, sin, use_cache=False)

        h = self.talker.norm(h)
        audio_logits = self.talker.lm_head(h)

        # Stack list -> tensor for ONNX compatibility
        return torch.stack(audio_logits, dim=1)


# ===========================================================================
# Export Functions
# ===========================================================================

def export_thinker(model: SimpleOmni, output_dir: str, config: SimpleOmniConfig):
    """Export Thinker to ONNX with dynamic sequence length."""
    wrapper = ThinkerWrapper(model).eval()

    dummy_input = torch.randint(0, config.vocab_size, (1, 16))
    output_path = os.path.join(output_dir, "thinker.onnx")

    torch.onnx.export(
        wrapper, (dummy_input,), output_path,
        input_names=["input_ids"],
        output_names=["text_logits", "bridge"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "text_logits": {1: "seq_len"},
            "bridge": {1: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Thinker exported: {output_path} ({file_size:.1f} MB)")


def export_talker(model: SimpleOmni, output_dir: str, config: SimpleOmniConfig):
    """Export Talker to ONNX with dynamic sequence length."""
    wrapper = TalkerWrapper(model).eval()

    dummy_bridge = torch.randn(1, 16, config.hidden_size)
    dummy_audio_ids = torch.randint(0, config.audio_vocab_size,
                                     (1, config.num_codebooks, 16))
    output_path = os.path.join(output_dir, "talker.onnx")

    torch.onnx.export(
        wrapper, (dummy_bridge, dummy_audio_ids), output_path,
        input_names=["bridge", "audio_ids"],
        output_names=["audio_logits"],
        dynamic_axes={
            "bridge": {1: "seq_len"},
            "audio_ids": {2: "seq_len"},
            "audio_logits": {2: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Talker exported: {output_path} ({file_size:.1f} MB)")


def verify_onnx(output_dir: str, model: SimpleOmni, config: SimpleOmniConfig):
    """Verify ONNX outputs match PyTorch outputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping verification")
        return

    print("\nVerifying ONNX outputs...")

    thinker_path = os.path.join(output_dir, "thinker.onnx")
    if os.path.exists(thinker_path):
        sess = ort.InferenceSession(thinker_path, providers=["CPUExecutionProvider"])
        dummy = np.random.randint(0, config.vocab_size, (1, 16)).astype(np.int64)
        ort_out = sess.run(None, {"input_ids": dummy})

        with torch.no_grad():
            wrapper = ThinkerWrapper(model).eval()
            pt_out = wrapper(torch.from_numpy(dummy))

        text_diff = np.abs(ort_out[0] - pt_out[0].float().numpy()).max()
        bridge_diff = np.abs(ort_out[1] - pt_out[1].float().numpy()).max()
        print(f"  Thinker text_logits max diff: {text_diff:.6f}")
        print(f"  Thinker bridge max diff:      {bridge_diff:.6f}")
        ok = text_diff < 1e-2 and bridge_diff < 1e-2
        print(f"  {'PASS' if ok else 'FAIL'}")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SimpleOmni to ONNX")
    parser.add_argument("--weight", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--output_dir", type=str, default="./onnx",
                        help="Output directory for ONNX files")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX outputs match PyTorch")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = SimpleOmniConfig()
    model = SimpleOmni(config)
    if os.path.exists(args.weight):
        state_dict = torch.load(args.weight, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {args.weight}")
    model.eval()

    export_thinker(model, args.output_dir, config)
    export_talker(model, args.output_dir, config)

    if args.verify:
        verify_onnx(args.output_dir, model, config)

    print("\nExport complete!")
