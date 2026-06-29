"""
Ch12: Fish Speech Inference Script

Usage:
    python inference.py --checkpoint ../checkpoints/fish_speech_final.pt --text "你好，世界"

Inference Pipeline:
    1. Load trained Dual-AR model
    2. Generate semantic tokens with Slow AR
    3. Generate acoustic tokens with Fast AR
    4. Decode tokens to mel spectrogram
    5. (Optional) Use vocoder to convert mel to waveform
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np

from model import FishSpeech


# --------------------------------------------------------
# Token to Mel Decoder (Simplified)
# --------------------------------------------------------

class TokenToMelDecoder(torch.nn.Module):
    """
    Simplified decoder: convert codec tokens to mel spectrogram.

    In real Fish Speech:
    - Use trained neural codec (EnCodec/DAC)
    - Decode tokens → waveform
    - Compute mel from waveform

    This simplified version:
    - Linear projection: tokens → mel
    - For demonstration only
    """

    def __init__(self, vocab_size: int, n_codebooks: int, n_mels: int = 80):
        super().__init__()
        self.codebook_embs = torch.nn.ModuleList([
            torch.nn.Embedding(vocab_size, n_mels)
            for _ in range(n_codebooks)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, K, T) codec tokens
        Returns: (B, n_mels, T) mel spectrogram
        """
        B, K, T = tokens.shape

        # Sum codebook embeddings
        mel = sum(self.codebook_embs[k](tokens[:, k]) for k in range(K))

        # Transpose to (B, n_mels, T)
        mel = mel.transpose(1, 2)

        return mel


# --------------------------------------------------------
# Inference
# --------------------------------------------------------

@torch.no_grad()
def generate_mel(
    model: FishSpeech,
    decoder: TokenToMelDecoder,
    text_tokens: torch.Tensor,
    max_len: int = 200,
    temperature: float = 1.0,
    top_k: int = 50,
) -> torch.Tensor:
    """
    Generate mel spectrogram from text tokens.

    Args:
        model: Fish Speech Dual-AR model
        decoder: Token to mel decoder
        text_tokens: (B, T_text) text token indices
        max_len: maximum generation length
        temperature: sampling temperature
        top_k: top-k sampling

    Returns:
        mel: (B, n_mels, T) mel spectrogram
    """
    model.eval()
    decoder.eval()

    # Step 1: Generate semantic tokens with Slow AR
    print(f"[inference] Generating semantic tokens...")
    semantic_tokens = model.generate(
        text_tokens,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k,
    )
    print(f"[inference] Generated {semantic_tokens.shape[1]} semantic tokens")

    # Step 2: Generate acoustic tokens with Fast AR
    # For simplicity, we'll use the semantic tokens as acoustic tokens
    # In practice, you'd use the Fast AR to generate the remaining codebooks
    B, T = semantic_tokens.shape
    K = model.n_codebooks

    # Expand to K codebooks (simplified)
    acoustic_tokens = semantic_tokens.unsqueeze(1).repeat(1, K, 1)

    print(f"[inference] Generated {K} codebooks of acoustic tokens")

    # Step 3: Decode tokens to mel
    print(f"[inference] Decoding tokens to mel...")
    mel = decoder(acoustic_tokens)
    print(f"[inference] Generated mel: {mel.shape}")

    return mel


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fish Speech Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--text", type=str, default="你好，世界", help="Input text")
    parser.add_argument("--output", type=str, default="output_mel.npy", help="Output mel file")
    parser.add_argument("--max-len", type=int, default=200, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Load checkpoint
    print(f"[load] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {})

    vocab_size = config.get("vocab_size", 1024)
    n_codebooks = config.get("n_codebooks", 4)

    # Create model
    model = FishSpeech(
        vocab_size=vocab_size,
        n_codebooks=n_codebooks,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[load] Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Create decoder
    decoder = TokenToMelDecoder(vocab_size, n_codebooks).to(device)

    # Prepare text tokens (simplified: random tokens for demo)
    # In practice, you'd use a real tokenizer
    text_tokens = torch.randint(0, vocab_size, (1, 10)).to(device)
    print(f"[input] Text tokens: {text_tokens.shape}")

    # Generate mel
    mel = generate_mel(
        model,
        decoder,
        text_tokens,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # Save output
    output_path = Path(args.output)
    np.save(output_path, mel.cpu().numpy())
    print(f"[save] Mel saved to: {output_path}")
    print(f"[save] Shape: {mel.shape}")

    # Statistics
    print(f"\n[stats] Mel statistics:")
    print(f"  Min: {mel.min().item():.4f}")
    print(f"  Max: {mel.max().item():.4f}")
    print(f"  Mean: {mel.mean().item():.4f}")
    print(f"  Std: {mel.std().item():.4f}")

    print("\n[done] Inference complete!")


if __name__ == "__main__":
    main()
