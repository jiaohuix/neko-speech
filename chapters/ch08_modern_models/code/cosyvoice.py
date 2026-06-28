"""
Ch08.2: CosyVoice — Zero-Shot TTS (Simplified)

Reference:
    CosyVoice: A Scalable Multilingual Zero-Shot Text-to-Speech Synthesizer
    Du et al., 2024 (Alibaba / FunAudioLLM)

    CosyVoice 2: Scalable Streaming Speech Synthesis with LLM
    Du et al., 2025

Core Idea — Zero-Shot Voice Cloning:
    Given 3 seconds of unseen reference audio, clone the speaker's voice.

    Pipeline:
        Text  ──→ Text Encoder ──→ text_emb ─┐
                                             ├──→ AR Transformer ──→ speech tokens ──→ Flow Decoder ──→ mel ──→ vocoder
        Ref Audio ──→ Speaker Encoder ──→ spk_emb ─┘

    How zero-shot works:
        1. Speaker encoder extracts a *speaker embedding* from 3s of reference audio.
           This is a single vector that captures voice identity (pitch, timbre, accent).
        2. The AR model is trained on thousands of speakers, conditioned on spk_emb.
        3. At inference, feed a NEW speaker's 3s audio → get their spk_emb → model
           generates in their voice, even though it never trained on that speaker.

    This is the same paradigm as VALL-E (Microsoft, 2023), which first showed
    that TTS can be framed as "language modeling over neural audio codes".

This Implementation:
    - Speaker encoder: simple CNN pooling over reference mel
    - AR model: decoder-only Transformer predicting mel frames
    - Flow decoder: simplified flow matching (same idea as F5-TTS)
    - Focus: demonstrating the zero-shot conditioning mechanism
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------
# Speaker Encoder: ref_mel → speaker embedding
# --------------------------------------------------------

class SpeakerEncoder(nn.Module):
    """
    Extract a fixed-size speaker embedding from reference mel.

    Architecture: Conv1d stack → temporal pooling → projection.

    In real CosyVoice, this is a pretrained model (e.g., CAM++, ECAPA-TDNN)
    trained on speaker verification. Here we use a simplified version.

    The key insight: temporal pooling (mean over time) makes the embedding
    *length-invariant* — works for any reference audio length.
    """

    def __init__(self, mel_dim=80, spk_dim=256):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(mel_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, spk_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(spk_dim),
            nn.ReLU(),
        )
        self.proj = nn.Linear(spk_dim, spk_dim)

    def forward(self, ref_mel, ref_mask=None):
        """
        Args:
            ref_mel:  (B, T, mel_dim)
            ref_mask: (B, T) True=padding

        Returns:
            spk_emb:  (B, spk_dim)
        """
        # Conv1d expects (B, C, T)
        x = ref_mel.transpose(1, 2)           # (B, mel_dim, T)
        x = self.convs(x)                     # (B, spk_dim, T)
        x = x.transpose(1, 2)                 # (B, T, spk_dim)

        # Masked mean pooling
        if ref_mask is not None:
            valid = (~ref_mask).float().unsqueeze(-1)  # (B, T, 1)
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)                 # (B, spk_dim)

        return self.proj(x)                   # (B, spk_dim)


# --------------------------------------------------------
# Autoregressive Transformer (Decoder-Only)
# --------------------------------------------------------

def _causal_mask(n):
    """Upper-triangular bool mask (True = masked)."""
    return torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)


class AutoregressiveDecoder(nn.Module):
    """
    Decoder-only Transformer that predicts mel frames one-by-one.

    Conditioning:
        - Speaker embedding (added to every token via projection)
        - Text embedding (cross-attention)

    At each step, the model sees [ref_mel_frames, generated_so_far]
    and predicts the next mel frame.

    In CosyVoice, this is actually a language model over discrete
    speech tokens (from a neural audio codec). We use continuous mel
    here for simplicity, predicting via MSE instead of cross-entropy.
    """

    def __init__(
        self,
        mel_dim=80,
        text_dim=512,
        spk_dim=256,
        dim=512,
        heads=8,
        n_layers=6,
        max_len=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.mel_dim = mel_dim
        self.dim = dim

        # Mel input projection
        self.mel_proj = nn.Linear(mel_dim, dim)

        # Speaker conditioning: project spk_emb → dim, add to every mel token
        self.spk_proj = nn.Linear(spk_dim, dim)

        # Positional encoding
        self.pos_emb = nn.Embedding(max_len, dim)

        # Cross-attention to text
        self.cross_attn_layers = nn.ModuleList()
        # Self-attention blocks
        self.self_attn_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.cross_attn_layers.append(nn.ModuleDict({
                "norm_q": nn.LayerNorm(dim),
                "norm_kv": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(
                    dim, heads, dropout=dropout, batch_first=True
                ),
            }))
            self.self_attn_layers.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(
                    dim, heads, dropout=dropout, batch_first=True
                ),
                "norm2": nn.LayerNorm(dim),
                "ff": nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                    nn.Dropout(dropout),
                ),
            }))

        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, mel_dim)

    def forward(self, mel_in, spk_emb, text_emb, text_mask):
        """
        Args:
            mel_in:    (B, T, mel_dim)  input mel frames (shifted right)
            spk_emb:   (B, spk_dim)     speaker embedding
            text_emb:  (B, T_text, text_dim)
            text_mask: (B, T_text) True=padding

        Returns:
            mel_pred:  (B, T, mel_dim)  predicted next frames
        """
        B, T, _ = mel_in.shape
        device = mel_in.device

        # Project mel + add speaker + position
        h = self.mel_proj(mel_in)                   # (B, T, dim)
        h = h + self.spk_proj(spk_emb)[:, None, :]  # broadcast speaker to all tokens
        pos = torch.arange(T, device=device)
        h = h + self.pos_emb(pos)[None, :, :]

        # Causal self-attention mask
        causal = _causal_mask(T).to(device)

        # Transformer layers
        kv = text_emb
        for cross_layer, self_layer in zip(self.cross_attn_layers, self.self_attn_layers):
            # Cross-attention to text
            q = cross_layer["norm_q"](h)
            kv_n = cross_layer["norm_kv"](kv)
            ca, _ = cross_layer["attn"](
                q, kv_n, kv_n,
                key_padding_mask=text_mask,
                need_weights=False,
            )
            h = h + ca

            # Causal self-attention
            n = self_layer["norm1"](h)
            sa, _ = self_layer["attn"](n, n, n, attn_mask=causal, need_weights=False)
            h = h + sa

            # FFN
            h = h + self_layer["ff"](self_layer["norm2"](h))

        return self.out_proj(self.out_norm(h))


# --------------------------------------------------------
# CosyVoice: Full Pipeline
# --------------------------------------------------------

class CosyVoice(nn.Module):
    """
    CosyVoice simplified: Zero-shot TTS.

    Training:
        1. Encode reference mel → spk_emb
        2. AR decoder predicts mel[t+1] from mel[:t] + spk_emb + text
        3. MSE loss on predicted mel frames

    Inference (zero-shot):
        1. Encode 3s reference → spk_emb
        2. AR generate frame-by-frame (teacher-forced first, then autoregressive)
    """

    def __init__(
        self,
        mel_dim=80,
        text_dim=512,
        spk_dim=256,
        dim=512,
        heads=8,
        n_layers=6,
    ):
        super().__init__()
        self.mel_dim = mel_dim
        self.speaker_encoder = SpeakerEncoder(mel_dim=mel_dim, spk_dim=spk_dim)
        self.ar_decoder = AutoregressiveDecoder(
            mel_dim=mel_dim, text_dim=text_dim, spk_dim=spk_dim,
            dim=dim, heads=heads, n_layers=n_layers,
        )

    def ar_loss(self, target_mel, target_mask, text_emb, text_mask, ref_mel, ref_mask):
        """
        Teacher-forced AR training loss.

        Input:  mel frames [0, 1, ..., T-2] (shifted right)
        Target: mel frames [1, 2, ..., T-1] (next frame prediction)
        """
        B, T, D = target_mel.shape

        # Speaker embedding from reference
        spk_emb = self.speaker_encoder(ref_mel, ref_mask)  # (B, spk_dim)

        # Shift: input = mel[:T-1], target = mel[1:]
        mel_in = target_mel[:, :-1, :]
        mel_tgt = target_mel[:, 1:, :]
        tgt_mask = target_mask[:, 1:]

        # Predict
        mel_pred = self.ar_decoder(mel_in, spk_emb, text_emb, text_mask)

        # MSE loss on valid frames
        diff = (mel_pred - mel_tgt) ** 2
        valid = (~tgt_mask).float().unsqueeze(-1)
        loss = (diff * valid).sum() / valid.sum().clamp(min=1) / D
        return loss

    @torch.no_grad()
    def sample(
        self,
        n_frames,
        text_emb, text_mask,
        ref_mel, ref_mask,
        prompt_mel=None,
    ):
        """
        Autoregressive generation.

        Args:
            n_frames:   total frames to generate
            text_emb:   (1, T_text, text_dim)
            text_mask:  (1, T_text)
            ref_mel:    (1, T_ref, mel_dim)  — the 3s reference
            ref_mask:   (1, T_ref)
            prompt_mel: (1, T_prompt, mel_dim)  — optional prompt to continue from

        Returns:
            mel: (1, n_frames, mel_dim)
        """
        spk_emb = self.speaker_encoder(ref_mel, ref_mask)  # (1, spk_dim)

        # Start from prompt or zeros
        if prompt_mel is not None:
            generated = [prompt_mel]
            remaining = n_frames - prompt_mel.size(1)
        else:
            generated = []
            remaining = n_frames

        for _ in range(remaining):
            if generated:
                ctx = torch.cat(generated, dim=1)
            else:
                # First frame: use mean of reference as a "soft start"
                ctx = ref_mel.mean(dim=1, keepdim=True)  # (1, 1, mel_dim)

            pred = self.ar_decoder(ctx, spk_emb, text_emb, text_mask)
            next_frame = pred[:, -1:, :]  # take last prediction
            generated.append(next_frame)

        return torch.cat(generated, dim=1)[:, :n_frames, :]


# --------------------------------------------------------
# Quick Sanity Check
# --------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    mel_dim, text_dim, spk_dim, dim = 80, 256, 128, 256
    B, T_text, T_ref, T_tgt = 2, 15, 30, 50

    # Dummy text encoder output
    text_emb = torch.randn(B, T_text, text_dim)
    text_mask = torch.zeros(B, T_text, dtype=torch.bool)

    ref_mel = torch.randn(B, T_ref, mel_dim)
    ref_mask = torch.zeros(B, T_ref, dtype=torch.bool)
    tgt_mel = torch.randn(B, T_tgt, mel_dim)
    tgt_mask = torch.zeros(B, T_tgt, dtype=torch.bool)

    model = CosyVoice(
        mel_dim=mel_dim, text_dim=text_dim,
        spk_dim=spk_dim, dim=dim, heads=4, n_layers=2,
    )

    # Training
    loss = model.ar_loss(tgt_mel, tgt_mask, text_emb, text_mask, ref_mel, ref_mask)
    print(f"[Train] AR Loss: {loss.item():.4f}")

    # Inference (zero-shot: use reference mel of unseen speaker)
    model.eval()
    gen = model.sample(
        n_frames=T_tgt,
        text_emb=text_emb[:1], text_mask=text_mask[:1],
        ref_mel=ref_mel[:1], ref_mask=ref_mask[:1],
    )
    print(f"[Infer] Generated mel shape: {tuple(gen.shape)}")
    print("[OK] CosyVoice sanity check passed.")
