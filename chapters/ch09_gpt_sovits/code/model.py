"""
Ch09: GPT-SoVITS -- Few-Shot Voice Cloning

Simplified implementation of GPT-SoVITS architecture.

GPT-SoVITS combines an autoregressive Transformer (GPT-style) with a
VITS-based vocoder for few-shot voice cloning.  The system has two stages:

    Stage 1 (AR / GPT):
        phoneme_ids ──> SimpleAR ──> semantic_token_ids
        The AR model learns to predict HuBERT-derived semantic tokens
        from text phonemes, conditioned on a short reference audio prompt.

    Stage 2 (SoVITS / VITS-based vocoder):
        semantic_token_ids + text + ref_audio ──> SimpleSoVITS ──> waveform
        The vocoder converts semantic tokens back into audio, using
        reference audio for timbre conditioning.

The critical design choice is using a **single-layer RVQ codebook** (n_q=1)
as the interface between stages.  This makes the AR problem much simpler
than predicting full codec tokens (as in VALL-E): only 1 token per frame
instead of 8.

Architecture overview:

    Reference Audio (3-10s)
          |
          v
    [HuBERT / SSL] -----> features (768-dim, 50Hz)
          |                       |
          v                       v
    [RVQ Quantizer]        (extract prompt tokens)
    n_q=1, bins=1024              |
          |                       |
          v                       v
    prompt_semantic_ids     +-----------+
                            |  SimpleAR |  <--- Text Phonemes
                            |  (GPT)    |
                            +-----------+
                                  |
                                  v
                          predicted_semantic_ids
                                  |
                                  v
                      +-----------------------+
                      |   SimpleSoVITS         |  <--- Text + Ref Mel
                      |   (VITS-based vocoder) |
                      +-----------------------+
                                  |
                                  v
                            Output Waveform

References:
    - Kim et al., 2021.  Conditional Variational Autoencoder with
      Adversarial Learning for End-to-End Text-to-Speech (VITS).
    - Wang et al., 2023.  Neural Codec Language Models for Zero-Shot TTS
      (VALL-E).
    - GPT-SoVITS open-source project (RVC-Boss).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_padding(kernel_size, dilation=1):
    """Compute 'same' padding for Conv1d."""
    return (kernel_size * dilation - dilation) // 2


LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    """Normal-distribution weight initialisation for Conv / Linear."""
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()


# ---------------------------------------------------------------------------
# Sine Positional Embedding (used by the AR model, following the original)
# ---------------------------------------------------------------------------

class SinePositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal positional encoding, added to the input.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """

    def __init__(self, dim, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: dim // 2 + dim % 2])
        # pe shape: (max_len, dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: (B, T, dim)
        Returns:
            x + PE  (B, T, dim)
        """
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# ===================================================================
# Stage 1: SimpleAR  (Autoregressive Transformer, GPT-style)
# ===================================================================

class CausalTransformerBlock(nn.Module):
    """
    Single Transformer decoder block used in the AR model.

    We use a standard pre-norm decoder block with:
      - Multi-head causal self-attention
      - FFN (4x hidden dim)
      - LayerNorm + residual

    In the full GPT-SoVITS the attention mask is more elaborate
    (bidirectional for text, causal for audio, masked cross-attention).
    Here we simplify: the *entire* sequence [text | audio] uses causal
    attention.  The text prefix can see all past text tokens, and each
    audio token sees all text + past audio -- which is correct behaviour
    for a decoder-only model.
    """

    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: (B, T, dim)
            attn_mask: (T, T) boolean causal mask
        Returns:
            (B, T, dim)
        """
        # Pre-norm self-attention (causal)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class SimpleAR(nn.Module):
    """
    Simplified autoregressive Transformer for text-to-semantic-token
    prediction.

    Input:
        phoneme_ids  (B, T_text)   -- text phoneme token IDs
        semantic_ids (B, T_audio)  -- teacher-forced semantic token IDs
                       (during inference, this is built autoregressively)

    Output:
        logits (B, T_audio, vocab_size)  -- predicted semantic token logits

    The model concatenates text and audio embeddings, applies positional
    encoding, and runs through a stack of causal Transformer blocks.
    The final projection predicts the *next* semantic token.

    Hyper-parameters (simplified from original 512-dim/12-layer/16-head):
        dim       = 384
        n_heads   = 8
        n_layers  = 8
        vocab_size = 1025  (1024 semantic tokens + 1 EOS)
    """

    def __init__(
        self,
        dim=384,
        n_heads=8,
        n_layers=8,
        phoneme_vocab_size=512,
        vocab_size=1025,
        dropout=0.0,
        max_seq_len=4096,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        # --- Embeddings ---
        self.text_embedding = nn.Embedding(phoneme_vocab_size, dim)
        self.audio_embedding = nn.Embedding(vocab_size, dim)
        self.text_position = SinePositionalEmbedding(dim, max_seq_len)
        self.audio_position = SinePositionalEmbedding(dim, max_seq_len)

        # --- Transformer stack ---
        self.blocks = nn.ModuleList(
            [CausalTransformerBlock(dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim)

        # --- Output projection ---
        self.predict_layer = nn.Linear(dim, vocab_size, bias=False)

    def _causal_mask(self, T, device):
        """Upper-triangular boolean mask (True = masked)."""
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

    def forward(self, phoneme_ids, semantic_ids):
        """
        Training forward pass (teacher forcing).

        Args:
            phoneme_ids:  (B, T_text)
            semantic_ids: (B, T_audio)

        Returns:
            logits: (B, T_audio, vocab_size)
        """
        B = phoneme_ids.size(0)
        T_text = phoneme_ids.size(1)
        T_audio = semantic_ids.size(1)

        # Embed text and audio
        text_emb = self.text_embedding(phoneme_ids)    # (B, T_text, dim)
        audio_emb = self.audio_embedding(semantic_ids)  # (B, T_audio, dim)

        # Positional encoding (applied separately, then concatenated)
        text_emb = self.text_position(text_emb)
        audio_emb = self.audio_position(audio_emb)

        # Concatenate: [text | audio]
        x = torch.cat([text_emb, audio_emb], dim=1)  # (B, T_text+T_audio, dim)

        # Causal attention mask over the full concatenated sequence
        T_total = T_text + T_audio
        mask = self._causal_mask(T_total, x.device)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        x = self.norm(x)

        # We only care about the audio portion of the output
        # (positions T_text:), which predict the next semantic token.
        audio_out = x[:, T_text:, :]  # (B, T_audio, dim)
        logits = self.predict_layer(audio_out)  # (B, T_audio, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        phoneme_ids,
        prompt_semantic_ids,
        max_new_tokens=500,
        top_k=5,
        temperature=1.0,
        eos_token=1024,
    ):
        """
        Autoregressive generation with top-k sampling.

        Args:
            phoneme_ids:        (1, T_text) -- input text
            prompt_semantic_ids: (1, T_prompt) -- reference audio tokens
            max_new_tokens:     max tokens to generate
            top_k:              top-k sampling width
            temperature:        sampling temperature
            eos_token:          EOS token id (default 1024)

        Returns:
            generated_ids: (1, T_generated) -- predicted semantic tokens
        """
        self.eval()
        # Start with the prompt tokens
        current_ids = prompt_semantic_ids.clone()  # (1, T_prompt)

        for _ in range(max_new_tokens):
            logits = self.forward(phoneme_ids, current_ids)  # (1, T, vocab)
            next_logits = logits[:, -1, :] / temperature  # last position

            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < values[:, [-1]]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            current_ids = torch.cat([current_ids, next_token], dim=1)

            # Stop if EOS
            if next_token.item() == eos_token:
                break

        # Return only the newly generated tokens (excluding prompt)
        generated = current_ids[:, prompt_semantic_ids.size(1):]
        return generated


# ===================================================================
# RVQ Quantizer (single codebook, frozen)
# ===================================================================

class SimpleRVQ(nn.Module):
    """
    Simplified Residual Vector Quantiser with a single codebook (n_q=1).

    In GPT-SoVITS this quantiser converts HuBERT SSL features (768-dim)
    into discrete semantic token IDs (0..1023).  During training the
    quantiser is frozen -- it acts as a fixed feature extractor.

    We implement a basic VQ-VAE style quantiser:
        encode: find nearest codebook entry, return its index
        decode: look up codebook embedding by index

    Args:
        bins:  codebook size (default 1024)
        dim:   feature dimension (default 768, matching HuBERT)
    """

    def __init__(self, bins=1024, dim=768):
        super().__init__()
        self.bins = bins
        self.dim = dim
        self.codebook = nn.Embedding(bins, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / bins, 1.0 / bins)

    def encode(self, x):
        """
        Quantise continuous features to token IDs.

        Args:
            x: (B, T, dim) or (B, dim, T)

        Returns:
            ids: (B, T) -- integer token IDs in [0, bins)
        """
        if x.dim() == 3 and x.size(1) == self.dim:
            x = x.transpose(1, 2)  # (B, dim, T) -> (B, T, dim)
        # Find nearest codebook entry
        # ||x - e||^2 = ||x||^2 - 2*x.e + ||e||^2
        x_flat = x.reshape(-1, self.dim)  # (B*T, dim)
        dist = (
            x_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * x_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1, keepdim=True).t()
        )
        ids = dist.argmin(dim=1)  # (B*T,)
        ids = ids.reshape(x.size(0), x.size(1))  # (B, T)
        return ids

    def decode(self, ids):
        """
        Look up codebook embeddings by token ID.

        Args:
            ids: (B, T) -- integer token IDs

        Returns:
            (B, dim, T) -- quantised feature vectors
        """
        emb = self.codebook(ids)  # (B, T, dim)
        return emb.transpose(1, 2)  # (B, dim, T)


# ===================================================================
# Stage 2: SimpleSoVITS  (VITS-based vocoder)
# ===================================================================

# --- TextEncoder (simplified, no MRTE) ---

class _TransformerEncoderLayer(nn.Module):
    """
    A lightweight Transformer encoder layer with relative position bias.
    Used inside the SoVITS TextEncoder.
    """

    def __init__(self, dim, n_heads, filter_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, filter_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(filter_channels, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        """x: (B, T, dim)"""
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=mask, need_weights=False)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class SoVITSTextEncoder(nn.Module):
    """
    Simplified TextEncoder for the SoVITS vocoder.

    In the original GPT-SoVITS, the TextEncoder has three sub-encoders
    (SSL encoder, text encoder, MRTE cross-attention).  We simplify to
    a single Transformer stack that encodes phoneme embeddings
    conditioned on a speaker embedding.

    Inputs:
        phoneme_ids: (B, T_text)
        speaker_emb: (B, speaker_dim)  -- global speaker conditioning

    Outputs:
        m_p, logs_p: (B, out_channels, T_text) -- prior distribution params
    """

    def __init__(
        self,
        phoneme_vocab_size=512,
        out_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        dropout=0.1,
        speaker_dim=256,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.embedding = nn.Embedding(phoneme_vocab_size, hidden_channels)
        self.speaker_proj = nn.Linear(speaker_dim, hidden_channels)

        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(
                hidden_channels, n_heads, filter_channels, kernel_size, dropout
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_channels)

        # Project to prior distribution parameters
        self.proj = nn.Linear(hidden_channels, out_channels * 2)

    def forward(self, phoneme_ids, speaker_emb):
        """
        Args:
            phoneme_ids: (B, T)
            speaker_emb: (B, speaker_dim)
        Returns:
            m_p:    (B, out_channels, T)
            logs_p: (B, out_channels, T)
        """
        x = self.embedding(phoneme_ids)  # (B, T, hidden)
        # Add speaker conditioning (broadcast over time)
        spk = self.speaker_proj(speaker_emb).unsqueeze(1)  # (B, 1, hidden)
        x = x + spk

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # Project to (m_p, logs_p)
        stats = self.proj(x)  # (B, T, out_channels*2)
        m_p, logs_p = stats.chunk(2, dim=-1)
        # Transpose to (B, C, T)
        return m_p.transpose(1, 2), logs_p.transpose(1, 2)


# --- PosteriorEncoder (WaveNet-based) ---

class _WNConvBlock(nn.Module):
    """Single dilated convolution block for the WaveNet-style encoder."""

    def __init__(self, hidden_channels, kernel_size, dilation_rate):
        super().__init__()
        padding = get_padding(kernel_size, dilation_rate)
        self.conv = weight_norm(
            nn.Conv1d(
                hidden_channels, hidden_channels * 2,
                kernel_size, dilation=dilation_rate, padding=padding,
            )
        )

    def forward(self, x):
        """x: (B, C, T) -> (B, C, T) gated output + skip"""
        h = self.conv(x)
        h1, h2 = h.chunk(2, dim=1)
        z = torch.tanh(h1) * torch.sigmoid(h2)
        return z


class PosteriorEncoder(nn.Module):
    """
    WaveNet-style dilated convolution encoder.

    Encodes the linear spectrogram into posterior distribution parameters
    (m_q, logs_q) and samples z_q via the reparameterisation trick.

    Only used during training; at inference time we sample from the prior.

    Args:
        in_channels:    spectrogram dimension (e.g. 1025 for n_fft=2048)
        out_channels:   latent dim (inter_channels, default 192)
        hidden_channels: WaveNet hidden dim (default 192)
        kernel_size:    conv kernel size (default 5)
        n_layers:       number of WaveNet layers (reduced from 16 to 8)
    """

    def __init__(
        self,
        in_channels=1025,
        out_channels=192,
        hidden_channels=192,
        kernel_size=5,
        n_layers=8,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList([
            _WNConvBlock(hidden_channels, kernel_size, dilation_rate=2 ** (i % 4))
            for i in range(n_layers)
        ])
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, spec):
        """
        Args:
            spec: (B, in_channels, T_spec) -- linear spectrogram
        Returns:
            z_q:    (B, out_channels, T_spec) -- sampled latent
            m_q:    (B, out_channels, T_spec)
            logs_q: (B, out_channels, T_spec)
        """
        x = self.pre(spec)  # (B, hidden, T)
        skip_sum = torch.zeros_like(x)
        for block in self.blocks:
            z = block(x)
            x = x + z
            skip_sum = skip_sum + z
        stats = self.proj(skip_sum)  # (B, out*2, T)
        m_q, logs_q = stats.chunk(2, dim=1)
        # Reparameterisation trick
        z_q = m_q + torch.randn_like(m_q) * torch.exp(logs_q)
        return z_q, m_q, logs_q


# --- Flow (normalising flow, affine coupling) ---

class _AffineCouplingLayer(nn.Module):
    """
    Single affine coupling layer.

    Splits input along channel dim into two halves (x1, x2).
    x1 passes through unchanged.  x2 is transformed:
        z2 = (x2 - t(x1)) * exp(-s(x1))
    where s, t are predicted by a small WaveNet.
    """

    def __init__(self, channels, hidden_channels, kernel_size=5, n_layers=4):
        super().__init__()
        self.half = channels // 2
        # Small WaveNet to predict scale and shift
        layers = []
        in_ch = self.half
        for i in range(n_layers):
            out_ch = hidden_channels if i < n_layers - 1 else channels
            layers.append(
                weight_norm(
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
                )
            )
            if i < n_layers - 1:
                layers.append(nn.ReLU())
            in_ch = hidden_channels
        self.net = nn.Sequential(*layers)
        # Initialise last conv to near-zero so initial transform ~ identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, reverse=False):
        """
        Args:
            x: (B, C, T)
            reverse: if True, run inverse transform
        Returns:
            z: (B, C, T)
            log_det: (B,) -- log |det Jacobian| (summed over T and C)
        """
        x1, x2 = x[:, : self.half], x[:, self.half:]
        h = self.net(x1)
        s, t = h[:, : self.half], h[:, self.half:]

        if not reverse:
            z2 = (x2 - t) * torch.exp(-s)
            log_det = (-s).sum(dim=(1, 2))
        else:
            z2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=(1, 2))

        z = torch.cat([x1, z2], dim=1)
        return z, log_det


class Flow(nn.Module):
    """
    Residual coupling block: K affine coupling layers with channel flips.

    Args:
        channels:        latent dimension
        hidden_channels: WaveNet hidden dim
        n_flows:         number of coupling layers (reduced from 4 to 2)
    """

    def __init__(self, channels=192, hidden_channels=192, n_flows=2):
        super().__init__()
        self.flows = nn.ModuleList([
            _AffineCouplingLayer(channels, hidden_channels) for _ in range(n_flows)
        ])

    def forward(self, z, reverse=False):
        """
        Args:
            z: (B, C, T)
            reverse: forward or inverse pass
        Returns:
            z_out: (B, C, T)
            total_log_det: (B,)
        """
        total_log_det = torch.zeros(z.size(0), device=z.device)
        if not reverse:
            for flow in self.flows:
                z, ld = flow(z, reverse=False)
                total_log_det = total_log_det + ld
                # Flip channels for next layer
                z = torch.flip(z, dims=[1])
        else:
            for flow in reversed(self.flows):
                z = torch.flip(z, dims=[1])
                z, ld = flow(z, reverse=True)
                total_log_det = total_log_det + ld
        return z, total_log_det


# --- Generator (HiFi-GAN Decoder) ---

class _ResBlock1(nn.Module):
    """
    HiFi-GAN ResBlock type 1.

    Three dilated conv layers with different dilation rates, each
    wrapped in a residual connection.
    """

    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(
                nn.Sequential(
                    nn.LeakyReLU(LRELU_SLOPE),
                    weight_norm(
                        nn.Conv1d(
                            channels, channels, kernel_size,
                            dilation=d, padding=get_padding(kernel_size, d),
                        )
                    ),
                    nn.LeakyReLU(LRELU_SLOPE),
                    weight_norm(
                        nn.Conv1d(
                            channels, channels, kernel_size,
                            dilation=1, padding=get_padding(kernel_size, 1),
                        )
                    ),
                )
            )

    def forward(self, x):
        for conv in self.convs:
            x = x + conv(x)
        return x


class Generator(nn.Module):
    """
    HiFi-GAN Generator: converts latent z into audio waveform.

    Architecture:
        z (B, initial_channel, T)
            -> Conv1d (pre)
            -> N x [ConvTranspose1d (upsample) + sum of ResBlock1]
            -> Conv1d (post) + tanh
        waveform (B, 1, T * prod(upsample_rates))

    Simplified from the original:
        upsample_initial_channel: 512 -> 256
        5 upsample stages with reduced kernel sizes
    """

    def __init__(
        self,
        initial_channel=192,
        upsample_initial_channel=256,
        upsample_rates=(10, 8, 2, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)

        self.conv_pre = weight_norm(
            nn.Conv1d(initial_channel, upsample_initial_channel, 7, padding=3)
        )

        # Upsampling layers
        self.ups = nn.ModuleList()
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(ch, ch // 2, k, stride=u, padding=(k - u) // 2)
                )
            )
            ch = ch // 2

        # Residual blocks (one set of 3 per upsample stage)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch_i = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilations):
                self.resblocks.append(_ResBlock1(ch_i, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

    def forward(self, z):
        """
        Args:
            z: (B, initial_channel, T)
        Returns:
            waveform: (B, 1, T * prod(upsample_rates))
        """
        x = self.conv_pre(z)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            # Sum outputs from all resblocks for this stage
            xs = torch.zeros_like(x)
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        """Remove weight normalisation for faster inference."""
        for up in self.ups:
            remove_weight_norm(up)
        for rb in self.resblocks:
            for layer in rb.convs:
                if isinstance(layer, (nn.Conv1d,)):
                    try:
                        remove_weight_norm(layer)
                    except ValueError:
                        pass
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


# --- ReferenceEncoder (extracts speaker embedding from reference audio) ---

class ReferenceEncoder(nn.Module):
    """
    Extracts a fixed-size speaker embedding from reference audio mel.

    Architecture:
        ref_mel (B, n_mels, T)
            -> Conv2d stack (downsample time & freq)
            -> Reshape to (B, C', T')
            -> GRU over time
            -> Linear -> speaker_emb (B, speaker_dim)
    """

    def __init__(self, n_mels=128, speaker_dim=256):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        # After 4 stride-2 convs: freq dim = n_mels // 16, channels = 64
        proj_dim = 64 * (n_mels // 16)
        self.proj = nn.Linear(proj_dim, 128)
        self.gru = nn.GRU(128, 128, batch_first=True)
        self.out = nn.Linear(128, speaker_dim)

    def forward(self, ref_mel):
        """
        Args:
            ref_mel: (B, n_mels, T) -- reference mel spectrogram
        Returns:
            speaker_emb: (B, speaker_dim)
        """
        x = ref_mel.unsqueeze(1)  # (B, 1, n_mels, T)
        x = self.convs(x)  # (B, 64, n_mels//16, T//16)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)  # (B, T', C*F)
        x = self.proj(x)  # (B, T', 128)
        _, h = self.gru(x)  # h: (1, B, 128)
        speaker_emb = self.out(h.squeeze(0))  # (B, speaker_dim)
        return speaker_emb


# --- Discriminator (Multi-Period Discriminator, simplified) ---

class _PeriodDiscriminator(nn.Module):
    """Single period sub-discriminator."""

    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 256, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(256, 512, (5, 1), stride=(1, 1), padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(512, 1, (3, 1), padding=(1, 0)))

    def forward(self, x):
        """
        Args:
            x: (B, 1, T) -- waveform
        Returns:
            decision: (B, 1, T', 1)
            features: list of intermediate feature maps
        """
        B, C, T = x.shape
        # Pad to multiple of period and reshape to 2D
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            x = F.pad(x, (0, pad_len))
            T = x.size(2)
        x = x.view(B, C, T // self.period, self.period)  # (B, 1, T/p, p)

        features = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return x, features


class Discriminator(nn.Module):
    """
    Multi-Period Discriminator (MPD) from HiFi-GAN.

    Uses prime periods [2, 3, 5, 7, 11] so each sub-discriminator
    sees different periodic structures in the waveform.
    """

    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [_PeriodDiscriminator(p) for p in periods]
        )

    def forward(self, real, fake):
        """
        Args:
            real: (B, 1, T) -- ground-truth waveform
            fake: (B, 1, T) -- generated waveform
        Returns:
            disc_loss: scalar
            gen_loss: scalar
            feat_loss: scalar
        """
        disc_loss = 0.0
        gen_loss = 0.0
        feat_loss = 0.0

        for disc in self.discriminators:
            # Real
            d_real, feats_real = disc(real)
            # Fake (detach to avoid backprop into generator from D)
            d_fake, feats_fake = disc(fake.detach())

            # Least-squares GAN loss
            disc_loss = disc_loss + F.mse_loss(d_real, torch.ones_like(d_real))
            disc_loss = disc_loss + F.mse_loss(d_fake, torch.zeros_like(d_fake))

            # Generator loss (want D(fake) -> 1)
            d_fake_g, feats_fake_g = disc(fake)
            gen_loss = gen_loss + F.mse_loss(d_fake_g, torch.ones_like(d_fake_g))

            # Feature matching loss
            for fr, ff in zip(feats_real, feats_fake_g):
                feat_loss = feat_loss + F.l1_loss(ff, fr.detach())

        return disc_loss, gen_loss, feat_loss


# ===================================================================
# Full GPT-SoVITS wrapper
# ===================================================================

class GPTSoVITS(nn.Module):
    """
    Combined GPT-SoVITS model.

    Wraps the AR model (Stage 1) and the SoVITS vocoder (Stage 2)
    into a single module for convenience.

    During training, the two stages are trained separately.
    During inference, they run in sequence:
        1. AR: text -> semantic tokens
        2. SoVITS: semantic tokens + text + ref audio -> waveform
    """

    def __init__(
        self,
        # AR params
        ar_dim=384,
        ar_n_heads=8,
        ar_n_layers=8,
        phoneme_vocab_size=512,
        semantic_vocab_size=1025,
        # SoVITS params
        hidden_channels=192,
        filter_channels=768,
        spec_channels=1025,
        n_mels=128,
        speaker_dim=256,
        # Generator params
        upsample_rates=(10, 8, 2, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4, 4),
        # RVQ params
        rvq_bins=1024,
        rvq_dim=768,
    ):
        super().__init__()

        # Stage 1: AR model
        self.ar = SimpleAR(
            dim=ar_dim,
            n_heads=ar_n_heads,
            n_layers=ar_n_layers,
            phoneme_vocab_size=phoneme_vocab_size,
            vocab_size=semantic_vocab_size,
        )

        # RVQ quantiser (frozen)
        self.quantizer = SimpleRVQ(bins=rvq_bins, dim=rvq_dim)
        for p in self.quantizer.parameters():
            p.requires_grad = False

        # Stage 2: SoVITS vocoder components
        self.ref_encoder = ReferenceEncoder(n_mels=n_mels, speaker_dim=speaker_dim)

        self.text_encoder = SoVITSTextEncoder(
            phoneme_vocab_size=phoneme_vocab_size,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=2,
            n_layers=6,
            speaker_dim=speaker_dim,
        )

        self.posterior_encoder = PosteriorEncoder(
            in_channels=spec_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_layers=8,
        )

        self.flow = Flow(
            channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_flows=2,
        )

        self.generator = Generator(
            initial_channel=hidden_channels,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
        )

        self.discriminator = Discriminator()

    def forward_stage1(self, phoneme_ids, semantic_ids):
        """
        Stage 1 forward: predict semantic tokens from text.

        Returns:
            logits: (B, T_audio, vocab_size)
        """
        return self.ar(phoneme_ids, semantic_ids)

    def forward_stage2(self, phoneme_ids, spec, ref_mel):
        """
        Stage 2 forward: vocoder training pass.

        Args:
            phoneme_ids: (B, T_text)
            spec:        (B, spec_channels, T_spec) -- linear spectrogram
            ref_mel:     (B, n_mels, T_ref) -- reference mel for speaker

        Returns:
            y_hat:   (B, 1, T_wav) -- generated waveform
            m_p, logs_p: prior params  (aligned to T_spec)
            m_q, logs_q: posterior params
        """
        # Speaker embedding from reference audio
        speaker_emb = self.ref_encoder(ref_mel)  # (B, speaker_dim)

        # Prior from text
        m_p, logs_p = self.text_encoder(phoneme_ids, speaker_emb)

        # Posterior from spectrogram
        z_q, m_q, logs_q = self.posterior_encoder(spec)

        # Align prior to posterior length via interpolation.
        # In the full VITS, MAS (Monotonic Alignment Search) handles this.
        # Here we use linear interpolation as a simplification.
        T_spec = m_q.size(2)
        T_text = m_p.size(2)
        if T_text != T_spec:
            m_p = F.interpolate(m_p, size=T_spec, mode='linear', align_corners=False)
            logs_p = F.interpolate(logs_p, size=T_spec, mode='linear', align_corners=False)

        # Flow: z_q -> z_p (should match prior)
        z_p, _ = self.flow(z_q, reverse=False)

        # Generate waveform from posterior sample
        y_hat = self.generator(z_q)

        return y_hat, m_p, logs_p, m_q, logs_q


# Alias for convenience: `from model import SoVITS`
SoVITS = GPTSoVITS


# ===================================================================
# Loss functions
# ===================================================================

def kl_loss(m_p, logs_p, m_q, logs_q):
    """
    KL divergence between two diagonal Gaussians.

    KL(q || p) = sum[ logs_p - logs_q + (exp(2*logs_q) + (m_q-m_p)^2)/(2*exp(2*logs_p)) - 0.5 ]

    Args:
        m_p, logs_p: (B, C, T) -- prior parameters
        m_q, logs_q: (B, C, T) -- posterior parameters

    Returns:
        scalar KL loss (mean over batch, sum over C and T)
    """
    kl = logs_p - logs_q - 0.5
    kl = kl + 0.5 * ((m_q - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = kl + 0.5 * torch.exp(2.0 * (logs_q - logs_p))
    return kl.sum(dim=(1, 2)).mean()


def mel_loss(y_real, y_fake, n_fft=2048, hop_length=640, win_length=2048, n_mels=128, sr=32000):
    """
    Mel-spectrogram L1 loss between real and generated waveforms.

    Args:
        y_real: (B, 1, T) -- ground truth waveform
        y_fake: (B, 1, T) -- generated waveform
    """
    import librosa
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_basis = torch.from_numpy(mel_basis).float().to(y_real.device)
    window = torch.hann_window(win_length, device=y_real.device)

    def _mel(wav):
        spec = torch.stft(
            wav.squeeze(1), n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, return_complex=True,
        )
        mag = torch.abs(spec)
        mel = torch.matmul(mel_basis, mag)
        return torch.log(torch.clamp(mel, min=1e-5))

    return F.l1_loss(_mel(y_fake), _mel(y_real))


# ===================================================================
# Main: shape verification tests
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GPT-SoVITS Shape Verification Tests")
    print("=" * 60)

    B = 2       # batch size
    T_text = 10  # text length
    T_audio = 50 # audio length (semantic tokens)
    T_spec = 50  # spectrogram frames
    T_ref = 40   # reference mel frames
    n_fft = 2048
    spec_channels = n_fft // 2 + 1  # 1025
    n_mels = 128

    device = "cpu"

    # --- AR model ---
    print("\n--- SimpleAR ---")
    ar = SimpleAR(dim=384, n_heads=8, n_layers=8).to(device)
    phoneme_ids = torch.randint(0, 512, (B, T_text), device=device)
    semantic_ids = torch.randint(0, 1024, (B, T_audio), device=device)

    logits = ar(phoneme_ids, semantic_ids)
    print(f"  phoneme_ids:  {phoneme_ids.shape}")
    print(f"  semantic_ids: {semantic_ids.shape}")
    print(f"  logits:       {logits.shape}")
    assert logits.shape == (B, T_audio, 1025), f"Expected ({B},{T_audio},1025), got {logits.shape}"
    ar_params = sum(p.numel() for p in ar.parameters())
    print(f"  params: {ar_params:,} ({ar_params/1e6:.1f}M)")

    # --- RVQ ---
    print("\n--- SimpleRVQ ---")
    rvq = SimpleRVQ(bins=1024, dim=768).to(device)
    ssl_feat = torch.randn(B, 768, T_audio, device=device)
    ids = rvq.encode(ssl_feat)
    decoded = rvq.decode(ids)
    print(f"  ssl_feat:  {ssl_feat.shape}")
    print(f"  ids:       {ids.shape}")
    print(f"  decoded:   {decoded.shape}")
    assert ids.shape == (B, T_audio)
    assert decoded.shape == (B, 768, T_audio)
    rvq_params = sum(p.numel() for p in rvq.parameters())
    print(f"  params: {rvq_params:,} ({rvq_params/1e6:.1f}M)")

    # --- TextEncoder ---
    print("\n--- SoVITSTextEncoder ---")
    text_enc = SoVITSTextEncoder(
        phoneme_vocab_size=512, out_channels=192, hidden_channels=192,
        filter_channels=768, n_layers=6, speaker_dim=256,
    ).to(device)
    speaker_emb = torch.randn(B, 256, device=device)
    m_p, logs_p = text_enc(phoneme_ids, speaker_emb)
    print(f"  phoneme_ids:  {phoneme_ids.shape}")
    print(f"  speaker_emb:  {speaker_emb.shape}")
    print(f"  m_p:          {m_p.shape}")
    print(f"  logs_p:       {logs_p.shape}")
    assert m_p.shape == (B, 192, T_text)
    te_params = sum(p.numel() for p in text_enc.parameters())
    print(f"  params: {te_params:,} ({te_params/1e6:.1f}M)")

    # --- PosteriorEncoder ---
    print("\n--- PosteriorEncoder ---")
    post_enc = PosteriorEncoder(
        in_channels=spec_channels, out_channels=192, hidden_channels=192, n_layers=8,
    ).to(device)
    spec = torch.randn(B, spec_channels, T_spec, device=device)
    z_q, m_q, logs_q = post_enc(spec)
    print(f"  spec:    {spec.shape}")
    print(f"  z_q:     {z_q.shape}")
    print(f"  m_q:     {m_q.shape}")
    print(f"  logs_q:  {logs_q.shape}")
    assert z_q.shape == (B, 192, T_spec)
    pe_params = sum(p.numel() for p in post_enc.parameters())
    print(f"  params: {pe_params:,} ({pe_params/1e6:.1f}M)")

    # --- Flow ---
    print("\n--- Flow ---")
    flow = Flow(channels=192, hidden_channels=192, n_flows=2).to(device)
    z_p, log_det = flow(z_q, reverse=False)
    z_q_back, log_det_inv = flow(z_p, reverse=True)
    print(f"  z_q (input):  {z_q.shape}")
    print(f"  z_p (output): {z_p.shape}")
    print(f"  log_det:      {log_det.shape}")
    assert z_p.shape == z_q.shape
    flow_params = sum(p.numel() for p in flow.parameters())
    print(f"  params: {flow_params:,} ({flow_params/1e6:.1f}M)")

    # --- Generator ---
    print("\n--- Generator ---")
    gen = Generator(
        initial_channel=192,
        upsample_initial_channel=256,
        upsample_rates=(10, 8, 2, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4, 4),
    ).to(device)
    wav = gen(z_q)
    hop = 10 * 8 * 2 * 2 * 2
    expected_T = T_spec * hop
    print(f"  z_q (input):  {z_q.shape}")
    print(f"  wav (output): {wav.shape}")
    print(f"  hop product:  {hop}")
    assert wav.shape == (B, 1, expected_T), f"Expected ({B},1,{expected_T}), got {wav.shape}"
    gen_params = sum(p.numel() for p in gen.parameters())
    print(f"  params: {gen_params:,} ({gen_params/1e6:.1f}M)")

    # --- ReferenceEncoder ---
    print("\n--- ReferenceEncoder ---")
    ref_enc = ReferenceEncoder(n_mels=n_mels, speaker_dim=256).to(device)
    ref_mel = torch.randn(B, n_mels, T_ref, device=device)
    spk = ref_enc(ref_mel)
    print(f"  ref_mel:     {ref_mel.shape}")
    print(f"  speaker_emb: {spk.shape}")
    assert spk.shape == (B, 256)
    ref_params = sum(p.numel() for p in ref_enc.parameters())
    print(f"  params: {ref_params:,} ({ref_params/1e6:.1f}M)")

    # --- Discriminator ---
    print("\n--- Discriminator ---")
    disc = Discriminator().to(device)
    real_wav = torch.randn(B, 1, expected_T, device=device)
    d_loss, g_loss, f_loss = disc(real_wav, wav)
    print(f"  disc_loss:  {d_loss.item():.4f}")
    print(f"  gen_loss:   {g_loss.item():.4f}")
    print(f"  feat_loss:  {f_loss.item():.4f}")
    disc_params = sum(p.numel() for p in disc.parameters())
    print(f"  params: {disc_params:,} ({disc_params/1e6:.1f}M)")

    # --- Full model ---
    print("\n--- GPTSoVITS (full model) ---")
    model = GPTSoVITS().to(device)

    # Stage 1
    logits = model.forward_stage1(phoneme_ids, semantic_ids)
    print(f"  Stage 1 logits: {logits.shape}")

    # Stage 2
    y_hat, m_p, logs_p, m_q, logs_q = model.forward_stage2(phoneme_ids, spec, ref_mel)
    print(f"  Stage 2 waveform: {y_hat.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params:     {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable params: {trainable_params:,} ({trainable_params/1e6:.1f}M)")

    # AR generation test
    print("\n--- AR generation test ---")
    prompt = torch.randint(0, 1024, (1, 5), device=device)
    text = torch.randint(0, 512, (1, 8), device=device)
    generated = model.ar.generate(text, prompt, max_new_tokens=10, top_k=5)
    print(f"  prompt:    {prompt.shape}")
    print(f"  generated: {generated.shape}")
    print(f"  tokens:    {generated[0].tolist()}")

    print("\n" + "=" * 60)
    print("All shape verification tests passed!")
    print("=" * 60)
