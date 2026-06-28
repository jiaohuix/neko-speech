# VoxCPM Architecture Deep Dive

> Source: arXiv:2509.24650, OpenBMB/VoxCPM codebase, local code at `/home/jhx/Projects/AIGC/Myprojs/VoxCPM/`

## 1. Executive Summary

VoxCPM is a **tokenizer-free** text-to-speech system that generates speech in a **continuous latent space** using diffusion autoregressive modeling. Instead of the standard `text → discrete tokens → audio` pipeline (like VITS, Bark, or XTTS), VoxCPM follows `text → continuous latent patches → audio waveform`, bypassing the information-loss bottleneck of discrete tokenization.

The model is built on the **MiniCPM-4** transformer backbone and uses a hierarchical semantic-acoustic architecture with **Conditional Flow Matching (CFM)** as the local diffusion decoder.

**Key numbers (VoxCPM-0.5B / v1):**
- Total parameters: ~0.5B
- Training data: 1.8M hours bilingual (EN+ZH)
- AudioVAE: 16 kHz input → 25 Hz latent rate, latent_dim=64
- Patch size: 2 (each patch = 2 latent frames)
- Autoregressive rate: ~12.5 steps/second
- Output: 16 kHz (v1), 48 kHz (v2)

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VoxCPM Full Pipeline                               │
└─────────────────────────────────────────────────────────────────────────────┘

  Text Tokens ──────────────────────────────────────────────────────────────┐
  (LlamaTokenizer)                                                          │
       │                                                                    ▼
       │                    ┌───────────────────┐                ┌────────────────────┐
       │                    │   AudioVAE        │  Audio Waveform│   AudioVAE         │
       │                    │   Encoder         │◄───────────────│   Decoder          │
       │                    │   (Causal CNN)    │                │   (Causal CNN)     │
       │                    │   16kHz → 25Hz    │                │   25Hz → 16/48kHz  │
       │                    │   latent_dim=64   │                │                    │
       │                    └────────┬──────────┘                └────────▲───────────┘
       │                             │                                    │
       │                     Latent: (D=64, T)                            │
       │                             │                                    │
       │                    ┌────────▼──────────┐                         │
       │                    │   Patchify        │                         │
       │                    │   patch_size=2    │                         │
       │                    │   → (T/2, P=2, D=64)                       │
       │                    └────────┬──────────┘                         │
       │                             │                                    │
       │                    ┌────────▼──────────┐                         │
       │                    │   LocEnc          │  Local Encoder           │
       │                    │   4-layer DiT     │  Compress P×D → hidden  │
       │                    │   hidden=1024     │  CLS token pooling       │
       │                    │   ffn=4096        │                         │
       │                    └────────┬──────────┘                         │
       │                             │                                    │
       │                    ┌────────▼──────────┐                         │
       │                    │   enc_to_lm_proj  │  Linear(1024 → hidden)  │
       │                    └────────┬──────────┘                         │
       │                             │                                    │
       │  ┌──────────────────────────▼────────────────────────────────┐   │
       │  │              TSLM (Text-Semantic Language Model)          │   │
       │  │              MiniCPM-4 Backbone                           │   │
       │  │              24 layers, hidden=896/1024, ffn=3584/4096    │   │
       │  │              GQA (16 heads, 8 kv_heads), LongRoPE        │   │
       │  │              Causal self-attention                        │   │
       │  │              Input: text_embed + audio_embed (interleaved)│   │
       │  └──────────────────────────┬────────────────────────────────┘   │
       │                             │                                    │
       │                    ┌────────▼──────────┐                         │
       │                    │   FSQ Layer        │  Finite Scalar Quant.   │
       │                    │   in=hidden        │  tanh → round*scale    │
       │                    │   latent=256/512   │  Straight-through est.  │
       │                    │   scale=9          │  Semantic bottleneck   │
       │                    └────────┬──────────┘                         │
       │                             │                                    │
       │  ┌──────────────────────────▼────────────────────────────────┐   │
       │  │              RALM (Residual Acoustic Language Model)      │   │
       │  │              MiniCPM-4 (shallow copy)                     │   │
       │  │              6 layers (v1) / 8 layers (v2)                │   │
       │  │              Same hidden_size as TSLM                     │   │
       │  │              Input: fusion(enc_out, audio_embed)          │   │
       │  └──────────────────────────┬────────────────────────────────┘   │
       │                             │                                    │
       │              ┌──────────────┴──────────────┐                      │
       │              │  dit_hidden = concat(        │                     │
       │              │    lm_to_dit(lm_hidden),     │                     │
       │              │    res_to_dit(res_hidden)     │                     │
       │              │  )                            │                     │
       │              └──────────────┬───────────────┘                     │
       │                             │                                    │
       │                    ┌────────▼──────────┐                         │
       │                    │   LocDiT + CFM     │  Local Diffusion        │
       │                    │   4 layers         │  Transformer            │
       │                    │   hidden=1024      │  Conditional Flow       │
       │                    │   ffn=4096         │  Matching               │
       │                    │   in_channels=64   │  Euler ODE solver       │
       │                    │   n_timesteps=10   │  CFG-Zero* guidance     │
       │                    └────────┬──────────┘                         │
       │                             │                                    │
       │                    Predicted latent patch (P=2, D=64)             │
       │                             │                                    │
       │                    ┌────────▼──────────┐                         │
       │                    │   Stop Predictor   │  3-layer MLP            │
       │                    │   hidden=1024      │  Binary: continue/stop  │
       │                    │   2-class head     │                         │
       │                    └───────────────────┘                         │
       │                                                                   │
       └───────────────────────────────────────────────────────────────────┘
```

---

## 3. Exact Parameter Configurations (from Code)

### 3.1 VoxCPM v1 (voxcpm.py)

```python
class VoxCPMConfig:
    patch_size: int = 2
    feat_dim: int = 64
    residual_lm_num_layers: int = 6
    scalar_quantization_latent_dim: int = 256
    scalar_quantization_scale: int = 9
    max_length: int = 4096

class VoxCPMEncoderConfig:  # LocEnc
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4

class VoxCPMDitConfig:  # LocDiT
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
```

### 3.2 VoxCPM v2 (voxcpm2.py)

```python
class VoxCPMConfig:
    patch_size: int = 4            # doubled from v1
    feat_dim: int = 64
    residual_lm_num_layers: int = 8  # increased from 6
    scalar_quantization_latent_dim: int = 512  # doubled from v1
    scalar_quantization_scale: int = 9
    max_length: int = 8192         # doubled from v1

class VoxCPMEncoderConfig:  # LocEnc (unchanged)
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4

class VoxCPMDitConfig:  # LocDiT (unchanged)
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
```

### 3.3 MiniCPM-4 Backbone (shared by TSLM and RALM)

The MiniCPM-4-0.5B backbone is loaded from a pretrained checkpoint. Based on the MiniCPM4-0.5B config:

```python
class MiniCPM4Config:
    hidden_size: int = 896          # from MiniCPM4-0.5B
    intermediate_size: int = 3584   # from MiniCPM4-0.5B (some configs show 4096)
    num_hidden_layers: int = 24     # TSLM uses all 24; RALM uses 6 (v1) or 8 (v2)
    num_attention_heads: int = 16
    num_key_value_heads: int = 8    # GQA: 2:1 ratio
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    use_mup: bool = True           # muP scaling
    scale_emb: float = 12.0        # embedding scale for muP
    dim_model_base: int = 256      # muP base dim
    scale_depth: float = 1.0       # depth scaling factor
    kv_channels: int = None        # head_dim = hidden_size // num_heads
```

### 3.4 AudioVAE v1

```python
class AudioVAEConfig:
    encoder_dim: int = 128
    encoder_rates: List[int] = [2, 5, 8, 8]   # product = 640
    latent_dim: int = 64
    decoder_dim: int = 1536
    decoder_rates: List[int] = [8, 8, 5, 2]   # product = 640
    depthwise: bool = True
    sample_rate: int = 16000
    use_noise_block: bool = False

# Derived:
# chunk_size = prod(encoder_rates) = 2*5*8*8 = 640 samples
# Latent frame rate = 16000 / 640 = 25 Hz
# Each patch = patch_size * chunk_size = 2 * 640 = 1280 samples = 80ms
# AR step rate = 25 / patch_size = 12.5 Hz
```

### 3.5 AudioVAE v2

```python
class AudioVAEConfig:  # AudioVAEConfigV2
    encoder_dim: int = 128
    encoder_rates: List[int] = [2, 5, 8, 8]   # product = 640 (same)
    latent_dim: int = 64
    decoder_dim: int = 2048                     # increased from 1536
    decoder_rates: List[int] = [8, 6, 5, 2, 2, 2]  # product = 1920
    depthwise: bool = True
    sample_rate: int = 16000                    # encoder input rate
    out_sample_rate: int = 48000                # decoder output rate (upscaled)
    use_noise_block: bool = False
    sr_bin_boundaries: List[int] = [20000, 30000, 40000]  # sample rate conditioning
    cond_type: str = "scale_bias"
    cond_dim: int = 128
    cond_out_layer: bool = False

# Derived:
# chunk_size (encoder) = 640 samples at 16kHz
# decode_chunk_size = prod(decoder_rates) = 1920 samples at 48kHz
# Effective upsampling: 48000/16000 = 3x
# Sample rate conditioning: 4 buckets based on sr_bin_boundaries
```

### 3.6 CFM Configuration

```python
class CfmConfig:
    sigma_min: float = 1e-6
    solver: str = "euler"
    t_scheduler: str = "log-norm"
    training_cfg_rate: float = 0.1        # 10% CFG dropout during training
    inference_cfg_rate: float = 1.0       # CFG scale at inference
    reg_loss_type: str = "l1"
    ratio_r_neq_t_range: Tuple[float, float] = (0.25, 0.75)
    noise_cond_prob_range: Tuple[float, float] = (0.0, 0.0)
    noise_cond_scale: float = 0.0

# Inference defaults:
# n_timesteps = 10 (Euler ODE steps)
# cfg_value = 2.0 (classifier-free guidance scale)
# temperature = 1.0 (initial noise scale)
# sway_sampling_coef = 1.0 (sway sampling schedule)
```

### 3.7 Training Configuration (from train script)

```python
# Optimizer
optimizer = AdamW(lr=1e-4, weight_decay=1e-2)

# Scheduler
scheduler = CosineWithWarmup(warmup_steps=1000, total_steps=100000)

# Loss weights
lambdas = {"loss/diff": 1.0, "loss/stop": 1.0}

# Other
batch_size = 1 (with gradient accumulation)
grad_accum_steps = configurable
num_iters = 100_000
max_grad_norm = 0.0 (disabled by default)
dtype = bfloat16
```

---

## 4. Tokenizer-Free Design: How It Works

### 4.1 The Tokenization Bottleneck Problem

Traditional TTS systems (VITS, Bark, XTTS, CosyVoice) use a two-stage pipeline:

```
Traditional:  text → [neural audio tokenizer] → discrete tokens → [AR model] → tokens → [decoder] → waveform
                        EnCodec/SoundStream/DAC         LM              codec decoder
```

These tokenizers (EnCodec, SoundStream, DAC) compress audio into **discrete codebook indices** via residual vector quantization (RVQ). The problem:
1. **Information loss**: Quantization is lossy — fine-grained acoustic details (prosody, timbre micro-variations) are irreversibly discarded
2. **Codebook collapse**: Many codebook entries go unused, reducing effective capacity
3. **Multi-stage complexity**: RVQ requires sequential prediction of multiple codebook levels (typically 8-12), slowing inference
4. **Rate-fidelity tradeoff**: Lower token rates lose quality; higher rates increase sequence length quadratically

### 4.2 VoxCPM's Continuous Latent Approach

VoxCPM replaces discrete tokenization with a **continuous latent space**:

```
VoxCPM:  text → [AudioVAE encoder] → continuous latents → [AR model + CFM] → latents → [AudioVAE decoder] → waveform
                    (causal CNN)          (TSLM+RALM+LocDiT)                  (causal CNN)
```

The **AudioVAE** encoder maps raw waveform to a continuous latent `z ∈ R^{D×T}` where D=64 and T=25 frames/sec. This latent is:
- **Continuous**: No quantization, full gradient flow
- **Low-dimensional**: 64 channels at 25 Hz (vs. EnCodec's 8 codebooks × 50 Hz)
- **Causal**: All convolutions are causal (left-padded), enabling streaming

### 4.3 The Patch Mechanism

Raw latents at 25 Hz would require very long sequences. VoxCPM groups consecutive latent frames into **patches**:

```
Latent z: [D=64, T=200]  (8 seconds of audio)
       ↓ patchify (patch_size=2 for v1, 4 for v2)
Patches:  [T/patch_size, patch_size, D] = [100, 2, 64]  (v1)
          [T/patch_size, patch_size, D] = [50, 4, 64]   (v2)
```

Each patch represents `patch_size × chunk_size` audio samples:
- v1: 2 × 640 = 1280 samples = 80ms per patch at 16 kHz
- v2: 4 × 640 = 2560 samples = 160ms per patch at 16 kHz

The AR model then generates **one patch per step**, making the effective generation rate:
- v1: 12.5 patches/sec
- v2: 6.25 patches/sec

### 4.4 Scalar Quantization as Semantic Bottleneck

Instead of discrete VQ, VoxCPM uses **Finite Scalar Quantization (FSQ)** between TSLM and RALM:

```python
class ScalarQuantizationLayer(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim=64, scale=9):
        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, out_dim)

    def forward(self, hidden):
        hidden = self.in_proj(hidden)           # project down to latent_dim
        hidden = torch.tanh(hidden)              # bound to [-1, 1]
        if self.training:
            quantized = torch.round(hidden * self.scale) / self.scale
            hidden = hidden + (quantized - hidden).detach()  # straight-through estimator
        else:
            hidden = torch.round(hidden * self.scale) / self.scale
        return self.out_proj(hidden)            # project back to full dim
```

This creates a **differentiable bottleneck** that:
1. **Compresses**: Projects from hidden_size (896/1024) → latent_dim (256/512) → hidden_size
2. **Discretizes lightly**: Round to `scale=9` levels, giving 19 possible values per dimension (in [-1, 1] with step 1/9)
3. **Separates semantics from acoustics**: The quantized output carries semantic information (coarse structure); the RALM recovers acoustic detail from the residual
4. **Remains differentiable**: Straight-through estimator allows gradient flow during training

**Key insight**: Unlike EnCodec's RVQ which aggressively quantizes audio, FSQ gently quantizes the LM's hidden state. The quantization is on the **semantic representation**, not the audio itself.

---

## 5. Hierarchical Semantic-Acoustic Architecture

### 5.1 TSLM (Text-Semantic Language Model)

The TSLM is the "brain" — it plans what to say at a semantic level:
- Initialized from pretrained MiniCPM-4-0.5B (24 layers)
- Processes interleaved text and audio embeddings
- Uses causal self-attention (autoregressive)
- Output: semantic representation of the next patch

### 5.2 RALM (Residual Acoustic Language Model)

The RALM is the "voice" — it adds acoustic detail:
- Shallow copy of MiniCPM-4 (6-8 layers, NOT pretrained)
- Input: fusion of TSLM output + audio embedding
- Causal self-attention
- Output: acoustic residual that complements the semantic signal

**v1 vs v2 fusion:**
```python
# v1: additive fusion
dit_hidden = lm_to_dit_proj(lm_hidden) + res_to_dit_proj(residual_hidden)

# v2: concatenation + projection
dit_hidden = fusion_concat_proj(concat(lm_to_dit_proj(lm_hidden), res_to_dit_proj(residual_hidden)))
# fusion_concat_proj: Linear(hidden*2 → hidden)
```

### 5.3 LocDiT (Local Diffusion Transformer)

The LocDiT generates the actual continuous latent patch via diffusion:
- 4-layer transformer (non-causal, processes the whole patch at once)
- Takes: conditioning vector (dit_hidden), noisy input, timestep, and previous patch as condition
- Output: predicted velocity field for CFM

**Sequence structure inside LocDiT:**
```
Input to LocDiT decoder:
[mu_token, time_token, cond_patches..., noisy_input_patches...]
     │          │              │                    │
     │          │              │                    └── in_proj(x): (P, hidden)
     │          │              └── cond_proj(cond): (P', hidden)
     │          └── time_embedding(t): (1, hidden)
     └── mu reshaped: (1 or D/hidden, hidden)
```

The transformer processes all tokens with **non-causal** (bidirectional) attention, then extracts only the output portion corresponding to the input patches.

---

## 6. Conditional Flow Matching (CFM) Mechanism

### 6.1 Overview

CFM is a simulation-free approach to training continuous normalizing flows. Instead of solving an ODE during training (expensive), it directly regresses on the velocity field:

```
Training objective:
  L = E_{t, x₁, z} [||v_θ(y_t, t, μ) - (z - x₁)||²]

where:
  x₁ = target latent patch (ground truth)
  z  = random noise ~ N(0, I)
  t  = time sampled from log-normal distribution
  y_t = (1-t)·x₁ + t·z    (interpolated noisy sample)
  v_θ = predicted velocity field (LocDiT output)
  μ  = conditioning from TSLM+RALM
```

### 6.2 Training Loss (from `unified_cfm.py`)

```python
def compute_loss(self, x1, mu, cond=None, tgt_mask=None, progress=0.0):
    # 1. Sample time steps (log-normal distribution)
    r, t = self.sample_r_t(x1)
    
    # 2. Sample noise
    z = torch.randn_like(x1)
    
    # 3. Interpolate: y_t = (1-t)*x1 + t*z
    y = (1 - t.view(-1,1,1)) * x1 + t.view(-1,1,1) * z
    
    # 4. Target velocity: v = z - x1  (optimal transport)
    v = z - x1
    
    # 5. Predict velocity
    u_pred = self.estimator(y, mu, t, cond, dt=t-r)
    
    # 6. MSE loss with adaptive weighting
    loss = F.mse_loss(u_pred, v.detach(), reduction='none').mean(dim=1)
    if tgt_mask is not None:
        weights = self.adaptive_loss_weighting(losses, tgt_mask)
        loss = (weights * losses).sum() / torch.clamp(torch.sum(tgt_mask), min=1.0)
```

### 6.3 Inference: Euler ODE Solver

At inference, the trained velocity field is integrated using Euler's method:

```python
def solve_euler(self, x, t_span, mu, cond, cfg_value=1.0):
    for step in range(1, len(t_span)):
        # Classifier-Free Guidance: run with and without conditioning
        dphi_dt = self.estimator(x_in, mu_in, t_in, cond_in, dt_in)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [b, b], dim=0)
        
        # CFG-Zero*: adaptive scale optimization
        st_star = self.optimized_scale(dphi_dt, cfg_dphi_dt)
        dphi_dt = cfg_dphi_dt * st_star + cfg_value * (dphi_dt - cfg_dphi_dt * st_star)
        
        # Euler step
        x = x - dt * dphi_dt
```

**Key features:**
- **Sway sampling**: `t_span = linspace(1, 0, n+1) + sway_coef * (cos(π/2 * t) - 1 + t)` — non-linear time schedule
- **CFG-Zero***: Adaptive scale `st_star = dot(v_cond, v_uncond) / ||v_uncond||²` — prevents over-exposure
- **Initial zero steps**: First 4% of steps use zero velocity for stability
- Only **10 Euler steps** needed (vs. 50-1000 for DDPM)

### 6.4 Autoregressive Diffusion

The key innovation: CFM generates **one patch at a time** autoregressively:

```
Step t=1: Generate patch_1 conditioned on text + silence
Step t=2: Generate patch_2 conditioned on text + patch_1
Step t=3: Generate patch_3 conditioned on text + patch_1 + patch_2
...
Step t=N: Generate patch_N conditioned on text + all previous patches
```

Each step involves:
1. TSLM forward_step (1 token, KV cache) → semantic hidden
2. FSQ quantization → quantized semantic hidden
3. RALM forward_step (1 token, KV cache) → acoustic hidden
4. Concat + project → conditioning vector μ
5. LocDiT + CFM (10 Euler steps) → predicted latent patch
6. Stop predictor → continue or halt

---

## 7. Key Code Patterns

### 7.1 Combined Text-Audio Embedding

```python
# Interleave text and audio in a single sequence
combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed
```

This elegant design allows the TSLM to process text and audio in the same sequence, with masks selecting which modality is active at each position.

### 7.2 Teacher-Forcing with Shifted Hidden

```python
# Training: shift right by 1 for autoregressive targets
lm_hidden = torch.cat((zeros_like(enc_outputs[:, 0:1, :]), enc_outputs[:, :-1, :]), dim=1)
```

The TSLM output at position `t-1` is used as conditioning to predict patch at position `t`.

### 7.3 LocEnc CLS Token Pooling

```python
class VoxCPMLocEnc(nn.Module):
    def forward(self, x):
        B, T, P, D = x.shape
        x = self.in_proj(x)                              # (B, T, P, hidden)
        special_tokens = self.special_token.expand(B, T, 1, -1)  # CLS token
        x = torch.cat([special_tokens, x], dim=2)        # (B, T, P+1, hidden)
        x = rearrange(x, "b t p c -> (b t) p c")
        outputs, _ = self.encoder(x, is_causal=False)    # bidirectional
        cls_output = outputs[:, 0, :]                    # pool CLS token
        return rearrange(cls_output, "(b t) c -> b t c", b=B)
```

Each patch of P×D features is compressed to a single hidden vector via CLS token pooling.

### 7.4 Streaming Inference with KV Cache

```python
# Pre-fill: process entire prompt in parallel
enc_outputs, kv_cache = self.base_lm(inputs_embeds=combined_embed, is_causal=True)

# Autoregressive generation: one step at a time
for i in range(max_len):
    lm_hidden = self.base_lm.forward_step(curr_embed, position_id)
    # KV cache updated incrementally
```

The MiniCPM-4 backbone uses a `StaticKVCache` (pre-allocated tensor) for O(1) per-step inference.

### 7.5 AudioVAE Streaming Decode

```python
class StreamingVAEDecoder:
    """Stateful streaming wrapper — patches causal conv buffers between calls"""
    def decode_chunk(self, z_chunk):
        return self._vae.decode(z_chunk)
```

The V2 AudioVAE supports true streaming decode by monkey-patching causal conv layers to maintain rolling buffers, eliminating redundant overlap computation.

---

## 8. Comparison: VoxCPM vs. Tokenizer-Based Systems

| Aspect | EnCodec/SoundStream | VoxCPM (AudioVAE) |
|--------|---------------------|-------------------|
| **Representation** | Discrete codebook indices (RVQ) | Continuous latent vectors |
| **Quantization** | Argmin over codebook (non-differentiable) | Scalar rounding (differentiable, straight-through) |
| **Latent dim** | 8-12 codebooks × 1024 entries each | 64 continuous dimensions |
| **Frame rate** | 50 Hz (EnCodec) / 75 Hz (SoundStream) | 25 Hz |
| **Information loss** | High (codebook quantization) | Low (continuous + gentle FSQ on hidden state) |
| **AR modeling** | Predict discrete tokens (CE loss) | Predict continuous patches (CFM/diffusion loss) |
| **Multi-codebook** | Sequential prediction of 8-12 levels | Single step (no codebook hierarchy) |
| **Fidelity ceiling** | Limited by codebook size and RVQ depth | Limited only by VAE reconstruction quality |
| **Gradient flow** | Blocked at quantization (requires tricks) | Full gradient through latent space |
| **Training stability** | Codebook collapse, index entropy issues | Standard MSE on continuous values |

### Why Tokenizer-Free Matters

1. **No codebook collapse**: EnCodec often has 30-50% unused codebook entries
2. **No RVQ error accumulation**: Each RVQ level adds quantization error
3. **Simpler AR model**: Predict one continuous vector vs. sequential discrete distributions
4. **Better voice cloning**: Continuous latents preserve speaker-specific micro-patterns
5. **End-to-end differentiable**: The entire pipeline (VAE encoder → LM → VAE decoder) is differentiable

---

## 9. Suggested Simplification for Teaching Version (~100M params)

### 9.1 Target Architecture: "MiniVoxCPM"

| Component | Original (0.5B) | Teaching (~100M) | Rationale |
|-----------|-----------------|-------------------|-----------|
| **TSLM** | 24L, hidden=896, ffn=3584 | 8L, hidden=512, ffn=2048 | ~30M params |
| **RALM** | 6L, hidden=896, ffn=3584 | 4L, hidden=512, ffn=2048 | ~12M params |
| **LocEnc** | 4L, hidden=1024, ffn=4096 | 2L, hidden=256, ffn=1024 | ~2M params |
| **LocDiT** | 4L, hidden=1024, ffn=4096 | 2L, hidden=256, ffn=1024 | ~2M params |
| **AudioVAE** | encoder_dim=128, latent=64 | encoder_dim=64, latent=32 | ~5M params |
| **FSQ** | latent=256, scale=9 | latent=64, scale=5 | Minimal |
| **Projections** | Various 896↔1024 | Various 512↔256 | ~1M |
| **AudioVAE Decoder** | dim=1536/2048 | dim=512 | ~40M |
| **Total** | ~500M | ~92M | |

### 9.2 Simplified AudioVAE

```python
class TeachingAudioVAEConfig:
    encoder_dim: int = 64
    encoder_rates: List[int] = [2, 4, 8]      # product = 64, simpler
    latent_dim: int = 32                        # halved
    decoder_dim: int = 512
    decoder_rates: List[int] = [8, 4, 2]       # product = 64, symmetric
    depthwise: bool = False                     # simpler architecture
    sample_rate: int = 16000
```

This gives: chunk_size = 64 samples, latent rate = 250 Hz (higher but simpler).

### 9.3 Simplified CFM

```python
class TeachingCfmConfig:
    sigma_min: float = 1e-5          # slightly higher for stability
    solver: str = "euler"
    t_scheduler: str = "uniform"     # simpler than log-normal
    training_cfg_rate: float = 0.1
    inference_cfg_rate: float = 1.0
    n_timesteps: int = 5             # fewer steps (5 instead of 10)
```

### 9.4 Teaching Version Key Simplifications

1. **Remove muP scaling**: Set `use_mup=False` to avoid scale_emb, scale_depth complexities
2. **Remove LongRoPE**: Use standard RoPE (no long/short factor switching)
3. **Remove GQA**: Use MHA (num_kv_heads = num_heads) for simpler attention
4. **Remove sample rate conditioning**: Use AudioVAE v1 (no sr_bin_boundaries)
5. **Remove streaming decode**: Use simple batched VAE decode
6. **Simplify FSQ**: Reduce latent_dim to 64, scale to 5
7. **Patch size = 1**: Each AR step generates 1 latent frame (no patchification)
8. **Remove stop predictor**: Fixed-length generation for teaching
9. **Remove reference audio tokens**: Only continuation mode (simpler inference)
10. **Use v1 additive fusion**: `dit_hidden = proj(lm) + proj(res)` instead of concat+project

### 9.5 Minimal Teaching Model Code Skeleton

```python
class MiniVoxCPM(nn.Module):
    def __init__(self):
        # Audio VAE (frozen)
        self.audio_vae = SimpleAudioVAE()
        
        # Local Encoder: patch → hidden
        self.loc_enc = nn.Sequential(
            nn.Linear(32, 256),
            nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, batch_first=True),
        )
        
        # TSLM: text + audio → semantic hidden
        self.tslm = SimpleTransformer(num_layers=8, hidden=512, ffn=2048)
        
        # FSQ: semantic bottleneck
        self.fsq = SimpleFSQ(in_dim=512, latent_dim=64, scale=5)
        
        # RALM: acoustic refinement
        self.ralm = SimpleTransformer(num_layers=4, hidden=512, ffn=2048)
        
        # LocDiT: diffusion decoder for one patch
        self.loc_dit = SimpleDiT(num_layers=2, hidden=256, in_channels=32)
        
    def forward_step(self, text_tokens, prev_latent):
        # 1. Encode previous latent patch
        enc = self.loc_enc(prev_latent)
        
        # 2. TSLM step
        semantic = self.tslm.forward_step(text_emb + enc)
        
        # 3. FSQ
        quantized = self.fsq(semantic)
        
        # 4. RALM step
        acoustic = self.ralm.forward_step(quantized + enc)
        
        # 5. CFM: generate next latent patch (5 Euler steps)
        mu = semantic + acoustic
        next_patch = self.cfm_sample(mu, n_steps=5)
        
        return next_patch
```

---

## 10. Parameter Count Breakdown

### 10.1 VoxCPM-0.5B (estimated)

| Component | Params (M) | Notes |
|-----------|-----------|-------|
| TSLM (MiniCPM-4-0.5B) | ~350 | 24L, h=896, ffn=3584, GQA |
| RALM | ~60 | 6L, same config (not pretrained) |
| LocEnc | ~18 | 4L, h=1024, ffn=4096 |
| LocDiT | ~18 | 4L, h=1024, ffn=4096 |
| AudioVAE encoder | ~3 | Causal CNN, dim=128 |
| AudioVAE decoder | ~20 | Causal CNN, dim=1536 |
| FSQ | ~1 | 2 linear projections |
| Projections | ~5 | enc_to_lm, lm_to_dit, res_to_dit |
| Stop predictor | ~2 | 3-layer MLP |
| **Total** | **~477** | |

### 10.2 VoxCPM2 (estimated ~2B)

VoxCPM2 scales up the backbone (likely using a larger MiniCPM-4 variant) with:
- Larger hidden_size and/or more layers
- patch_size=4 (fewer AR steps, more computation per step)
- FSQ latent=512 (richer semantic bottleneck)
- RALM 8 layers (deeper acoustic refinement)
- AudioVAE v2 with 48 kHz output and sample rate conditioning

---

## 11. Key Insights for neko-speech

### 11.1 What to Learn from VoxCPM

1. **Continuous > Discrete for audio**: The tokenizer-free approach avoids the entire class of codebook-related problems
2. **FSQ is elegant**: A simple `tanh → round → straight-through` creates a semantic bottleneck without discrete tokens
3. **Hierarchical LM design**: TSLM (semantics) + RALM (acoustics) naturally decomposes the generation problem
4. **CFM is efficient**: 10 Euler steps vs. 50-1000 DDPM steps, with better quality
5. **Patch-based AR**: Generating multi-frame patches reduces sequence length while maintaining quality
6. **Causal AudioVAE**: Enables true streaming inference

### 11.2 What to Simplify for Teaching

1. Start with **AudioVAE only** (Chapter: "Audio Fundamentals") — understand continuous latent spaces
2. Add **LocEnc + simple AR** (Chapter: "Autoregressive Modeling") — predict latents from text
3. Add **FSQ** (Chapter: "Semantic-Acoustic Separation") — understand the bottleneck
4. Add **CFM/LocDiT** (Chapter: "Diffusion for Audio") — understand flow matching
5. Add **streaming** (Chapter: "Real-Time Inference") — KV cache + streaming VAE

### 11.3 Suggested Chapter Integration

| neko-speech Chapter | VoxCPM Component | Teaching Goal |
|---------------------|------------------|---------------|
| Ch01: Audio Fundamentals | AudioVAE encode/decode | Understand latent space, mel vs. latent |
| Ch02: Tacotron2 Pipeline | Contrast with VoxCPM | Why seq2seq + vocoder is limiting |
| Ch03: Neural Audio Codecs | AudioVAE vs EnCodec | Continuous vs discrete, RVQ vs VAE |
| Ch04: Language Models for Audio | TSLM + RALM | Hierarchical AR modeling |
| Ch05: Diffusion Models | LocDiT + CFM | Flow matching for audio generation |
| Ch06: Voice Cloning | Reference tokens + CFG | Zero-shot speaker adaptation |
| Ch07: Streaming | KV cache + streaming VAE | Real-time TTS systems |

---

## 12. File Index

| File | Purpose |
|------|---------|
| `src/voxcpm/model/voxcpm2.py` | Main VoxCPM2 model (v2, latest) |
| `src/voxcpm/model/voxcpm.py` | Original VoxCPM model (v1) |
| `src/voxcpm/modules/minicpm4/model.py` | MiniCPM-4 transformer backbone |
| `src/voxcpm/modules/minicpm4/config.py` | MiniCPM-4 configuration |
| `src/voxcpm/modules/minicpm4/cache.py` | Static KV cache for inference |
| `src/voxcpm/modules/audiovae/audio_vae.py` | AudioVAE v1 (16kHz I/O) |
| `src/voxcpm/modules/audiovae/audio_vae_v2.py` | AudioVAE v2 (16kHz→48kHz, SR conditioning) |
| `src/voxcpm/modules/locdit/unified_cfm.py` | Conditional Flow Matching implementation |
| `src/voxcpm/modules/locdit/local_dit.py` | Local DiT v1 |
| `src/voxcpm/modules/locdit/local_dit_v2.py` | Local DiT v2 (mu reshaping) |
| `src/voxcpm/modules/locenc/local_encoder.py` | Local Encoder (CLS pooling) |
| `src/voxcpm/modules/layers/scalar_quantization_layer.py` | FSQ implementation |
| `src/voxcpm/modules/layers/lora.py` | LoRA for fine-tuning |
| `scripts/train_voxcpm_finetune.py` | Training script |

---

## Sources

- [VoxCPM Paper (arXiv:2509.24650)](https://arxiv.org/abs/2509.24650)
- [VoxCPM Paper HTML](https://arxiv.org/html/2509.24650v1)
- [OpenBMB/VoxCPM GitHub](https://github.com/OpenBMB/VoxCPM)
- [VoxCPM Demo Page](https://openbmb.github.io/VoxCPM-demopage/)
- [VoxCPM Website](https://voxcpm.net/)
- [openbmb/VoxCPM2 on Hugging Face](https://huggingface.co/openbmb/VoxCPM2)
- [MiniCPM4-0.5B Config](https://huggingface.co/openbmb/MiniCPM4-0.5B/blob/refs%2Fpr%2F6/config.json)
- [VoxCPM OpenReview (ICLR)](https://openreview.net/forum?id=h5KLpGoqzC)
- [VoxCPM ReadTheDocs](https://voxcpm.readthedocs.io/en/latest/models/architecture.html)
