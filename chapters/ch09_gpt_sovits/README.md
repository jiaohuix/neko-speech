# Ch09: GPT-SoVITS -- Few-Shot Voice Cloning

> In previous chapters, Neko learned to synthesize speech from text (Ch02-Ch05),
> compress audio into discrete tokens (Ch06),
> and even clone voices with VALL-E's multi-layer codec approach (Ch07).
>
> But VALL-E's AR model must predict 8 codec layers x 50Hz = 400 tokens/second.
> That's slow. Can we do better?
>
> This chapter: **GPT-SoVITS** -- voice cloning with just **1 semantic token per frame**.

---

## Chapter Guide

### Why GPT-SoVITS?

Recall VALL-E's architecture from Ch07:

| Problem | VALL-E's Approach | Cost |
|---------|-------------------|------|
| How to tokenize audio? | EnCodec multi-layer codebook (n_q=8) | 8 tokens per frame |
| What does the AR model predict? | Layer-by-layer (AR + NAR) | Sequence = 50Hz x 8 layers |
| Reference audio needed? | 3-10 seconds | Requires codec pretraining |

GPT-SoVITS's core insight: **not all token layers are equally important**.

HuBERT's first quantization codebook already captures most of the **semantic information**
(what is said), while **timbre information** (who says it) can be passed directly
to the vocoder via reference audio.

```
VALL-E:     Text + Ref -> AR(8-layer tokens) -> Codec Decoder -> Waveform
                          Slow, long sequences

GPT-SoVITS: Text + Ref -> AR(1-layer tokens) -> VITS Vocoder  -> Waveform
                          8x faster, better quality
```

### Key Innovations

1. **Single-layer semantic token (n_q=1)**: The AR model predicts only 1 token/frame, 8x shorter sequences
2. **VITS-based vocoder**: Higher quality than codec decoders, end-to-end waveform generation
3. **Two-stage separation**: AR handles "what to say", vocoder handles "how to say it"
4. **Zero-shot voice cloning**: 3-10 seconds of reference audio is enough, no fine-tuning required

### Learning Roadmap

| Section | Topic | Goal |
|---------|-------|------|
| 9.1 | System architecture overview | Understand the two-stage data flow |
| 9.2 | Semantic tokens: why n_q=1 suffices | Semantic vs acoustic tokens |
| 9.3 | AR model: GPT-style Transformer | Attention mask design |
| 9.4 | SoVITS vocoder: VITS variant | TextEncoder, VAE + Flow + HiFi-GAN |
| 9.5 | Two-stage training | Why separate training? |
| 9.6 | Inference pipeline | End-to-end: text + reference -> waveform |
| 9.7 | Code walkthrough | Implementation details |
| 9.8 | Production comparison | Teaching version vs full GPT-SoVITS |
| 9.9 | ONNX export | Deployment optimization |

---

## 9.1 System Architecture Overview

### Inference Data Flow

```
  Reference Audio (3-10s, 16kHz)
        |
        v
  [HuBERT / SSL Model]  --> SSL features (768-dim, 50Hz)
        |                           |
        v                           v
  [RVQ Quantizer]             (extract prompt tokens)
  n_q=1, bins=1024                  |
        |                           |
        v                           v
  prompt_semantic_ids        +--------------+
  (e.g. [42, 187, 3, ...])   |  SimpleAR    | <-- Text Phonemes
                             |  (GPT-style) |
                             +--------------+
                                     |
                                     v
                             predicted_semantic_ids
                             (e.g. [42, 187, 3, 55, 201, ...])
                                     |
                                     v
                        +---------------------------+
                        |   SoVITS Vocoder           | <-- Text + Ref Mel
                        |   (VITS-based)             |
                        |                            |
                        |  TextEncoder -> Prior      |
                        |  Flow^{-1} -> z            |
                        |  Generator -> Waveform     |
                        +---------------------------+
                                     |
                                     v
                               Output Waveform (32kHz)
```

### Key Signal Dimensions

| Signal | Rate | Shape | Notes |
|--------|------|-------|-------|
| Text Phonemes | ~10-20 Hz | `(T_text,)` | Discrete IDs, vocab=512 |
| SSL Features | 50 Hz | `(768, T_ssl)` | HuBERT output |
| Semantic Tokens | 50 Hz | `(T_audio,)` | Discrete IDs, vocab=1025 (incl. EOS) |
| Linear Spectrogram | 50 Hz | `(1025, T_spec)` | n_fft=2048 |
| Mel Spectrogram | 50 Hz | `(128, T_mel)` | For reference encoder |
| Waveform | 32 kHz | `(T_wav,)` | Final output |

### Training vs Inference

```
Training:
  Stage 1 (AR):    phoneme_ids + semantic_ids -> predict next semantic_id
  Stage 2 (SoVITS): phoneme_ids + spec + ref_mel -> reconstruct waveform (VAE+Flow+GAN)

Inference:
  1. AR:    phoneme_ids + prompt_tokens -> autoregressive semantic_ids
  2. SoVITS: phoneme_ids + speaker_emb -> sample z + inverse flow + generate waveform
  (PosteriorEncoder and Discriminator are only used during training)
```

---

## 9.2 Semantic Tokens: Why n_q=1 Suffices

### 9.2.1 The Problem with Multi-Layer Codec Tokens

In Ch06 we learned about neural audio codecs like EnCodec. They use **multi-layer RVQ**
to compress audio into stacked token sequences:

```
Audio Waveform
    |  Encoder
    v
Continuous Features (50Hz, 128-dim)
    |  RVQ (n_q=8)
    v
8 Token Layers:
  Layer 0: [42, 187, 3, 55, ...]     <-- Semantic layer (what is said)
  Layer 1: [12, 89, 201, 7, ...]     <-- Acoustic detail layer
  Layer 2: [45, 123, 8, 99, ...]     <-- Finer details
  ...
  Layer 7: [1, 0, 3, 2, ...]         <-- Finest residual
```

Deeper layers carry increasingly "acoustic" information (timbre, formant details)
and less "semantic" information.

### 9.2.2 HuBERT + Single-Layer Quantization

GPT-SoVITS doesn't perform full audio encoding/decoding. It does one thing:

> **Extract semantic features with HuBERT, then quantize with a single VQ codebook.**

```
Audio (16kHz)
    |  HuBERT (frozen SSL model)
    v
SSL features (768-dim, 50Hz)    <-- Semantically rich continuous representation
    |  RVQ (n_q=1, bins=1024)
    v
semantic_ids (50Hz)              <-- 1 integer token per frame
```

**Why are HuBERT features naturally "semantic"?**

HuBERT's training objective is **masked prediction of phoneme clusters**.
It learns to understand *what is said*, not *how it's said*.
So even the first VQ layer captures the primary semantic content.

### 9.2.3 What About Timbre?

Semantic tokens lose timbre information. GPT-SoVITS has an elegant solution:

```
                   Semantic Tokens (timbre lost)
                        |
                        v
SoVITS Vocoder <-- Reference Audio Mel Spectrogram (timbre preserved)
```

The vocoder receives both:
1. **Semantic tokens** -> knows "what is said"
2. **Reference audio** -> knows "who says it" (via ReferenceEncoder -> speaker embedding)

Timbre is injected directly into the vocoder -- the AR model never needs to predict it.

### 9.2.4 Efficiency Gains from n_q=1

| Approach | Token Rate | AR Sequence (1 second) | AR Prediction Target |
|----------|------------|----------------------|---------------------|
| VALL-E (EnCodec) | 50Hz x 8 = 400 tok/s | 400 | 8 codebook layers |
| **GPT-SoVITS** | **50Hz x 1 = 50 tok/s** | **50** | **1 codebook layer** |

AR sequence length reduced **8x**, inference speed improved **8x**.

> **Note**: This isn't a "free lunch." The cost is that the vocoder must be
> "smarter" -- it has to reconstruct high-quality waveform from coarse semantic
> tokens + reference audio. That's why GPT-SoVITS uses VITS instead of a
> simple codec decoder.

---

## 9.3 AR Model: GPT-style Transformer

### 9.3.1 Core Idea

The AR model's task is identical in structure to language modeling:

> **Given text phonemes and existing semantic tokens, predict the next semantic token.**

```
GPT:    "The weather today" -> predict -> "is nice"
AR:     phonemes + [42, 187, 3] -> predict -> 55
```

### 9.3.2 Architecture

```
  SimpleAR Architecture
  ======================

  phoneme_ids (B, T_text)
      |
  Text Embedding (vocab=512 -> dim=384)
      |
  + Sine Positional Encoding
      |
      |  concat
      v
  [text_emb | audio_emb]  (B, T_text + T_audio, 384)
      |
  8 x CausalTransformerBlock
  (causal self-attention, 8 heads, 4x FFN)
      |
  LayerNorm
      |
  Extract audio portion (positions T_text:)
      |
  Linear(384, 1025, bias=False)  <-- prediction layer
      |
  logits (B, T_audio, 1025)
```

### 9.3.3 Attention Mask Design

The attention mask in GPT-SoVITS is one of its most elegant design choices.

**Original implementation** uses a hybrid mask over the concatenated `[text | audio]` sequence:

```
  Attention Mask Layout (original GPT-SoVITS):

             text    audio
  text   [ bidirectional  |   masked   ]   <- text attends to all text
         [                |            ]      text does NOT see audio
  audio  [ all-to-text    |   causal   ]   <- audio sees all text
         [                |  (upper    ]      audio sees only past audio
                            triangular)
```

**Why this design?**

1. **Text bidirectional**: Phonemes have no causal relationship; mutual attention gives better text representations
2. **Audio causal**: Each semantic token can only depend on past tokens (autoregressive constraint)
3. **Audio->Text full attention**: Each audio token sees the complete text (knows what to say)
4. **Text ignores Audio**: Text representations don't need audio (prevents information leakage)

> Our **simplified implementation** uses pure causal attention for the entire sequence.
> Since text is short, the bidirectional-vs-causal difference is minor, and the
> implementation is much cleaner.

### 9.3.4 Hyperparameter Comparison

| Parameter | Original GPT-SoVITS | Teaching Version (ours) | Change |
|-----------|---------------------|------------------------|--------|
| hidden_dim | 512 | 384 | -25% |
| num_layers | 12-24 | 8 | -33% |
| num_heads | 16 | 8 | -50% |
| vocab_size | 1025 | 1025 | same |
| phoneme_vocab | 512-732 | 512 | same |
| BERT features | 1024-dim RoBERTa | Not used | simplified |
| Parameters | ~40M | ~15.2M | -62% |

### 9.3.5 Inference: Autoregressive Generation with Top-k Sampling

```python
# Pseudocode: AR inference
def generate(phoneme_ids, prompt_tokens, max_tokens=500):
    current = prompt_tokens.clone()

    for step in range(max_tokens):
        logits = ar_model(phoneme_ids, current)   # (1, T, 1025)
        next_logits = logits[:, -1, :]             # last position

        # Top-k filtering
        top_values = topk(next_logits, k=5)
        next_logits[next_logits < top_values[-1]] = -inf

        # Sampling
        probs = softmax(next_logits / temperature)
        next_token = multinomial(probs)

        current = concat(current, next_token)

        if next_token == EOS:  # token 1024
            break

    return current[len(prompt_tokens):]  # strip prompt
```

### 9.3.6 KV-Cache Acceleration (Production Only)

The production GPT-SoVITS uses **KV-Cache** for efficient inference:

```
Without KV-Cache:
  step 1: process [text, token_0]                    -> predict token_1
  step 2: process [text, token_0, token_1]            -> predict token_2
  step 3: process [text, token_0, token_1, token_2]   -> predict token_3
  Complexity: O(T^2) per step, O(T^3) total

With KV-Cache:
  step 0: process [text, prompt_tokens]  -> cache K,V
  step 1: process [new_token] + cached K,V -> predict, update cache
  step 2: process [new_token] + cached K,V -> predict, update cache
  Complexity: O(T) per step, O(T^2) total
```

Our teaching version omits KV-Cache for clarity (at the cost of slower inference).

---

## 9.4 SoVITS Vocoder: VITS Variant

SoVITS is a modified VITS with three key differences:
- **No DurationPredictor** (the AR model implicitly handles duration)
- **Semantic tokens as input** (instead of raw phonemes only)
- **ReferenceEncoder** for speaker embedding (enables zero-shot cloning)

### 9.4.1 Overall Architecture

```
  SoVITS Vocoder Architecture (Training)
  ========================================

  phoneme_ids --> TextEncoder --> (mu_p, log_sigma_p)   Prior distribution
       ^                              |
  speaker_emb ------------------------+  (speaker conditioning)
       ^                                | KL divergence
  ref_mel --> ReferenceEncoder --> speaker_emb
                                        |
  spec ------> PosteriorEncoder --> (mu_q, log_sigma_q)  Posterior distribution
                                        |
                                  z_q = mu_q + eps * sigma_q  (reparameterization)
                                        |
                                  Flow(z_q) -> z_p  (normalizing flow)
                                        |
                                  Generator(z_q) -> y_hat  (HiFi-GAN)
                                        |
                                  Discriminator(y, y_hat)  (adversarial training)
```

### 9.4.2 TextEncoder (Simplified, No MRTE)

The original GPT-SoVITS TextEncoder has three sub-encoders:

```
Original (production):
  SSL features -> SSL Encoder (3 layers) -> content
  phonemes     -> Text Encoder (6 layers) -> text
  ref_audio    -> Reference Encoder       -> speaker_emb (ge)

  MRTE (Multi-Reference Timbre Encoder):
    content x text (cross-attention) + ge -> fused
  Final Encoder (3 layers) -> (mu_p, log_sigma_p)
```

**MRTE** is the most sophisticated component -- it uses cross-attention to align
acoustic content with linguistic text, while the speaker embedding injects timbre
additively. This is the core mechanism for zero-shot voice cloning.

Our simplified version merges all three sub-encoders into one:

```
Simplified (teaching):
  phonemes -> Embedding -> + speaker_proj(speaker_emb)
                        |
                  6-layer Transformer Encoder
                        |
                  Linear -> (mu_p, log_sigma_p)
```

### 9.4.3 PosteriorEncoder (WaveNet Dilated Convolutions)

```
  Linear Spectrogram (B, 1025, T)
      |  Conv1d(1025, 192)
      |  8x WaveNet Block (dilations: 1,2,4,8,1,2,4,8)
      |    each: Conv1d(192, 384) -> split -> tanh x sigmoid
      |  Conv1d(192, 384) -> (mu_q, log_sigma_q)
      |  Reparameterize: z_q = mu_q + eps * sigma_q
  z_q (B, 192, T)
```

Only used during training. At inference time, we sample from the prior.

### 9.4.4 Flow (Normalizing Flow, Affine Coupling)

```
  2 x (AffineCouplingLayer + Flip)

  Each AffineCouplingLayer:
    x = [x1, x2]  (channel split)
    s, t = WaveNet(x1)
    z2 = (x2 - t) * exp(-s)
    z = [x1, z2]
```

Training: z_q -> z_p (posterior -> prior space)
Inference: z_p -> z_q (prior sample -> decoder space)

### 9.4.5 Generator (HiFi-GAN)

```
  z (B, 192, T)
      |  Conv1d(192, 256, k=7)
      |  5 x [ConvTranspose1d (upsample) + 3x ResBlock1]
      |    upsample_rates: [10, 8, 2, 2, 2]
      |    total upsampling: 10 x 8 x 2 x 2 x 2 = 640
      |  Conv1d -> tanh
  waveform (B, 1, T*640)
```

### 9.4.6 ReferenceEncoder

```
  ref_mel (B, 128, T)
      |  4x Conv2d (stride=2, progressive downsampling)
      |  Reshape (B, T', C*F)
      |  Linear -> GRU -> Linear
  speaker_emb (B, 256)
```

### 9.4.7 Discriminator (Multi-Period Discriminator)

```
  Multi-Period Discriminator (MPD)
  periods = [2, 3, 5, 7, 11]

  Each sub-discriminator:
    waveform (1D) -> reshape to 2D (period x subsequence)
    -> 4x Conv2d -> LeakyReLU
    -> Conv2d -> real/fake decision
```

Each sub-discriminator sees different periodic structures in the waveform
(prime-numbered periods ensure no overlap in the periodicities captured).

---

## 9.5 Two-Stage Training

### 9.5.1 Why Two Stages?

The two models solve fundamentally different problems:

| | AR Model | SoVITS Vocoder |
|---|----------|---------------|
| **Input** | Text + history tokens | Text + reference audio |
| **Output** | Semantic tokens | Waveform |
| **Loss** | CrossEntropy | GAN (mel + KL + adversarial + feature) |
| **Training** | Standard autoregressive | GAN adversarial training |
| **Frame rate** | 50 Hz | 50 Hz -> 32 kHz |

Joint training would be complex and the two stages' gradients would interfere.

### 9.5.2 Stage 1: AR Model Training

```
  Stage 1 Training Pipeline
  ==========================

  Data Preparation:
    1. Extract HuBERT SSL features for all audio
    2. RVQ quantization -> semantic_ids ("semantic labels" for each clip)
    3. G2P conversion: text -> phoneme_ids

  Training Loop:
    Input:  phoneme_ids + semantic_ids[:-1]  (teacher forcing)
    Target: semantic_ids[1:]                  (right-shifted by 1)

    Loss = CrossEntropy(logits, targets, reduction="sum")

  Optimizer: AdamW (simplified; original uses ScaledAdam from k2)
    - betas: (0.9, 0.95)
    - warmup: 2000 steps
    - cosine decay to 0.0001
    - gradient clipping: 1.0

  ScaledAdam (production):
    - Parameter-aware learning rate scaling
    - Peak LR: 0.01, init LR: 0.00001
    - Warmup: 2000 steps, total decay: 40000 steps

  Gradient Accumulation:
    - Batch size: 8 (per GPU)
    - Accumulation steps: 4
    - Effective batch size: 32
    - This allows training on GPUs with limited memory while maintaining
      stable gradient estimates
```

### 9.5.3 Stage 2: SoVITS Vocoder Training

```
  Stage 2 Training Pipeline (GAN Training)
  ==========================================

  Each batch:
    1. Generator forward:
       ref_mel -> ReferenceEncoder -> speaker_emb
       phonemes + speaker_emb -> TextEncoder -> (mu_p, log_sigma_p)
       spec -> PosteriorEncoder -> z_q, (mu_q, log_sigma_q)
       Flow(z_q) -> z_p  (posterior -> prior)
       Generator(z_q) -> y_hat  (synthesize waveform)

    2. Discriminator step:
       L_D = MSE(D(real), 1) + MSE(D(fake.detach()), 0)

    3. Generator step:
       L_G = gen_loss + feat_loss + 45*mel_loss + 1*kl_loss

  Optimizer: AdamW x 2 (one for G, one for D)
    - betas: (0.8, 0.99)
    - lr: 1e-4
    - Exponential decay: 0.999875 per epoch

  Differential Learning Rates (production):
    - Text embedding, encoder_text, MRTE: lr * 0.4  (slower updates)
    - All other parameters: lr  (base rate)
    - Rationale: text-related parameters converge faster and need
      smaller updates to avoid catastrophic forgetting
```

### 9.5.4 Loss Function Weights

```
L_gen = L_adv_G + L_feat + 45 * L_mel + 1 * L_KL
```

| Loss | Weight | Meaning |
|------|--------|---------|
| L_mel | 45 | Mel-spectrogram reconstruction (most important) |
| L_KL | 1 | KL divergence regularization (prior-posterior match) |
| L_adv_G | 1 | Adversarial loss (improves naturalness) |
| L_feat | 1 | Feature matching (stabilizes training) |

---

## 9.6 Inference Pipeline

### 9.6.1 Complete Inference Steps

```python
# Pseudocode: GPT-SoVITS inference

def synthesize(text, ref_audio_path):
    # 1. Preprocessing
    phoneme_ids = g2p(text)               # text -> phoneme IDs
    ref_wav = load_audio(ref_audio_path)   # load reference audio

    # 2. Extract reference features
    ref_mel = mel_spectrogram(ref_wav)
    speaker_emb = ref_encoder(ref_mel)     # speaker embedding

    # 3. Extract prompt tokens
    ssl_feat = hubert(ref_wav)             # HuBERT features
    prompt_tokens = rvq.encode(ssl_feat)   # quantize to semantic IDs

    # 4. AR generation
    generated_tokens = ar_model.generate(
        phoneme_ids, prompt_tokens,
        top_k=5, temperature=1.0,
    )

    # 5. SoVITS vocoder
    m_p, logs_p = text_encoder(phoneme_ids, speaker_emb)
    z_p = m_p + randn * exp(logs_p) * 0.667   # sample from prior
    z = flow.inverse(z_p)                       # inverse transform
    waveform = generator(z)                     # generate waveform

    return waveform
```

### 9.6.2 Key Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| top_k | 5 | AR sampling width. Lower = more deterministic, higher = more diverse |
| temperature | 1.0 | Sampling temperature. >1 = more random, <1 = more conservative |
| noise_scale | 0.667 | VAE sampling noise. Controls variation in generated waveform |
| max_tokens | 500 | Max generated tokens (50Hz x 10s) |
| repetition_penalty | 1.35 | Penalizes repeated tokens (production only) |

### 9.6.3 Real-Time Factor (RTF)

```
RTF = generation_time / audio_duration

RTF < 1:  Faster than real-time (ideal goal)
RTF = 1:  Real-time
RTF > 1:  Slower than real-time
```

GPT-SoVITS RTF is dominated by the AR model (autoregressive is the bottleneck):

| Component | Time Share | Optimization |
|-----------|-----------|-------------|
| AR autoregressive | ~80% | KV-Cache, parallel decoding |
| SoVITS vocoder | ~20% | Single forward pass, already fast |

---

## 9.7 Code Walkthrough

### 9.7.1 File Structure

```
ch09_gpt_sovits/
|-- README.md              # This tutorial
|-- code/
|   |-- model.py           # All model components + loss functions (1281 lines)
|   |-- train.py           # Two-stage training script
|   |-- inference.py       # End-to-end inference
|   +-- export_onnx.py     # ONNX export
|-- checkpoints/           # Saved model weights
|-- outputs/               # Generated audio
+-- onnx_models/           # Exported ONNX models
```

### 9.7.2 model.py Core Classes

```python
# --- Stage 1: AR Model ---
class SimpleAR:
    """384-dim, 8-layer, 8-head Transformer decoder (~15.2M params)"""
    def forward(phoneme_ids, semantic_ids) -> logits
    def generate(phoneme_ids, prompt, top_k, temperature) -> tokens

# --- RVQ Quantizer ---
class SimpleRVQ:
    """Single-codebook quantizer (n_q=1, bins=1024, dim=768) (~0.8M params)"""
    def encode(ssl_features) -> token_ids
    def decode(token_ids) -> quantised_features

# --- Stage 2: SoVITS Vocoder Components ---
class SoVITSTextEncoder:       # text -> (mu_p, log_sigma_p)     (~2.9M)
class PosteriorEncoder:        # spectrogram -> z_q (train only) (~3.2M)
class Flow:                    # invertible transform z_q <-> z_p (~1.3M)
class Generator:               # z -> waveform (HiFi-GAN)        (~3.8M)
class ReferenceEncoder:        # ref_mel -> speaker_emb          (~0.3M)
class Discriminator:           # MPD discriminator (train only)  (~4.2M)

# --- Combined Model ---
class GPTSoVITS:
    """Wrapper: AR + RVQ + SoVITS (~31.6M total)"""
    def forward_stage1(phoneme_ids, semantic_ids) -> logits
    def forward_stage2(phoneme_ids, spec, ref_mel) -> waveform, stats
```

### 9.7.3 Quick Start: Shape Verification

Run the built-in shape tests to verify all components:

```bash
cd chapters/ch09_gpt_sovits/code
python model.py
```

Expected output:

```
============================================================
GPT-SoVITS Shape Verification Tests
============================================================

--- SimpleAR ---
  phoneme_ids:  torch.Size([2, 10])
  semantic_ids: torch.Size([2, 50])
  logits:       torch.Size([2, 50, 1025])
  params: 15,2xx,xxx (15.2M)

--- SimpleRVQ ---
  ssl_feat:  torch.Size([2, 768, 50])
  ids:       torch.Size([2, 50])
  decoded:   torch.Size([2, 768, 50])
  params: 786,432 (0.8M)

--- GPTSoVITS (full model) ---
  Stage 1 logits: torch.Size([2, 50, 1025])
  Stage 2 waveform: torch.Size([2, 1, 32000])
  Total params:     31,6xx,xxx (31.6M)
  Trainable params: 30,8xx,xxx (30.8M)

============================================================
All shape verification tests passed!
============================================================
```

### 9.7.4 Code Example: AR Generation

```python
from model import SimpleAR

# Create AR model
ar = SimpleAR(dim=384, n_heads=8, n_layers=8)

# Prepare inputs
phoneme_ids = torch.randint(0, 512, (1, 15))   # 15 phoneme tokens
prompt_tokens = torch.randint(0, 1024, (1, 25)) # 0.5s of reference audio

# Generate semantic tokens
generated = ar.generate(
    phoneme_ids, prompt_tokens,
    max_new_tokens=100,  # up to 2 seconds
    top_k=5,
    temperature=1.0,
)
print(f"Generated {generated.shape[1]} semantic tokens")
```

### 9.7.5 Code Example: Full Pipeline

```python
from model import GPTSoVITS

model = GPTSoVITS()

# Stage 1: AR training step
logits = model.forward_stage1(phoneme_ids, semantic_ids)
loss = F.cross_entropy(logits.reshape(-1, 1025), semantic_ids.reshape(-1))

# Stage 2: Vocoder training step
y_hat, m_p, logs_p, m_q, logs_q = model.forward_stage2(
    phoneme_ids, spectrogram, ref_mel
)
kl = kl_loss(m_p, logs_p, m_q, logs_q)
```

### 9.7.6 Parameter Breakdown

| Component | Parameters | Role |
|-----------|-----------|------|
| SimpleAR | ~15.2M | Phonemes -> semantic tokens |
| SimpleRVQ | ~0.8M (frozen) | SSL features -> discrete IDs |
| SoVITSTextEncoder | ~2.9M | Text -> prior distribution |
| PosteriorEncoder | ~3.2M | Spectrogram -> posterior (train only) |
| Flow | ~1.3M | Invertible z transform |
| Generator | ~3.8M | Latent -> waveform (HiFi-GAN) |
| ReferenceEncoder | ~0.3M | Reference mel -> speaker embedding |
| Discriminator | ~4.2M (train only) | Adversarial training |
| **Total** | **~31.6M** | |
| **Trainable** | **~30.8M** | (RVQ is frozen) |

---

## 9.8 Teaching Version vs Production GPT-SoVITS

### 9.8.1 Scale Comparison

```
  Production GPT-SoVITS (~70M)
  ============================

  AR Model (~40M):
    dim=512, layers=12, heads=16
    BERT features (1024-dim)
    KV-Cache inference
    DPO loss (optional)

  SoVITS (~30M):
    hidden_channels=192, filter_channels=768
    PosteriorEncoder: 16 layers
    Flow: 4 coupling layers
    Generator: upsample_initial=512
    MRTE (full cross-attention)

  External (frozen, not counted):
    HuBERT-base: ~95M
    RoBERTa-large: ~330M


  Teaching Version (~31.6M)
  ==========================

  SimpleAR (~15.2M):
    dim=384, layers=8, heads=8
    No BERT (learned positional only)
    Naive autoregressive inference
    CE loss only

  SoVITS (~16.4M):
    hidden_channels=192, filter_channels=768
    PosteriorEncoder: 8 layers
    Flow: 2 coupling layers
    Generator: upsample_initial=256
    Simplified TextEncoder (no MRTE)

  External:
    HuBERT: ~95M (still needed)
    BERT: Not used
```

### 9.8.2 Detailed Comparison Table

| Component | Production | Teaching | Rationale |
|-----------|-----------|----------|-----------|
| **AR dim** | 512 | 384 | -25% width, saves ~10M params |
| **AR layers** | 12-24 | 8 | Fewer layers, still captures long-range deps |
| **AR heads** | 16 | 8 | Proportional to dim reduction |
| **BERT features** | 1024-dim RoBERTa | Removed | Eliminates 330M external dependency |
| **Attention mask** | Hybrid (bi+causal) | Pure causal | Simpler implementation |
| **KV-Cache** | Yes | No | Clearer code, slower inference |
| **DPO loss** | Optional | No | Simpler training objective |
| **Optimizer** | ScaledAdam (k2) | AdamW | Standard PyTorch optimizer |
| **PosteriorEncoder** | 16 layers | 8 layers | -50% WaveNet depth |
| **Flow layers** | 4 | 2 | Fewer coupling layers |
| **Generator upsample** | 512 channels | 256 channels | Smaller HiFi-GAN |
| **MRTE** | Full cross-attention | Removed | Simplified text encoder |
| **Total params** | ~70M | ~31.6M | 55% reduction |

### 9.8.3 What We Sacrificed

| Aspect | Impact | Mitigation |
|--------|--------|-----------|
| Smaller AR | Lower capacity for complex prosody | Fine-tune longer on target data |
| No BERT | Less linguistic information | Phoneme-only is sufficient for most use cases |
| No KV-Cache | Slower inference (~O(T^2) vs O(T)) | Acceptable for teaching/demo |
| Simpler TextEncoder | No explicit content-text-timbre fusion | Speaker conditioning via addition works for single-speaker |
| Smaller Generator | Slightly lower audio quality | Acceptable for learning purposes |

### 9.8.4 What We Preserved

- **Core two-stage pipeline**: AR + VITS vocoder
- **Single-codebook RVQ (n_q=1)**: The key efficiency innovation
- **VAE + Flow + GAN training**: Full generative modeling pipeline
- **Multi-Period Discriminator**: Same adversarial training approach
- **Reference encoder**: Speaker embedding from reference audio
- **All loss functions**: CE, mel, KL, adversarial, feature matching

---

## 9.9 ONNX Export

### 9.9.1 Export Strategy

GPT-SoVITS splits into two ONNX models for deployment:

```
  ONNX Export Architecture
  =========================

  1. AR Model: ar_model.onnx
     Inputs:  phoneme_ids, semantic_ids
     Output:  logits
     Opset:   16

  2. SoVITS Vocoder: sovits_model.onnx
     Inputs:  phoneme_ids, speaker_emb
     Output:  waveform
     Opset:   17
     (Includes TextEncoder + Flow^{-1} + Generator)
     (Excludes PosteriorEncoder/Discriminator -- training only)
```

### 9.9.2 Dynamic Axes

All ONNX models support variable-length inputs:

```python
dynamic_axes = {
    'phoneme_ids':  {1: 'text_len'},   # variable text length
    'semantic_ids': {1: 'audio_len'},  # variable audio length
    'waveform':     {2: 'wav_len'},    # variable output length
}
```

### 9.9.3 ONNX Runtime Inference

```python
import onnxruntime as ort

ar_session = ort.InferenceSession('ar_model.onnx')
vits_session = ort.InferenceSession('sovits_model.onnx')

# AR inference (single forward pass per step)
logits = ar_session.run(None, {
    'phoneme_ids': phoneme_ids,
    'semantic_ids': current_tokens,
})

# Vocoder inference (single forward pass)
waveform = vits_session.run(None, {
    'phoneme_ids': phoneme_ids,
    'speaker_emb': speaker_emb,
})
```

ONNX Runtime is typically **1.5-2x faster** than PyTorch (via operator fusion and hardware optimization).

---

## 9.10 Training and Inference Commands

### Quick Start

```bash
cd chapters/ch09_gpt_sovits/code

# Verify model shapes
python model.py

# Train Stage 1 (AR model)
python train.py --stage 1 --epochs 10 --batch-size 4 --lr 1e-4

# Train Stage 2 (SoVITS vocoder)
python train.py --stage 2 --epochs 10 --batch-size 4 --lr 1e-4

# Inference (basic)
python inference.py --text "hello world" --output output.wav

# Inference with trained model + reference audio
python inference.py \
    --ar-checkpoint ../checkpoints/ar_model.pt \
    --sovits-checkpoint ../checkpoints/sovits_model.pt \
    --ref-audio reference.wav \
    --text "hello world" \
    --output clone.wav

# Benchmark inference speed
python inference.py --text "hello world" --output output.wav --benchmark

# Export to ONNX
python export_onnx.py --output-dir ../onnx_models

# Export with benchmark
python export_onnx.py --output-dir ../onnx_models --benchmark
```

---

## 9.11 Chapter Summary

### GPT-SoVITS's Core Contributions

| Problem | GPT-SoVITS's Solution |
|---------|----------------------|
| Multi-layer tokens make AR slow | Single-layer semantic token (n_q=1) |
| Codec decoder quality is limited | VITS vocoder (VAE + Flow + GAN) |
| Requires massive pretraining data | Pretrain + few-shot fine-tune (3-10 min) |
| Timbre and semantics are coupled | Semantic tokens + reference audio separation |
| High inference latency | 8x shorter sequences + KV-Cache |

### Open Problems

1. **Autoregressive bottleneck**: Even with 1-layer tokens, AR generation is sequential -> non-autoregressive alternatives
2. **HuBERT dependency**: Requires a pretrained HuBERT model (~95M) -> lightweight SSL models
3. **Long text quality degradation**: Longer generation = more accumulated error -> segmented synthesis + concatenation
4. **Limited emotion control**: Emotion is inherited from reference audio -> explicit emotion conditioning (Ch08)

---

## Exercises

1. **Information-theoretic analysis of n_q=1**: EnCodec uses n_q=8 codebooks (1024 entries each), total capacity = 1024^8. GPT-SoVITS uses n_q=1 (1024 entries), capacity = 1024. What does this mean? How does GPT-SoVITS compensate for the information loss?

2. **Attention mask design**: In the AR model, why does the text portion use bidirectional attention while audio uses causal attention? What happens if everything is causal? What if everything is bidirectional?

3. **KL loss role**: In Stage 2 training, KL loss weight is 1.0 while mel loss weight is 45.0. What happens if KL loss is removed? What if KL weight is raised to 45?

4. **Flow necessity**: What happens if you remove the Flow (sample directly from the prior and feed to the Generator)? What fundamental problem does the Flow solve?

5. **Comparative experiment**: Using the same text and reference audio, synthesize with Ch07 (VALL-E) and Ch09 (GPT-SoVITS). Compare:
   - Inference speed (RTF)
   - Speaker similarity
   - Naturalness (MOS)
   - Generated token sequence length

---

## References

1. Kim et al., 2021. *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech* (VITS). [arXiv:2106.06103](https://arxiv.org/abs/2106.06103)
2. Wang et al., 2023. *Neural Codec Language Models for Zero-Shot TTS* (VALL-E). [arXiv:2301.02111](https://arxiv.org/abs/2301.02111)
3. Hsu et al., 2021. *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*. [arXiv:2106.07447](https://arxiv.org/abs/2106.07447)
4. Defossez et al., 2022. *High Fidelity Neural Audio Compression* (EnCodec). [arXiv:2210.13438](https://arxiv.org/abs/2210.13438)
5. Kong et al., 2020. *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis*. [arXiv:2010.05646](https://arxiv.org/abs/2010.05646)
6. GPT-SoVITS open-source project. [github.com/RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
7. Dinh et al., 2016. *Density estimation using Real-NVP*. (Normalizing Flows). [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)

---

## Directory Structure

```
ch09_gpt_sovits/
|-- README.md              # This tutorial (you are here)
|-- code/
|   |-- model.py           # All model components (~1281 lines)
|   |-- train.py           # Two-stage training script
|   |-- inference.py       # End-to-end inference
|   +-- export_onnx.py     # ONNX export
|-- checkpoints/           # Saved model weights
|-- outputs/               # Generated audio
+-- onnx_models/           # Exported ONNX models
```
