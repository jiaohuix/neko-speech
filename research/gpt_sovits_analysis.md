# GPT-SoVITS Architecture Deep Analysis

> Research date: 2026-06-29
> Source: GPT-SoVITS codebase + VITS paper (Kim et al., 2021)

---

## 1. System Overview

GPT-SoVITS is a **two-stage zero-shot TTS system** combining:
- **Stage 1 (GPT/AR)**: Autoregressive Transformer that maps phonemes -> semantic tokens
- **Stage 2 (SoVITS)**: VITS-based vocoder that maps semantic tokens + text -> waveform

The key insight is using a **single-layer RVQ codebook** (n_q=1) as the "semantic token" interface between the two stages, making the AR problem much simpler than predicting full acoustic features.

```
                    GPT-SoVITS Inference Pipeline
                    =============================

  Reference Audio (3-10s)
        |
        v
  [Chinese HuBERT] -----> SSL features (768-dim, 50Hz)
        |                       |
        v                       v
  [RVQ Quantizer]        (extract prompt tokens)
  n_q=1, bins=1024              |
        |                       |
        v                       v
  prompt_semantic_ids     +-----------+
                          |  AR Model |  <--- Text Phonemes + BERT features
                          |  (GPT)    |
                          +-----------+
                                |
                                v
                        predicted_semantic_ids
                                |
                                v
                    +-----------------------+
                    |   SoVITS (VITS-based) |  <--- Text Phonemes + Ref Mel
                    +-----------------------+
                                |
                                v
                          Output Waveform
```

---

## 2. AR Model (Text-to-Semantic / GPT)

### 2.1 Architecture

The AR model is a **decoder-only Transformer** (GPT-style) that autoregressively predicts semantic token IDs from phoneme IDs + BERT features.

```
  AR Model Internal Architecture
  ==============================

  Input:
    phoneme_ids [B, T_text]     -----> ar_text_embedding (Embedding)
    bert_feature [B, 1024, T]   -----> bert_proj (Linear 1024->512) + transpose
                                        |
                                        v
                                   [sum with text embedding]
                                        |
                                        v
                                   ar_text_position (SinePositionalEmbedding)
                                        |
    semantic_ids [B, T_audio]   -----> ar_audio_embedding (Embedding)
                                        |
                                        v
                                   ar_audio_position (SinePositionalEmbedding)
                                        |
                                        v
                              [concat text_features + audio_features]
                                        |
                                        v
                            TransformerEncoder (bidirectional attn on text,
                             causal attn on audio, cross-attn text->audio)
                                        |
                                        v
                              ar_predict_layer (Linear, no bias)
                                        |
                                        v
                              logits [B, T_audio, vocab_size=1025]
```

### 2.2 Exact Parameter Configurations

Multiple config variants exist in the codebase:

| Parameter            | s1.yaml (base) | s1big.yaml | s1big2.yaml | s1longer.yaml | s1longer-v2.yaml |
|---------------------|---------------|------------|-------------|---------------|-----------------|
| embedding_dim       | 512           | 512        | 1024        | 512           | 512             |
| hidden_dim          | 512           | 512        | 1024        | 512           | 512             |
| num_head            | 16            | 16         | 16          | 16            | 16              |
| num_layers (n_layer)| 12            | 24         | 16          | 24            | 24              |
| vocab_size          | 1025          | 1025       | 1025        | 1025          | 1025            |
| phoneme_vocab_size  | 512           | 512        | 512         | 512           | 732             |
| dropout             | 0             | 0          | 0           | 0             | 0               |
| EOS token           | 1024          | 1024       | 1024        | 1024          | 1024            |
| num_codebook        | 8 (default)   | 8          | 8           | 8             | 8               |

**Default config** (from `t2s_model.py` hardcoded defaults):
- embedding_dim = 512
- hidden_dim = 512
- num_head = 8 (note: configs override to 16)
- num_layers = 12
- num_codebook = 8
- vocab_size = 1024 + 1 = 1025
- phoneme_vocab_size = 512
- EOS = 1024

**Key note**: `num_codebook=8` in the config is the RVQ depth, but the AR model only predicts the **first codebook** (n_q=1). The remaining 7 codebooks are handled by the SoVITS quantizer internally.

### 2.3 Key Components (from code analysis)

**Embeddings**:
- `ar_text_embedding`: TokenEmbedding(512, 512) - phoneme ID -> 512-dim
- `ar_audio_embedding`: TokenEmbedding(512, 1025) - semantic ID -> 512-dim
- `bert_proj`: Linear(1024, 512) - project BERT features to model dim
- `ar_text_position` / `ar_audio_position`: SinePositionalEmbedding(512)

**Transformer**:
- Standard PyTorch TransformerEncoder with custom attention masking
- Text portion: bidirectional self-attention (all-to-all)
- Audio portion: causal self-attention (upper triangular mask)
- Cross-attention: text can attend to audio, audio cannot attend to text (masked)
- FFN dimension: hidden_dim * 4 = 2048

**Output projection**:
- `ar_predict_layer`: Linear(512, 1025, bias=False)

**KV-Cache for inference** (optimized path):
- `T2STransformer` wraps `T2SBlock` list with `@torch.jit.script`
- `process_prompt()`: processes the full prompt in one pass, builds KV cache
- `decode_next_token()`: appends new token using cached K,V (no re-computation)
- Significantly faster than naive autoregressive loop

### 2.4 Training Configuration

From `s1_train.py` + `t2s_lightning_module.py`:

| Parameter           | Value                              |
|--------------------|------------------------------------|
| Optimizer          | ScaledAdam (from k2/fsa)           |
| Peak LR            | 0.01                               |
| Init LR            | 0.00001                            |
| End LR             | 0.0001                             |
| Warmup steps       | 2000                               |
| Total decay steps  | 40000                              |
| LR schedule        | WarmupCosineLRSchedule             |
| Batch size         | 8                                  |
| Gradient accum     | 4 (effective batch = 32)           |
| Epochs             | 20-300 (config-dependent)          |
| Precision          | 16-mixed (fp16)                    |
| Gradient clip      | 1.0                                |
| Betas              | (0.9, 0.95)                        |
| Loss               | CrossEntropyLoss(reduction="sum")  |
| Optional           | + DPO loss (beta=0.2)              |

### 2.5 Inference

- Top-k sampling with k=5 (default) or k=15 (v2 config)
- Temperature = 1.0
- Repetition penalty = 1.35
- Early stop at 50 Hz * max_sec (typically 54s = 2700 tokens)
- Minimum 11 tokens before EOS allowed (0.4s minimum)
- Batch inference with dynamic batch reduction (removes finished sequences)

---

## 3. SoVITS Model (VITS-based Vocoder)

### 3.1 Architecture

```
  SoVITS Model Architecture
  ==========================

  Inputs:
    semantic_codes [B, 1, T]  (from AR model, n_q=1)
    text_phonemes  [B, T_text]
    reference_mel  [B, C, T_ref]

  +---------------------+
  | quantizer.decode()  |  codes -> quantized [B, 768, T]
  +---------------------+
          |
          v  (interpolate 25Hz -> 50Hz if needed)
          |
  +---------------------+     +-------------------+
  |    TextEncoder       |     | ReferenceEncoder  |
  | (enc_p)              |     | (ref_enc)         |
  |                      |     |                   |
  | ssl_proj: 768->192  |     | Conv2d x6 layers  |
  | encoder_ssl: 3 lyr  |     | + GRU(128)        |
  | text_embedding       |     | + Linear->512     |
  | encoder_text: 6 lyr  |     |                   |
  | MRTE (cross-attn)    |<----| ge (speaker emb)  |
  | encoder2: 3 layers   |     +-------------------+
  | proj -> (m_p, logs_p)|
  +---------------------+
          |
          v  z_p = m_p + randn * exp(logs_p) * noise_scale
          |
  +---------------------+
  | ResidualCouplingBlock|  (flow, reverse=True)
  | 4 flows + Flip       |  z_p -> z
  +---------------------+
          |
          v
  +---------------------+
  | Generator (HiFi-GAN) |
  | conv_pre: 192->512   |
  | upsample x5 stages   |
  | resblocks x3 per up  |
  | conv_post -> tanh    |
  +---------------------+
          |
          v
     waveform [B, 1, T*hop]
```

### 3.2 Exact Parameter Configurations

From `s2.json`:

| Parameter                | Value                          |
|-------------------------|--------------------------------|
| **Audio**               |                                |
| sampling_rate           | 32000                          |
| filter_length (n_fft)   | 2048                           |
| hop_length              | 640                            |
| win_length              | 2048                           |
| n_mel_channels          | 128                            |
| segment_size            | 20480 samples                  |
| **TextEncoder (enc_p)** |                                |
| inter_channels          | 192                            |
| hidden_channels         | 192                            |
| filter_channels         | 768                            |
| n_heads                 | 2                              |
| n_layers                | 6 (split: 3 ssl + 6 text + 3)  |
| kernel_size             | 3                              |
| p_dropout               | 0.1                            |
| **RVQ Quantizer**       |                                |
| dimension               | 768                            |
| n_q                     | 1                              |
| bins (codebook size)    | 1024                           |
| freeze_quantizer        | True                           |
| semantic_frame_rate     | "25hz"                         |
| **ReferenceEncoder**    |                                |
| gin_channels            | 512                            |
| MelStyleEncoder input   | spec_channels (v1) or 704 (v2) |
| **PosteriorEncoder**    |                                |
| in_channels             | spec_channels (1025)           |
| out_channels            | 192 (inter_channels)           |
| hidden_channels         | 192                            |
| kernel_size             | 5                              |
| dilation_rate           | 1                              |
| n_layers                | 16                             |
| **Flow**                |                                |
| ResidualCouplingBlock   | 4 flows                        |
| channels                | 192                            |
| hidden_channels         | 192                            |
| kernel_size             | 5                              |
| dilation_rate           | 1                              |
| n_layers (WN)           | 4                              |
| **Generator (HiFi-GAN)**|                                |
| resblock                | "1" (ResBlock1)                |
| resblock_kernel_sizes   | [3, 7, 11]                     |
| resblock_dilation_sizes | [[1,3,5], [1,3,5], [1,3,5]]   |
| upsample_rates          | [10, 8, 2, 2, 2]              |
| upsample_initial_channel| 512                            |
| upsample_kernel_sizes   | [16, 16, 8, 2, 2]             |
| **Discriminator**       |                                |
| MultiPeriodDiscriminator| periods=[2,3,5,7,11] + ScaleD |

### 3.3 TextEncoder Internal Flow

The TextEncoder is the most interesting component -- it fuses three information sources:

1. **SSL features** (from HuBERT): 768-dim -> projected to 192 -> encoded by 3-layer Transformer
2. **Text phonemes**: embedded -> encoded by 6-layer Transformer
3. **Speaker embedding** (from ReferenceEncoder): global conditioning vector

These are fused by **MRTE (Multi-Reference Timbre Encoder)**:
- Cross-attention: SSL features (query) attend to text features (key/value)
- Add: SSL features + speaker embedding (ge)
- Result: content + timbre + linguistic information combined

Then a final 3-layer Transformer refines the fused representation before projecting to (m_p, logs_p) for the prior distribution.

### 3.4 Training Configuration

From `s2_train.py` + `s2.json`:

| Parameter           | Value                                    |
|--------------------|------------------------------------------|
| Optimizer          | AdamW                                     |
| Learning rate      | 0.0001                                    |
| Betas              | [0.8, 0.99]                               |
| Eps                | 1e-9                                      |
| LR decay           | 0.999875 (ExponentialLR per epoch)        |
| Batch size         | 32                                        |
| Epochs             | 100                                       |
| FP16               | True (GradScaler)                         |
| Loss weights       | c_mel=45, c_kl=1.0                       |
| Text low LR rate   | 0.4 (text layers at 40% base LR)          |
| Segment size       | 20480 samples (0.64s at 32kHz)            |

**Loss function** (Generator):
```
loss_gen_all = generator_loss + feature_loss + mel_loss*45 + commit_loss*1 + kl_loss*1
```

**Loss function** (Discriminator):
```
loss_disc_all = discriminator_loss(real, fake)
```

**Differential learning rates**:
- Text embedding, encoder_text, MRTE: `lr * 0.4` (slower updates for text-related params)
- All other params: `lr` (base learning rate)

---

## 4. RVQ Quantizer

### 4.1 Architecture

The RVQ uses only **n_q=1** (single codebook), which is the critical design choice:

| Parameter                  | Value   |
|---------------------------|---------|
| dimension                 | 768     |
| n_q (number of quantizers)| 1       |
| bins (codebook size)      | 1024    |
| decay (EMA)               | 0.99    |
| kmeans_init               | True    |
| kmeans_iters              | 50      |
| threshold_ema_dead_code   | 2       |

### 4.2 Why n_q=1?

Traditional audio tokenizers (EnCodec, DAC) use n_q=8 to capture both semantic and acoustic information across multiple codebook layers. GPT-SoVITS makes a different choice:

- **n_q=1**: Only the first (most semantic) codebook is used
- The AR model only needs to predict 1 token per frame instead of 8
- This makes the AR problem **8x simpler** in terms of sequence length
- Acoustic detail is handled by the SoVITS model through conditioning on reference audio

The quantizer is **frozen** during SoVITS training (`freeze_quantizer=True`), treating it as a fixed feature extractor.

---

## 5. Two-Stage Training Pipeline

### 5.1 Stage 1: AR Model Training (s1_train.py)

```
  Stage 1: Text-to-Semantic (GPT model)
  ======================================

  Data Preparation:
    1. Extract HuBERT features from all audio -> semantic_ids via RVQ
    2. Convert text to phoneme_ids via G2P
    3. Extract BERT features from text

  Training Loop:
    Input:  phoneme_ids + bert_feature + semantic_ids (teacher forcing)
    Target: semantic_ids (shifted by 1, with EOS padding)

    Loss: CrossEntropy(logits, targets, reduction="sum")
          + optional DPO loss

  Key Details:
    - Uses PyTorch Lightning with manual optimization
    - ScaledAdam optimizer (parameter-aware scaling)
    - Warmup + Cosine decay schedule
    - Gradient accumulation over 4 steps
    - No validation step (limit_val_batches=0)
    - Saves half-precision weights for inference
```

### 5.2 Stage 2: SoVITS Training (s2_train.py)

```
  Stage 2: SoVITS Vocoder Training
  =================================

  Data Preparation:
    1. SSL features from HuBERT (768-dim, 50Hz)
    2. Linear spectrograms (1025-dim)
    3. Raw audio waveforms
    4. Phoneme sequences from text

  Training Loop (GAN training):
    Generator forward:
      ssl -> ssl_proj -> quantizer -> TextEncoder -> prior (m_p, logs_p)
      spec -> PosteriorEncoder -> posterior (m_q, logs_q)
      z = m_q + randn * exp(logs_q)  [reparameterization]
      z_p = flow(z)                   [flow matching]
      z_slice = random_segment(z)
      y_hat = Generator(z_slice)      [HiFi-GAN decoder]

    Discriminator:
      MultiPeriodDiscriminator(real_audio, fake_audio)

    Losses:
      D: discriminator_loss
      G: gen_loss + feature_loss + mel_loss*45 + kl_loss + commit_loss

  Key Details:
    - DDP training with distributed bucket sampler
    - Bucket sampler groups similar-length samples for efficiency
    - Differential LR: text layers at 0.4x base rate
    - Pretrained model loading (both G and D)
    - FP16 mixed precision with GradScaler
```

### 5.3 Fine-tuning Workflow

For a new speaker, the workflow is:
1. Prepare 3-10 minutes of reference audio
2. Run data preprocessing (HuBERT extraction, G2P, BERT features)
3. Fine-tune Stage 1 (AR model) from pretrained weights: ~20 epochs
4. Fine-tune Stage 2 (SoVITS) from pretrained weights: ~30 epochs
5. Result: a voice clone that can generate speech from any text

---

## 6. ONNX Export Approach

### 6.1 Export Strategy

The ONNX export splits the system into **5 separate models** for deployment:

```
  ONNX Export Architecture
  =========================

  1. SSL Model (external): Chinese HuBERT -> ssl_content [1, 768, T]
     (not exported by GPT-SoVITS, loaded separately)

  2. T2S Encoder:  {project}_t2s_encoder.onnx
     Inputs:  ref_seq, text_seq, ref_bert, text_bert, ssl_content
     Outputs: x (encoded text), prompts (reference semantic tokens)
     Opset:   16

  3. T2S First Stage Decoder:  {project}_t2s_fsdec.onnx
     Inputs:  x, prompts
     Outputs: y, k_cache, v_cache, y_emb, x_example
     Opset:   16
     Purpose: Process the full prompt, build initial KV cache

  4. T2S Stage Decoder:  {project}_t2s_sdec.onnx
     Inputs:  iy, ik, iv, iy_emb, ix_example
     Outputs: y, k_cache, v_cache, y_emb, logits, samples
     Opset:   16
     Purpose: Autoregressive loop, one token at a time

  5. VITS Model:  {project}_vits.onnx
     Inputs:  text_seq, pred_semantic, ref_audio
     Outputs: audio (waveform)
     Opset:   17
```

### 6.2 Key Export Details

- **Dynamic axes**: All models support variable-length inputs via `dynamic_axes` parameter
- **KV-cache exposed**: The autoregressive decoder explicitly passes K,V caches as ONNX I/O tensors
- **TorchScript bridge**: `T2SBlock` and `T2STransformer` use `@torch.jit.script` for ONNX compatibility
- **No PosteriorEncoder in ONNX**: The exported VITS model removes `enc_q` (only needed for training)
- **Opset versions**: 16 for T2S models, 17 for VITS (needs newer ONNX ops)
- **MoeVS config**: Exports a JSON config for the MoeVS inference engine

### 6.3 Inference Runtime (ONNX)

```
  ONNX Runtime Inference:
  1. SSL: HuBERT(ref_audio_16k) -> ssl_content
  2. T2S Encoder: encode(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
  3. T2S First Stage: process_prompt(x, prompts) -> initial state + KV cache
  4. T2S Stage Decoder: loop { decode_next_token(state) } until EOS
  5. VITS: synthesize(text_seq, pred_semantic, ref_audio) -> waveform
```

---

## 7. Parameter Count Estimation

### 7.1 AR Model (default: 512-dim, 12 layers)

| Component              | Approx Params |
|-----------------------|---------------|
| ar_text_embedding      | 512 * 512 = 262K |
| ar_audio_embedding     | 512 * 1025 = 525K |
| bert_proj              | 1024 * 512 + 512 = 525K |
| Positional embeddings  | ~negligible (sine) |
| Transformer (12 layers)| 12 * (4*512^2 + 2*512*2048 + norms) ~ 38M |
| ar_predict_layer       | 512 * 1025 = 525K |
| **Total AR**           | **~40M** |

### 7.2 SoVITS Model

| Component              | Approx Params |
|-----------------------|---------------|
| TextEncoder (enc_p)    | ~7M (3+6+3 transformer layers + MRTE) |
| PosteriorEncoder       | ~4M (16-layer WaveNet) |
| ResidualCouplingBlock  | ~2M (4 flows) |
| Generator (HiFi-GAN)   | ~14M (5 upsample stages + resblocks) |
| ReferenceEncoder       | ~2M (Conv2d stack + GRU) |
| RVQ Quantizer          | ~0.8M (768*1024 embedding) |
| Discriminator (train)  | ~6M (not needed for inference) |
| **Total SoVITS (infer)**| **~30M** |

### 7.3 Full System

| Component | Params |
|-----------|--------|
| AR Model  | ~40M   |
| SoVITS    | ~30M   |
| **Total** | **~70M** |
| External: HuBERT | ~95M (frozen, not trained) |
| External: BERT   | ~330M (frozen, not trained) |

---

## 8. Key Design Patterns and Innovations

### 8.1 MRTE (Multi-Reference Timbre Encoder)

The MRTE is the core mechanism for zero-shot voice cloning:

```
  MRTE Architecture:
    content = ssl_features [B, 192, T]     (from HuBERT, encoded)
    text    = text_features [B, 192, T']   (from phoneme encoder)
    ge      = speaker_emb [B, 512, 1]      (from reference encoder)

    c_pre: 192 -> 512
    text_pre: 192 -> 512
    cross_attention: content(Q) x text(K,V) -> fused [B, 512, T]
    output = fused + content + ge
    c_post: 512 -> 192
```

Key insight: The cross-attention aligns acoustic content with linguistic text, while the speaker embedding `ge` injects timbre information additively.

### 8.2 BERT Feature Conditioning

The AR model uses BERT features as additional text conditioning:
- BERT (Chinese RoBERTa-wwm-ext-large) extracts 1024-dim features per token
- `bert_proj`: Linear(1024, embedding_dim) projects to model dimension
- Added (not concatenated) to the text embedding before positional encoding
- Provides prosodic and linguistic information beyond phoneme identity

### 8.3 Attention Mask Design (AR Model)

The attention mask is crucial for the GPT-style training:

```
  Attention Mask Layout (for concatenated [text, audio] sequence):

             text    audio
  text   [  all     all-to  ]   <- text attends to all text (bidirectional)
          -to-text   -none      <- text does NOT attend to audio
         [                  ]
  audio  [  all     causal  ]   <- audio attends to all text
          -to-text  (upper     <- audio attends causally to past audio
                    triangular)
```

### 8.4 VITS Adaptations from Original Paper

Compared to the original VITS (Kim et al., 2021):
1. **No duration predictor**: Replaced by the AR model which implicitly handles alignment
2. **Semantic tokens as input**: Instead of raw phonemes, uses HuBERT-derived semantic codes
3. **MRTE instead of simple conditioning**: More sophisticated speaker/content fusion
4. **Frozen quantizer**: The RVQ codebook is pretrained and frozen
5. **Speed control**: Optional interpolation in TextEncoder for variable speaking rate

---

## 9. Suggested Simplification for Teaching Version (~50M params)

### 9.1 Target: Reduced AR Model + Same SoVITS

| Change | Original | Teaching | Params Saved |
|--------|----------|----------|-------------|
| AR hidden_dim | 512 | 384 | ~10M |
| AR num_layers | 12 | 8 | ~13M |
| AR num_heads | 16 | 8 | 0 (adjustment) |
| BERT features | 1024-dim | Remove or distill | ~0.5M |
| DPO loss | Yes | No | 0 (simplification) |
| SoVITS | Same | Same | 0 |

Result: AR ~17M + SoVITS ~30M = ~47M total

### 9.2 Alternative: Keep AR, Simplify SoVITS

| Change | Original | Teaching | Params Saved |
|--------|----------|----------|-------------|
| SoVITS hidden_channels | 192 | 128 | ~8M |
| SoVITS filter_channels | 768 | 512 | ~4M |
| Flow layers | 4 | 2 | ~1M |
| Generator upsample_initial | 512 | 256 | ~8M |
| PosteriorEncoder layers | 16 | 8 | ~2M |

Result: AR ~40M + SoVITS ~7M = ~47M total (but quality drops significantly)

### 9.3 Recommended Approach for neko-speech

**Best balance of quality and teachability:**

1. **AR Model**: 384-dim, 8-layer, 8-head Transformer (~17M)
   - Remove BERT dependency (use learned positional instead)
   - Keep KV-cache inference
   - Remove DPO loss
   - Keep ScaledAdam or switch to AdamW for simplicity

2. **SoVITS Model**: Keep close to original but with modest reductions
   - hidden_channels=192, filter_channels=768 (keep)
   - Reduce flow layers from 4 to 2 (-1M)
   - Reduce PosteriorEncoder from 16 to 12 layers (-1M)
   - Keep HiFi-GAN generator as-is (critical for quality)

3. **External Models**: Replace with lighter alternatives
   - HuBERT-base (95M) -> Wav2Vec2-small or distil-HuBERT (~20M)
   - RoBERTa-large (330M) -> Remove entirely, use phoneme-only

**Estimated teaching version: ~50M trainable params + ~20M frozen SSL model**

### 9.4 Code Simplification Priorities

1. Remove KV-cache complexity for initial teaching (naive autoregressive loop is clearer)
2. Remove DPO training path
3. Remove batch inference optimizations
4. Replace ScaledAdam with standard AdamW
5. Simplify attention mask construction (use standard causal mask)
6. Remove v1/v2 version branching
7. Replace MRTE with simpler cross-attention for teaching clarity
8. Combine the 3-sub-encoder TextEncoder into a single encoder for teaching

---

## 10. Summary Table

| Aspect | AR (GPT) | SoVITS (VITS) |
|--------|----------|---------------|
| Role | Phonemes -> Semantic tokens | Semantic tokens -> Waveform |
| Architecture | Decoder-only Transformer | VAE + Normalizing Flow + HiFi-GAN |
| Key Input | phoneme_ids + BERT + prompt | semantic_codes + text + ref_mel |
| Key Output | predicted semantic_ids | audio waveform |
| Params | ~40M | ~30M |
| Training | CE loss, ScaledAdam | GAN loss, AdamW |
| Sampling Rate | 50 Hz (token rate) | 32 kHz (audio) |
| Inference | Autoregressive w/ KV-cache | Single forward pass |
| Pretrained dep. | HuBERT, BERT | HuBERT (frozen quantizer) |
