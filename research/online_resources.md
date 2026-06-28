# TTS Model Online Resources Research

> Research Date: 2026-06-29
> Scope: Classical to SOTA TTS models, neural audio codecs, flow matching, voice cloning, and omni models

---

## Table of Contents

1. [Parameter Comparison Table](#1-parameter-comparison-table)
2. [Training Recipes Summary](#2-training-recipes-summary)
3. [Common Implementation Patterns](#3-common-implementation-patterns)
4. [Key Insights from Community](#4-key-insights-from-community)
5. [MiniMind-O Analysis](#5-minimind-o-analysis)
6. [Neural Audio Codecs](#6-neural-audio-codecs)
7. [Flow Matching for Speech](#7-flow-matching-for-speech)
8. [Voice Cloning Techniques](#8-voice-cloning-techniques)
9. [Recommended Simplification Approaches](#9-recommended-simplification-approaches)
10. [Chinese Community Resources](#10-chinese-community-resources)
11. [References](#11-references)

---

## 1. Parameter Comparison Table

| Model | Year | Type | Params | Hidden Dim | Layers | Heads | FFN Dim | Vocoder | Inference |
|-------|------|------|--------|------------|--------|-------|---------|---------|-----------|
| **Tacotron 2** | 2017 | AR Seq2Seq | ~28M | 1024 (LSTM) | 3 conv + 2 LSTM (enc), 2 LSTM (dec) | - | - | WaveGlow/WaveNet | ~30x RTF |
| **FastSpeech 2** | 2020 | Non-AR FF | ~25M | 256 | 4 (enc) + 4 (dec) | 2 | 1024 | HiFi-GAN (ext.) | ~100x RTF |
| **VITS** | 2021 | E2E VAE+Flow+GAN | ~83M | 192 | 6 (text enc), 4 (posterior enc) | 2 | 768 | Built-in HiFi-GAN | ~67x RTF |
| **VALL-E** | 2023 | AR+NAR Codec LM | ~300M | 1024 | 12 (AR) + 12 (NAR) | 16 | 4096 | EnCodec decoder | ~20x RTF |
| **F5-TTS** | 2024 | Non-AR Flow Match | ~300M | 1024 (DiT) | 22 (DiT), 4 (text enc) | 16 | 2048 | BigVGAN | ~15x RTF |
| **CosyVoice 2** | 2024 | LLM + Flow Match | ~500M | - | - | - | - | Chunk-aware FM | ~150ms latency |
| **IndexTTS** | 2025 | AR GPT-style | ~300M+ | - | - | - | - | BigVGAN2 | Real-time |
| **MiniMind-O** | 2026 | Omni Thinker-Talker | ~115M | - | - | - | - | Streaming speech | CPU inference |

### Detailed Hyperparameter Breakdown

#### Tacotron 2 (NVIDIA Reference)
| Parameter | Value | Source |
|-----------|-------|--------|
| encoder_embedding_dim | 512 | hparams.py |
| decoder_rnn_dim | 1024 | hparams.py |
| prenet_dim | 256 | hparams.py |
| attention_rnn_dim | 1024 | hparams.py |
| attention_location_n_filters | 32 | hparams.py |
| attention_location_kernel_size | 31 | hparams.py |
| postnet_embedding_dim | 512 | hparams.py |
| postnet_conv_dim_1 | 512 | hparams.py |
| n_frames_per_step | 1 (reduced from 3) | hparams.py |
| sample_rate | 22050 | hparams.py |
| n_mel_channels | 80 | hparams.py |
| hop_length | 256 | hparams.py |
| win_length | 1024 | hparams.py |

#### FastSpeech 2 (ming024 Implementation)
| Parameter | Value | Source |
|-----------|-------|--------|
| encoder_hidden | 256 | model.yaml |
| encoder_layers | 4 | model.yaml |
| encoder_heads | 2 | model.yaml |
| decoder_hidden | 256 | model.yaml |
| decoder_layers | 4 | model.yaml |
| decoder_heads | 2 | model.yaml |
| conv_filter_size | 1024 | model.yaml |
| conv_kernel_size | [9, 1] | model.yaml |
| pitch_feature_level | phoneme_level | model.yaml |
| energy_feature_level | phoneme_level | model.yaml |
| variance_predictor_kernel | 3 | model.yaml |
| variance_predictor_layers | 2 | model.yaml |

#### VITS (jaywalnut310)
| Parameter | Value | Source |
|-----------|-------|--------|
| inter_channels | 192 | configs/ljs_base.json |
| hidden_channels | 192 | configs/ljs_base.json |
| filter_channels | 768 | configs/ljs_base.json |
| n_heads | 2 | configs/ljs_base.json |
| n_layers | 6 | configs/ljs_base.json |
| kernel_size | 3 | configs/ljs_base.json |
| p_dropout | 0.1 | configs/ljs_base.json |
| n_flow_layers | 4 | configs/ljs_base.json |
| resblock | "1" | configs/ljs_base.json |
| resblock_kernel_sizes | [3, 7, 11] | configs/ljs_base.json |
| resblock_dilation_sizes | [[1,3,5],[1,3,5],[1,3,5]] | configs/ljs_base.json |
| upsample_rates | [8, 8, 2, 2] | configs/ljs_base.json |
| upsample_initial_channel | 512 | configs/ljs_base.json |
| segment_size | 8192 | configs/ljs_base.json |

#### VALL-E
| Parameter | Value | Source |
|-----------|-------|--------|
| AR layers | 12 | Paper (2301.02111) |
| AR heads | 16 | Paper |
| AR embedding_dim | 1024 | Paper |
| AR ff_dim | 4096 | Paper |
| NAR layers | 12 | Paper |
| NAR heads | 16 | Paper |
| EnCodec codebooks | 8 | EnCodec paper |
| EnCodec codebook_size | 1024 | EnCodec paper |
| EnCodec vector_dim | 128 | EnCodec paper |
| Tokens/second | ~50 | Paper |

#### F5-TTS
| Parameter | Value | Source |
|-----------|-------|--------|
| DiT layers | 22 | Paper (2410.06885) |
| DiT heads | 16 | Paper |
| DiT embedding_dim | 1024 | Paper |
| DiT FFN dim | 2048 | Paper |
| Text encoder layers | 4 | Paper |
| Text encoder dim | 512 | Paper |
| Training data | 100K hours | Paper |
| Vocoder | BigVGAN | GitHub repo |

#### CosyVoice 2
| Parameter | Value | Source |
|-----------|-------|--------|
| Total params | ~500M | HuggingFace |
| Architecture | LLM + Chunk-aware FM | Paper (2412.10117) |
| Semantic tokens rate | 25Hz / 50Hz | AIModels.fyi |
| Token type | Supervised semantic + FSQ | datarootlabs.com |
| Streaming latency | ~150ms | GitHub repo |
| Languages | 5+ | Paper |

#### IndexTTS
| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | GPT-style AR + XTTS/Tortoise | Paper (2502.05512) |
| Conditioning encoder | Conformer-based | Paper |
| Vocoder | BigVGAN2 (24KHz) | Paper |
| Training data | ~34,000 hours | Paper |
| Token decomposition | Semantic + Acoustic (v2.5) | Paper (2601.03888) |
| Training stages | 3 (pretrain, SFT, alignment) | Paper |

---

## 2. Training Recipes Summary

### Tacotron 2 (NVIDIA)
| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| LR decay | Exponential, starting at 45k steps |
| Weight decay | 1e-6 |
| Batch size | 64 |
| Gradient clip | 1.0 |
| Epochs / Steps | ~1500 epochs / 500k steps |
| Mixed precision | Supported (AMP) |
| Data preprocessing | STFT -> mel-spectrogram, Griffin-Lim for monitoring |
| Teacher forcing | Used during training |
| Hardware | 8x V100 for full training |

### FastSpeech 2 (ming024)
| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| LR schedule | Warmup + decay |
| Batch size | 16 |
| Total steps | 300,000 |
| Data preprocessing | MFA alignment, pitch/energy extraction |
| Duration extraction | Montreal Forced Aligner (MFA) |
| Hardware | Single GPU (12GB+), ~4 days |

### VITS (jaywalnut310)
| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate (G) | 2e-4 |
| Learning rate (D) | 2e-4 |
| Warmup steps | 4000 |
| Batch size | 32-64 (effective) |
| Loss | KL + reconstruction + adversarial + duration |
| Alignment | Monotonic Alignment Search (MAS) |
| Segment size | 8192 samples |
| Hardware | Single GPU (24GB, e.g. 4090) |

### VALL-E
| Setting | Value |
|---------|-------|
| Training data | 60,000 hours (LibriLight) |
| AR training steps | 800,000 |
| NAR training steps | 560,000 |
| Batch size | 6,000 codec tokens |
| GPUs | 16 |
| Audio tokenizer | EnCodec (8 codebooks) |
| Text input | Phonemes (G2P) |

### F5-TTS
| Setting | Value |
|---------|-------|
| Framework | Conditional Flow Matching (CFM) |
| Training data | 100K hours (multilingual) |
| Optimizer | AdamW (typical) |
| Inference steps | 10-26 ODE steps (with Sway Sampling) |
| Vocoder | BigVGAN |

### CosyVoice 2
| Setting | Value |
|---------|-------|
| Architecture | LLM + chunk-aware causal flow matching |
| Training data | Large-scale multilingual |
| Semantic tokens | Supervised + FSQ |
| Sample rate | 25Hz / 50Hz token rate |
| Streaming | Chunk-aware, 150ms latency |

### IndexTTS
| Setting | Value |
|---------|-------|
| Training paradigm | 3-stage (Pretrain, SFT, Alignment) |
| Training data | ~34,000 hours |
| Input sequence | [conditioning, phonemes, ...] |
| Vocoder | BigVGAN2 |
| Zero-shot | Yes (3+ seconds reference) |

---

## 3. Common Implementation Patterns

### 3.1 Audio Representation Pipeline

```
Raw Audio (16-48kHz)
    |
    v
[STFT / Mel Filterbank]  <-- Classical TTS (Tacotron2, FastSpeech2)
    |
    v
Mel Spectrogram (80-dim, 22050Hz)
    |
    v
[Neural Codec]  <-- Modern TTS (VALL-E, CosyVoice, IndexTTS)
    |
    v
Discrete Tokens (8 codebooks, 50 tokens/sec)
```

### 3.2 Three Generations of TTS Architecture

| Generation | Pattern | Models | Alignment |
|------------|---------|--------|-----------|
| **1st: AR + Vocoder** | Seq2Seq -> mel -> vocoder | Tacotron 2 + WaveGlow | Location-sensitive attention |
| **2nd: Non-AR + Vocoder** | Duration -> mel -> vocoder | FastSpeech 2 + HiFi-GAN | External aligner (MFA) |
| **2.5: E2E** | VAE + Flow + GAN -> waveform | VITS | MAS (Monotonic Alignment Search) |
| **3rd: Codec LM** | LM on discrete tokens | VALL-E, CosyVoice, IndexTTS | No explicit alignment |
| **3.5: Flow Matching** | ODE on continuous features | F5-TTS, Matcha-TTS | No explicit alignment |

### 3.3 Shared Design Patterns

1. **Text Frontend**
   - All models use phoneme input (G2P conversion)
   - BERT-style text embeddings increasingly common (IndexTTS 2.5, CosyVoice)
   - Character-level fallback for out-of-vocabulary

2. **Speaker Conditioning**
   - Tacotron 2: Speaker embedding table
   - FastSpeech 2: Speaker embedding + variance adaptor
   - VITS: Speaker embedding (sid)
   - VALL-E/CosyVoice/IndexTTS: 3-5 second audio prompt (zero-shot)

3. **Vocoder Integration**
   - Separate: Tacotron 2 + WaveGlow, FastSpeech 2 + HiFi-GAN
   - Integrated: VITS (HiFi-GAN as decoder)
   - Codec-based: VALL-E (EnCodec decoder), IndexTTS (BigVGAN2)

4. **Loss Functions**
   | Model | Loss Components |
   |-------|-----------------|
   | Tacotron 2 | L1 mel loss + gate loss |
   | FastSpeech 2 | MSE mel + duration + pitch + energy |
   | VITS | KL + L1 + feature match + adv + dur |
   | VALL-E | Cross-entropy (token prediction) |
   | F5-TTS | Flow matching velocity loss |

---

## 4. Key Insights from Community

### 4.1 Attention Alignment Pitfalls (Tacotron 2)
- **Problem**: Attention "wanders" or jumps, causing garbled output
- **Solutions**:
  - Use **guided attention loss** to constrain to approximately monotonic
  - Apply **Double Decoder Consistency (DDC)** for robust, fast-converging alignment
  - **Remove silence** from audio preprocessing (beginning/end)
  - Start with `n_frames_per_step=3`, reduce to 1 later
  - Use location-sensitive attention with `n_filters=32`, `kernel_size=31`

### 4.2 Duration Predictor Challenges (FastSpeech 2)
- **Problem**: Duration predictor is the bottleneck of quality
- **Solutions**:
  - Use **Montreal Forced Aligner (MFA)** for ground truth durations
  - JETS: Jointly train alignment module, skip MFA dependency
  - Pitch/energy normalization critical for stable training
  - Variance adaptor hyperparameters sensitive to dataset

### 4.3 GAN Training Instability (VITS)
- **Problem**: Generator and discriminator out of balance
- **Solutions**:
  - Use **multi-period + multi-scale discriminators** (from HiFi-GAN)
  - **Feature matching loss** stabilizes training
  - Careful **learning rate ratio** (G and D should be similar)
  - **KL loss weight** (c_kl) needs tuning; too high = robotic, too low = unstable
  - MAS alignment may fail on noisy data; clean data essential

### 4.4 Codec Token Quality (VALL-E / CosyVoice / IndexTTS)
- **Problem**: EnCodec tokens may lack semantic content, causing "babbling"
- **Solutions**:
  - Use **supervised semantic tokens** (CosyVoice 2, IndexTTS 2.5)
  - Train on more data (>10K hours) for AR model to generalize
  - **Codec merging** (VALL-E R): merge codebook layers to reduce token rate
  - **FSQ (Finite Scalar Quantization)** as alternative to VQ (CosyVoice 2)

### 4.5 Flow Matching Training Tips
- **Problem**: ODE solver steps are slow at inference
- **Solutions**:
  - **Sway Sampling** (F5-TTS): non-uniform time steps, training-free speedup
  - **Consistency distillation**: reduce to 1-4 steps
  - **Classifier-free guidance** can be removed with proper training
  - 10-26 steps usually sufficient with good ODE solver (Euler/RK4)

### 4.6 Data Quality > Data Quantity
- Community consensus: **100 hours of clean data > 1000 hours of noisy data**
- Critical preprocessing steps:
  1. Audio normalization (-23dB to -26dB LUFS)
  2. Silence trimming (leading/trailing)
  3. Sample rate standardization (22050 or 24000 Hz)
  4. Remove clips with background noise, clipping, or reverberation
  5. Text normalization (numbers, abbreviations, punctuation)

---

## 5. MiniMind-O Analysis

### Overview
- **Repository**: [github.com/jingyaogong/minimind-o](https://github.com/jingyaogong/minimind-o)
- **Paper**: [arXiv:2605.03937](https://arxiv.org/html/2605.03937v1) (May 2026)
- **Author**: Jingyao Gong (jingyaogong)
- **Scale**: ~0.1B (115M) active parameters

### What Is MiniMind-O?
MiniMind-O is an **open-source, small-scale omni model** that processes **text, speech, and image** inputs and produces both **text and streaming speech** outputs. It is built on the MiniMind language model and represents one of the smallest complete omni model implementations available.

### Architecture: Thinker-Talker

```
         [Text] [Speech] [Image]
              |     |      |
              v     v      v
         +-------------------+
         |     Thinker       |   <-- Multimodal understanding
         |   (MiniMind LLM)  |   <-- Reasoning & text generation
         +-------------------+
              |
              v
         +-------------------+
         |     Talker        |   <-- Speech generation
         |  (Speech Decoder) |   <-- Streaming output
         +-------------------+
              |
              v
         [Text + Streaming Speech Output]
```

Key design decisions:
1. **Decoupled design**: Multimodal understanding (Thinker) and speech generation (Talker) are separate
2. **Speech-native**: Not bolted-on; speech is a first-class modality
3. **Streaming**: Talker produces speech incrementally
4. **Tiny**: Trainable on consumer GPUs, supports CPU-only inference

### Integration into Speech Textbook

| Chapter | Integration Point |
|---------|-------------------|
| Ch 2-3 (Fundamentals) | MiniMind-O's audio tokenizer as example of learned audio representation |
| Ch 5 (Codec LM) | Thinker-Talker as simplified VALL-E/CosyVoice architecture |
| Ch 7 (Omni Models) | Full case study: smallest omni model, reproducible training |
| Ch 8 (Projects) | Student project: fine-tune MiniMind-O on custom voice data |

### Why MiniMind-O Matters for the Book
1. **Pedagogical value**: 0.1B params = fully understandable, inspectable, trainable
2. **Complete pipeline**: Text -> speech -> text in one model (the "omni" story)
3. **Open source**: All code, data, and weights available
4. **Consumer-friendly**: Doesn't require A100 clusters
5. **Modern architecture**: Represents the latest thinking (Thinker-Talker paradigm)

---

## 6. Neural Audio Codecs

### EnCodec (Meta/Facebook Research)
- **Repository**: [github.com/facebookresearch/encodec](https://github.com/facebookresearch/encodec)
- **Architecture**: SEANet encoder + RVQ bottleneck + SEANet decoder
- **Models**: 24kHz (1.5-24kbps) and 48kHz (3-24kbps)
- **Key innovation**: Multi-bandwidth training (single model, variable bitrate)
- **Training**: Adversarial with multi-scale discriminators + reconstruction loss + commitment loss
- **Impact**: Foundation for VALL-E, MusicGen, AudioGen

### SoundStream (Google Research)
- **Paper**: End-to-End Neural Audio Codec (2021)
- **Architecture**: Convolutional encoder + RVQ + Convolutional decoder
- **Key innovation**: First neural codec for speech + music, real-time on smartphone
- **RVQ cascade**: Progressive quantization, coarse-to-fine
- **Impact**: Basis for Lyra v2, influenced EnCodec design

### RVQ (Residual Vector Quantization) Deep Dive
```
Input latent z
    |
    v
[VQ_1] --> q_1 (coarse, e.g., semantic content)
    |
    v
z - q_1 = residual_1
    |
    v
[VQ_2] --> q_2 (medium, e.g., prosody)
    |
    v
residual_1 - q_2 = residual_2
    |
    v
[VQ_3] --> q_3 (fine, e.g., acoustic detail)
    |
    ...
    v
Reconstruction = q_1 + q_2 + q_3 + ... + q_N
```

**Key insight for textbook**: The first few codebooks capture semantic content (language, words), while later codebooks capture acoustic detail (speaker identity, emotion, background). This is why VALL-E's AR model predicts only the first codebook and NAR predicts the rest.

### Modern Codec Variants
| Codec | Year | Innovation | Used By |
|-------|------|------------|---------|
| SoundStream | 2021 | First E2E neural codec | Google/Lyra v2 |
| EnCodec | 2022 | Multi-bandwidth, open source | VALL-E, MusicGen |
| RVQGAN | 2023 | Improved training (NeurIPS 2023) | SNAC, ImageBind |
| FSQ | 2024 | Finite Scalar Quantization | CosyVoice 2 |
| SpeechTokenizer | 2023 | Speech-optimized | Various research |
| SNAC | 2024 | Multi-scale codec | Research |

---

## 7. Flow Matching for Speech

### What Is Flow Matching?
Flow matching is a simulation-free training approach for continuous normalizing flows (CNFs). Instead of solving ODEs during training (expensive), it directly regresses on the **velocity field** that transports noise to data.

```
Training:
  v_t(x) = velocity field (what direction to move at time t)
  Loss = ||v_theta(x_t, t) - u_t(x_t)||^2

Inference:
  Start from noise x_0 ~ N(0, I)
  Integrate: dx/dt = v_theta(x_t, t) from t=0 to t=1
  Result: x_1 ~ data distribution
```

### Key Models Using Flow Matching
| Model | Application | DiT Backbone | Steps |
|-------|------------|--------------|-------|
| **Matcha-TTS** | Non-AR TTS | Lightweight | 10-50 |
| **F5-TTS** | Zero-shot TTS | 22-layer DiT | 10-26 |
| **E2 TTS** | Flat UNet variant | UNet | 10-50 |
| **CosyVoice 2** | Streaming TTS | Chunk-aware FM | Streaming |
| **SpeechFlow** | Pre-training | Meta AI (ICLR 2024) | Pre-train |

### Why Flow Matching > Diffusion for TTS
1. **Faster training**: No ODE solver needed during training
2. **Straight paths**: Optimal transport paths are straighter than diffusion paths
3. **Fewer steps**: 10-26 steps vs 50-1000 for diffusion
4. **Better quality**: Sharper, more natural output
5. **Unified framework**: Encompasses both diffusion and flow-based approaches

### Practical Implementation Pattern (F5-TTS style)
```python
# Simplified flow matching training loop
for batch in dataloader:
    text, mel, duration = batch
    
    # Sample random time
    t = torch.rand(B, 1, 1)  # [0, 1]
    
    # Interpolate between noise and data
    noise = torch.randn_like(mel)
    x_t = (1 - t) * noise + t * mel  # Linear interpolation (OT path)
    
    # Target velocity
    u_t = mel - noise  # Constant velocity for linear path
    
    # Predict velocity
    v_pred = DiT(x_t, t, text_condition)
    
    # Loss
    loss = F.mse_loss(v_pred, u_t)
    loss.backward()
```

### References
- [NeurIPS 2024 Tutorial: Flow Matching for Generative Modeling](https://neurips.cc/virtual/2024/tutorial/99531)
- [Matcha-TTS: Fast TTS with Conditional Flow Matching](https://github.com/shivammehta25/Matcha-TTS)
- [Meta AI: Generative Pre-training for Speech with Flow Matching (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/27c546ab1e4f1d7d638e6a8dfbad9a07-Paper-Conference.pdf)

---

## 8. Voice Cloning Techniques

### Taxonomy of Voice Cloning Approaches

| Approach | Mechanism | Examples | Data Needed |
|----------|-----------|----------|-------------|
| **Fine-tuning** | Adapt all/some params to target speaker | VITS fine-tune | 10-30 min |
| **Speaker embedding** | Condition on learned speaker vector | Tacotron 2 + GE2 | 5-30 sec |
| **In-context learning** | Prompt with reference audio | VALL-E, CosyVoice | 3-10 sec |
| **Retrieval-augmented** | kNN over speaker database | kNN-TTS (NAACL 2025) | Reference set |
| **Codec cloning** | Clone via codec tokens | GPT-SoVITS | 5 sec - 1 min |

### State of the Art (2025-2026)
- **VEVO** (ICLR 2025): Controllable zero-shot voice imitation via large-scale in-context learning
- **IndexTTS 2**: First AR TTS with precise duration control + zero-shot cloning
- **CosyVoice 2**: Multilingual zero-shot with streaming capability
- **F5-TTS**: Flow matching based, zero-shot with code-switching
- **GPT-SoVITS**: Open-source, 5-second zero-shot or 1-minute fine-tuning

### Key Techniques
1. **Speaker Encoder**: Extract speaker embedding from reference audio (e.g., GE2, ECAPA-TDNN, CAM++)
2. **Prompt Engineering**: Use 3-10 seconds of reference audio as "prompt" for LM-based TTS
3. **Semantic Tokens**: Disentangle content from speaker characteristics
4. **Flow Matching Decoder**: Generate continuous features conditioned on speaker + text

### Reference
- [Voice Cloning: Comprehensive Survey (arXiv, May 2025)](https://arxiv.org/html/2505.00579v1)
- [GPT-SoVITS (GitHub)](https://github.com/RVC-Boss/GPT-SoVITS)

---

## 9. Recommended Simplification Approaches

For each model, a recommended approach for textbook implementation (educational, simplified versions):

### Tacotron 2 -- "Mini-Tacotron"
| Original | Simplified | Rationale |
|----------|------------|-----------|
| Location-sensitive attention | Scaled dot-product attention | Simpler, fewer params |
| 3 conv + 2 LSTM encoder | 1 conv + 1 LSTM | Reduce complexity |
| 2 LSTM decoder | 1 LSTM decoder | Faster training |
| WaveGlow vocoder | HiFi-GAN (pretrained) | Simpler, better quality |
| n_frames_per_step=1 | n_frames_per_step=3 | Faster convergence |
**Estimated params**: ~5M | **Training**: Single GPU, 1-2 days

### FastSpeech 2 -- "Mini-FastSpeech"
| Original | Simplified | Rationale |
|----------|------------|-----------|
| 4-layer FFT (enc + dec) | 2-layer Transformer each | Faster training |
| Variance adaptor (pitch+energy) | Duration predictor only | Simpler first |
| MFA alignment | Pre-computed durations | Skip alignment tool |
| hidden=256, heads=2 | hidden=128, heads=2 | Smaller model |
**Estimated params**: ~5M | **Training**: Single GPU, <1 day

### VITS -- "Mini-VITS"
| Original | Simplified | Rationale |
|----------|------------|-----------|
| 6-layer text encoder | 3-layer text encoder | Fewer params |
| Stochastic duration predictor | Deterministic duration predictor | Simpler |
| 4 flow layers | 2 flow layers | Faster |
| Full HiFi-GAN decoder | Lightweight HiFi-GAN | Smaller |
| hidden_channels=192 | hidden_channels=96 | Half size |
**Estimated params**: ~20M | **Training**: Single GPU (12GB+), 2-3 days

### VALL-E -- "Mini-VALL-E"
| Original | Simplified | Rationale |
|----------|------------|-----------|
| 12-layer AR + 12-layer NAR | 4-layer AR + 4-layer NAR | Dramatic reduction |
| 8 EnCodec codebooks | 4 codebooks | Fewer predictions |
| 60K hours training | 1K hours (small dataset) | Educational scale |
| hidden=1024, heads=16 | hidden=256, heads=4 | 16x smaller |
| 50 tokens/sec | 25 tokens/sec | Slower rate, easier |
**Estimated params**: ~15M | **Training**: Single GPU, 3-5 days

### F5-TTS -- "Mini-F5"
| Original | Simplified | Rationale |
|----------|------------|-----------|
| 22-layer DiT | 6-layer DiT | Smaller backbone |
| hidden=1024, heads=16 | hidden=256, heads=4 | 16x smaller |
| 100K hours training | 100-500 hours | Educational scale |
| BigVGAN vocoder | HiFi-GAN (lightweight) | Simpler |
| 26 ODE steps | 10 ODE steps | Faster inference |
**Estimated params**: ~15M | **Training**: Single GPU, 2-3 days

### CosyVoice -- "Mini-CosyVoice"
| Original | Simplified | Rationale |
|----------|------------|-----------|
| 500M LLM | 50M small LM | Educational |
| Chunk-aware FM | Simple FM decoder | Skip streaming |
| Supervised semantic tokens | Unsupervised (EnCodec) | Simpler pipeline |
| Multilingual | Single language | Scope reduction |
**Estimated params**: ~50M | **Training**: Multi-GPU, 1 week

### MiniMind-O -- Already Minimal!
MiniMind-O is already designed as a minimal omni model (0.1B). For the textbook:
- Use as-is for demonstration
- Possible further simplification: remove image modality, keep text+speech only
- **Estimated params**: ~115M (already minimal)
- **Training**: Consumer GPU (as designed)

---

## 10. Chinese Community Resources

### Zhihu (知乎) Articles

| Title | URL | Topics |
|-------|-----|--------|
| LLM时代的可控语音合成综述 | [zhihu.com/p/18891206551](https://zhuanlan.zhihu.com/p/18891206551) | Modern TTS landscape |
| FastSpeech——高速end-to-end语音合成 | [zhihu.com/p/362716246](https://zhuanlan.zhihu.com/p/362716246) | FastSpeech architecture |
| 语音合成之VITS | [zhihu.com/p/703579506](https://zhuanlan.zhihu.com/p/703579506) | VITS deep dive |
| 手把手教你打造端到端语音合成系统 | [zhihu.com/p/114212581](https://zhuanlan.zhihu.com/p/114212581) | Tacotron tutorial |
| TTS开源调研与测评 | [zhihu.com/p/687094556](https://zhuanlan.zhihu.com/p/687094556) | Open-source TTS comparison |
| 声音克隆与TTS技术全方位分析 | [zhihu.com/p/29698958517](https://zhuanlan.zhihu.com/p/29698958517) | Voice cloning survey |
| VALL-E: 第一个基于语言模型的TTS | [zhihu.com/p/692312842](https://zhuanlan.zhihu.com/p/692312842) | VALL-E explanation |
| JETS: FastSpeech2 + HiFi-GAN端到端 | [zhihu.com/p/554520284](https://zhuanlan.zhihu.com/p/554520284) | JETS architecture |
| F5-TTS细读：用流模型构建TTS | [zhihu.com/p/2007498244293424765](https://zhuanlan.zhihu.com/p/2007498244293424765) | F5-TTS deep dive |

### CSDN Blog Posts
| Title | URL | Topics |
|-------|-----|--------|
| FastSpeech2论文阅读 | [CSDN](https://blog.csdn.net/pied_piperG/article/details/135719625) | FastSpeech2 details |
| 端到端TTS模型的演进 | [CSDN](https://blog.csdn.net/shichaog/article/details/147523028) | Tacotron to modern TTS |
| 基于条件变分自编码器的VITS | [CSDN](https://blog.csdn.net/m0_56942491/article/details/136536601) | VITS implementation |

### Bilibili & Video Resources
| Title | Platform | Topics |
|-------|----------|--------|
| IndexTTS V2 Interleaving使用教程 | [Bilibili](https://www.bilibili.com/video/BV1S7HXzTEve/) | IndexTTS advanced |
| End-to-End Adversarial TTS讲解 | [YouTube](https://www.youtube.com/watch?v=WTB2p4bqtXU) | VITS architecture |
| IndexTTS v2 Trainer Tutorial | [YouTube](https://www.youtube.com/watch?v=fu3S2n0bwUc) | Custom language training |

---

## 11. References

### Primary Repositories
| Model | Repository | Stars | License |
|-------|-----------|-------|---------|
| Tacotron 2 | [NVIDIA/tacotron2](https://github.com/nvidia/tacotron2) | 8k+ | BSD |
| FastSpeech 2 | [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) | 2k+ | MIT |
| VITS | [jaywalnut310/vits](https://github.com/jaywalnut310/vits) | 3k+ | MIT |
| VALL-E | [lifeiteng/vall-e](https://github.com/lifeiteng/vall-e) | 2k+ | MIT |
| F5-TTS | [swivid/f5-tts](https://github.com/swivid/f5-tts) | 14k+ | CC-BY-NC |
| CosyVoice | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | 10k+ | Apache 2.0 |
| IndexTTS | [index-tts/index-tts](https://github.com/index-tts/index-tts) | 5k+ | Custom |
| MiniMind-O | [jingyaogong/minimind-o](https://github.com/jingyaogong/minimind-o) | New | Open |
| EnCodec | [facebookresearch/encodec](https://github.com/facebookresearch/encodec) | 4k+ | MIT |
| GPT-SoVITS | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | 40k+ | MIT |

### Key Papers
| Paper | Year | arXiv/Venue |
|-------|------|-------------|
| Tacotron 2 | 2017 | 1712.05884 |
| FastSpeech 2 | 2020 | 2006.04558 |
| VITS | 2021 | 2106.06103 |
| EnCodec | 2022 | 2210.13438 |
| VALL-E | 2023 | 2301.02111 |
| VALL-E 2 | 2024 | 2406.05370 |
| F5-TTS | 2024 | 2410.06885 (ACL 2025) |
| CosyVoice 2 | 2024 | 2412.10117 |
| IndexTTS | 2025 | 2502.05512 |
| IndexTTS 2.5 | 2026 | 2601.03888 |
| MiniMind-O | 2026 | 2605.03937 |
| Flow Matching Tutorial | 2024 | NeurIPS 2024 |
| Matcha-TTS | 2023 | 2309.03199 |
| RVQGAN | 2023 | NeurIPS 2023 |
| Voice Cloning Survey | 2025 | 2505.00579 |
| VEVO | 2025 | ICLR 2025 |

### Community Resources
| Resource | URL |
|----------|-----|
| NVIDIA NGC Tacotron2 | [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/tacotron_2_and_waveglow_for_pytorch) |
| PyTorch Hub Tacotron2 | [pytorch.org/hub](https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/) |
| HuggingFace VITS Docs | [huggingface.co/docs](https://huggingface.co/docs/transformers/model_doc/vits) |
| Meta AI SpeechFlow | [ai.meta.com](https://ai.meta.com/research/publications/generative-pre-training-for-speech-with-flow-matching/) |
| CosyVoice Deep Dive | [onlyvoice.art](https://onlyvoice.art/blog/deep-dive-cosyvoice-speech-synthesis-pipeline-20260415) |
| VALL-E Microsoft | [microsoft.com](https://www.microsoft.com/en-us/research/project/vall-e-x/) |
| SoundStream Google | [research.google](https://research.google/blog/soundstream-an-end-to-end-neural-audio-codec/) |
| TTS Practice Tutorial | [apxml.com](https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-4-advanced-text-to-speech-synthesis/practice-advanced-tts-training) |
