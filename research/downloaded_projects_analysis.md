# Downloaded TTS Projects - Complete Analysis Report

> Date: 2026-06-29
> Scope: All downloaded TTS projects (existing + newly extracted from zips)
> Purpose: Extract educational insights for the neko-speech textbook

---

## Executive Summary

This report consolidates analysis of **10 TTS-related projects** found across the local filesystem.
Four projects were newly extracted from zip archives at `/mnt/d/down/Qwen3.5-0.8B-MNN/`:
**Fish Speech**, **FireRedTTS2**, **IndexTTS2**, and **Bert-VITS2**.
Six projects were previously analyzed: **GPT-SoVITS**, **VoxCPM/VoxCPM2**, **Bert-VITS2-MNN**,
**Qwen3.5-0.8B-MNN**, **qwen35_08b_nekoneko-MNN**, and **MNN Framework**.

---

## 1. Project Inventory

| # | Project | Location | Status | Origin | Key Innovation |
|---|---------|----------|--------|--------|----------------|
| 1 | **Fish Speech S2** | `extracted_projects/fish-speech-main/` | NEW - Full source | Fish Audio (ex-Bert-VITS2 team) | Dual-AR (4B slow + 400M fast), RL alignment, 80+ languages |
| 2 | **FireRedTTS2** | `extracted_projects/FireRedTTS2-main/` | NEW - Full source | XiaoHongShu (Little Red Book) | Dual-transformer CSM-like, 12.5Hz streaming, multi-speaker dialogue |
| 3 | **IndexTTS2** | `extracted_projects/index-tts-main/` | NEW - Full source | Bilibili | Emotion disentanglement, duration control, GPT latent features |
| 4 | **Bert-VITS2** | `extracted_projects/Bert-VITS2-master/` | NEW - Full source | Fish Audio (deprecated) | VITS2 + multilingual BERT, predecessor to Fish Speech |
| 5 | GPT-SoVITS | `/home/jhx/Projects/AIGC/GPT-SoVITS/` | Analyzed | RVC-Boss | Two-stage AR + VITS, single-codebook semantic tokens |
| 6 | VoxCPM/VoxCPM2 | `/home/jhx/Projects/AIGC/Myprojs/VoxCPM/` | Analyzed | OpenBMB | Tokenizer-free, continuous latent, CFM/DiT |
| 7 | Bert-VITS2-MNN | `/home/jhx/Projects/nlp/MNN/transformers/Bert-VITS2-MNN/` | Analyzed | MNN Team | Full VITS on Android via MNN, RTF 0.357 |
| 8 | Qwen3.5-0.8B-MNN | `/mnt/d/down/Qwen3.5-0.8B-MNN/` | Analyzed | ModelScope | 4-bit quantized LLM for TTS backbone |
| 9 | MNN Framework | `/home/jhx/Projects/nlp/MNN/` | Analyzed | Alibaba | Model conversion & mobile inference framework |
| 10 | MiniMind-O | (GitHub reference only) | Planned | jingyaogong | Omni Thinker-Talker, ~0.1B params |

---

## 2. Newly Analyzed Projects (Detailed)

### 2.1 Fish Speech S2 Pro

**Paper:** arXiv:2603.08823 (2026), arXiv:2411.01156 (2024)
**License:** Fish Audio Research License
**Scale:** 4B parameters (S2 Pro)
**Training Data:** 10M+ hours, 80+ languages

#### Architecture: Dual-Autoregressive (Dual-AR)

```
Fish Speech S2 Dual-AR Architecture
====================================

Text Input (80+ languages, no phonemes needed)
    |
    v
[Slow AR - 4B params]                  Master Transformer
    |  - Decoder-only Transformer          (time-axis generation)
    |  - Predicts primary semantic         Qwen3-based backbone
    |    codebook (codebook 0)             ~21 Hz frame rate
    |  - Operates along time axis
    |
    v
[Fast AR - 400M params]                Slave Transformer
    |  - Smaller, faster Transformer       (codebook-axis generation)
    |  - Predicts remaining 9 codebooks    RVQ codec (10 codebooks)
    |    at each time step                 codebook_size ~160
    |  - Reconstructs acoustic details
    |
    v
[RVQ Audio Codec - 10 codebooks, ~21 Hz]
    |  - Modded DAC architecture
    |  - 10 residual codebook layers
    |
    v
Waveform Output

Key Config (from llama.py DualARModelArgs):
  Slow AR: n_layer=32, n_head=32, dim=4096 (LLaMA-scale)
  Fast AR: n_fast_layer=4, fast_dim=configurable
  Codebook: codebook_size=160, num_codebooks=4 (base), 10 (S2 Pro)
  RoPE base: 10000
  Supports: LoRA fine-tuning (r=32, alpha=16)
```

#### Key Innovations

1. **Dual-AR Architecture**: Master-slave design separates semantic planning (slow, large) from acoustic rendering (fast, small). This is structurally isomorphic to standard LLMs, enabling use of LLM inference engines (SGLang, vLLM).

2. **RL Alignment (GRPO)**: Group Relative Policy Optimization for post-training. Uses the same model suite for data cleaning and annotation as Reward Models. Multi-dimensional rewards: semantic accuracy, instruction adherence, acoustic preference, timbre similarity.

3. **Inline Emotion Tags**: 15,000+ natural language tags like `[whisper]`, `[excited]`, `[angry]`, `[singing]`, `[laughing]`. Sub-word level prosody control at any position in text.

4. **Multi-Speaker Generation**: `<|speaker:i|>` tokens allow multiple speakers in a single generation, no separate reference audio per speaker needed.

5. **Extreme Streaming**: RTF 0.195 on H200, TTFA ~100ms, 3000+ tokens/s throughput. Compatible with SGLang Continuous Batching, Paged KV Cache, CUDA Graph, RadixAttention.

#### Benchmarks (S2 Pro)

| Benchmark | Score |
|-----------|-------|
| Seed-TTS Eval WER (Chinese) | 0.54% (best overall) |
| Seed-TTS Eval WER (English) | 0.99% (best overall) |
| Audio Turing Test | 0.515 posterior mean |
| EmergentTTS-Eval Win Rate | 81.88% |

#### What We Learn for the Textbook

- **Dual-AR is the production architecture**: Fish Speech S2 represents the current SOTA for open-source TTS. The slow/fast AR split is a generalizable pattern.
- **LLM inference engines apply directly to TTS**: The structural isomorphism with LLMs means we can teach TTS using the same infrastructure as LLM deployment.
- **RL alignment for speech**: GRPO for TTS is a novel concept worth teaching -- applying RLHF ideas to speech quality.
- **Teaching simplification**: The Dual-AR pattern can be simplified to ~50M params (Slow AR: 8 layers, dim=512; Fast AR: 2 layers, dim=256; 4 codebooks instead of 10).

---

### 2.2 FireRedTTS2

**Paper:** arXiv:2509.02020 (Sep 2025)
**License:** Apache 2.0
**Origin:** XiaoHongShu (Little Red Book / RED)
**Sample Rate:** 24 kHz output, 16 kHz input

#### Architecture: Dual-Transformer (CSM-Inspired)

```
FireRedTTS2 Architecture
=========================

Based on Sesame CSM (Conversational Speech Model) structure.

Input Format (interleaved text-speech sequence):
  tokens: [B, seq_len, n_codebooks+1]  (17 columns: 16 audio + 1 text)
  tokens_mask: [B, seq_len, n_codebooks+1]

  Each frame has:
    - 16 audio token columns (codebook indices from RedCodec)
    - 1 text column (Qwen2.5 tokenizer IDs)

[Backbone Transformer]         (Qwen2.5-based)
  |  - Text embeddings + Audio embeddings summed
  |  - Causal self-attention
  |  - Predicts codebook 0 + text tokens
  |
  v
[Decoder Transformer]          (smaller)
  |  - Takes backbone embedding + codebook embeddings
  |  - Causal attention over codebook dimension
  |  - Predicts codebooks 1-15
  |
  v
[RedCodec Decoder]             (Vocos-based acoustic decoder)
  |  - RVQ with Whisper encoder backbone
  |  - SSL adaptor for feature extraction
  |  - 12.5 Hz streaming rate
  |
  v
24 kHz Waveform

Key Config (from modules.py FLAVORS):
  qwen2_200M: 4 layers, 12 heads, 2 kv_heads, dim=1536, ffn=8960
  qwen2_500M: 24 layers, 14 heads, 2 kv_heads, dim=896, ffn=4864
  qwen2_1.5B: 28 layers, 12 heads, 2 kv_heads, dim=1536, ffn=8960
  qwen2_3B:   36 layers, 16 heads, 2 kv_heads, dim=2048, ffn=11008
  Audio codec: 16 codebooks, 12.5 Hz frame rate
```

#### Key Innovations

1. **Multi-Speaker Dialogue Generation**: First TTS to natively support multi-speaker conversations (up to 4 speakers, 3+ minutes). Uses `[S1]`, `[S2]` speaker tags in text.

2. **12.5Hz Streaming Tokenizer**: RedCodec achieves ultra-low frame rate (12.5 Hz) enabling first-packet latency as low as 140ms on L20 GPU. Each chunk is 0.08 seconds.

3. **Text-Speech Interleaved Sequence**: Inspired by Sesame CSM, text and audio tokens coexist in the same sequence with separate heads for each modality. Backbone handles time-axis, decoder handles codebook-axis.

4. **Whisper-Based Codec**: The RedCodec uses Whisper encoder layers as its backbone, with an SSL adaptor for feature extraction -- a novel choice that leverages speech understanding capabilities.

5. **bf16 Inference**: Reduces VRAM from 14GB to 9GB, enabling consumer GPU deployment.

#### Code Architecture (from fireredtts2.py)

```python
class ModelArgs:
    backbone_flavor: str       # e.g., "qwen2_1.5B"
    decoder_flavor: str        # e.g., "qwen2_200M"
    text_vocab_size: int       # Qwen2.5 vocab
    audio_vocab_size: int      # codec codebook size
    audio_num_codebooks: int   # 16
    decoder_loss_weight: float # balance between c0 and c1-c15 loss
    use_text_loss: bool        # whether to predict text tokens too

# Loss function:
# loss = 2 * ((1-w)*c0_loss + w*c_loss) + 0.01*text_loss
# where c0_loss = first codebook CE, c_loss = remaining codebooks CE
```

#### What We Learn for the Textbook

- **CSM architecture is accessible**: FireRedTTS2 is based on Sesame CSM's open structure, making it a teachable dual-transformer pattern.
- **Multi-speaker modeling**: The speaker-tag approach is simpler and more practical than speaker embeddings for dialogue.
- **Ultra-low frame rate (12.5 Hz)**: Shows that aggressive compression is possible with the right codec design.
- **Teaching simplification**: Use qwen2_200M backbone (4 layers) + small decoder, reduce to 4-8 codebooks instead of 16.

---

### 2.3 IndexTTS2

**Paper:** arXiv:2506.21619 (Jun 2025), arXiv:2502.05512 (Feb 2025)
**License:** Custom (contact bilibili for commercial use)
**Origin:** Bilibili
**Versions:** IndexTTS 1.0 -> 1.5 -> 2.0

#### Architecture: GPT-style AR + Emotion Disentanglement

```
IndexTTS2 Architecture
=======================

[GPT-style AR Model] (based on GPT-2 / Tortoise-TTS / XTTS)
  |
  |-- Conditioning Encoder (Conformer-based)
  |     - Extracts features from reference audio
  |     - Perceiver Resampler for variable-length compression
  |
  |-- Speaker/Emotion Disentanglement
  |     - Speaker features from timbre prompt
  |     - Emotion features from style prompt (separate audio)
  |     - Independent control over timbre vs emotion
  |
  |-- GPT Backbone (GPT2InferenceModel)
  |     - KV-cache enabled for fast inference
  |     - Predicts VQ-VAE tokens autoregressively
  |
  |-- Duration Control (IndexTTS2 novel)
  |     - Mode 1: Specify token count for precise duration
  |     - Mode 2: Free AR generation preserving prompt prosody
  |
  v
[VQ-VAE / BigVGAN2 Vocoder]
  |  - XTTS-style DVAE for tokenization
  |  - BigVGAN2 for waveform generation (24 kHz)
  |  - ECAPA-TDNN for speaker embedding

Emotion Control (IndexTTS2 novel):
  - emo_audio_prompt: separate reference audio for emotion
  - emo_alpha: 0.0-1.0 scale for emotion intensity
  - emo_vector: 8-float [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
  - use_emo_text: text-based emotion guidance (Qwen3 fine-tuned)
  - emo_text: separate text description for emotion
```

#### Key Innovations

1. **Duration Control for AR TTS**: First autoregressive TTS with precise synthesis duration control. Two modes: explicit token count or free generation. Critical for video dubbing.

2. **Emotion-Speaker Disentanglement**: Separate prompts for timbre (who speaks) and emotion (how they speak). The model can clone a voice while changing the emotional expression.

3. **Multi-Modal Emotion Control**:
   - Audio emotion prompt (reference emotional speech)
   - Emotion vector (8-dimensional float array)
   - Text emotion description (fine-tuned Qwen3 converts text to emotion vector)
   - Automatic text-based emotion inference (from the TTS script itself)

4. **GPT Latent Representations**: Incorporates GPT-style latent features for improved speech clarity during emotional expressions.

5. **Three-Stage Training Paradigm**:
   - Stage 1: Pretrain (large-scale, stability)
   - Stage 2: SFT (supervised fine-tuning, quality)
   - Stage 3: Alignment (stability + emotion control)

#### Code Architecture (from model_v2.py)

```python
class GPT2InferenceModel(GPT2PreTrainedModel):
    # Wraps GPT-2 with:
    # - Cached mel embeddings (reference audio)
    # - KV-cache for autoregressive generation
    # - Text position embeddings
    # - Custom sampling (typical sampling, beam search)

class ConditioningEncoder:
    # Conformer-based encoder for reference audio
    # spec_dim -> embedding_dim via Conv1d + AttentionBlocks
    # Perceiver Resampler for variable-length -> fixed-length

# VQ-VAE tokenization via XTTS DVAE
# BigVGAN2 vocoder for waveform synthesis
```

#### What We Learn for the Textbook

- **Emotion control is a practical feature**: The multi-modal emotion control system (audio/vector/text) is a complete design pattern worth teaching.
- **Duration control solves a real problem**: AR TTS models are notoriously hard to control in duration -- IndexTTS2's approach is practical.
- **GPT-2 as TTS backbone**: Shows that standard language model architectures work well for speech when combined with the right conditioning.
- **Teaching simplification**: Use GPT-2 small (117M) + simplified conditioning encoder + basic VQ-VAE.

---

### 2.4 Bert-VITS2

**Status:** DEPRECATED - README recommends switching to Fish Speech
**License:** GPLv3
**Origin:** Fish Audio (same team as Fish Speech)

#### Architecture: VITS2 + Multilingual BERT

```
Bert-VITS2 Architecture
========================

Based on: VITS2 (daniilrobnikov) + MassTTS (anyvoiceai)

Text Input (ZH / JP / EN / mixed)
    |
    v
[BERT Feature Extraction]
  - Chinese: chinese-roberta-wwm-ext-large
  - Japanese: bert-base-japanese-v3
  - English: bert-base-uncased
    |
    v
[VITS2 Backbone]
  - Text Encoder: Transformer + BERT features
  - Duration Predictor: Stochastic
  - Posterior Encoder: WaveNet-based
  - Flow: Residual Coupling
  - Decoder: HiFi-GAN
    |
    v
Waveform (22.05 kHz or 44.1 kHz)

Key Features:
  - Multi-speaker (speaker embedding table)
  - Emotion tagging (emotional-vits extension)
  - ONNX export support
  - WebUI for training and inference
```

#### Historical Significance

Bert-VITS2 was the **bridge between classical VITS and modern AR TTS**:
- Added BERT embeddings to VITS for better text understanding
- Introduced zero-shot voice cloning to the VITS framework
- Its team went on to create Fish Speech (the Dual-AR successor)
- The MNN mobile deployment (Bert-VITS2-MNN) remains the gold standard for on-device TTS

#### What We Learn

- **BERT + VITS is a natural combination**: The BERT feature conditioning pattern is simple and effective.
- **The evolution path**: Bert-VITS2 -> Fish Speech shows how the field moved from VITS-based to AR-based approaches.
- **Deprecation as a teaching moment**: The explicit recommendation to switch to Fish Speech documents how fast the field moves.

---

## 3. Architecture Pattern Taxonomy (Updated)

### 3.1 Five TTS Architecture Paradigms (2024-2026)

```
Pattern A: AR + Vocoder (GPT-SoVITS)
  Text -> [AR Transformer] -> Discrete Tokens -> [VITS/HiFi-GAN] -> Audio
  ~70M params, proven, fast training, good few-shot

Pattern B: VITS End-to-End (Bert-VITS2, Bert-VITS2-MNN)
  Text -> [BERT] -> [Encoder + Flow + Decoder] -> Audio
  ~83M params, best single-speaker quality, harder for zero-shot

Pattern C: Diffusion AR (VoxCPM2)
  Text -> [LLM backbone] -> Continuous Latents -> [CFM/DiT] -> [AudioVAE] -> Audio
  ~2B params, tokenizer-free, highest quality, most compute-intensive

Pattern D: Dual-AR Codec LM (Fish Speech S2, FireRedTTS2)
  Text -> [Slow AR] -> Codebook 0 -> [Fast AR] -> Codebooks 1-N -> [Codec Decoder] -> Audio
  4B+400M params (Fish) or 1.5B+200M (FireRed), production SOTA

Pattern E: Omni Thinker-Talker (MiniMind-O)
  [Text+Speech+Image] -> [Thinker] -> [Talker] -> [Codec] -> Audio + Text
  ~115M params, multi-modal, streaming speech output
```

### 3.2 Comprehensive Comparison Table

| Feature | GPT-SoVITS | VoxCPM2 | Fish S2 | FireRedTTS2 | IndexTTS2 | Bert-VITS2-MNN |
|---------|------------|---------|---------|-------------|-----------|-----------------|
| **Architecture** | AR + VITS | MiniCPM-4 + CFM | Dual-AR | Dual-Transformer | GPT + VQ-VAE | VITS + BERT |
| **Params** | ~70M | ~2B | ~4.4B | ~1.7B | ~300M+ | ~30MB (mobile) |
| **Tokenization** | RVQ (n_q=1) | None (continuous) | RVQ (10 CB) | RVQ (16 CB) | VQ-VAE | N/A (E2E) |
| **Token Rate** | 50 Hz | 25 Hz (patches) | ~21 Hz | 12.5 Hz | ~25 Hz | N/A |
| **Zero-shot** | Yes (3-10s) | Yes | Yes (10-30s) | Yes | Yes (3s+) | Limited |
| **Emotion Control** | No | No | **Yes (tags)** | No | **Yes (multi-modal)** | No |
| **Duration Control** | No | No | No | No | **Yes (novel)** | No |
| **Multi-speaker** | Single | Single | **Yes (native)** | **Yes (dialogue)** | Single | Single |
| **Streaming** | Partial | Yes | **Yes (100ms TTFA)** | **Yes (140ms)** | No | **Yes (RTF 0.36)** |
| **Languages** | 5 | 30 | **80+** | 7 | 3 (ZH/EN/JP) | 3 (ZH/JP/EN) |
| **Sample Rate** | 32 kHz | 48 kHz | 24 kHz | 24 kHz | 24 kHz | 22.05 kHz |
| **RL Alignment** | No | No | **Yes (GRPO)** | No | No | No |
| **Mobile Deploy** | No | No | No | No | No | **Yes (MNN)** |
| **Training Data** | User-provided | 2M+ hours | **10M+ hours** | Large-scale | ~34K hours | User-provided |
| **License** | MIT | Apache 2.0 | Research | Apache 2.0 | Custom | GPL |

---

## 4. Key Architectural Innovations Worth Teaching

### 4.1 Dual-Autoregressive Pattern (Fish Speech)

The most important production architecture pattern of 2025-2026:

```
Why Dual-AR?
=============

Problem: RVQ codecs have 8-16 codebook layers. Predicting all of them
at each time step is expensive (N * codebook_vocab logits per step).

Solution: Split into two Transformers:
  1. Slow AR (large): Predicts codebook 0 (semantic) along time axis
     - This is the "what to say" model
     - Large, high capacity, processes full context
  2. Fast AR (small): Predicts codebooks 1-N given codebook 0
     - This is the "how to say it" model
     - Small, fast, operates per-frame

Benefit: The slow AR sees the full time context for semantic coherence.
The fast AR only needs the current frame's semantic info to fill in
acoustic detail. Total compute is much lower than a single model
predicting all codebooks.

Teaching Version (~50M):
  Slow AR: 8 layers, dim=512, 8 heads (~20M)
  Fast AR: 2 layers, dim=256, 4 heads (~5M)
  Codec: 4 codebooks, vocab=256
```

### 4.2 Emotion-Speaker Disentanglement (IndexTTS2)

```
Emotion Control Architecture
=============================

Traditional: speaker_embedding = f(reference_audio)
  -> Single vector controls both timbre AND emotion

IndexTTS2:
  timbre_embedding = speaker_encoder(timbre_reference_audio)
  emotion_embedding = emotion_encoder(style_reference_audio)

  # Independent control:
  output = TTS(text, timbre=timbre_embedding, emotion=emotion_embedding)

  # Additional control modes:
  emotion_vector = [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
  emotion_text -> Qwen3 fine-tuned -> emotion_vector

Teaching Application:
  - Show students how disentangling factors of variation improves controllability
  - Compare single-prompt vs dual-prompt voice cloning
  - Demonstrate emotion transfer: same voice, different emotions
```

### 4.3 Text-Speech Interleaved Sequences (FireRedTTS2/CSM)

```
Interleaved Sequence Format
============================

Each position in the sequence has N+1 columns:
  [audio_cb0, audio_cb1, ..., audio_cbN-1, text_token]

For text-only positions:
  [0, 0, ..., 0, text_id]     # audio columns are zero/masked

For audio-only positions:
  [cb0_id, cb1_id, ..., cbN-1_id, 0]  # text column is zero/masked

For mixed positions (during training):
  [cb0_id, cb1_id, ..., cbN-1_id, text_id]  # both present

Key insight: A single Transformer can model both text and audio in
the same sequence, with the backbone handling time-axis dependencies
and the decoder handling codebook-axis dependencies.

Teaching Application:
  - Elegant way to unify modalities without cross-attention
  - Compare with VoxCPM's additive masking approach
  - Shows how Sesame CSM's architecture generalizes
```

### 4.4 RL Alignment for Speech (Fish Speech S2)

```
GRPO for TTS
=============

Traditional TTS training:
  Loss = CrossEntropy(predicted_tokens, target_tokens)

Fish Speech S2 adds post-training alignment:
  1. Generate multiple candidate outputs for same input
  2. Score each with multi-dimensional reward model:
     - Semantic accuracy (WER/CER)
     - Instruction adherence (tag compliance)
     - Acoustic quality (MOS prediction)
     - Timbre similarity (speaker embedding cosine)
  3. Apply GRPO (Group Relative Policy Optimization):
     - Normalize scores within group
     - Update policy to prefer higher-scoring outputs

Teaching Application:
  - Connects RLHF from LLMs to speech generation
  - Shows how "alignment" concepts transfer across modalities
  - Practical example of reward modeling for audio
```

### 4.5 Duration Control for AR TTS (IndexTTS2)

```
Duration Adaptation Scheme
===========================

Problem: AR TTS generates token-by-token, making duration unpredictable.

IndexTTS2 Solution:
  Mode 1 (Controlled):
    - User specifies target_duration_ms
    - Model computes target_token_count = target_duration_ms * token_rate
    - Generation stops at exactly target_token_count tokens
    - Padding/truncation with prosody preservation

  Mode 2 (Free):
    - Standard AR generation with EOS token
    - But prosody features from prompt are faithfully reproduced
    - Duration naturally matches the reference style

Teaching Application:
  - Practical solution to a real-world TTS problem
  - Connects to video dubbing, podcast generation use cases
  - Simple to implement, high impact
```

---

## 5. Deployment Patterns

### 5.1 Mobile Deployment (Bert-VITS2-MNN)

```
Complete mobile TTS pipeline:
  PyTorch -> ONNX -> MNN (int8 weight quant) -> Android JNI

Results on Snapdragon 888:
  Total model: ~30MB
  RTF: 0.357 (faster than real-time)
  Audio quality: 22.05 kHz
```

### 5.2 Server Deployment (Fish Speech S2)

```
Production server pipeline:
  PyTorch -> SGLang/vLLM (Continuous Batching, Paged KV Cache)

Results on H200:
  RTF: 0.195
  TTFA: ~100ms
  Throughput: 3000+ tokens/s
```

### 5.3 Consumer GPU (FireRedTTS2)

```
Consumer deployment:
  PyTorch bf16 inference

Results:
  VRAM: 9GB (from 14GB with bf16)
  First packet: 140ms on L20
  Compatible with consumer GPUs (RTX 3090/4090)
```

---

## 6. Code Snippets Worth Referencing

### 6.1 Fish Speech: Dual-AR Forward Pass Pattern

```python
# From llama.py DualARModelArgs
# Key insight: slow AR generates codebook 0, fast AR fills in the rest

@dataclass
class DualARModelArgs(BaseModelArgs):
    model_type: str = "dual_ar"
    n_fast_layer: int = 4           # Fast AR depth
    fast_dim: int | None = None     # Defaults to slow AR dim
    fast_n_head: int | None = None  # Defaults to slow AR heads
    codebook_size: int = 160
    num_codebooks: int = 4
    norm_fastlayer_input: bool = False
```

### 6.2 FireRedTTS2: Multi-Token Prediction with Decoder

```python
# From fireredtts2/llm/llm.py
# Key insight: backbone predicts c0, decoder predicts c1-cN

# Codebook 0 prediction (from backbone hidden state)
c0_logits = self.codebook0_head(audio_h)  # [audio_len, audio_vocab_size]
c0_loss = F.cross_entropy(c0_logits, c0_target)

# Remaining codebooks (decoder with causal attention over codebooks)
decoder_embeds = torch.cat([audio_h.unsqueeze(1), c_embeds], dim=1)
decoder_h = self.decoder(self.projection(decoder_embeds), ...)
c_logits = torch.einsum("bsd,sdv->bsv", decoder_h[:, 1:, :], self.audio_head)
c_loss = F.cross_entropy(c_logits.reshape(-1, ...), target_tokens.reshape(-1))

# Combined loss
loss = 2 * ((1-w)*c0_loss + w*c_loss) + 0.01*text_loss
```

### 6.3 IndexTTS2: Emotion Vector Control

```python
# 8-dimensional emotion space
emo_vector = [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
# Each value 0.0-1.0, can be specified manually or derived from audio/text

# Text-to-emotion via fine-tuned Qwen3
emo_text = "You scared me to death! Are you a ghost?"
# Automatically converts to appropriate emotion vector
```

### 6.4 FireRedTTS2: Sampling Without CUDA Synchronization

```python
# From fireredtts2/llm/llm.py
# Clever trick: avoid CUDA sync during multinomial sampling

def _multinomial_sample_one_no_sync(probs):
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)
```

---

## 7. Chapter Integration Recommendations

### 7.1 Which Projects Map to Which Chapters

| Chapter | Primary Reference | Secondary Reference |
|---------|-------------------|-------------------|
| Ch01: Audio Fundamentals | VoxCPM AudioVAE | All codec implementations |
| Ch02: Tacotron 2 | (our implementation) | Bert-VITS2 (evolution comparison) |
| Ch03: WaveNet / Vocoders | GPT-SoVITS HiFi-GAN | IndexTTS BigVGAN2 |
| Ch04: FastSpeech 2 | (our implementation) | FireRedTTS2 (contrast: no duration predictor needed) |
| Ch05: VITS | Bert-VITS2-MNN | GPT-SoVITS SoVITS stage |
| Ch06: Neural Codec | Fish Speech RVQ | FireRedTTS2 RedCodec, IndexTTS VQ-VAE |
| Ch07: VALL-E / Codec LM | Fish Speech Dual-AR | FireRedTTS2 dual-transformer |
| Ch08: Modern Models | Fish S2, IndexTTS2, FireRedTTS2 | All production systems |
| Ch09: GPT-SoVITS | GPT-SoVITS (full) | (our teaching version) |
| Ch10: VoxCPM | VoxCPM2 (full) | (our teaching version) |
| Ch11: MiniMind-O | MiniMind-O | Fish Speech multi-speaker |

### 7.2 New Teaching Opportunities from Newly Analyzed Projects

1. **Chapter 6 (Neural Codec)**: Compare three codec designs:
   - Fish Speech: Modded DAC (10 codebooks, ~21 Hz)
   - FireRedTTS2: RedCodec with Whisper backbone (16 codebooks, 12.5 Hz)
   - IndexTTS: XTTS DVAE (VQ-VAE style)

2. **Chapter 7 (Codec LM)**: Teach the Dual-AR pattern as the "production" architecture:
   - Fish Speech: Master-slave with different scales
   - FireRedTTS2: Backbone + decoder with codebook-axis attention
   - Compare with VALL-E's AR + NAR approach

3. **Chapter 8 (Modern Models)**: Emotion and control features:
   - IndexTTS2 emotion disentanglement
   - Fish Speech inline tags
   - FireRedTTS2 multi-speaker dialogue

4. **Deployment Appendix**: Three deployment targets:
   - Mobile: Bert-VITS2-MNN (MNN, int8, Android)
   - Server: Fish Speech S2 (SGLang, H200)
   - Consumer: FireRedTTS2 (bf16, 9GB VRAM)

---

## 8. Simplification Strategies for Teaching Versions

### 8.1 "Mini-FishSpeech" (~50M params)

| Component | Original (S2 Pro) | Teaching | Rationale |
|-----------|-------------------|----------|-----------|
| Slow AR | 4B, 32 layers, dim=4096 | 8 layers, dim=512 | ~20M |
| Fast AR | 400M, 4 layers | 2 layers, dim=256 | ~5M |
| Codec | 10 CB, vocab=160 | 4 CB, vocab=256 | Simpler |
| Languages | 80+ | 2 (ZH/EN) | Scope |
| RL Alignment | Yes (GRPO) | No | Advanced topic |
| Emotion Tags | 15000+ | 5 basic tags | Simpler |

### 8.2 "Mini-FireRedTTS" (~60M params)

| Component | Original | Teaching | Rationale |
|-----------|----------|----------|-----------|
| Backbone | qwen2_1.5B (28L) | qwen2_200M (4L) | ~15M |
| Decoder | qwen2_200M (4L) | 2 layers, dim=384 | ~5M |
| Codec | 16 CB, 12.5Hz | 4 CB, 25Hz | Simpler |
| Speakers | 4 | 2 | Simpler |
| Streaming | Yes | Optional | Complex |

### 8.3 "Mini-IndexTTS" (~50M params)

| Component | Original | Teaching | Rationale |
|-----------|----------|----------|-----------|
| GPT backbone | GPT-2 large | GPT-2 small (117M -> 50M reduced) | Smaller |
| Conditioning | Conformer + Perceiver | Simple attention encoder | Simpler |
| Emotion | Multi-modal | Vector only (8-dim) | Simpler |
| Duration | Two modes | Fixed-length only | Simpler |
| Vocoder | BigVGAN2 | HiFi-GAN (lightweight) | Simpler |

---

## 9. File Locations Reference

| Project | Root Path | Key Files |
|---------|-----------|-----------|
| Fish Speech | `research/extracted_projects/fish-speech-main/` | `fish_speech/models/text2semantic/llama.py` (Dual-AR), `fish_speech/models/dac/` (codec) |
| FireRedTTS2 | `research/extracted_projects/FireRedTTS2-main/` | `fireredtts2/llm/llm.py` (dual-transformer), `fireredtts2/codec/` (RedCodec) |
| IndexTTS2 | `research/extracted_projects/index-tts-main/` | `indextts/gpt/model_v2.py` (GPT model), `indextts/vqvae/` (VQ-VAE) |
| Bert-VITS2 | `research/extracted_projects/Bert-VITS2-master/` | `models.py` (VITS2), `bert/` (BERT features) |
| GPT-SoVITS | `/home/jhx/Projects/AIGC/GPT-SoVITS/` | `GPT_SoVITS/AR/models/t2s_model.py`, `module/models.py` |
| VoxCPM | `/home/jhx/Projects/AIGC/Myprojs/VoxCPM/` | `src/voxcpm/model/voxcpm2.py`, `modules/` |
| Bert-VITS2-MNN | `/home/jhx/Projects/nlp/MNN/transformers/Bert-VITS2-MNN/` | `distill/`, `bertvits2-jni/` |
| MNN | `/home/jhx/Projects/nlp/MNN/` | `transformers/llm/export/llmexport.py` |
| Existing research | `research/` | `downloaded_projects.md`, `gpt_sovits_analysis.md`, `voxcpm_analysis.md` |

---

## 10. Recommendations for Next Steps

1. **Immediate**: Read the Fish Speech technical report (arXiv:2603.08823) for Dual-AR details
2. **Immediate**: Read the FireRedTTS2 paper (arXiv:2509.02020) for CSM-style architecture
3. **Short-term**: Implement "Mini-FishSpeech" as Ch07 teaching code (Dual-AR is the most important new pattern)
4. **Short-term**: Add emotion control demo to Ch08 using IndexTTS2 patterns
5. **Medium-term**: Build the MNN deployment pipeline using Bert-VITS2-MNN as reference
6. **Long-term**: Consider Fish Speech S2 or FireRedTTS2 as basis for the "Modern Models" survey chapter
