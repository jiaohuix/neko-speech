# Chapter 11 Plan: MiniMind-O — Neko Learns to Listen, See, and Speak

> **Status:** Planning
> **Paper:** arXiv:2605.03937 (May 2026)
> **Code:** https://github.com/jingyaogong/minimind-o (Apache 2.0)
> **Target:** ~50M params teaching version, trainable on RTX 3060 (12GB)

---

## 1. Why MiniMind-O for Chapter 11?

### 1.1 Positioning in the Curriculum

| Chapter | Model | Core Lesson |
|---------|-------|-------------|
| Ch01 | Audio Fundamentals | Sound = waves, sampling, STFT |
| Ch02 | Tacotron 2 | Seq2seq TTS, attention, Mel prediction |
| Ch03 | WaveNet | Autoregressive waveform modeling |
| Ch04 | FastSpeech 2 | Non-autoregressive, variance adaptors |
| Ch05 | VITS | VAE + normalizing flows + GAN |
| Ch06 | Neural Codec | RVQ-VAE, EnCodec, discrete audio tokens |
| Ch07 | VALL-E | TTS as language modeling, zero-shot cloning |
| Ch08 | Modern Models | CosyVoice, FishSpeech, ChatTTS overview |
| Ch09 | GPT-SoVITS | Few-shot cloning, VITS + AR + SoVITS |
| Ch10 | VoxCPM | Tokenizer-free, continuous latent, CFM |
| **Ch11** | **MiniMind-O** | **Omni model: text+speech+image → text+streaming speech** |

MiniMind-O is the natural culmination of the entire book. Every previous chapter teaches one piece of the puzzle:
- Ch01-05: How to generate speech
- Ch06: How to compress speech into tokens
- Ch07-09: How to use language models for speech
- Ch10: How to skip tokens entirely
- **Ch11: How to unify ALL modalities into a single model**

### 1.2 What Makes MiniMind-O Different from Other Omni Models?

| Model | Params | Open Source | Architecture | Streaming |
|-------|--------|-------------|--------------|-----------|
| GPT-4o | ~1.8T (est.) | No | Unknown | Yes |
| Gemini 2.0 Flash | Unknown | No | Unknown | Yes |
| Qwen2.5-Omni | ~7B | Partial | Thinker-Talker | Yes |
| Moshi | ~7B | Yes | Inner Monologue | Yes |
| GLM-4-Voice | ~9B | Partial | Dual-track | Yes |
| Mini-Omni2 | ~1B | Yes | Multi-encoder | Yes |
| **MiniMind-O** | **~0.1B** | **Yes (full)** | **Thinker-Talker** | **Yes** |

Key differentiators:
1. **Smallest functional omni model** — 1000x smaller than GPT-4o
2. **Fully open source** — code, weights, data, training scripts all available
3. **Consumer GPU trainable** — single RTX 3090 in 2 hours (mini dataset)
4. **Pure PyTorch** — no high-level framework abstractions
5. **Thinker-Talker architecture** — same paradigm as Qwen3-Omni at 1/100th the scale

---

## 2. Architecture Overview

### 2.1 High-Level Diagram

```
                        MiniMind-O Full Architecture
                        ============================

  INPUTS                                    MODEL                         OUTPUTS
  ======                                    =====                         =======

  Text ──────────────────────┐
  (tokenizer)                │
                             │
  Speech ─────────┐          │              ┌─────────────────────┐
  (16 kHz wav)    │          │              │                     │
                  ▼          │              │    THINKER           │      Text Output
  SenseVoice ──► Audio ──►  │   ┌────────► │    (8 layers,        │────────► (token stream)
  (frozen)       Projector──┼───┘          │     hidden=768)      │
                  (MLP)      │              │                     │
                             │              │    bridge_layer=3    │
  Image ──────────┐          │              │         │           │
  (256x256)       │          │              │         │ bridge    │
                  ▼          │              │         │ states    │
  SigLIP2 ────► Vision ──►  │              │         ▼           │
  (frozen)      Projector───┼──────────────│                     │
                  (MLP)      │              └─────────────────────┘
                             │                        │
                             │              ┌─────────▼───────────┐
                             │              │                     │
                             │              │    TALKER            │      Audio Output
  Speaker ──────────────────►│──────────────│    (4 layers,        │────────► (8 Mimi code
  Embedding                  │              │     hidden=768)      │          streams)
  (CAM++, 192-d)             │              │                     │
                             │              │    MTP Head          │
                             │              │    (shared base +    │
                             │              │     8 LR adapters)   │
                             │              └─────────┬───────────┘
                             │                        │
                             │                        ▼
                             │              ┌─────────────────────┐
                             │              │    Mimi Decoder      │      24 kHz
                             │              │    (frozen, 8 CB)    │────────► Waveform
                             │              └─────────────────────┘
```

### 2.2 Module Parameter Breakdown (Original)

| Module | Implementation | Key Config | Status | Params (dense) |
|--------|---------------|------------|--------|----------------|
| Thinker | MiniMind Transformer | 8 layers, hidden=768 | trainable | 63.91M |
| Talker | Standalone MiniMind blocks | 4 layers, 8 CB heads | trainable | 47.05M |
| Audio Projector | 2-layer MLP | 512 → 768 | trainable | 0.99M |
| Vision Projector | 2-layer MLP | 768 → 768 | trainable | 1.18M |
| Audio Encoder | SenseVoice-Small | 16 kHz features | frozen | 234.00M |
| Vision Encoder | SigLIP2 base-p32-256 | 256x256, 64 tokens | frozen | 94.55M |
| Speech Codec | Mimi | 8 CB, 12.5 Hz, 24 kHz | frozen | 96.15M |
| Speaker Condition | CAM++ embedding | 192-d vector | precomputed | — |
| **Trainable total** | | | | **113.13M** |
| **Frozen total** | | | | **424.70M** |

### 2.3 Thinker-Talker Interaction (Detailed)

```
Step-by-step forward pass:
==========================

1. Text tokens → Thinker embed_tokens → hidden_states
2. Audio features (from SenseVoice + AudioProjector) injected at placeholder positions
3. Image features (from SigLIP2 + VisionProjector) injected at placeholder positions
4. Thinker processes all layers; bridge_states captured at layer 3 (middle layer)
5. Talker receives:
   - text_condition = embed_proj(bridge_states) * text_scale
   - audio_history  = codec_proj(talker_embed(audio_ids)) * audio_scale
   - hidden = text_condition + audio_history
6. Talker processes through 4 transformer layers
7. TalkerHead outputs 8 sets of logits (one per Mimi codebook):
   - base_logits = base_linear(hidden)
   - for each codebook i: logits[i] = base_logits + adapter_i(hidden)

Why bridge at middle layer?
- Embedding layer: too little semantic information
- Final layer: too shaped toward next-token prediction
- Middle layer: best balance of contextual + cross-modal information
```

### 2.4 Sequence Format

```
Input layout (9 streams, same seq_len):
========================================

Stream 8 (text):    [sys][user_text][<audio>...][assistant_text][EOS][pad...]
Stream 0 (CB-0):    [pad...pad][ref_codes][spk][pad...pad][code_0_t1][code_0_t2]...[stop][pad]
Stream 1 (CB-1):    [pad...pad][ref_codes][spk][pad...pad][pad][code_1_t1][code_1_t2]...[stop][pad]
Stream 2 (CB-2):    [pad...pad][ref_codes][spk][pad...pad][pad][pad][code_2_t1]...[stop][pad]
...
Stream 7 (CB-7):    [pad...pad][ref_codes][spk][pad...pad][pad×7][code_7_t1]...[stop][pad]

Key:
- text stream goes through Thinker
- audio streams 0-7 go through Talker
- audio codes are STAGGERED: layer 0 starts first, layer 7 starts last
- this stagger creates a "delay pattern" enabling streaming generation
- spk = speaker token position (replaced with speaker embedding projection)
- ref_codes = reference audio codes for voice cloning (right-aligned before assistant)
```

### 2.5 Streaming Generation

```
Generation loop (simplified):
=============================

for step in range(max_new_tokens):
    # 1. Thinker generates text token (always one step ahead)
    text_token = sample(thinker_logits)
    
    # 2. Talker generates audio codes with delay pattern
    for codebook_i in range(8):
        if audio_step >= codebook_i:  # delay by codebook index
            code_i = sample(talker_logits[codebook_i])
            audio_codes[codebook_i].append(code_i)
    
    # 3. Once all 8 codebooks have produced at least 1 frame,
    #    Mimi decoder can reconstruct audio incrementally
    if audio_step >= 7:
        frame = [codes[i][step-7+i] for i in range(8)]  # diagonal read
        audio_chunk = mimi.decode(frame)
        play(audio_chunk)  # streaming playback!
```

### 2.6 MTP (Multi-Token Prediction) Head

```
TalkerHead architecture:
========================

Input: hidden_states (B, T, 768)
       │
       ├──► base_linear(768 → 2112) ──────────────────────┐
       │                                                    │
       ├──► adapter_0: Linear(768→256) → GELU → Linear(256→2112) ──► logits_0
       ├──► adapter_1: Linear(768→256) → GELU → Linear(256→2112) ──► logits_1
       ├──► adapter_2: Linear(768→256) → GELU → Linear(256→2112) ──► logits_2
       ...                                                    │
       └──► adapter_7: Linear(768→256) → GELU → Linear(256→2112) ──► logits_7
                                                              │
Final: logits_i = base_logits + adapter_i_output             │
       (preserves shared knowledge + codebook-specific bias)

Why this design?
- 8 separate heads would multiply params by 8×
- Shared base captures distributional commonalities
- Low-rank adapters (rank=256) capture codebook-specific differences
- Total overhead: 1 base + 8 adapters ≈ 1.5× a single head (not 8×)
```

### 2.7 TalkerEmbedding (Mirror of TalkerHead)

```
TalkerEmbedding architecture:
==============================

Input: audio_ids (B, 8, T)  — 8 codebook streams
       │
       ├──► base_embed(audio_ids) → base_out (B, 8, T, 768)
       │
       ├──► adapter_0_embed(audio_ids[:,0,:]) → adapter_out_0
       ├──► adapter_1_embed(audio_ids[:,1,:]) → adapter_out_1
       ...
       └──► adapter_7_embed(audio_ids[:,7,:]) → adapter_out_7

Output: mean(base_out[:,i,:] + adapter_i_out for i in range(8))
        → single embedding (B, T, 768) that fuses all codebook info
```

### 2.8 Voice Control Pipeline

```
Voice cloning flow:
===================

Reference audio (3-5 sec)
    │
    ├──► Mimi encoder → ref_codes (8 × T_ref)  — placed in audio stream before assistant
    │
    └──► CAM++ encoder → spk_emb (192-d)       — projected to talker_hidden, 
                                                 placed at spk_token position

During training:
- 50% chance to drop ref_codes (keep only spk_emb) — teaches model to rely on speaker embedding
- 5 seen voices + 7 unseen voices

During inference:
- Swap voice by changing ref_codes and spk_emb
- Thinker prompt and Talker weights remain unchanged
```

---

## 3. Simplification Strategy for Teaching Version (~50M params)

### 3.1 Design Principles

1. **Preserve the architecture** — same Thinker-Talker paradigm, same sequence format
2. **Reduce dimensions** — smaller hidden sizes and fewer layers
3. **Drop one modality initially** — focus on text+speech, add image as extension
4. **Use simpler frozen modules** — replace heavy frozen encoders with lighter alternatives
5. **Fewer codebooks** — 4 instead of 8, reducing Talker complexity

### 3.2 Proposed Configuration

```python
class MiniOmniConfig:
    # Thinker (language backbone)
    hidden_size = 384           # vs 768 original
    num_hidden_layers = 6       # vs 8 original
    num_attention_heads = 6     # vs 8
    num_key_value_heads = 2     # vs 4 (GQA)
    intermediate_size = 1024    # vs ~2048
    vocab_size = 6400           # same as MiniMind
    
    # Talker
    talker_hidden_size = 384    # vs 768
    num_talker_layers = 2       # vs 4
    
    # Audio codec (Mimi or simplified)
    num_codebooks = 4           # vs 8 — halves Talker output complexity
    audio_vocab_size = 2082     # 2048 codes + 34 specials
    audio_frame_rate = 12.5     # Hz (same)
    sample_rate = 24000         # Hz (same)
    
    # Projectors
    audio_hidden_size = 512     # SenseVoice output dim (same)
    image_hidden_size = 768     # SigLIP2 output dim (same, optional)
    
    # Bridge
    bridge_layer = 2            # vs 3 (num_hidden_layers // 2 - 1)
    
    # Speaker
    spk_emb_size = 192          # CAM++ dim (same)
```

### 3.3 Parameter Budget

| Module | Original | Teaching Version | Notes |
|--------|----------|-----------------|-------|
| Thinker | 63.91M | ~18M | 6 layers, hidden=384 |
| Talker | 47.05M | ~12M | 2 layers, hidden=384, 4 CB heads |
| Audio Projector | 0.99M | ~0.5M | 512→384 |
| Vision Projector | 1.18M | ~0.6M | 768→384 (optional) |
| **Trainable total** | **113M** | **~31M** | |
| SenseVoice (frozen) | 234M | ~234M | Keep frozen, no memory issue |
| Mimi (frozen) | 96M | ~96M | Keep frozen, use 4 of 8 codebooks |
| SigLIP2 (frozen) | 94.5M | 0 or ~94.5M | Optional |
| **Runtime total** | **538M** | **~361M** | Fits in 12GB easily |

Memory estimate for training:
- Model parameters (fp32): ~31M × 4 bytes = 124 MB
- Optimizer states (AdamW): ~31M × 8 bytes = 248 MB  
- Gradients: ~31M × 4 bytes = 124 MB
- Activations (batch_size=4, seq_len=512): ~2 GB
- Frozen encoders (inference only): ~1.3 GB
- **Total: ~4 GB** — comfortably fits RTX 3060 12GB

### 3.4 Simplification Trade-offs

| Aspect | Original | Teaching | Impact |
|--------|----------|----------|--------|
| Hidden size | 768 | 384 | Lower quality but same architecture |
| Thinker layers | 8 | 6 | Slightly weaker understanding |
| Talker layers | 4 | 2 | Less stable speech generation |
| Codebooks | 8 | 4 | Lower audio fidelity |
| MoE | Optional | Dropped | Simpler, no auxiliary loss |
| Voice cloning | Full | Simplified | Fewer voices, basic demo |
| Image modality | Full | Optional extension | Can skip for core lesson |
| Training data | ~2000h | ~50h (mini subset) | Quick iteration |

---

## 4. Training Pipeline

### 4.1 Original Training Schedule

```
Stage 1: T2A (Text-to-Audio)
├── Data: sft_t2a (1636h output speech)
├── Mode: all (Thinker + Talker + projectors)
├── LR: 5e-4, batch_size: varies, max_seq_len: 512
├── Goal: Align text with speech output
├── Thinker learns semantic conditions
└── Talker learns to generate Mimi codes

Stage 2: A2A (Audio-to-Audio) — Phase 1
├── Data: sft_a2a (1712h input, 423h output)
├── Mode: audio_proj (only audio projector)
├── LR: 5e-4
├── Goal: Align audio input pathway
└── Prevent speech input from disrupting learned T2A

Stage 3: A2A (Audio-to-Audio) — Phase 2
├── Data: sft_a2a (same)
├── Mode: all (full model)
├── LR: 2e-5 (much smaller!)
├── max_seq_len: 768 (longer sequences)
├── Goal: Fine-tune full model for speech-in → speech-out
└── Lower LR protects previously learned capabilities

Stage 4: I2T (Image-to-Text) — Optional
├── Data: sft_i2t (image instruction data)
├── Mode: vision_proj (only vision projector)
├── Goal: Add visual understanding
└── Isolated training prevents overwriting speech abilities
```

### 4.2 Simplified Training Schedule (Teaching Version)

```
Stage 1: T2A — "Teach Neko to speak" (~1-2 hours on RTX 3060)
├── Data: sft_t2a_mini (470h English)
├── Mode: all
├── LR: 5e-4, batch_size: 8, max_seq_len: 256
└── Goal: Basic text → speech generation

Stage 2: A2A — "Teach Neko to listen and respond" (~1-2 hours)
├── Data: sft_a2a_mini (75h input, 57h output)
├── Mode: audio_proj first (warmup), then all
├── LR: 1e-4, batch_size: 4, max_seq_len: 384
└── Goal: Speech → speech interaction
```

### 4.3 Loss Functions

```python
# Text loss (standard cross-entropy on Thinker output)
text_loss = CrossEntropy(thinker_logits, text_labels)  # ignore_index=-100

# Audio loss (per-codebook cross-entropy on Talker output)
audio_loss = 0
for i in range(num_codebooks):  # 4 or 8
    layer_loss = CrossEntropy(audio_logits[i], audio_labels[i])
    # Weight stop token 10× higher — critical for knowing when to stop
    stop_mask = (audio_labels[i] == AUDIO_STOP_TOKEN)
    weighted_loss = layer_loss * (1 + stop_mask * 9)
    audio_loss += weighted_loss
audio_loss /= num_codebooks

# Total loss
loss = text_loss + audio_loss  # (+ MoE aux_loss if applicable)
```

---

## 5. Key Concepts to Teach

### 5.1 Concept Map

```
                    Concepts Taught in Chapter 11
                    =============================

From Previous Chapters:
├── Ch06: Neural Audio Codec (RVQ, codebooks) → Mimi codec
├── Ch07: Language Modeling for Speech → Unified text+audio sequences
├── Ch01-05: Speech features → SenseVoice as encoder
└── Ch08: Modern architectures → Transformer, MoE, streaming

New Concepts in Chapter 11:
├── Omni-modal modeling: one model, multiple modalities
├── Thinker-Talker paradigm: semantic vs acoustic pathways
├── Bridge layer: why middle layers are better for conditioning
├── Multi-Token Prediction: parallel codebook prediction
├── Delay pattern: staggered generation for streaming
├── In-context voice cloning: reference codes + speaker embedding
├── VAD + barge-in: real-time interaction engineering
├── Scheduled sampling: training with noisy history
└── Frozen encoders: the "feature extraction" paradigm
```

### 5.2 Section-by-Section Outline

| Section | Title | Key Question | Code Component |
|---------|-------|-------------|----------------|
| 11.1 | From TTS to Omni | Why chain ASR+LLM+TTS when one model can do it all? | Conceptual |
| 11.2 | The Thinker-Talker Idea | How do you separate understanding from speaking? | Architecture diagram |
| 11.3 | Audio Input: Frozen Encoders | Why freeze SenseVoice instead of training it? | `encode_audio_inputs()` |
| 11.4 | The Bridge Layer | Why not use the final layer? | `bridge_states` capture |
| 11.5 | Audio Output: Mimi Codec | How do 8 codebooks represent sound? | Mimi encode/decode |
| 11.6 | Multi-Token Prediction | Why predict all codebooks at once? | `TalkerHead`, `TalkerEmbedding` |
| 11.7 | Sequence Format | How do text and 8 audio streams coexist? | `OmniDataset.__getitem__()` |
| 11.8 | Streaming Generation | How can you play audio before the model finishes? | `stream_generate()` |
| 11.9 | Voice Cloning | How does a reference clip control the output voice? | Speaker conditioning |
| 11.10 | Training Pipeline | What order should capabilities be introduced? | `train_sft_omni.py` |
| 11.11 | VAD and Barge-In | How does the model handle interruptions? | `SileroVAD`, `RealtimeSession` |
| 11.12 | Building SimpleOmni | Complete implementation walkthrough | `code/model.py` |
| 11.13 | Experiments | Training, evaluation, comparison | `code/train.py` |

---

## 6. Comparison with Other Omni Models

### 6.1 Architecture Comparison

| Feature | GPT-4o | Gemini Live | Qwen2.5-Omni | Moshi | MiniMind-O |
|---------|--------|-------------|-------------|-------|------------|
| **Scale** | ~1.8T | ~10B+ | ~7B | ~7B | ~0.1B |
| **Speech Input** | Native | Native | SenseVoice | Custom ASR | SenseVoice |
| **Speech Output** | Streaming | Streaming | Streaming | Streaming | Streaming |
| **Vision** | Yes | Yes | Yes | No | Yes (SigLIP2) |
| **Codec** | Unknown | Unknown | CosyVoice | Custom | Mimi |
| **Architecture** | Unknown | Unknown | Thinker-Talker | Inner Monologue | Thinker-Talker |
| **Open Source** | No | No | Partial | Yes | Full |
| **Trainable at Home** | No | No | No | Requires multi-GPU | Yes (1 GPU) |

### 6.2 The Omni Model Design Space

```
              Understanding Capability
              ▲
              │
    GPT-4o ●  │  ● Gemini Live
              │
              │     ● Qwen2.5-Omni
              │
              │  ● Moshi
              │
              │           ● Mini-Omni2
              │
              │              ● MiniMind-O  ← Our focus
              │
              └────────────────────────────────► Scale (params)
              0.1B    1B     7B     70B    1T+

MiniMind-O: "I can't reason like GPT-4o, but you can TRAIN me 
from scratch on your laptop and understand every line of code."
```

---

## 7. Code Plan

### 7.1 `code/model.py` — SimpleOmni (~50M params)

```
Components to implement:
├── RMSNorm, RoPE, Attention, FeedForward — reuse from MiniMind
├── SimpleThinker — 6-layer Transformer (hidden=384)
├── SimpleTalker — 2-layer Transformer (hidden=384)
├── SimpleTalkerHead — shared base + 4 LR adapters
├── SimpleTalkerEmbedding — shared base + 4 adapter embeddings  
├── SimpleAudioProjector — 2-layer MLP (512→384)
├── SimpleOmni — top-level model with forward() and generate()
└── stream_generate() — streaming generation with delay pattern

What to simplify:
- Drop MoE (FeedForward only, no expert routing)
- 4 codebooks instead of 8
- Smaller dimensions throughout
- Simplified speaker conditioning

What to keep:
- Thinker-Talker architecture (core lesson)
- Bridge layer mechanism (non-obvious insight)
- MTP head design (important pattern)
- Delay pattern for streaming (key engineering)
- Sequence format with multiple streams (essential)
```

### 7.2 `code/train.py` — Training Script

```
Features:
├── Stage 1: T2A training
├── Stage 2: A2A training  
├── Loss computation (text + audio + stop-token weighting)
├── Gradient accumulation
├── Mixed precision (bf16)
├── Checkpointing
└── Logging (text_loss, audio_loss, total_loss)

Simplifications:
- No DDP (single GPU)
- No wandb integration
- Simpler data loading (in-memory for mini dataset)
- No torch.compile
```

### 7.3 `code/inference.py` — Inference Demo

```
Features:
├── Text-to-Speech generation
├── Audio-to-Audio generation (speech in → speech out)
├── Streaming output (play audio while generating)
├── Voice cloning with reference audio
└── Simple CLI interface

Dependencies:
- Mimi model for audio decode
- SenseVoice for audio encode (or simplified alternative)
```

### 7.4 `code/export_onnx.py` — ONNX Export

```
Features:
├── Export Thinker to ONNX (for deployment)
├── Export Talker to ONNX
├── Quantization options (INT8, FP16)
├── Verification: compare PyTorch vs ONNX outputs
└── Notes on streaming inference with ONNX
```

---

## 8. Exercises

### 8.1 Conceptual Exercises

1. **Bridge Layer Ablation**: Move the bridge layer from layer 2 to layer 0 and to the final layer. Compare generation quality. Why does the middle layer work best?

2. **Codebook Reduction**: Compare 1, 2, 4, and 8 codebooks. How does audio quality (PESQ/MOS) change? How does training time change?

3. **Delay Pattern Visualization**: Plot the delay pattern showing when each codebook starts generating. What happens if you remove the delay (all codebooks start at step 0)?

4. **Scheduled Sampling Rate**: Train with scheduled_sampling = 0, 0.05, 0.1, 0.2. How does this affect generation stability?

### 8.2 Implementation Exercises

1. **Add Image Modality**: Extend SimpleOmni with a vision projector and SigLIP2 encoder. Implement the image injection pathway.

2. **Implement Barge-In**: Use SileroVAD to detect when the user starts speaking during model output. Implement graceful interruption.

3. **Multi-Turn Conversation**: Extend inference.py to support multi-turn dialogue with KV-cache persistence across turns.

4. **MoE Talker**: Replace the Talker's FeedForward with MoE. Does it improve quality without increasing active parameters?

### 8.3 Research Exercises

1. **Compare with Qwen2.5-Omni**: Read the Qwen2.5-Omni technical report. How does their Thinker-Talker differ from MiniMind-O?

2. **Alternative Codecs**: Replace Mimi with Encodec or DAC. How does the codec choice affect generation quality?

3. **Scaling Laws**: Train SimpleOmni at 10M, 30M, 50M, and 100M params. Plot CER vs. params. Where are the diminishing returns?

---

## 9. Integration with Existing Chapters

### 9.1 Backward References (Ch11 builds on)

| Chapter | Concept Used in Ch11 | How |
|---------|---------------------|-----|
| Ch01 | Sampling, STFT, Mel | Audio preprocessing for SenseVoice |
| Ch02 | Seq2seq, attention | Thinker is a seq2seq model |
| Ch03 | Autoregressive generation | Talker generates codes autoregressively |
| Ch04 | Variance adaptors | Comparison: omni models don't need duration predictors |
| Ch05 | VAE, normalizing flows | Mimi codec uses similar compression ideas |
| Ch06 | RVQ, codebooks | Mimi IS a neural codec (8-layer RVQ) |
| Ch07 | Audio as language | Core insight: audio tokens in LM sequence |
| Ch08 | CosyVoice, FishSpeech | Industrial omni models comparison |
| Ch09 | Voice cloning | In-context voice cloning with reference codes |
| Ch10 | Continuous latents | VoxCPM vs Mimi: continuous vs discrete codecs |

### 9.2 Forward Connections (Ch11 enables)

- **Capstone Project**: Build a real-time voice assistant using SimpleOmni + WebSocket streaming
- **Appendix**: Full omni model deployment on edge devices (ONNX Runtime)
- **Future chapter**: Video understanding (extending to temporal visual modality)

---

## 10. Estimated Implementation Effort

| Task | Estimated Time | Difficulty | Notes |
|------|---------------|------------|-------|
| Research & paper reading | 2 days | Medium | Done (this plan) |
| `model.py` implementation | 5-7 days | High | Core architecture, most complex |
| `train.py` implementation | 2-3 days | Medium | Adapt from original training script |
| `inference.py` implementation | 2-3 days | Medium | Streaming generation is tricky |
| `export_onnx.py` | 1-2 days | Low | Standard export with verification |
| `README.md` writing | 5-7 days | Medium | ~500 lines, catgirl theme |
| Testing & debugging | 3-5 days | High | Multi-stream generation has edge cases |
| Data preparation | 1-2 days | Low | Use MiniMind-O mini dataset |
| **Total** | **21-29 days** | | **~1 month** |

### 10.1 Priority Order

1. **model.py** — Get the architecture right first
2. **train.py** — Verify the model trains (even on toy data)
3. **inference.py** — Verify end-to-end generation works
4. **README.md** — Write while implementation is fresh
5. **export_onnx.py** — Polish last

---

## 11. References

### 11.1 Primary Sources

1. Gong, J. (2026). "MiniMind-O Technical Report: An Open Small-Scale Speech-Native Omni Model." arXiv:2605.03937
2. MiniMind-O GitHub: https://github.com/jingyaogong/minimind-o
3. MiniMind (LLM): https://github.com/jingyaogong/minimind
4. MiniMind-V (VLM): https://github.com/jingyaogong/minimind-v

### 11.2 Component References

5. SenseVoice (speech encoder): FunASR project
6. SigLIP2 (vision encoder): Google Research
7. Mimi (neural audio codec): Kyutai — https://huggingface.co/docs/transformers/model_doc/mimi
8. CAM++ (speaker embedding): ModelScope/DashScope
9. SileroVAD (voice activity detection): Silero Team

### 11.3 Related Omni Models

10. Qwen2.5-Omni Technical Report: arXiv:2503.20215
11. Moshi: https://github.com/kyutai-labs/moshi
12. Mini-Omni2: https://github.com/gpt-omni/mini-omni2
13. GLM-4-Voice: Zhipu AI

### 11.4 Foundational Papers

14. Vaswani et al. (2017). "Attention Is All You Need." arXiv:1706.03762
15. Defossez et al. (2022). "High Fidelity Neural Audio Compression." arXiv:2210.13438 (EnCodec)
16. Wang et al. (2023). "VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers." arXiv:2301.02111
17. Meta (2024). "Multi-Token Prediction." (MTP methodology)

---

## 12. Answers to Key Questions

### Q1: What makes MiniMind-O different from other omni models?

MiniMind-O is the **smallest fully-functional open-source omni model** at 0.1B parameters. While GPT-4o, Gemini Live, and Qwen2.5-Omni operate at billions to trillions of parameters and require massive compute, MiniMind-O can be trained on a single consumer GPU (RTX 3090) in about 2 hours. The key insight is that the Thinker-Talker architecture — originally designed for much larger models — can be scaled down dramatically if you: (1) use frozen pretrained encoders for feature extraction, (2) share parameters between codebook predictions via the MTP head, and (3) bridge at the middle layer rather than duplicating the full backbone.

### Q2: How does the Thinker-Talker architecture work?

The Thinker is a standard causal language model (MiniMind Transformer) that processes text tokens, audio features (from SenseVoice), and image features (from SigLIP2) in a unified sequence. It generates text output and produces hidden states. The Talker is a separate, smaller Transformer that reads the Thinker's **middle-layer hidden states** (not the final layer!) through a bridge connection, combined with autoregressive audio codebook history, and outputs predictions for 8 parallel Mimi codebook streams. The separation allows the Thinker to focus on semantic understanding while the Talker focuses on acoustic rendering.

### Q3: What speech encoder/decoder does it use?

- **Encoder (input):** SenseVoice-Small (FunASR), a frozen 234M parameter model that converts 16 kHz speech into 512-dimensional feature vectors. These are projected to the Thinker's hidden dimension via a 2-layer MLP.
- **Decoder (output):** Mimi (Kyutai), a frozen 96M parameter neural audio codec with 8 codebook layers at 12.5 Hz frame rate. It converts discrete Mimi codes back to 24 kHz waveform. The Talker predicts these codes; Mimi decodes them.

### Q4: How is streaming speech output achieved?

Three mechanisms work together:
1. **Delay pattern**: The 8 codebook streams are staggered — codebook 0 starts generating first, codebook 7 starts last. This means after 7 steps, a complete frame (all 8 codes) is available for decoding.
2. **Incremental Mimi decoding**: The Mimi decoder can reconstruct audio from partial code sequences, so playback starts as soon as the first complete frame is ready.
3. **Text-ahead generation**: The Thinker generates text tokens one step ahead of the Talker's audio codes, providing semantic conditioning before the acoustic generation catches up.

### Q5: What's the minimal viable implementation for teaching?

The teaching version (~50M params) preserves the full Thinker-Talker architecture while reducing:
- Thinker: 8 layers → 6 layers, hidden 768 → 384
- Talker: 4 layers → 2 layers, 8 codebooks → 4
- Same bridge mechanism, same MTP head, same delay pattern
- Same frozen SenseVoice and Mimi (no memory concern since frozen)
- Trainable on RTX 3060 (12GB) with batch_size=4-8
- Training completes in ~2-4 hours on mini dataset
