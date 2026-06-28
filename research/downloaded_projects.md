# Downloaded TTS Projects - Exploration Report

> Date: 2026-06-29
> Scope: D drive downloads, local AIGC projects, MNN framework

---

## 1. Inventory Summary

| # | Project | Location | Status | Type |
|---|---------|----------|--------|------|
| 1 | GPT-SoVITS | `/home/jhx/Projects/AIGC/GPT-SoVITS/` | Full source | Two-stage TTS (AR + VITS) |
| 2 | VoxCPM2 | `/home/jhx/Projects/AIGC/Myprojs/VoxCPM/` | Full source | Tokenizer-free diffusion AR TTS |
| 3 | Bert-VITS2-MNN | `/home/jhx/Projects/nlp/MNN/transformers/Bert-VITS2-MNN/` | Full source + Android | VITS + MNN mobile deployment |
| 4 | Qwen3.5-0.8B-MNN | `/mnt/d/down/Qwen3.5-0.8B-MNN/` | MNN weights + config | 4-bit quantized LLM (TTS component) |
| 5 | qwen35_08b_nekoneko-MNN | `/mnt/d/down/qwen35_08b_nekoneko-MNN/` | MNN weights + config | Custom fine-tuned Qwen3.5-0.8B |
| 6 | MNN Framework | `/home/jhx/Projects/nlp/MNN/` | Full source | Model conversion & inference framework |
| 7 | IndexTTS | `/mnt/d/down/index-tts-main.zip.crdownload` | Incomplete download | Bilibili TTS (not available) |
| 8 | Fish Speech | `/mnt/d/down/fish-speech-main.zip` | Zip exists, extract failed | VQ-based TTS (not analyzed) |
| 9 | FireRedTTS2 | `/mnt/d/down/FireRedTTS2-main.zip` | Zip exists, extract failed | XiaoHongShu TTS (not analyzed) |
| 10 | Bert-VITS2 | `/mnt/d/down/Bert-VITS2-master.zip.crdownload` | Incomplete download | Multi-lingual VITS (not available) |

---

## 2. Detailed Project Analysis

### 2.1 GPT-SoVITS

**Location:** `/home/jhx/Projects/AIGC/GPT-SoVITS/`
**License:** MIT
**Paper:** Based on VITS (Kim et al., 2021) + VALL-E inspired AR

#### Architecture: Two-Stage Pipeline

```
Stage 1 (AR/GPT): Phonemes -> Semantic Tokens
  - Text2SemanticDecoder (Transformer)
  - Config: hidden_dim=512, num_heads=8, num_layers=12
  - Vocab: phoneme_vocab=512, codebook=1024+1(EOS)
  - Input: phoneme_ids + bert_features + ref_semantic_ids
  - Output: semantic token sequence (single codebook, n_q=1)
  - Rate: 50Hz (hz=50), max_sec from config

Stage 2 (SoVITS/VITS): Semantic Tokens + Text -> Waveform
  - SynthesizerTrn (VITS architecture)
  - Components: Encoder + SDP/DP + Flow + Decoder (HiFi-GAN)
  - Multi-speaker via gin_channels (speaker embedding)
  - MRTE (Multi-Reference Text Encoder) for cross-lingual
  - ResidualVectorQuantizer for latent encoding

Reference Audio Processing:
  - Chinese HuBERT -> SSL features (768-dim, 50Hz)
  - RVQ Quantizer (n_q=1, bins=1024) -> semantic tokens
  - BERT (chinese-roberta-wwm-ext-large) -> text features
```

#### Parameter Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| AR hidden_dim | 512 | `t2s_model.py` default_config |
| AR num_heads | 8 | default_config |
| AR num_layers | 12 | default_config |
| AR codebook | 8 (config), 1 used | default_config |
| Phoneme vocab | 512 | default_config |
| Semantic vocab | 1024 + 1 (EOS) | default_config |
| HuBERT dim | 768 | chinese-hubert-base |
| BERT dim | 1024 | roberta-wwm-ext-large |
| Sample rate | 32000 Hz | SoVITS default |
| AR rate | 50 Hz | config.py hz=50 |

#### Training Scripts

| Script | Purpose |
|--------|---------|
| `s1_train.py` | Stage 1: AR Transformer training (PyTorch Lightning) |
| `s2_train.py` | Stage 2: SoVITS VITS training (DDP, mixed precision) |

#### Export Capabilities

| Format | File | Status |
|--------|------|--------|
| ONNX | `onnx_export.py` | Available (T2S + SoVITS modules) |
| TorchScript | `export_torch_script.py` | Available |
| MNN | Not directly | Via ONNX -> MNNConvert |

#### Key Files

```
GPT_SoVITS/
  AR/models/t2s_model.py          # AR Transformer (semantic token prediction)
  AR/models/t2s_model_onnx.py     # ONNX-compatible AR model
  module/models.py                 # VITS SynthesizerTrn
  module/models_onnx.py           # ONNX-compatible VITS
  module/mrte_model.py            # Multi-Reference Text Encoder
  module/quantize.py              # Residual Vector Quantizer
  onnx_export.py                  # ONNX export script
  export_torch_script.py          # TorchScript export
  s1_train.py                     # Stage 1 training
  s2_train.py                     # Stage 2 training
  config.py                       # Global configuration
```

---

### 2.2 VoxCPM2

**Location:** `/home/jhx/Projects/AIGC/Myprojs/VoxCPM/`
**License:** Apache 2.0
**Paper:** arXiv:2509.24650
**Version:** VoxCPM2 (2B params, 30 languages, 48kHz output)

#### Architecture: Tokenizer-Free Diffusion Autoregressive

```
Pipeline:
  Text (LlamaTokenizer, multichar Chinese masking)
    |
    v
  MiniCPM-4 (base_lm) -- autoregressive backbone
    |                         |
    |                    ScalarQuantizationLayer
    |                    (FSQ: latent_dim=512, scale=9)
    |                         |
    v                         v
  Residual LM (8 layers)   Stop Predictor
    |                      (Linear -> SiLU -> Linear(2))
    v
  LM-to-DiT Projection
    |
    v
  UnifiedCFM (Conditional Flow Matching)
    |  - Estimator: VoxCPMLocDiTV2
    |  - Solver: Euler
    |  - sigma_min: 1e-6
    v
  AudioVAE V2
    - Encoder: 16kHz input, asymmetric
    - Decoder: 48kHz output (built-in super-resolution)
    - chunk-based encoding/decoding
```

#### Parameter Configuration

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Base LM (MiniCPM-4)** | hidden_size | from config (Qwen3-like: 1024) |
| | intermediate_size | 3072 |
| | num_attention_heads | 16 |
| | num_key_value_heads | 8 |
| | num_hidden_layers | 28 |
| | vocab_size | 151936 |
| | max_position_embeddings | 40960 |
| | use_mup | true |
| | rope_theta | configurable |
| **Residual LM** | num_hidden_layers | 8 |
| | hidden_size | same as base_lm |
| **Local Encoder** | hidden_dim | 1024 |
| | ffn_dim | 4096 |
| | num_heads | 16 |
| | num_layers | 4 |
| **Local DiT** | hidden_dim | 1024 |
| | ffn_dim | 4096 |
| | num_heads | 16 |
| | num_layers | 4 |
| **CFM** | sigma_min | 1e-6 |
| | solver | euler |
| | t_scheduler | log-norm |
| | training_cfg_rate | 0.1 |
| | inference_cfg_rate | 1.0 |
| **FSQ** | latent_dim | 512 |
| | scale | 9 |
| **AudioVAE V1** | encoder_dim | 128 |
| | encoder_rates | [2, 5, 8, 8] |
| | latent_dim | 64 |
| | decoder_dim | 1536 |
| | decoder_rates | [8, 8, 5, 2] |
| | sample_rate | 16000 |
| **Global** | patch_size | 4 |
| | feat_dim | 64 |
| | max_length | 8192 |
| | sample_rate (in) | 16000 |
| | sample_rate (out) | 48000 (v2) |

#### Training Configuration (v2 LoRA)

| Parameter | Value |
|-----------|-------|
| batch_size | 2 |
| grad_accum_steps | 8 (effective bs=16) |
| num_iters | 1000 |
| learning_rate | 1e-4 |
| weight_decay | 0.01 |
| warmup_steps | 100 |
| max_grad_norm | 1.0 |
| LoRA r | 32 |
| LoRA alpha | 32 |
| LoRA targets (LM) | q_proj, v_proj, k_proj, o_proj |
| LoRA targets (DiT) | q_proj, v_proj, k_proj, o_proj |
| Loss weights | diff=1.0, stop=1.0 |

#### Training Configuration (v2 Full Fine-tune)

| Parameter | Value |
|-----------|-------|
| learning_rate | 1e-5 (10x lower than LoRA) |
| All other params | Same as LoRA config |

#### Export Capabilities

| Format | Status | Notes |
|--------|--------|-------|
| ONNX | Not directly | Complex diffusion architecture |
| MNN | Via MNN framework | LLM part convertible; DiT/VAE need custom work |
| GGUF | Available | `gguf2mnn.py` conversion script |
| SafeTensors | Available | `safetensors2mnn.py` conversion script |

#### Key Files

```
src/voxcpm/
  model/voxcpm2.py              # Main VoxCPM2 model (2B params)
  model/voxcpm.py               # VoxCPM v1 model
  modules/minicpm4/             # MiniCPM-4 backbone (config + model + cache)
  modules/audiovae/             # AudioVAE v1 + v2
  modules/locdit/               # Local DiT + UnifiedCFM
  modules/locenc/               # Local Encoder
  modules/layers/               # FSQ, LoRA
  training/                     # Training infrastructure
gguf2mnn.py                     # GGUF to MNN conversion
safetensors2mnn.py              # SafeTensors to MNN conversion
conf/voxcpm_v2/                 # v2 training configs
```

---

### 2.3 Bert-VITS2-MNN

**Location:** `/home/jhx/Projects/nlp/MNN/transformers/Bert-VITS2-MNN/`
**License:** GPLv3 (based on Bert-VITS2)
**Based on:** Bert-VITS2 v2.3 (commit 13424595)
**Platform:** Android (offline, JNI + MNN)

#### Architecture: Full VITS Pipeline on Mobile

```
Input Text (ZH / JP / EN / ZH+EN Mix)
   |
   v
Tokenization + G2P
  - Chinese: cppjieba (C++ port of jieba)
  - Japanese: openjtalk (C++ G2P)
  - English: cpptokenizer (HuggingFace tokenizer-cpp)
  - Kotlin: BV2-specific text preprocessing
   |
   v
BERT Embedding (Distilled Models)
  - Chinese: roberta-wwm-ext-large distilled -> 4 layers, hidden=384, heads=6 (~30MB)
  - Japanese: similarly distilled
  - English: similarly distilled
   |
   v
BV2 Inference via MNN
  - Encoder (Text Encoder)
  - Emb (Speaker Embedding)
  - SDP/DP (Stochastic/Standard Duration Predictor)
  - Flow (Normalizing Flow)
  - Decoder (HiFi-GAN vocoder)
   |
   v
Waveform Output (.wav)
```

#### Performance Benchmarks

| Metric | Value |
|--------|-------|
| Test Device | Snapdragon 888 |
| Model Sample Rate | 22050 Hz |
| Total Model Size | ~29.7 MB (all BV2 modules, int8 weight quant) |
| E2E Latency | ~1856 ms (text preprocess + Encoder + Flow + Decoder) |
| Audio Duration | ~5.20s (22050 Hz x 114688 frames) |
| **RTF** | **~0.357** (< 1 = faster than real-time) |
| Throughput | ~2.80s audio per second |

#### BERT Distillation Details

| Parameter | Teacher | Student |
|-----------|---------|---------|
| Model | chinese-roberta-wwm-ext-large | Distilled BertModel |
| Layers | 24 | 4 |
| Hidden Size | 1024 | 384 |
| Intermediate | 4096 | 1536 |
| Attention Heads | 16 | 6 |
| Training Data | ~10M texts (Wikipedia CN + SkyPile) | Same |
| Loss | - | MSE (hidden state matching) |
| Optimizer | - | AdamW (lr=5e-5) |
| Scheduler | - | StepLR (step=1000, gamma=0.9) |

#### Model Conversion Pipeline

```
PyTorch (.pth)
    |
    v
ONNX Export (Bert-VITS2 export_onnx.py)
    |
    v
MNN Convert:
  ./MNNConvert --modelFile model.onnx --MNNModel model.mnn \
    --framework ONNX --bizCode MNN \
    --weightQuantBits 8 --weightQuantAsymmetric
    |
    v
Android Assets (bertvits2-jni/src/main/assets/)
```

#### Key Files

```
Bert-VITS2-MNN/
  distill/
    BertModelDistill.py         # BERT distillation training
    bert_distill_onnx_export.py  # Distilled BERT -> ONNX
    combine_bert_model.py        # Merge linear layers
    preprocess_text.py           # Text preprocessing for distillation
  app/                           # Android demo app
  bertvits2-infer-wrapper/       # Inference AAR (reusable)
  bertvits2-jni/                 # JNI native inference code
  base_model_22k/                # 22kHz pretrained base model
  text-preprocess/               # ZH/JP/EN text preprocessing
  cppjieba/                      # Chinese word segmentation
  cpptokenizer/                  # HuggingFace tokenizer C++ port
  openjtalk/                     # Japanese G2P
  third_party/MNN/               # MNN framework (submodule)
```

---

### 2.4 Qwen3.5-0.8B-MNN

**Location:** `/mnt/d/down/Qwen3.5-0.8B-MNN/`
**Source:** ModelScope `MNN/Qwen3.5-0.8B-MNN`
**Base:** Qwen/Qwen3.5-0.8B (4-bit quantized)

#### Configuration

| Parameter | Value |
|-----------|-------|
| model_type | qwen3_5 |
| hidden_size | 1024 |
| attention_mask | float |
| attention_type | full |
| is_mrope | true (multi-dimensional RoPE) |
| is_visual | true (multimodal: text + vision) |
| has_deepstack | true |
| image_size | 420 |
| num_grid_per_side | 48 |
| quant_bit | 4 |
| quant_block | 64 |
| backend_type | cpu |
| thread_num | 4 |
| precision | low |
| memory | low |
| max_new_tokens | 8192 |
| sampler | mixed (penalty -> topK -> topP -> min_p -> temperature) |
| temperature | 1.0 |
| topP | 0.95 |
| topK | 20 |
| penalty | 1.1 |

#### Files

```
Qwen3.5-0.8B-MNN/
  config.json         # Inference config (sampling params, backend)
  llm_config.json     # Model architecture config
  llm.mnn             # Model structure
  llm.mnn.weight      # Quantized weights (~470MB)
  llm.mnn.json        # Weight metadata
  visual.mnn          # Vision encoder structure
  visual.mnn.weight   # Vision weights (~63MB)
  tokenizer.txt       # Tokenizer data
```

**Relevance to TTS:** This is a text generation LLM. In a TTS pipeline, it could serve as the text understanding / semantic token prediction backbone (similar to how VoxCPM uses MiniCPM-4 or GPT-SoVITS uses its AR Transformer).

---

### 2.5 qwen35_08b_nekoneko-MNN

**Location:** `/mnt/d/down/qwen35_08b_nekoneko-MNN/`
**Source:** ModelScope `jiaohui/qwen35_08b_nekoneko-MNN`
**Note:** User's own fine-tuned model (uploaded to ModelScope)

#### Differences from Stock Qwen3.5-0.8B-MNN

| Parameter | Stock | NekoNeko |
|-----------|-------|----------|
| attention_type | full | **mix** (sliding window + full) |
| sliding_window | N/A | **4** |
| layer_nums | not in config | **24** |
| tokenizer | tokenizer.txt | **tokenizer.mtok** |
| max_new_tokens | 8192 | not set (uses default) |
| sampler pipeline | 5-stage | simpler (temperature + top_k + top_p + min_p) |
| temperature | 1.0 | **0.8** |
| top_k | 20 | **40** |
| top_p | 0.95 | **0.9** |
| min_p | 0 | **0.05** |
| repetition_penalty | 1.1 | **1.0** |
| tie_embeddings | array format | **object format** (with offsets) |
| quant_bit | 4 | 4 (same) |
| quant_block | 64 | 64 (same) |

**Relevance:** This appears to be a custom fine-tune, possibly for the neko-speech project's text processing pipeline. The `mix` attention type with sliding_window=4 is notable.

---

### 2.6 MNN Framework & Conversion Tools

**Location:** `/home/jhx/Projects/nlp/MNN/`

#### LLM Export Tool (`transformers/llm/export/llmexport.py`)

The primary tool for converting HuggingFace LLMs to MNN format.

**Supported model types:** Qwen, LLaMA, ChatGLM, Phi, GPT, Gemma, and many more.

**Key export options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--path` | Source model path (HF id or local) | Required |
| `--type` | Model type override | Auto-detect |
| `--dst_path` | Output directory | `./model` |
| `--export` | Export format (onnx/mnn) | None |
| `--quant_bit` | Weight quantization bits | 4 |
| `--quant_block` | Quantization block size | 64 |
| `--lm_quant_bit` | LM head quant bits | same as quant_bit |
| `--awq` | AWQ quantization | False |
| `--smooth` | SmoothQuant | False |
| `--hqq` | HQQ quantization | False |
| `--omni` | OmniQuant | False |
| `--sym` | Symmetric quantization | False |
| `--onnx_slim` | ONNX graph optimization | False |
| `--transformer_fuse` | Fuse vision transformer ops | False |
| `--scale_bit` | Scale/zero-point precision | 16 (fp16) |
| `--embed_bit` | Embedding precision | 16 |
| `--lora_split` | Export LoRA separately | False |
| `--generate_for_npu` | NPU deployment format | False |

**Export pipeline:**

```
HuggingFace Model (safetensors / pytorch)
    |
    v
LlmExporter.load_model()
    |
    v
(Optional) AWQ/SmoothQuant/HQQ calibration
    |
    v
ONNX Export (torch.onnx.export)
    |  - Dynamic axes for seq_len
    |  - FakeLinear for quantization simulation
    |  - ONNX rebuilder for graph optimization
    v
MNN Convert (MNNConvert binary or pymnn)
    |  - Weight quantization (2/3/4/8 bit)
    |  - Block-wise quantization
    |  - Asymmetric quantization
    v
Output:
  - llm.mnn (model structure)
  - llm.mnn.weight (quantized weights)
  - llm_config.json (architecture config)
  - config.json (inference config)
  - tokenizer file
```

**Usage example (from Qwen3.5-0.8B-MNN README):**

```bash
cd MNN
mkdir build && cd build
cmake .. -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true \
         -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true
make -j

# Convert
python llmexport.py --path Qwen/Qwen3.5-0.8B --export mnn \
  --quant_bit 4 --quant_block 64 --dst_path ./Qwen3.5-0.8B-MNN

# Run
./llm_demo /path/to/config.json prompt.txt
```

#### Diffusion Export (`transformers/diffusion/export/`)

| File | Purpose |
|------|---------|
| `convert_mnn.py` | Stable Diffusion -> MNN conversion |
| `convert_mnn_sd35.py` | SD 3.5 specific conversion |
| `sana_convert_mnn.py` | Sana model conversion |
| `onnx_export.py` | Generic ONNX export for diffusion |

**Relevance to TTS:** VoxCPM2's DiT (Diffusion Transformer) component uses similar architecture to these diffusion models. The conversion patterns here could be adapted for VoxCPM2's `UnifiedCFM` + `VoxCPMLocDiTV2` modules.

#### MNNConvert (Low-Level Converter)

**Location:** `/home/jhx/Projects/nlp/MNN/tools/converter/`

```bash
# Build
cd MNN && mkdir build && cd build
cmake .. -DMNN_BUILD_CONVERTER=true
make

# Usage
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN
./MNNConvert -f TF --modelFile model.pb --MNNModel model.mnn --bizCode MNN
./MNNConvert -f CAFFE --modelFile model.caffemodel --prototxt deploy.prototxt --MNNModel model.mnn
./MNNConvert -f TFLITE --modelFile model.tflite --MNNModel model.mnn

# With quantization (for Bert-VITS2-MNN style)
./MNNConvert --modelFile model.onnx --MNNModel model.mnn \
  --framework ONNX --bizCode MNN \
  --weightQuantBits 8 --weightQuantAsymmetric
```

---

## 3. Architecture Pattern Comparison

### 3.1 TTS Architecture Paradigms Found

```
Pattern A: AR + Vocoder (GPT-SoVITS)
  Text -> [AR Transformer] -> Discrete Tokens -> [VITS/HiFi-GAN] -> Audio
  Proven, fast training, good few-shot, but limited by tokenization quality

Pattern B: VITS End-to-End (Bert-VITS2-MNN)
  Text -> [BERT] -> [Encoder + Flow + Decoder] -> Audio
  Classic VITS, best for single-speaker quality, harder for zero-shot

Pattern C: Diffusion AR (VoxCPM2)
  Text -> [LLM backbone] -> Continuous Latents -> [CFM/DiT] -> [AudioVAE] -> Audio
  Tokenizer-free, highest quality, but most compute-intensive

Pattern D: LLM as Component (Qwen-MNN models)
  LLM handles text understanding/semantic prediction
  Separate audio modules handle waveform generation
  Flexible, leverages LLM scaling laws
```

### 3.2 Comparison Table

| Feature | GPT-SoVITS | VoxCPM2 | Bert-VITS2-MNN |
|---------|------------|---------|-----------------|
| **Params** | ~100M (AR) + ~80M (VITS) | ~2B | ~30MB distilled BERT + BV2 |
| **Architecture** | AR Transformer + VITS | MiniCPM-4 + CFM/DiT + AudioVAE | VITS + BERT embeddings |
| **Tokenization** | Discrete (RVQ, n_q=1) | **None** (continuous latent) | N/A (end-to-end VITS) |
| **Zero-shot** | Yes (3-10s ref audio) | Yes (ref audio or text description) | Limited (fine-tune required) |
| **Few-shot** | Yes (1 min training) | Yes (LoRA, ~1000 iters) | Yes (BV2 training pipeline) |
| **Languages** | ZH, EN, JA, KO, Cantonese | **30 languages** | ZH, JP, EN |
| **Sample Rate** | 32 kHz | **48 kHz** (v2) | 22.05 kHz / 44.1 kHz |
| **Voice Design** | No | **Yes** (text description) | No |
| **Streaming** | Partial | Yes (RTF ~0.3 on 4090) | **Yes** (RTF ~0.36 on SD888) |
| **Training Data** | User-provided | 2M+ hours | User-provided |
| **Mobile Deploy** | Not optimized | Not yet | **Android native (MNN)** |
| **ONNX Export** | Yes | No | Yes (then -> MNN) |
| **MNN Export** | Via ONNX | Partial (LLM part) | **Full pipeline** |
| **LoRA Support** | No | **Yes** (LM + DiT + proj) | No |
| **License** | MIT | Apache 2.0 | GPL (BV2) |

### 3.3 Key Architectural Insights for Neko-Speech

#### What to learn from each:

1. **From GPT-SoVITS:**
   - Two-stage design simplifies training (separate AR + vocoder)
   - Single-codebook semantic tokens (50Hz) are a good AR target
   - ONNX export pattern for both stages
   - HuBERT for reference audio encoding

2. **From VoxCPM2:**
   - Tokenizer-free approach avoids quantization artifacts
   - MiniCPM-4 backbone shows LLM scaling applies to TTS
   - CFM (Conditional Flow Matching) as local diffusion decoder
   - AudioVAE V2 with asymmetric encode/decode (16kHz in, 48kHz out)
   - ScalarQuantizationLayer (FSQ) as a middle ground between discrete and continuous
   - LoRA fine-tuning at multiple levels (LM, DiT, projections)
   - Voice Design from text description (unique capability)

3. **From Bert-VITS2-MNN:**
   - Complete mobile deployment reference implementation
   - BERT distillation: 24-layer -> 4-layer, 1024 -> 384 hidden, ~30MB total
   - ONNX -> MNN conversion with int8 weight quantization
   - RTF 0.357 on Snapdragon 888 proves VITS is viable on mobile
   - C++ text preprocessing pipeline (cppjieba, openjtalk, tokenizer-cpp)
   - AAR packaging for Android integration

4. **From MNN Framework:**
   - `llmexport.py` can convert any HuggingFace LLM to MNN (4-bit quantized)
   - Supports Qwen3.5 (directly relevant to neko-speech's text backbone)
   - Diffusion model export patterns (applicable to VoxCPM2's DiT)
   - Multiple quantization methods: AWQ, SmoothQuant, HQQ, OmniQuant
   - NPU deployment support

---

## 4. MNN Conversion Reference

### 4.1 For LLM Components (Text Backbone)

```bash
# Convert Qwen3.5-0.8B to MNN (4-bit)
cd /home/jhx/Projects/nlp/MNN/transformers/llm/export/
python llmexport.py \
  --path /path/to/Qwen3.5-0.8B \
  --export mnn \
  --quant_bit 4 \
  --quant_block 64 \
  --dst_path /path/to/output/
```

### 4.2 For VITS/Vocoder Components

```bash
# Step 1: Export PyTorch to ONNX
python export_onnx.py  # Project-specific

# Step 2: Convert ONNX to MNN
cd /home/jhx/Projects/nlp/MNN/build/
./MNNConvert \
  --modelFile model.onnx \
  --MNNModel model.mnn \
  --framework ONNX \
  --bizCode MNN \
  --weightQuantBits 8 \
  --weightQuantAsymmetric
```

### 4.3 For Diffusion Components (DiT/CFM)

```bash
# Reference: transformers/diffusion/export/convert_mnn.py
# VoxCPM2's UnifiedCFM + VoxCPMLocDiTV2 would follow similar pattern:
# 1. Export DiT to ONNX (with CFM solver steps baked in)
# 2. Convert ONNX to MNN
# 3. AudioVAE can be exported separately as a standalone decoder
```

### 4.4 Existing MNN Conversion Code Locations

| Code | Path | Purpose |
|------|------|---------|
| LLM Export | `MNN/transformers/llm/export/llmexport.py` | HuggingFace LLM -> MNN |
| GGUF to MNN | `MNN/transformers/llm/export/gguf2mnn.py` | GGUF format -> MNN |
| SafeTensors to MNN | `MNN/transformers/llm/export/safetensors2mnn.py` | SafeTensors -> MNN |
| Diffusion Export | `MNN/transformers/diffusion/export/` | SD/Sana -> MNN |
| BERT Distill ONNX | `MNN/transformers/Bert-VITS2-MNN/distill/` | BERT distillation + ONNX |
| MNNConvert | `MNN/tools/converter/` | Low-level format converter |
| VoxCPM GGUF | `VoxCPM/gguf2mnn.py` | VoxCPM-specific GGUF conversion |
| VoxCPM SafeTensors | `VoxCPM/safetensors2mnn.py` | VoxCPM-specific ST conversion |

---

## 5. Pending Downloads (Not Yet Available)

These projects were found as incomplete downloads or failed extractions:

### 5.1 IndexTTS (Bilibili)
- **Expected:** `/mnt/d/down/index-tts-main.zip.crdownload` (incomplete)
- **Known:** Bilibili's TTS system, likely based on VITS architecture with improvements for Chinese
- **GitHub:** `https://github.com/index-tts/index-tts`

### 5.2 Fish Speech
- **Expected:** `/mnt/d/down/fish-speech-main.zip` (exists but extraction failed)
- **Known:** VQ-VAE based TTS, fast inference, multi-language
- **Note:** Fish Speech is the successor to Bert-VITS2 (same team: fishaudio)
- **GitHub:** `https://github.com/fishaudio/fish-speech`

### 5.3 FireRedTTS2 (XiaoHongShu / Little Red Book)
- **Expected:** `/mnt/d/down/FireRedTTS2-main.zip` (exists but extraction failed)
- **Known:** XiaoHongShu's TTS system
- **Likely features:** Chinese-focused, possibly diffusion-based

### 5.4 Bert-VITS2
- **Expected:** `/mnt/d/down/Bert-VITS2-master.zip.crdownload` (incomplete)
- **Known:** Multi-lingual VITS with BERT embeddings, by fishaudio
- **Note:** Bert-VITS2-MNN already provides the mobile deployment path

---

## 6. Recommendations for Neko-Speech

Based on this survey, the most relevant patterns for the neko-speech project:

1. **Text Backbone:** Qwen3.5-0.8B (already have MNN version) for semantic understanding
2. **Architecture:** VoxCPM2's tokenizer-free approach is state-of-the-art but complex; GPT-SoVITS two-stage is more practical for iteration
3. **Mobile Target:** Bert-VITS2-MNN proves VITS on mobile is viable at RTF 0.36
4. **Training:** VoxCPM2's LoRA fine-tuning pattern (r=32, alpha=32, targeting q/k/v/o projections) is the most flexible
5. **Conversion Pipeline:** MNN `llmexport.py` for LLM, `MNNConvert` for VITS/vocoder modules
6. **Audio Output:** VoxCPM2's AudioVAE V2 (16kHz -> 48kHz asymmetric) is worth studying for high-quality output

### Suggested Architecture for Neko-Speech v1

```
Text Input
    |
    v
Qwen3.5-0.8B-MNN (text understanding + semantic prediction)
    |
    v
[Semantic-to-Acoustic Module] (TBD: VITS or CFM-based)
    |
    v
AudioVAE Decoder -> 48kHz Waveform
```

This combines the available MNN-quantized LLM with a learnable acoustic module, following the "LLM as Component" pattern.
