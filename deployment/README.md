# Neko Speech Deployment Toolkit

> **"猫娘学习部署技能"** -- From training to the edge, the catgirl's journey.

This toolkit converts the TTS models from each chapter into production-ready
formats for desktop, server, and mobile deployment.

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   Training (PyTorch)                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│   │ ch02     │  │ ch04     │  │ ch05     │                      │
│   │Tacotron2 │  │FastSpeech│  │  VITS    │                      │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘                     │
│        │             │             │                              │
│        ▼             ▼             ▼                              │
│   ┌─────────────────────────────────────┐                        │
│   │      export_onnx.py (ONNX Export)   │                        │
│   └──────────────┬──────────────────────┘                        │
│                  │                                                │
│        ┌─────────┼─────────┐                                     │
│        ▼         ▼         ▼                                     │
│   ┌────────┐ ┌────────┐ ┌────────┐                              │
│   │  ONNX  │ │  MNN   │ │sherpa- │                              │
│   │Runtime │ │(Mobile)│ │onnx    │                              │
│   │(Server)│ │        │ │(Edge)  │                              │
│   └────────┘ └────────┘ └────────┘                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Export all models to ONNX
python export_onnx.py --model all --output_dir ./onnx_models --verify --benchmark

# 3. Convert to MNN (mobile)
python export_mnn.py --onnx_dir ./onnx_models --output_dir ./mnn_models --fp16

# 4. Run benchmarks
python benchmark.py --onnx_dir ./onnx_models --mnn_dir ./mnn_models --output benchmark_results.md

# 5. Try sherpa-onnx TTS
pip install sherpa-onnx
python sherpa_onnx_demo.py --list_models
```

---

## Table of Contents

1. [What is ONNX and Why Does It Matter?](#what-is-onnx)
2. [Export Pipeline: PyTorch to ONNX](#export-pipeline)
3. [Mobile Deployment with MNN](#mobile-mnn)
4. [Desktop TTS with sherpa-onnx](#desktop-sherpa-onnx)
5. [Format Comparison](#format-comparison)
6. [Benchmark Results](#benchmark-results)
7. [Troubleshooting](#troubleshooting)

---

## What is ONNX? <a id="what-is-onnx"></a>

ONNX (Open Neural Network Exchange) is an open format for representing
machine learning models. Think of it as a "universal translator" between
frameworks.

### Why ONNX matters for TTS:

| Benefit | Explanation |
|---------|-------------|
| **Cross-platform** | Run the same model on Linux, Windows, macOS, Android, iOS |
| **Optimized inference** | ONNX Runtime applies graph optimizations (operator fusion, constant folding) |
| **Hardware acceleration** | CPU (AVX-512, ARM NEON), GPU (CUDA, DirectML), NPU |
| **Framework independence** | Train in PyTorch, deploy with ONNX Runtime |
| **Smaller footprint** | No PyTorch dependency at inference time |

### The ONNX Ecosystem:

```
PyTorch / TensorFlow / JAX
         │
         ▼
    ONNX Format (.onnx)
         │
    ┌────┼────────────┬─────────────┐
    ▼    ▼            ▼             ▼
 ONNX   ONNX         MNN         CoreML
Runtime Runtime     (Mobile)    (Apple)
 (CPU)   (GPU)
```

### ONNX Opsets

Each ONNX model uses a specific "opset" version (like an API version).
Higher opsets support more operators but may not be supported by all runtimes.

- **Opset 14**: Good balance of features and compatibility
- **Opset 17+**: Latest ops, but limited runtime support
- We use **opset 14** as default for maximum compatibility

---

## Export Pipeline: PyTorch to ONNX <a id="export-pipeline"></a>

### Step 1: Export Models

```bash
# Export all models
python export_onnx.py --model all --output_dir ./onnx_models

# Export a specific model
python export_onnx.py --model vits --output_dir ./onnx_models

# With verification and benchmarking
python export_onnx.py --model all --output_dir ./onnx_models --verify --benchmark
```

### What gets exported:

| Chapter | Model | ONNX Files | Notes |
|---------|-------|------------|-------|
| ch02 | Tacotron2 | `tacotron2_encoder.onnx`, `tacotron2_postnet.onnx` | Autoregressive: encoder + postnet exported separately. Decoder runs step-by-step in Python. |
| ch04 | FastSpeech2 | `fastspeech2.onnx` | Full inference graph (encoder + length regulator + decoder). May fall back to encoder-only if dynamic shapes fail. |
| ch05 | VITS | `vits_text_encoder.onnx`, `vits_generator.onnx`, `vits_full.onnx` | Components + full inference pipeline. |

### Step 2: Verify

ONNX verification checks that the exported model produces the same outputs
as the original PyTorch model:

```python
# Verification process:
# 1. Run PyTorch model on dummy input -> get reference output
# 2. Run ONNX model on same input -> get ONNX output
# 3. Compare: max_diff < 1e-4 -> PASS
```

### Step 3: Optimize (optional)

```bash
# Install onnxslim for graph optimization
pip install onnxslim

# Optimize ONNX model
onnxslim tacotron2_encoder.onnx tacotron2_encoder_slim.onnx
```

Optimization techniques:
- **Constant folding**: Pre-compute operations on static weights
- **Operator fusion**: Combine sequential ops (Conv+BN -> single Conv)
- **Dead code elimination**: Remove unused nodes

### Special Considerations per Model

#### Tacotron2 (ch02) -- Autoregressive Challenge

Tacotron2's decoder generates one mel frame at a time in a loop. This is
inherently sequential and hard to export as a single ONNX graph.

Our approach:
- Export **encoder** (text -> encoder features) as one ONNX model
- Export **postnet** (mel refinement) as another ONNX model
- The autoregressive decoder loop runs in Python/ONNX Runtime step-by-step

This is the standard approach used by NVIDIA's TensorRT-Tacotron2 as well.

#### FastSpeech2 (ch04) -- Dynamic Length

FastSpeech2's Length Regulator uses `repeat_interleave` with predicted
durations, which creates dynamic output shapes.

Our approach:
- Try exporting the full inference graph first
- If ONNX export fails (common with dynamic shapes), fall back to
  encoder-only export (features + duration/pitch/energy predictions)
- The length regulation and decoding can be done in ONNX Runtime
  with a second pass

#### VITS (ch05) -- Complex Pipeline

VITS has multiple interconnected components:
1. TextEncoder: text -> prior parameters (mu, sigma)
2. DurationPredictor: features -> duration per token
3. Expand: repeat encoder output by duration
4. Flow (inverse): prior sample -> posterior latent
5. Generator (HiFi-GAN): latent -> waveform

Our approach:
- Export each component separately for maximum flexibility
- Also try a full end-to-end export (may fail due to dynamic loops)
- The Generator is the most important component to export -- it's
  a pure convolutional network, very ONNX-friendly

---

## Mobile Deployment with MNN <a id="mobile-mnn"></a>

MNN (Mobile Neural Network) is Alibaba's lightweight inference engine,
optimized for mobile ARM CPUs.

### Prerequisites

```bash
# Build MNNConvert from source
cd /path/to/MNN
mkdir build && cd build
cmake .. -DMNN_BUILD_CONVERTER=ON
make -j$(nproc)

# Verify
./MNNConvert --version
```

### Conversion Pipeline

```bash
# Step 1: Export to ONNX first
python export_onnx.py --model all --output_dir ./onnx_models

# Step 2: Convert ONNX to MNN
python export_mnn.py --onnx_dir ./onnx_models --output_dir ./mnn_models

# With FP16 weights (reduces size ~2x)
python export_mnn.py --onnx_dir ./onnx_models --output_dir ./mnn_models --fp16

# With weight quantization (reduces size ~4x)
python export_mnn.py --onnx_dir ./onnx_models --output_dir ./mnn_models --quant 8
```

### MNN Conversion Options

| Option | Effect | Use Case |
|--------|--------|----------|
| `--fp16` | FP16 weight storage | Mobile with limited storage |
| `--quant 8` | 8-bit weight quantization | Smallest model size |
| `--optimizeLevel 2` | Aggressive graph optimization | Fastest inference |

### Android Deployment

MNN provides a TTS SDK for Android:

```
MNN/apps/frameworks/mnn_tts/
├── src/
│   ├── piper/          # Piper TTS engine
│   ├── bertvits2/      # Bert-VITS2 engine
│   └── supertonic/     # Supertonic engine
├── android/            # JNI bindings
└── demo/               # Demo app
```

The MNN TTS SDK supports:
- **Piper TTS**: Fast, lightweight, many languages
- **Bert-VITS2**: High quality Chinese+English
- **Custom models**: Load your own MNN-converted models

### iOS Deployment

MNN provides an iOS framework:
```swift
// Load MNN model
let interpreter = MNNInterpreter(modelPath: "vits_generator.mnn")
// Run inference
let output = interpreter.run(input: textFeatures)
```

### MNN Limitations

- **No dynamic loops**: Autoregressive models (Tacotron2 decoder) need
  step-by-step inference from the app code
- **Limited ops**: Some PyTorch ops may not have MNN equivalents
- **No training**: Inference only

---

## Desktop TTS with sherpa-onnx <a id="desktop-sherpa-onnx"></a>

[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) is a cross-platform
speech toolkit with built-in TTS support using ONNX models.

### Installation

```bash
pip install sherpa-onnx
```

Available for:
- Linux (x86_64, aarch64)
- macOS (x86_64, arm64)
- Windows (x86_64)
- Android (via JNI)
- iOS (via framework)

### Supported TTS Models

| Model | Description | Quality | Speed |
|-------|-------------|---------|-------|
| **VITS** | End-to-end TTS (same as our ch05) | High | Fast |
| **Piper** | Lightweight VITS variant | Good | Very fast |
| **Kokoro-82M** | 82M parameter model | Very high | Moderate |
| **Matcha-TTS** | Flow-based TTS | High | Fast |

### Quick Demo

```bash
# List available models
python sherpa_onnx_demo.py --list_models

# Download a Chinese VITS model
python sherpa_onnx_demo.py --download vits-zh-hf-theresa

# Generate speech
python sherpa_onnx_demo.py \
    --model_dir ./sherpa-onnx-vits-zh-hf-theresa \
    --text "猫娘老师，你好！今天学习什么呢？" \
    --output neko_hello.wav

# Adjust speed
python sherpa_onnx_demo.py \
    --model_dir ./sherpa-onnx-vits-zh-hf-theresa \
    --text "Hello, world!" \
    --speed 1.2
```

### Architecture: Our Models vs sherpa-onnx

```
Our ch05 VITS (educational)          sherpa-onnx VITS (production)
┌─────────────────────┐              ┌─────────────────────────┐
│ TextEncoder (small) │              │ TextEncoder (full)      │
│   vocab=200         │              │   vocab=1000+           │
│   hidden=192        │              │   hidden=192            │
│   layers=2          │              │   layers=6              │
├─────────────────────┤              ├─────────────────────────┤
│ Flow (2 layers)     │              │ Flow (4 layers)         │
├─────────────────────┤              ├─────────────────────────┤
│ Generator (HiFi-GAN)│              │ Generator (HiFi-GAN)    │
│   upsample: 256x    │              │   upsample: 256x        │
├─────────────────────┤              ├─────────────────────────┤
│ No text frontend    │              │ Full G2P + lexicon      │
│ (assumes phoneme    │              │ + tone handling          │
│  IDs as input)      │              │                         │
└─────────────────────┘              └─────────────────────────┘
```

The architecture is the same -- the difference is scale and text processing.

### Using Our Models with sherpa-onnx

Our educational models cannot be directly loaded by sherpa-onnx because:
1. sherpa-onnx expects a specific ONNX graph structure
2. Our models lack the text frontend (G2P, lexicon)
3. sherpa-onnx models include speaker embeddings

To bridge this gap:
1. **For learning**: Use `export_onnx.py` + ONNX Runtime directly
2. **For production**: Use sherpa-onnx's pre-trained models
3. **For custom models**: Train our VITS on real data, then convert
   to sherpa-onnx format (requires text frontend integration)

---

## Format Comparison <a id="format-comparison"></a>

| Aspect | PyTorch | ONNX | MNN | sherpa-onnx |
|--------|---------|------|-----|-------------|
| **Purpose** | Training + research | Cross-platform inference | Mobile inference | TTS applications |
| **GPU support** | CUDA, ROCm | CUDA, DirectML | OpenCL, Vulkan | CPU only |
| **Model size** | Large (.pt) | Medium (.onnx) | Small (.mnn) | Medium (.onnx) |
| **Inference speed** | Slow (CPU) | Fast | Fastest (ARM) | Fast |
| **Training** | Yes | No | No | No |
| **Dynamic shapes** | Full | Limited | Very limited | Limited |
| **Autoregressive** | Full support | Step-by-step | Step-by-step | Built-in |
| **Text frontend** | Manual | Manual | Manual | Built-in |
| **Best for** | Research, GPU | Server, desktop | Android, iOS | Quick TTS apps |

### When to Use What

```
Need to train?                     -> PyTorch
Need to deploy on server?          -> ONNX Runtime
Need to deploy on Android/iOS?     -> MNN
Need quick TTS app?                -> sherpa-onnx
Need maximum quality?              -> PyTorch GPU inference
Need minimum latency?              -> MNN on mobile ARM
Need smallest model?               -> MNN with 8-bit quantization
```

---

## Benchmark Results <a id="benchmark-results"></a>

Run the benchmark to get actual numbers for your machine:

```bash
python benchmark.py --device cpu --output benchmark_results.md
python benchmark.py --device cuda --output benchmark_results_gpu.md
```

### Expected Results (RTX 3060 / AMD Ryzen 5)

These are approximate numbers for educational-sized models (not production scale):

| Model | PyTorch CPU | ONNX CPU | PyTorch GPU | RTF |
|-------|------------|----------|-------------|-----|
| Tacotron2 | ~500ms | ~350ms | ~50ms | 0.05-0.1 |
| FastSpeech2 | ~50ms | ~30ms | ~10ms | 0.005-0.02 |
| VITS | ~200ms | ~150ms | ~30ms | 0.02-0.05 |

*Note: RTF (Real-Time Factor) < 1.0 means faster than real-time.*

### Key Insights

1. **FastSpeech2 is the fastest**: Non-autoregressive = parallel generation
2. **ONNX is 1.5-2x faster than PyTorch on CPU**: Graph optimizations help
3. **GPU gives 5-10x speedup**: For batch inference, GPU dominates
4. **VITS has the best speed/quality tradeoff**: End-to-end, no vocoder needed
5. **MNN on mobile ARM**: Typically 2-3x faster than ONNX Runtime on same CPU

---

## Troubleshooting <a id="troubleshooting"></a>

### ONNX Export Failures

**Problem**: `RuntimeError: Only tuples, lists or Variables are supported as JIT output`

**Solution**: Wrap model output in a tuple or use a wrapper module that
returns a single tensor.

---

**Problem**: `TracerWarning: Converting a tensor to a Python boolean`

**Solution**: This is a warning about dynamic control flow. If the model
uses `if` statements that depend on tensor values, those branches will
be "baked in" at export time. For TTS, this is usually fine since we
export the inference path.

---

**Problem**: FastSpeech2 LengthRegulator fails to export

**Solution**: The `repeat_interleave` with dynamic durations is not fully
ONNX-compatible. The script falls back to encoder-only export. You can
implement the length regulation in ONNX Runtime or post-processing.

### MNN Conversion Failures

**Problem**: `MNNConvert` not found

**Solution**: Build MNN from source with converter support:
```bash
git clone https://github.com/alibaba/MNN.git
cd MNN && mkdir build && cd build
cmake .. -DMNN_BUILD_CONVERTER=ON
make -j$(nproc)
```

---

**Problem**: Unsupported op error during MNN conversion

**Solution**: Some ONNX ops are not supported by MNN. Try:
1. Using a different opset version (`--opset 13`)
2. Simplifying the ONNX graph first (`onnxslim`)
3. Replacing unsupported ops with supported equivalents

### ONNX Runtime Issues

**Problem**: `InvalidArgument: Unexpected input data type`

**Solution**: Check input dtypes. ONNX is strict about types:
- `int64` for token IDs
- `float32` for continuous inputs
- Use `numpy.astype()` to ensure correct types

---

**Problem**: Very slow ONNX inference

**Solution**: Check that you're using the right execution provider:
```python
# CPU (default)
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# GPU (much faster)
sess = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
```

### sherpa-onnx Issues

**Problem**: `ModuleNotFoundError: No module named 'sherpa_onnx'`

**Solution**: `pip install sherpa-onnx`

---

**Problem**: Model not found / tokens.txt missing

**Solution**: Download the full model archive and extract it. The model
directory should contain:
- `*.onnx` (the model)
- `tokens.txt` (token vocabulary)
- `lexicon.txt` (optional, for phoneme-based models)

---

## File Structure

```
deployment/
├── export_onnx.py          # Unified ONNX export for all models
├── export_mnn.py           # MNN conversion pipeline
├── benchmark.py            # Comprehensive benchmarking
├── sherpa_onnx_demo.py     # sherpa-onnx TTS demo
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── benchmark_results.md    # Generated benchmark report
├── onnx_models/            # Exported ONNX models
│   ├── tacotron2_encoder.onnx
│   ├── tacotron2_postnet.onnx
│   ├── fastspeech2.onnx
│   ├── vits_text_encoder.onnx
│   ├── vits_generator.onnx
│   └── vits_full.onnx
└── mnn_models/             # Converted MNN models
    ├── tacotron2_encoder.mnn
    ├── tacotron2_postnet.mnn
    ├── fastspeech2.mnn
    └── vits_generator.mnn
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >= 2.0 | Model loading, PyTorch benchmarks |
| `onnx` | >= 1.14 | ONNX model validation |
| `onnxruntime` | >= 1.16 | ONNX inference (CPU) |
| `onnxruntime-gpu` | >= 1.16 | ONNX inference (CUDA, optional) |
| `onnxslim` | >= 0.1 | ONNX graph optimization (optional) |
| `psutil` | >= 5.9 | Memory tracking |
| `tabulate` | >= 0.9 | Table formatting |
| `librosa` | >= 0.10 | Audio processing |
| `soundfile` | >= 0.12 | WAV file I/O |
| `MNN` | latest | MNN inference (optional) |
| `sherpa-onnx` | latest | TTS deployment (optional) |

---

## References

- [ONNX Documentation](https://onnx.ai/onnx/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [MNN GitHub](https://github.com/alibaba/MNN)
- [sherpa-onnx GitHub](https://github.com/k2-fsa/sherpa-onnx)
- [sherpa-onnx TTS Models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)
- [MNN Bert-VITS2 Conversion](https://github.com/alibaba/MNN/tree/master/transformers/Bert-VITS2-MNN)
- [Tacotron2 Paper](https://arxiv.org/abs/1712.05884)
- [FastSpeech2 Paper](https://arxiv.org/abs/2006.04558)
- [VITS Paper](https://arxiv.org/abs/2106.06103)
