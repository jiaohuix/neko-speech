# Neko Speech Deployment Benchmark Results

**Date:** 2026-06-29 00:56:24
**Machine:** DESKTOP-C7OEKOU (x86_64)
**PyTorch:** 2.3.0+cu121
**CUDA:** Available (NVIDIA GeForce RTX 3060)

## Summary Table

| Model | Framework | Device | Size (MB) | Params | Load (ms) | Infer (ms) | Audio (s) | RTF | Memory (MB) |
|-------|-----------|--------|-----------|--------|-----------|------------|-----------|-----|-------------|
| VITS | PyTorch | cpu | - | 17.0M | 529.9 | 442.1 +/- 175.3 | 0.4 | 1.256 | 64 |
| vits_full.onnx | ONNX | CPU | 56.2 | - | 1172.1 | 147.3 +/- 26.3 | - | - | - |
| vits_generator.onnx | ONNX | CPU | 48.7 | - | 123.8 | 241.7 +/- 17.8 | - | - | - |
| vits_text_encoder.onnx | ONNX | CPU | 2.7 | - | 45.3 | 1.8 +/- 0.5 | - | - | - |

## Key Observations

- **RTF (Real-Time Factor):** < 1.0 means faster than real-time
- **FastSpeech2** is non-autoregressive -> fastest inference
- **Tacotron2** is autoregressive -> slowest but highest quality potential
- **VITS** is end-to-end -> good balance of speed and quality
- **ONNX** typically 1.5-3x faster than PyTorch on CPU
- **MNN** optimized for mobile ARM CPUs

## Format Comparison

| Format | Best For | Pros | Cons |
|--------|----------|------|------|
| PyTorch | Research, training | Full flexibility, autograd | Slow inference, large |
| ONNX | Desktop/server deploy | Cross-platform, optimized | No training, static graph |
| MNN | Mobile (Android/iOS) | ARM optimized, tiny | Limited ops, no dynamic |
| sherpa-onnx | TTS applications | Ready-to-use TTS API | Limited model support |
