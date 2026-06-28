# Neko Speech

An open-source textbook for learning modern Audio AI from scratch.

Learn speech synthesis by rebuilding classic and modern models with minimal, readable PyTorch implementations. Every chapter produces working code, not just theory.

The long-term goal: a lightweight, local AI catgirl voice assistant.

---

## Progress

| Chapter | Topic | Status | Key Output |
|---------|-------|--------|------------|
| Ch01 | Audio Fundamentals | ✅ Done | Waveform, FFT, STFT, Mel, Griffin-Lim |
| Ch02 | Tacotron2 | 🔄 Training | End-to-end TTS pipeline |
| Ch03 | WaveNet | 🚧 In Progress | Neural vocoder |
| Ch04 | FastSpeech2 | 🚧 In Progress | Non-autoregressive TTS |
| Ch05 | VITS | 🚧 In Progress | End-to-end with flow |
| Ch06 | Neural Audio Codec | 🚧 In Progress | EnCodec, RVQ-VAE |
| Ch07 | VALL-E | 🚧 In Progress | Codec language model |
| Ch08 | Modern Models | 🚧 In Progress | F5-TTS, CosyVoice, IndexTTS |
| Ch09 | Voice Cloning | Planned | GPT-SoVITS, zero-shot |
| Ch10 | Deployment | Planned | Local catgirl assistant |

---

## Quick Start

### 1. Setup

```bash
git clone <repo>
cd neko-speech
pip install torch numpy soundfile librosa scipy tqdm
```

### 2. Download Data

```bash
cd data
python download_neko_1k.py --output-dir processed --num-samples 1000
```

### 3. Train Tacotron2

```bash
cd chapters/ch02_tacotron/code
python train.py \
    --data-dir ../../data/processed \
    --epochs 50 \
    --batch-size 4
```

### 4. Test TTS

```bash
# With trained checkpoint
python test_tts.py \
    --checkpoint ../checkpoints/tacotron_epoch_50.pt \
    --text "你好，我是猫娘。" \
    --output ../outputs/neko_output.wav

# Quick pipeline test (random weights — expect noise)
python test_tts.py \
    --text "测试一下" \
    --output ../outputs/test_random.wav
```

---

## Project Structure

```
neko-speech/
├── README.md                          # This file
├── AGENTS.md                          # Project governance principles
├── data/
│   ├── download_neko_1k.py            # Dataset downloader (ModelScope)
│   └── processed/                     # Audio + manifest (generated)
├── chapters/
│   ├── ch01_audio_fundamentals/       # FFT, STFT, Mel, Griffin-Lim
│   │   ├── README.md
│   │   └── code/
│   │       ├── 01_waveform.py
│   │       ├── 02_fft.py
│   │       ├── 03_stft.py
│   │       ├── 04_mel.py
│   │       └── 05_reconstruct.py
│   └── ch02_tacotron/
│       ├── README.md
│       ├── code/
│       │   ├── model.py               # Tacotron2 architecture
│       │   ├── train.py               # Training loop
│       │   ├── inference.py           # Autoregressive inference
│       │   └── test_tts.py            # End-to-end TTS test
│       ├── checkpoints/               # Saved models (.gitignore)
│       └── outputs/                   # Generated audio
└── skills/
    └── image-gen/                     # Neko illustration generator
```

---

## Governance

See [AGENTS.md](AGENTS.md) for the 9 core principles that guide this project.

Key tenets:
- **Learn in Public** — every step is documented
- **Fundamentals First** — understand before using
- **Build, Don't Wrap** — implement from scratch
- **Every Chapter Serves One Final Product** — all chapters converge on the catgirl assistant

---

## License

MIT
