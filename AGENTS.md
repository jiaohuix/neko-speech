# AGENTS

## What This Project Is

An open-source **textbook**, not a framework.
A **world-building project**, not a model zoo.

The reader is following a story: **a catgirl learning to speak**.
Every chapter, every model, every line of code serves that single goal.

---

## Core Principles

### Principle 0: Learn in Public

This is not "learn first, output later."
It is a continuous loop:

```
Learn → Organize → Output → Get Feedback → Learn Again
```

The author is the first reader. Every chapter must be run, reviewed, and iterated by the author before moving on.
Output (GitHub, PDF, video, community) is part of the learning process, not a postscript.

### Principle 1: Fundamentals First

Audio Fundamentals (signal → sampling → STFT → Mel → Codec → Tokens) is the common foundation for ALL modern Audio AI.

Tacotron is the first neural TTS model. It is NOT Chapter 1.

If we start from Tacotron, every subsequent model (GPT-SoVITS, VALL-E, CosyVoice, Omni) forces us to loop back and patch gaps. The reader experience collapses.

### Principle 2: The Catgirl Is the World

Neko is not decoration. She is the unified worldview:

```
Neko Teacher  →  Neko Dataset  →  Neko Voice  →  Neko Desktop  →  Neko Assistant
```

Every demo, every dataset, every deployment, every video is Neko.
The reader is not learning isolated models. They are **accompanying a catgirl from silence to fluent conversation**.

### Principle 3: Every Chapter Serves One Final Product

Not:
- "Today we learn Tacotron"
- "Tomorrow we learn WaveNet"

But:
- "Today we learn STFT — so Neko can understand sound"
- "Today we learn Tacotron — so Neko can speak her first words"
- "Today we learn WaveNet — so Neko's voice sounds human"
- "Today we learn sherpa-onnx — so Neko can live on your desktop"

This gives the entire book a narrative arc that no model-centric tutorial can match.

### Principle 4: Build First, Polish Later

A complete but iterating textbook is more valuable than one perfect chapter and seven blank ones.

A chapter is ready when it can:
1. Explain why this model is needed
2. Run a minimal PyTorch implementation
3. Complete a simple experiment

Then move on. Perfectionism kills momentum.

### Principle 5: Version Everything

Chapters are not final. They are versions:
- v0.1: it runs
- v0.3: theory added
- v0.6: code polished
- v0.8: figures added
- v1.0: formal textbook chapter

Design for sustainable iteration, not one-shot completion.

### Principle 6: Every Model Answers the Same Six Questions

Do not just introduce the model. Every chapter must answer:

1. Why is it needed?
2. What problem did the previous generation leave unsolved?
3. What is the core idea?
4. How to implement it with minimal PyTorch?
5. How to verify it actually works?
6. What new problems does it create for the next generation?

This creates a unified learning thread across the entire book.

### Principle 7: Evidence First

This is not a blog post. Every conclusion must be traceable.

Not:
> "I think Tacotron struggles with long sentences..."

But:
> "Tacotron's location-sensitive attention fails on long inputs (Wang et al., 2017). Guided attention loss (Tachibana et al., 2018) was introduced to solve this."

Every chapter must cite:
- The original paper(s)
- Key follow-up works that explain why it works or why it fails
- Your own experimental verification

This makes the GitHub repo, the PDF, and the videos all equally rigorous.

### Principle 8: Every Chapter Must Produce Assets

A chapter is not "done" when the code runs. It is done when it produces a reusable asset package:

```
chapter/
├── README.md          # standalone explanation
├── model.py           # minimal PyTorch implementation
├── train.py           # training script
├── experiment/        # results, logs, sample outputs
├── figure.png         # Neko Teacher illustration
├── script.md          # video script outline
└── (later) pdf.md     # content for the formal PDF
```

One source of truth. Multiple outputs (GitHub, PDF, Bilibili, Xiaohongshu, course).

If a chapter does not leave behind assets, it does not exist for the reader outside this session.

---

## Output Format (Every Chapter)

1. **Why** — the pain point (connect to previous chapter)
2. **Theory** — intuition first, then math
3. **PyTorch Implementation** — minimal, readable, pure PyTorch/Transformers
4. **Experiment** — train script + verification
5. **Summary** — what was solved, what remains
6. **Hook** — the problem that leads to the next chapter

---

## Code Style

- Pure PyTorch / Transformers. No external TTS frameworks.
- Small files. Core model < 300 lines. Train script < 200 lines.
- Readability over engineering complexity.
- No unnecessary abstractions.
- Every chapter runnable independently.

---

## Figures

- Every important concept gets a figure.
- Use the Neko teacher illustration style (see `skills/image-gen/`).
- The catgirl explains. She is never the focus over technical content.
- Academic correctness > artistic style.
