# PROMPTS: Neko Teacher Chapter Illustrations

## Prompt Template Structure

```
[UNIVERSAL_STYLE_PREFIX] + [SCENE_DESCRIPTION] + [TECHNICAL_CONTENT] + [COMPOSITION_NOTES]
```

---

## Chapter-Specific Prompts

### Ch00: Environment Setup
**Scene**: Neko Teacher welcoming the reader to her classroom.
> A cute chibi anime catgirl teacher with pink hair, cat ears, glasses, wearing a white lab coat with pink bow, standing in front of a blackboard showing "Hello, Audio World!", holding a laptop with PyTorch logo, welcoming expression, soft hand-drawn illustration style mixed with professional academic diagram, pastel pink-purple-blue color palette, kawaii 2D anime style with light watercolor and marker texture, clean educational layout, high clarity, professional yet adorable, white background with subtle sparkles and cherry blossom elements

### Ch01: Audio Fundamentals — What is Sound?
**Scene**: Neko pointing at a waveform turning into frequency spectrum.
> A cute chibi anime catgirl teacher with pink hair, cat ears, glasses, wearing a white lab coat with pink bow, pointing at a large diagram showing a sound wave transforming from time domain to frequency domain via FFT, clear sine waves and spectrum bars, mathematical formulas floating around, arrows and labels in both Chinese and English, pastel cute color scheme, professional educational illustration with kawaii elements, high clarity, soft hand-drawn lines, adorable and precise

### Ch01: STFT and Mel Spectrogram
**Scene**: Neko explaining how STFT windows slide over a waveform.
> A cute chibi anime catgirl teacher with pink hair, cat ears, glasses, wearing a white lab coat with pink bow, explaining STFT window sliding over a waveform diagram, showing overlapping frames, Hann window function, and the resulting spectrogram, step-by-step flowchart with clear mathematical notations, hand-drawn academic style with cute chibi elements, pastel colors, professional yet super adorable educational illustration, high readability

### Ch01: Mel Scale
**Scene**: Neko showing the non-linear mapping from Hz to Mel.
> A cute chibi anime catgirl teacher with pink hair, cat ears, glasses, wearing a white lab coat with pink bow, pointing at a graph showing linear frequency scale vs Mel frequency scale, with filter banks visualized as overlapping triangles, formula mel = 2595 * log10(1 + f/700) displayed prominently, hand-drawn academic diagram style, cute chibi elements, clear labels in Chinese and English, pastel colors, professional yet adorable

### Ch02: Tacotron — Seq2Seq + Attention
**Scene**: Neko at a large Transformer-like architecture diagram.
> A cute chibi anime catgirl teacher with pink hair, cat ears, glasses, wearing a white lab coat with pink bow, pointing at a large Tacotron2 architecture diagram, showing Encoder (Conv + BiLSTM), Location-Sensitive Attention, Decoder (LSTM + PreNet + PostNet), and mel spectrogram output, colorful attention maps, mathematical formulas floating around, arrows and labels in both Chinese and English, pastel cute color scheme, professional educational illustration with kawaii elements, high clarity, soft hand-drawn lines, adorable and precise

### Ch02: Alignment Problem
**Scene**: Neko showing good vs bad attention alignment.
> A cute chibi anime catgirl teacher with pink hair, cat ears, glasses, wearing a white lab coat with pink bow, showing two attention heatmaps side by side — one diagonal and clean (good alignment), one scattered and chaotic (bad alignment), explaining why Tacotron sometimes repeats or skips words, hand-drawn academic style, clear labels, pastel colors, professional yet adorable educational illustration

### Ch03: WaveNet — Causal Dilated Convolution
**Scene**: Neko explaining how dilated convolutions expand receptive field.
> A cute chibi anime catgirl teacher with pink hair, cat ears, glasses, wearing a white lab coat with pink bow, explaining a dilated causal convolution diagram showing exponentially increasing dilation rates (1, 2, 4, 8, 16), with the catgirl pointing at how the receptive field grows without increasing parameters, waveforms flowing through the network, hand-drawn academic diagram style, cute chibi elements, clear mathematical notations, pastel colors, professional yet adorable

### Ch04: FastSpeech — Non-Autoregressive Parallel Generation
**Scene**: Neko showing how length regulator stretches phoneme sequences.
> A cute chibi anime catgirl teacher with pink hair, cat ears, glasses, wearing a white lab coat with pink bow, explaining a FastSpeech architecture diagram with explicit duration predictor, length regulator expanding phoneme embeddings to match mel frames, and parallel transformer decoder, showing the speed comparison (autoregressive vs parallel), hand-drawn academic style, cute chibi elements, clear labels, pastel colors, professional yet adorable educational illustration

### Ch05: VITS — Flow + VAE + GAN
**Scene**: Neko showing the end-to-end text-to-waveform pipeline.
> A cute chibi anime catgirl teacher with pink hair, cat ears, glasses, wearing a white lab coat with pink bow, explaining a VITS architecture diagram showing text encoder, posterior encoder, normalizing flow, decoder/generator, and discriminator, with stochastic duration predictor, highlighting how it eliminates the separate vocoder stage, hand-drawn academic diagram style, cute chibi elements, clear labels, pastel colors, professional yet adorable educational illustration

### Ch06: GPT-SoVITS — Voice Cloning
**Scene**: Neko showing how a few seconds of reference audio clones a voice.
> A cute chibi anime catgirl teacher with pink hair, cat ears, glasses, wearing a white lab coat with pink bow, showing a GPT-SoVITS diagram with reference audio → SSL feature extraction → VQ codebook → AR/LLM generation → vocoder, demonstrating few-shot voice cloning concept with a small audio waveform icon transforming into a cloned voice, hand-drawn academic diagram style, cute chibi elements, clear labels, pastel colors, professional yet adorable educational illustration

### Ch07: Modern TTS — CosyVoice / IndexTTS
**Scene**: Neko at a large language model diagram with audio tokens.
> A cute chibi anime catgirl teacher with pink hair, cat ears, glasses, wearing a white lab coat with pink bow, pointing at a modern LLM-based TTS architecture showing text tokens and audio tokens coexisting in the same Transformer, with flow matching or diffusion for audio generation, large colorful tokens floating around, hand-drawn academic style with cute chibi elements, clear labels in Chinese and English, pastel colors, professional yet adorable educational illustration

### Ch08: Omni — Multimodal Speech
**Scene**: Neko surrounded by text, audio, and image tokens.
> A cute chibi anime catgirl teacher with pink hair, cat ears, glasses, wearing a white lab coat with pink bow, standing in the center of a unified multimodal diagram showing text tokens, audio tokens, and image tokens all feeding into a single Transformer backbone, with speech understanding and generation flowing in both directions, futuristic but hand-drawn academic style, cute chibi elements, clear labels, pastel colors, professional yet adorable educational illustration

---

## Cover Art

### Book Cover
> Cute pink-haired catgirl TTS teacher holding a big book titled "手搓猫娘 TTS：从0到超级小模型", surrounded by floating mel-spectrograms, Transformer blocks and waveform icons, hand-drawn academic kawaii style, professional yet super adorable 2D anime illustration, pastel colors, sparkles, high detail, clean composition

---

## Editing Prompts (for image.edit)

When using `images.edit` to maintain character consistency across scenes:

**Always include**:
> 完全保留原粉色头发猫娘老师五官、脸型、猫耳、眼镜与白大褂特征，...

This ensures Neko Teacher looks identical across all chapter illustrations.
