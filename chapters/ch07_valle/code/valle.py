"""
Ch07: VALL-E — Neural Codec Language Models for Zero-Shot TTS

Simplified implementation of VALL-E (Wang et al., 2023).

Core idea: treat audio codec tokens as a "language" and use GPT-style
autoregressive modeling to generate speech from text.

Architecture:
    Text tokens ──────┐
                      ├──→ AR Transformer ──→ Level-0 tokens
    Reference tokens ─┘         (GPT-style, causal)
                      │
    Level-0 tokens ───┤
                      ├──→ NAR Transformer ──→ Level 1..3 tokens
    Reference tokens ─┘     (bidirectional, parallel)
                      │
    All tokens ──────────→ Codec Decoder ──→ Waveform

AR model: predicts level-0 tokens one at a time, left to right.
    Captures temporal structure (rhythm, prosody, duration).

NAR model: given level-0 tokens, predicts levels 1-3 in parallel.
    Captures acoustic detail (timbre, fine spectral structure).

Reference:
    Wang et al., 2023. Language Models Are General-Purpose Interfaces.
    (a.k.a. VALL-E: Neural Codec Language Models for Zero-Shot TTS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Text Encoder (character-level)
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """
    Simple character-level text encoder.
    Converts text token IDs to continuous embeddings via embedding + conv.
    """

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        # Two conv layers for local context (like Tacotron encoder)
        self.convs = nn.Sequential(
            nn.Conv1d(dim, dim, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 5, padding=2),
            nn.ReLU(),
        )

    def forward(self, text_ids):
        """
        Args:
            text_ids: (B, T_text) — character token IDs

        Returns:
            (B, T_text, dim) — text embeddings
        """
        x = self.embedding(text_ids)            # (B, T_text, dim)
        x = self.convs(x.transpose(1, 2)).transpose(1, 2)  # conv on time axis
        return x


# ---------------------------------------------------------------------------
# AR Model (Autoregressive Transformer for Level-0 tokens)
# ---------------------------------------------------------------------------

class ARTransformer(nn.Module):
    """
    GPT-style autoregressive Transformer.

    Given:
        [text_tokens | audio_prompt_tokens | generated_tokens]

    Predicts the next level-0 audio token at each position.
    Uses causal masking so each position only sees past tokens.

    This is the heart of VALL-E: "predicting the next audio token,
    just like GPT predicts the next word."
    """

    def __init__(
        self,
        vocab_size=256,
        audio_codebook_size=256,
        dim=256,
        num_heads=4,
        num_layers=6,
        max_seq_len=2048,
    ):
        super().__init__()
        self.audio_codebook_size = audio_codebook_size
        self.dim = dim

        # Text embeddings (frozen from TextEncoder)
        self.text_proj = nn.Linear(dim, dim)

        # Audio token embedding (level-0 codebook)
        self.audio_embed = nn.Embedding(audio_codebook_size, dim)

        # BOS token (beginning of audio sequence)
        self.bos_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Positional encoding
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        # Causal Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers,
        )

        # Output head: predict next audio token
        self.out_proj = nn.Linear(dim, audio_codebook_size)

    def _build_causal_mask(self, T, device):
        """Build upper-triangular causal mask (True = masked)."""
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, text_emb, text_tokens, audio_codes_l0):
        """
        Training forward pass.

        Args:
            text_emb: (B, T_text, dim) — from TextEncoder
            text_tokens: (B, T_text) — not used directly, for shape
            audio_codes_l0: (B, T_audio) — level-0 ground truth tokens

        Returns:
            logits: (B, T_audio, audio_codebook_size)
        """
        B, T_text, _ = text_emb.shape
        T_audio = audio_codes_l0.shape[1]
        device = text_emb.device

        # Text positions
        text_pos = self.pos_embed(torch.arange(T_text, device=device))  # (T_text, dim)
        text_in = self.text_proj(text_emb) + text_pos.unsqueeze(0)

        # Audio positions: BOS + tokens[:-1] (shifted right)
        audio_emb = self.audio_embed(audio_codes_l0[:, :-1])  # (B, T_audio-1, dim)
        bos = self.bos_embed.expand(B, -1, -1)                 # (B, 1, dim)
        audio_in = torch.cat([bos, audio_emb], dim=1)           # (B, T_audio, dim)

        # Positional encoding for audio (offset by text length)
        audio_pos = self.pos_embed(
            torch.arange(T_text, T_text + T_audio, device=device)
        )
        audio_in = audio_in + audio_pos.unsqueeze(0)

        # Concatenate: [text | audio]
        memory = text_in                                       # (B, T_text, dim)
        tgt = audio_in                                          # (B, T_audio, dim)

        # Causal mask on audio positions
        tgt_mask = self._build_causal_mask(T_audio, device)

        # Transformer decoder
        output = self.transformer(
            tgt, memory, tgt_mask=tgt_mask,
        )  # (B, T_audio, dim)

        logits = self.out_proj(output)  # (B, T_audio, codebook_size)
        return logits

    @torch.no_grad()
    def generate(self, text_emb, prompt_codes_l0, max_new_tokens=200, temperature=1.0, top_k=50):
        """
        Autoregressive generation of level-0 tokens.

        Args:
            text_emb: (B, T_text, dim) — text embeddings
            prompt_codes_l0: (B, T_prompt) — reference audio tokens (voice prompt)
            max_new_tokens: how many new tokens to generate
            temperature: sampling temperature (1.0 = default)
            top_k: keep only top-k logits for sampling

        Returns:
            all_codes: (B, T_prompt + max_new_tokens) — prompt + generated tokens
        """
        B, T_text, _ = text_emb.shape
        T_prompt = prompt_codes_l0.shape[1]
        device = text_emb.device

        # Text as memory
        text_pos = self.pos_embed(torch.arange(T_text, device=device))
        memory = self.text_proj(text_emb) + text_pos.unsqueeze(0)

        # Start with prompt tokens
        generated = prompt_codes_l0.clone()  # (B, T_so_far)

        for _ in range(max_new_tokens):
            T_cur = generated.shape[1]

            # Build input: BOS + generated tokens
            audio_emb = self.audio_embed(generated)
            bos = self.bos_embed.expand(B, -1, -1)
            audio_in = torch.cat([bos, audio_emb[:, :-1]], dim=1)

            audio_pos = self.pos_embed(
                torch.arange(T_text, T_text + T_cur, device=device)
            )
            audio_in = audio_in + audio_pos.unsqueeze(0)

            tgt_mask = self._build_causal_mask(T_cur, device)
            output = self.transformer(audio_in, memory, tgt_mask=tgt_mask)

            # Get logits for the last position
            logits = self.out_proj(output[:, -1, :]) / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated


# ---------------------------------------------------------------------------
# NAR Model (Non-Autoregressive Transformer for Levels 1..N)
# ---------------------------------------------------------------------------

class NARTransformer(nn.Module):
    """
    Non-autoregressive Transformer for predicting codec levels 1-3.

    Unlike the AR model which generates tokens one by one,
    the NAR model predicts ALL tokens at ALL levels 1-3 in parallel,
    conditioned on level-0 tokens and the reference audio prompt.

    This makes it much faster (no sequential generation), but it
    cannot capture temporal dependencies as well as AR.

    Strategy: for each target level l ∈ {1, 2, 3}, the model sees
    levels 0..l-1 and predicts level l. This is done iteratively
    during inference but the model itself is trained on all levels.
    """

    def __init__(
        self,
        audio_codebook_size=256,
        dim=256,
        num_heads=4,
        num_layers=6,
        num_levels=4,
        max_seq_len=2048,
    ):
        super().__init__()
        self.audio_codebook_size = audio_codebook_size
        self.num_levels = num_levels
        self.dim = dim

        # Text projection (from TextEncoder dim)
        self.text_proj = nn.Linear(dim, dim)

        # Shared audio embedding (for condition levels 0..l-1)
        self.audio_embed = nn.Embedding(audio_codebook_size, dim)

        # Level embedding: tells the model which level it's predicting
        self.level_embed = nn.Embedding(num_levels, dim)

        # Positional encoding
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        # Bidirectional Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Output head
        self.out_proj = nn.Linear(dim, audio_codebook_size)

    def forward(self, text_emb, all_codes, target_level):
        """
        Training: predict target_level given levels 0..target_level-1.

        Args:
            text_emb: (B, T_text, dim)
            all_codes: (B, num_levels, T_audio) — ground truth for all levels
            target_level: int — which level to predict (1, 2, or 3)

        Returns:
            logits: (B, T_audio, audio_codebook_size)
        """
        B, T_text, _ = text_emb.shape
        T_audio = all_codes.shape[2]
        device = text_emb.device

        # Text input
        text_pos = self.pos_embed(torch.arange(T_text, device=device))
        text_in = self.text_proj(text_emb) + text_pos.unsqueeze(0)

        # Audio input: sum of embeddings from levels 0..target_level-1
        audio_in = torch.zeros(B, T_audio, self.dim, device=device)
        for lv in range(target_level):
            audio_in = audio_in + self.audio_embed(all_codes[:, lv])

        # Add level embedding (which level we're predicting)
        level_emb = self.level_embed(
            torch.tensor(target_level, device=device)
        )
        audio_in = audio_in + level_emb

        # Positional encoding (offset by text length)
        audio_pos = self.pos_embed(
            torch.arange(T_text, T_text + T_audio, device=device)
        )
        audio_in = audio_in + audio_pos.unsqueeze(0)

        # Concatenate [text | audio] — bidirectional attention
        full_seq = torch.cat([text_in, audio_in], dim=1)  # (B, T_text+T_audio, dim)

        output = self.transformer(full_seq)  # (B, T_text+T_audio, dim)

        # Extract audio positions
        audio_out = output[:, T_text:, :]  # (B, T_audio, dim)

        logits = self.out_proj(audio_out)
        return logits

    @torch.no_grad()
    def generate(self, text_emb, codes_l0, target_level):
        """
        Predict all tokens at a given level, conditioned on level-0.

        For simplicity, we always condition on level-0 tokens only
        (even for levels 2 and 3). This is a simplification — the real
        VALL-E uses iterative prediction.

        Args:
            text_emb: (B, T_text, dim)
            codes_l0: (B, T_audio) — level-0 tokens (from AR or ground truth)
            target_level: int — which level to predict

        Returns:
            tokens: (B, T_audio) — predicted tokens for target_level
        """
        B, T_text, _ = text_emb.shape
        T_audio = codes_l0.shape[1]
        device = text_emb.device

        text_pos = self.pos_embed(torch.arange(T_text, device=device))
        text_in = self.text_proj(text_emb) + text_pos.unsqueeze(0)

        audio_in = self.audio_embed(codes_l0)
        level_emb = self.level_embed(
            torch.tensor(target_level, device=device)
        )
        audio_in = audio_in + level_emb

        audio_pos = self.pos_embed(
            torch.arange(T_text, T_text + T_audio, device=device)
        )
        audio_in = audio_in + audio_pos.unsqueeze(0)

        full_seq = torch.cat([text_in, audio_in], dim=1)
        output = self.transformer(full_seq)

        audio_out = output[:, T_text:, :]
        logits = self.out_proj(audio_out)

        tokens = logits.argmax(dim=-1)  # greedy decoding
        return tokens


# ---------------------------------------------------------------------------
# VALL-E (Full Pipeline: AR + NAR)
# ---------------------------------------------------------------------------

class VALLE(nn.Module):
    """
    VALL-E: Neural Codec Language Models for Zero-Shot TTS.

    The complete pipeline:
    1. Text → TextEncoder → text embeddings
    2. Reference audio → Codec.encode → prompt tokens (all levels)
    3. text_emb + prompt_level0 → AR.generate → generated level-0 tokens
    4. For each level l in 1..3:
         text_emb + generated_level0 → NAR.generate(l) → level-l tokens
    5. All generated tokens → Codec.decode → mel → waveform

    Zero-shot voice cloning works because the prompt tokens encode
    the speaker's voice characteristics. The AR model, conditioned on
    these prompt tokens, generates new tokens that match the same voice.
    """

    def __init__(
        self,
        vocab_size=256,
        audio_codebook_size=256,
        dim=256,
        num_heads=4,
        ar_layers=6,
        nar_layers=6,
        num_levels=4,
        max_seq_len=2048,
    ):
        super().__init__()
        self.audio_codebook_size = audio_codebook_size
        self.num_levels = num_levels

        self.text_encoder = TextEncoder(vocab_size, dim)
        self.ar_model = ARTransformer(
            vocab_size=vocab_size,
            audio_codebook_size=audio_codebook_size,
            dim=dim,
            num_heads=num_heads,
            num_layers=ar_layers,
            max_seq_len=max_seq_len,
        )
        self.nar_model = NARTransformer(
            audio_codebook_size=audio_codebook_size,
            dim=dim,
            num_heads=num_heads,
            num_layers=nar_layers,
            num_levels=num_levels,
            max_seq_len=max_seq_len,
        )

    def forward_ar(self, text_ids, audio_codes):
        """
        Training forward pass for AR model (level-0 only).

        Args:
            text_ids: (B, T_text)
            audio_codes: (B, num_levels, T_audio) — all levels, only uses level 0

        Returns:
            logits: (B, T_audio, audio_codebook_size) — level-0 predictions
        """
        text_emb = self.text_encoder(text_ids)
        codes_l0 = audio_codes[:, 0, :]  # (B, T_audio) — level 0
        return self.ar_model(text_emb, text_ids, codes_l0)

    def forward_nar(self, text_ids, audio_codes, target_level):
        """
        Training forward pass for NAR model.

        Args:
            text_ids: (B, T_text)
            audio_codes: (B, num_levels, T_audio)
            target_level: int — which level to predict

        Returns:
            logits: (B, T_audio, audio_codebook_size)
        """
        text_emb = self.text_encoder(text_ids)
        return self.nar_model(text_emb, audio_codes, target_level)

    @torch.no_grad()
    def generate(self, text_ids, prompt_codes, max_new_tokens=200, temperature=1.0, top_k=50):
        """
        Zero-shot generation: given text + reference audio prompt,
        generate new speech tokens.

        Args:
            text_ids: (B, T_text) — text to synthesize
            prompt_codes: (B, num_levels, T_prompt) — reference audio tokens
            max_new_tokens: number of new tokens to generate
            temperature: sampling temperature for AR
            top_k: top-k sampling for AR

        Returns:
            all_codes: (B, num_levels, T_prompt + max_new_tokens)
        """
        B = text_ids.shape[0]
        T_prompt = prompt_codes.shape[2]
        device = text_ids.device

        # Encode text
        text_emb = self.text_encoder(text_ids)

        # Step 1: AR model generates level-0 tokens
        prompt_l0 = prompt_codes[:, 0, :]  # (B, T_prompt)
        generated_l0 = self.ar_model.generate(
            text_emb, prompt_l0,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )  # (B, T_prompt + max_new_tokens)

        # Step 2: NAR model fills in levels 1..N-1
        T_total = generated_l0.shape[1]
        all_codes = torch.zeros(
            B, self.num_levels, T_total,
            dtype=torch.long, device=device,
        )
        all_codes[:, 0, :] = generated_l0

        # Copy prompt tokens for all levels
        all_codes[:, :, :T_prompt] = prompt_codes

        # Generate remaining levels
        for level in range(1, self.num_levels):
            new_tokens = self.nar_model.generate(
                text_emb, generated_l0, target_level=level,
            )  # (B, T_total)
            # Only replace the non-prompt positions
            all_codes[:, level, T_prompt:] = new_tokens[:, T_prompt:]

        return all_codes


# ---------------------------------------------------------------------------
# Shape Verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing VALL-E shapes...")

    model = VALLE(
        vocab_size=256,
        audio_codebook_size=256,
        dim=256,
        num_heads=4,
        ar_layers=4,
        nar_layers=4,
        num_levels=4,
    )

    B, T_text, T_audio = 2, 15, 50

    text_ids = torch.randint(0, 256, (B, T_text))
    audio_codes = torch.randint(0, 256, (B, 4, T_audio))

    # AR forward
    ar_logits = model.forward_ar(text_ids, audio_codes)
    print(f"AR logits:  {ar_logits.shape}")    # (B, T_audio, 256)

    # NAR forward (level 1)
    nar_logits = model.forward_nar(text_ids, audio_codes, target_level=1)
    print(f"NAR logits: {nar_logits.shape}")    # (B, T_audio, 256)

    # Generation
    prompt_codes = audio_codes[:, :, :20]  # use first 20 frames as prompt
    generated = model.generate(
        text_ids, prompt_codes, max_new_tokens=30, temperature=1.0,
    )
    print(f"Generated:  {generated.shape}")    # (B, 4, 50)  = 20 prompt + 30 new

    total_params = sum(p.numel() for p in model.parameters())
    ar_params = sum(p.numel() for p in model.ar_model.parameters())
    nar_params = sum(p.numel() for p in model.nar_model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"  AR model:  {ar_params:,}")
    print(f"  NAR model: {nar_params:,}")
    print("All shapes OK!")
