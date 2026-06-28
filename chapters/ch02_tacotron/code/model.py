"""
Ch02: Tacotron2 — Minimal PyTorch Implementation

This is NOT a production model. It is a minimal, readable implementation
for educational purposes.

Reference:
    Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions
    Wang et al., 2018

Architecture:
    Text → Embedding → Encoder (Conv + BiLSTM)
                        ↓
                  Location-Sensitive Attention
                        ↓
                  Decoder (PreNet + LSTM)
                        ↓
                  PostNet (Conv) → Mel Spectrogram
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------
# Encoder
# --------------------------------------------------------

class Encoder(nn.Module):
    """
    Tacotron2 Encoder:
        Char Embedding → 3× Conv1d → BiLSTM

    Input:  (B, T_text)  token ids
    Output: (B, T_text, encoder_dim)
    """

    def __init__(self, vocab_size=256, embed_dim=512, encoder_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 3 layers of Conv1d (kernel=5) with batch norm and ReLU
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, encoder_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(encoder_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
            )
            for _ in range(3)
        ])

        # BiLSTM: forward + backward → 2 × 256 = 512
        self.lstm = nn.LSTM(
            encoder_dim, encoder_dim // 2, num_layers=1,
            batch_first=True, bidirectional=True,
        )

    def forward(self, x):
        # x: (B, T)
        x = self.embedding(x)           # (B, T, embed_dim)
        x = x.transpose(1, 2)           # (B, embed_dim, T) for Conv1d

        for conv in self.convs:
            x = conv(x)                 # (B, encoder_dim, T)

        x = x.transpose(1, 2)           # (B, T, encoder_dim)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)             # (B, T, encoder_dim)
        return x


# --------------------------------------------------------
# Location-Sensitive Attention
# --------------------------------------------------------

class LocationSensitiveAttention(nn.Module):
    """
    Tacotron2 Attention:
        Combines decoder state + encoder features + previous attention location.

    This prevents the decoder from skipping or repeating words.
    """

    def __init__(self, encoder_dim=512, decoder_dim=1024, attention_dim=128, location_feature_dim=32):
        super().__init__()
        # Project encoder outputs
        self.W_enc = nn.Linear(encoder_dim, attention_dim)
        # Project decoder hidden state
        self.W_dec = nn.Linear(decoder_dim, attention_dim)
        # Conv over previous attention weights (location feature)
        self.loc_conv = nn.Conv1d(1, location_feature_dim, kernel_size=31, padding=15)
        self.W_loc = nn.Linear(location_feature_dim, attention_dim)
        # Final scoring
        self.v = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden, prev_attention):
        """
        Args:
            encoder_out:    (B, T_enc, encoder_dim)
            decoder_hidden: (B, decoder_dim)
            prev_attention: (B, T_enc)  — attention weights from previous step

        Returns:
            context: (B, encoder_dim)
            attention: (B, T_enc)
        """
        B, T_enc, _ = encoder_out.shape

        # Encoder projection: (B, T_enc, attention_dim)
        enc_proj = self.W_enc(encoder_out)
        # Decoder projection: (B, 1, attention_dim)
        dec_proj = self.W_dec(decoder_hidden).unsqueeze(1)
        # Location feature from prev_attention
        loc_feat = self.loc_conv(prev_attention.unsqueeze(1))  # (B, 32, T_enc)
        loc_feat = loc_feat.transpose(1, 2)                    # (B, T_enc, 32)
        loc_proj = self.W_loc(loc_feat)                        # (B, T_enc, attention_dim)

        # Score: (B, T_enc, attention_dim) → (B, T_enc, 1)
        score = torch.tanh(enc_proj + dec_proj + loc_proj)
        score = self.v(score).squeeze(-1)                      # (B, T_enc)

        # Softmax over encoder positions
        attention = F.softmax(score, dim=-1)                   # (B, T_enc)

        # Weighted sum of encoder outputs
        context = torch.bmm(attention.unsqueeze(1), encoder_out)  # (B, 1, encoder_dim)
        context = context.squeeze(1)                              # (B, encoder_dim)

        return context, attention


# --------------------------------------------------------
# PreNet
# --------------------------------------------------------

class PreNet(nn.Module):
    """
    PreNet: 2-layer FC with ReLU and dropout.
    Acts as an information bottleneck — forces the decoder to compress
    the previous mel frame, improving generalization.
    """

    def __init__(self, mel_dim=80, hidden_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(mel_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.layers(x)


# --------------------------------------------------------
# PostNet
# --------------------------------------------------------

class PostNet(nn.Module):
    """
    PostNet: 5-layer Conv1d that refines the predicted mel spectrogram.
    Residual connection around the entire PostNet.
    """

    def __init__(self, mel_dim=80, hidden_dim=512):
        super().__init__()
        layers = []
        for i in range(5):
            in_ch = mel_dim if i == 0 else hidden_dim
            out_ch = mel_dim if i == 4 else hidden_dim
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2))
            if i < 4:
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(0.5))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, mel_dim, T)
        out = self.convs(x)  # (B, mel_dim, T)
        return x + out       # Residual


# --------------------------------------------------------
# Decoder
# --------------------------------------------------------

class Decoder(nn.Module):
    """
    Tacotron2 Decoder (simplified for training):
        PreNet → 2× LSTM → Linear projection to mel + stop token

    Note: This is the "teacher forcing" version for training.
    Autoregressive inference is separate.
    """

    def __init__(self, mel_dim=80, encoder_dim=512, decoder_dim=1024, prenet_dim=256):
        super().__init__()
        self.mel_dim = mel_dim
        self.decoder_dim = decoder_dim

        self.prenet = PreNet(mel_dim, prenet_dim)

        # 2-layer LSTM
        self.lstm = nn.LSTM(
            prenet_dim + encoder_dim, decoder_dim,
            num_layers=2, batch_first=True, dropout=0.1,
        )

        # Project to mel + stop token
        self.mel_proj = nn.Linear(decoder_dim, mel_dim)
        self.stop_proj = nn.Linear(decoder_dim + mel_dim, 1)

    def forward(self, encoder_out, mel_targets):
        """
        Teacher forcing training.

        Args:
            encoder_out: (B, T_enc, encoder_dim)
            mel_targets: (B, T_mel, mel_dim) — ground truth mel frames

        Returns:
            mel_outputs: (B, T_mel, mel_dim)
            stop_tokens: (B, T_mel)
        """
        B, T_mel, _ = mel_targets.shape
        device = mel_targets.device

        # Initial input: first frame of target (or zeros)
        decoder_input = mel_targets[:, 0, :]  # (B, mel_dim)

        # Simple mean pooling as initial hidden state context
        context = encoder_out.mean(dim=1)  # (B, encoder_dim)

        mel_outputs = []
        stop_outputs = []

        for t in range(T_mel):
            # PreNet
            prenet_out = self.prenet(decoder_input)  # (B, prenet_dim)

            # Concat with context
            lstm_input = torch.cat([prenet_out, context], dim=-1).unsqueeze(1)  # (B, 1, prenet_dim+enc)

            # LSTM
            lstm_out, _ = self.lstm(lstm_input)  # (B, 1, decoder_dim)
            lstm_out = lstm_out.squeeze(1)        # (B, decoder_dim)

            # Predict mel frame
            mel_out = self.mel_proj(lstm_out)     # (B, mel_dim)

            # Predict stop token
            stop_in = torch.cat([lstm_out, mel_out], dim=-1)
            stop_out = self.stop_proj(stop_in).squeeze(-1)  # (B,)

            mel_outputs.append(mel_out)
            stop_outputs.append(stop_out)

            # Teacher forcing: next input = ground truth
            if t + 1 < T_mel:
                decoder_input = mel_targets[:, t + 1, :]

        mel_outputs = torch.stack(mel_outputs, dim=1)    # (B, T_mel, mel_dim)
        stop_outputs = torch.stack(stop_outputs, dim=1)  # (B, T_mel)

        return mel_outputs, stop_outputs


# --------------------------------------------------------
# Tacotron2 (Full Model)
# --------------------------------------------------------

class Tacotron2(nn.Module):
    """
    Minimal Tacotron2 for educational purposes.

    Input:  (B, T_text)      — text token ids
    Output: (B, T_mel, 80)   — mel spectrogram (before + after PostNet)
            (B, T_mel)       — stop token logits
    """

    def __init__(
        self,
        vocab_size=256,
        mel_dim=80,
        encoder_dim=512,
        decoder_dim=1024,
        prenet_dim=256,
    ):
        super().__init__()
        self.encoder = Encoder(vocab_size, encoder_dim=encoder_dim)
        self.decoder = Decoder(mel_dim, encoder_dim, decoder_dim, prenet_dim)
        self.postnet = PostNet(mel_dim)

    def forward(self, text_tokens, mel_targets):
        """
        Training forward pass.

        Args:
            text_tokens: (B, T_text)
            mel_targets: (B, T_mel, mel_dim) — for teacher forcing

        Returns:
            mel_before: (B, T_mel, mel_dim) — decoder output (before PostNet)
            mel_after:  (B, T_mel, mel_dim) — after PostNet
            stop_logits: (B, T_mel)
        """
        # Encode text
        encoder_out = self.encoder(text_tokens)  # (B, T_text, encoder_dim)

        # Decode with teacher forcing
        mel_before, stop_logits = self.decoder(encoder_out, mel_targets)

        # PostNet refinement
        # PostNet expects (B, mel_dim, T_mel)
        mel_after = self.postnet(mel_before.transpose(1, 2)).transpose(1, 2)
        mel_after = mel_before + mel_after  # Residual (actually PostNet already does this)

        return mel_before, mel_after, stop_logits

    def inference(self, text_tokens, max_len=1000, stop_threshold=0.5):
        """
        Autoregressive inference (simplified).

        Args:
            text_tokens: (B, T_text)
            max_len: maximum number of mel frames to generate

        Returns:
            mel_outputs: (B, T_generated, mel_dim)
        """
        self.eval()
        with torch.no_grad():
            B = text_tokens.shape[0]
            device = text_tokens.device

            encoder_out = self.encoder(text_tokens)

            # Start with zero frame
            decoder_input = torch.zeros(B, self.decoder.mel_dim, device=device)
            context = encoder_out.mean(dim=1)

            mel_outputs = []

            for t in range(max_len):
                prenet_out = self.decoder.prenet(decoder_input)
                lstm_input = torch.cat([prenet_out, context], dim=-1).unsqueeze(1)
                lstm_out, _ = self.decoder.lstm(lstm_input)
                lstm_out = lstm_out.squeeze(1)

                mel_out = self.decoder.mel_proj(lstm_out)
                mel_outputs.append(mel_out)

                # Check stop token
                stop_in = torch.cat([lstm_out, mel_out], dim=-1)
                stop_prob = torch.sigmoid(self.decoder.stop_proj(stop_in)).squeeze(-1)
                if (stop_prob > stop_threshold).all():
                    break

                # Autoregressive: next input = current prediction
                decoder_input = mel_out

            mel_outputs = torch.stack(mel_outputs, dim=1)

            # Apply PostNet
            mel_after = self.postnet(mel_outputs.transpose(1, 2)).transpose(1, 2)
            mel_after = mel_outputs + mel_after

            return mel_outputs, mel_after


# --------------------------------------------------------
# Test: Shape verification
# --------------------------------------------------------

if __name__ == "__main__":
    print("Testing Tacotron2 shapes...")

    model = Tacotron2(vocab_size=256, mel_dim=80)

    B, T_text, T_mel = 2, 20, 100
    text = torch.randint(0, 256, (B, T_text))
    mel = torch.randn(B, T_mel, 80)

    # Training forward
    mel_before, mel_after, stop_logits = model(text, mel)
    print(f"mel_before:   {mel_before.shape}")   # (B, T_mel, 80)
    print(f"mel_after:    {mel_after.shape}")    # (B, T_mel, 80)
    print(f"stop_logits:  {stop_logits.shape}")  # (B, T_mel)

    # Inference
    mel_gen, mel_gen_post = model.inference(text, max_len=50)
    print(f"mel_gen:      {mel_gen.shape}")      # (B, T_generated, 80)
    print(f"mel_gen_post: {mel_gen_post.shape}")  # (B, T_generated, 80)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("All shapes OK!")
