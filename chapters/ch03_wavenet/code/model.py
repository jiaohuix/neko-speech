"""
Ch03: WaveNet — 神经声码器

WaveNet 是一种自回归生成模型，用因果扩张卷积建模音频波形的条件概率分布：

    P(y) = ∏ P(y_t | y_{<t}, mel)

核心组件：
    1. 因果卷积 (Causal Conv1d) — 只看过去，不看未来
    2. 扩张卷积 (Dilated Conv1d) — 指数增长的感受野
    3. 门控激活 (Gated Activation) — tanh × sigmoid
    4. 残差 + 跳跃连接 — 深层训练稳定性

参考：
    van den Oord et al., 2016. WaveNet: A Generative Model for Raw Audio.

接口：
    输入: mel [B, n_mels, T_mel]
    输出: logits [B, n_mu_law, T_wav]   (训练)
          waveform [B, T_wav]            (推理)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------
# Causal Conv1d
# --------------------------------------------------------

class CausalConv1d(nn.Module):
    """
    因果卷积：位置 t 的输出只依赖 ≤ t 的输入。

    标准 Conv1d 是对称 padding（左右各 d*(K-1)/2），会"看到未来"。
    因果卷积改为只在左侧 padding d*(K-1)，右侧不 padding。

        input:   [x0, x1, x2, x3, x4]
        pad(2d): [0, ..., 0, x0, x1, x2, x3, x4]
        conv:    [y0, y1, y2, y3, y4]

    y[0] 只看 padding 和 x[0]（过去），不会看到 x[1]（未来）。
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        # x: (B, C, T)
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))   # 左侧补零
        out = self.conv(x)                # 输出长度 = 输入长度
        return out


# --------------------------------------------------------
# Residual Block
# --------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    WaveNet 残差块：

        wav_in ─→ [Dilated Causal Conv] ─→ split ─→ tanh ─┐
        mel_cond ─→ [1×1 Conv] ──────────→ split ─→ sig  ─┤×─→ h
                                                            │
                                    h ─→ [1×1] ─→ residual ─→ + ─→ wav_out
                                    h ─→ [1×1] ─→ skip ─────────→ skip_out

    门控激活 (Gated Activation):
        z = tanh(W_f * x + V_f * c) × sigmoid(W_g * x + V_g * c)

    tanh 是信息通道（-1 到 1），sigmoid 是门（0 到 1）。
    比 ReLU 更有表达力：可以同时放大和抑制信号。
    """

    def __init__(self, res_channels, skip_channels, mel_channels, dilation):
        super().__init__()
        # 扩张因果卷积：输出 2×res_channels（前一半 filter，后一半 gate）
        self.dil_conv = CausalConv1d(
            res_channels, 2 * res_channels, kernel_size=3, dilation=dilation
        )
        # Mel 条件注入（1×1 卷积，同样输出 2×res_channels）
        self.mel_proj = nn.Conv1d(mel_channels, 2 * res_channels, 1)
        # 残差输出投影
        self.res_proj = nn.Conv1d(res_channels, res_channels, 1)
        # 跳跃连接输出投影
        self.skip_proj = nn.Conv1d(res_channels, skip_channels, 1)

    def forward(self, x, mel_cond):
        """
        Args:
            x:       (B, res_channels, T) — 波形特征
            mel_cond: (B, mel_channels, T) — 上采样后的 mel 条件

        Returns:
            residual: (B, res_channels, T)
            skip:     (B, skip_channels, T)
        """
        # 扩张因果卷积 + mel 条件
        h = self.dil_conv(x) + self.mel_proj(mel_cond)  # (B, 2*res, T)

        # 门控激活：tanh × sigmoid
        h_filter, h_gate = h.chunk(2, dim=1)
        h = torch.tanh(h_filter) * torch.sigmoid(h_gate)  # (B, res, T)

        # 残差 + 跳跃
        residual = self.res_proj(h)  # (B, res, T)
        skip = self.skip_proj(h)     # (B, skip, T)

        return x + residual, skip


# --------------------------------------------------------
# WaveNet
# --------------------------------------------------------

class WaveNet(nn.Module):
    """
    WaveNet 声码器：Mel 频谱 → 波形。

    结构:
        Mel → Upsample (TransposedConv) → mel_condition (B, mel_ch, T_wav)
        Wav → Input Projection → [ResBlock_1, ..., ResBlock_N]
              → sum(skip_i) → ReLU → 1×1 → 1×1 → logits

    感受野计算（多层 cycle）:
        RF = num_cycles × (2^blocks_per_cycle - 1) × (K-1) + 1

    Args:
        n_mels:        Mel 频谱维度（默认 80）
        res_channels:  残差通道数
        skip_channels: 跳跃连接通道数
        n_blocks:      每个 cycle 的残差块数
        n_cycles:      cycle 重复次数
        n_mu_law:      mu-law 量化级数（256 → 8-bit）
        hop_length:    帧移（mel 帧间距，用于上采样）
    """

    def __init__(
        self,
        n_mels=80,
        res_channels=64,
        skip_channels=128,
        n_blocks=10,
        n_cycles=3,
        n_mu_law=256,
        hop_length=256,
    ):
        super().__init__()
        self.n_mu_law = n_mu_law
        self.hop_length = hop_length
        self.res_channels = res_channels

        # Mel 上采样：多级 TransposedConv 把 T_mel → T_mel × hop_length
        # 用 16×16=256 替代单次 stride=256，参数量从 3.3M 降到 ~52K
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(n_mels, n_mels, kernel_size=32, stride=16, padding=8),
            nn.ConvTranspose1d(n_mels, n_mels, kernel_size=32, stride=16, padding=8),
        )

        # 波形输入投影
        self.input_proj = nn.Conv1d(n_mu_law, res_channels, 1)

        # 残差块（多个 cycle，每个 cycle 内 dilation 指数增长）
        self.blocks = nn.ModuleList()
        for c in range(n_cycles):
            for i in range(n_blocks):
                dilation = 2 ** i
                self.blocks.append(
                    ResidualBlock(res_channels, skip_channels, n_mels, dilation)
                )

        # 输出头：sum(skips) → 2-layer 1×1 conv → n_mu_law classes
        self.skip_proj = nn.Conv1d(skip_channels, skip_channels, 1)
        self.output_proj = nn.Conv1d(skip_channels, n_mu_law, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mel, wav):
        """
        训练前向传播（Teacher Forcing）。

        Args:
            mel: (B, n_mels, T_mel) — 条件 Mel 频谱
            wav: (B, T_wav) — mu-law 编码的波形，值域 [0, n_mu_law)

        Returns:
            logits: (B, n_mu_law, T_wav) — 每个时间步的类别 logits
        """
        T_wav = wav.shape[1]

        # 上采样 mel → 波形分辨率
        mel_up = self.upsample(mel)                  # (B, n_mels, ~T_wav)
        if mel_up.shape[2] > T_wav:
            mel_up = mel_up[:, :, :T_wav]
        elif mel_up.shape[2] < T_wav:
            mel_up = F.pad(mel_up, (0, T_wav - mel_up.shape[2]))

        # One-hot 波形输入
        wav_onehot = F.one_hot(wav.long(), self.n_mu_law).float()  # (B, T, 256)
        x = wav_onehot.transpose(1, 2)                            # (B, 256, T)
        x = self.input_proj(x)                                    # (B, res, T)

        # 残差块链
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x, mel_up)
            skip_sum = skip_sum + skip

        # 输出
        out = F.relu(skip_sum)
        out = F.relu(self.skip_proj(out))
        logits = self.output_proj(out)  # (B, n_mu_law, T)

        return logits

    @torch.no_grad()
    def generate(self, mel, n_samples, temperature=1.0):
        """
        自回归生成（教学版，逐步生成，非常慢）。

        每一步：
        1. 把已生成的波形通过整个网络
        2. 取最后一个时间步的输出分布
        3. 采样下一个采样点

        Args:
            mel: (1, n_mels, T_mel)
            n_samples: 生成采样点数
            temperature: 采样温度（<1 更确定，>1 更随机）

        Returns:
            waveform: (n_samples,) — mu-law 编码值 [0, 255]
        """
        self.eval()
        device = mel.device

        mel_up = self.upsample(mel)
        if mel_up.shape[2] > n_samples:
            mel_up = mel_up[:, :, :n_samples]
        elif mel_up.shape[2] < n_samples:
            mel_up = F.pad(mel_up, (0, n_samples - mel_up.shape[2]))

        # 从中间值（mu-law 128 ≈ 0.0）开始
        samples = torch.full((1, n_samples), self.n_mu_law // 2,
                             dtype=torch.long, device=device)

        for t in range(n_samples):
            wav_slice = samples[:, :t + 1]
            mel_slice = mel_up[:, :, :t + 1]
            logits = self.forward(mel_slice, wav_slice)   # (1, 256, t+1)
            logits_t = logits[:, :, -1] / temperature      # (1, 256)
            probs = F.softmax(logits_t, dim=-1)
            next_sample = torch.multinomial(probs, 1).squeeze(-1)
            if t + 1 < n_samples:
                samples[:, t + 1] = next_sample

        return samples.squeeze(0)


# --------------------------------------------------------
# mu-law encoding / decoding
# --------------------------------------------------------

def mu_law_encode(waveform, n_mu_law=256):
    """
    波形 [-1, 1] → mu-law 编码 [0, n_mu_law).

    公式：f(x) = sgn(x) · ln(1 + μ|x|) / ln(1 + μ)

    mu-law 压缩大振幅、保留小振幅细节，适合语音的动态范围。
    """
    mu = n_mu_law - 1
    x = waveform.clamp(-1.0, 1.0)
    encoded = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log(torch.tensor(mu + 1.0))
    quantized = ((encoded + 1.0) / 2.0 * mu + 0.5).long()
    return quantized.clamp(0, mu)


def mu_law_decode(encoded, n_mu_law=256):
    """mu-law 编码 [0, n_mu_law) → 波形 [-1, 1]."""
    mu = n_mu_law - 1
    x = 2.0 * encoded.float() / mu - 1.0
    decoded = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log(torch.tensor(mu + 1.0))) - 1.0) / mu
    return decoded


# --------------------------------------------------------
# Test: Shape verification
# --------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Ch03 WaveNet — Shape Verification")
    print("=" * 60)

    # Config
    B, n_mels = 2, 80
    T_mel = 20
    hop_length = 256
    n_mu_law = 256
    n_blocks, n_cycles = 10, 3

    model = WaveNet(
        n_mels=n_mels, res_channels=64, skip_channels=128,
        n_blocks=n_blocks, n_cycles=n_cycles,
        n_mu_law=n_mu_law, hop_length=hop_length,
    )

    # -- Test 1: Upsample --
    T_wav = T_mel * hop_length
    mel = torch.randn(B, n_mels, T_mel)
    wav = torch.randint(0, n_mu_law, (B, T_wav))

    mel_up = model.upsample(mel)
    print(f"\n[1] Upsample:")
    print(f"    Input:  {list(mel.shape)}")
    print(f"    Output: {list(mel_up.shape)}  (expect ~{T_wav})")

    # -- Test 2: Causal property --
    print(f"\n[2] Causal property:")
    cconv = CausalConv1d(n_mu_law, 32, kernel_size=3, dilation=4)
    wav_test = torch.randn(1, n_mu_law, 50)
    out_full = cconv(wav_test)
    wav_trunc = wav_test.clone()
    wav_trunc[:, :, 40:] = 0
    out_trunc = cconv(wav_trunc)
    # Position 20 should be unaffected (RF = 20 + 2*4 = 28 < 40)
    print(f"    Position 20 unchanged: {torch.allclose(out_full[:, :, 20], out_trunc[:, :, 20])}")
    # Position 45 should be affected (45 >= 40, within RF)
    print(f"    Position 45 changed:   {not torch.allclose(out_full[:, :, 45], out_trunc[:, :, 45])}")

    # -- Test 3: Forward pass --
    logits = model(mel, wav)
    print(f"\n[3] Forward pass:")
    print(f"    Mel:    {list(mel.shape)}")
    print(f"    Wav:    {list(wav.shape)}")
    print(f"    Logits: {list(logits.shape)}  (expect [2, 256, {T_wav}])")
    assert logits.shape == (B, n_mu_law, T_wav), f"Shape mismatch! {logits.shape}"

    # -- Test 4: Autoregressive generation (tiny) --
    print(f"\n[4] Autoregressive generation:")
    gen_n = 64
    gen = model.generate(mel[:1, :, :2], n_samples=gen_n, temperature=1.0)
    print(f"    Generated: {list(gen.shape)}  (expect [{gen_n}])")
    print(f"    Value range: [{gen.min().item()}, {gen.max().item()}]")
    assert gen.shape == (gen_n,)

    # -- Test 5: mu-law round-trip --
    print(f"\n[5] mu-law encode/decode:")
    x = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0])
    enc = mu_law_encode(x)
    dec = mu_law_decode(enc)
    print(f"    Original:  {x.tolist()}")
    print(f"    Encoded:   {enc.tolist()}")
    print(f"    Decoded:   {[round(v, 3) for v in dec.tolist()]}")
    print(f"    Max error: {(x - dec).abs().max().item():.4f}")

    # -- Receptive field --
    rf = n_cycles * (2**n_blocks - 1) * 2 + 1
    print(f"\n[6] Receptive field: {rf:,} samples ({rf / 16000:.3f}s @ 16kHz)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {total_params:,}")

    print(f"\n{'=' * 60}")
    print("All shapes OK!")
