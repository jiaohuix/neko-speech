"""
Ch05 VITS — 子模块库

VITS (Conditional Variational Autoencoder with Adversarial Learning
for End-to-End Text-to-Speech) 的所有子模块。

Reference:
    Kim et al., 2021. "Conditional Variational Autoencoder with
    Adversarial Learning for End-to-End Text-to-Speech"

Components:
    1. TextEncoder       — 文本 → (μ_p, log σ_p)  先验分布参数
    2. PosteriorEncoder  — 频谱 → (μ_q, log σ_q)  后验分布参数
    3. Flow              — 可逆变换 z_q → z_p     归一化流
    4. ResBlock          — HiFi-GAN 残差块
    5. Generator         — z → 波形 (HiFi-GAN Decoder)
    6. DurationPredictor — 时长预测 (stochastic)
    7. Discriminator     — 多周期判别器 (MPD)
    8. StochasticDurationPredictor — 随机时长预测器 (简化版)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm


# --------------------------------------------------------
# 工具函数
# --------------------------------------------------------

def init_weights(m, mean=0.0, std=0.01):
    """正态分布初始化权重"""
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()


def get_padding(kernel_size, dilation=1):
    """计算 'same' padding"""
    return (kernel_size * dilation - dilation) // 2


LRELU_SLOPE = 0.1


# --------------------------------------------------------
# 1. TextEncoder — 文本 → 先验分布参数
# --------------------------------------------------------

class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码 (Shaw et al., 2018)

    与绝对位置编码不同，相对编码只关心 token 之间的距离。
    这使得模型对不同长度的序列泛化更好。

    实现方式：为每个相对位置 [-clip, +clip] 学习一个 embedding。
    """

    def __init__(self, dim, max_len=5000, clip=4):
        super().__init__()
        self.dim = dim
        self.clip = clip
        # 2*clip+1 个相对位置：-clip, ..., 0, ..., +clip
        self.emb = nn.Embedding(2 * clip + 1, dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, length):
        """
        Args:
            length: 序列长度 T

        Returns:
            (T, T, dim) 相对位置编码矩阵
        """
        positions = torch.arange(length, device=self.emb.weight.device)
        relative = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        relative = relative.clamp(-self.clip, self.clip) + self.clip
        return self.emb(relative)  # (T, T, dim)


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力 (带相对位置编码)

    标准 MHA + 相对位置偏置。
    """

    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, rel_pos_enc=None, mask=None):
        """
        Args:
            x: (B, T, dim)
            rel_pos_enc: (T, T, dim) 相对位置编码
            mask: (B, 1, T) 或 None

        Returns:
            (B, T, dim)
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, H, T, D)

        # 标准注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # 加上相对位置偏置
        if rel_pos_enc is not None:
            # rel_pos_enc: (T, T, dim) → (T, T, H, D) → (H, T, T, D)
            rel = rel_pos_enc.view(T, T, self.n_heads, self.head_dim)
            rel = rel.permute(2, 0, 1, 3)  # (H, T, T, D)
            # q: (B, H, T, D) → 计算 q 与相对位置的点积
            rel_score = torch.einsum("bhtd,htsd->bhts", q, rel) * self.scale
            attn = attn + rel_score

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.dim)
        return self.out_proj(out)


class FFN(nn.Module):
    """前馈网络 (Position-wise Feed-Forward)"""

    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class EncoderBlock(nn.Module):
    """
    Transformer 编码块 (Pre-Norm 风格)

    LayerNorm → Self-Attention → Residual
    LayerNorm → FFN           → Residual
    """

    def __init__(self, dim, n_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, ffn_dim)

    def forward(self, x, rel_pos_enc=None, mask=None):
        x = x + self.attn(self.norm1(x), rel_pos_enc, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TextEncoder(nn.Module):
    """
    文本编码器：文本序列 → 先验分布参数 (μ_p, log σ_p)

    架构：
        Token Embedding
            ↓
        相对位置编码 + N × Transformer Encoder Block
            ↓
        线性投影 → (μ_p, log σ_p)

    输入：text_ids (B, T_text)
    输出：x (B, hidden, T_text), μ_p (B, hidden, T_text),
          log σ_p (B, hidden, T_text), text_mask (B, 1, T_text)

    设计选择：
    - 使用 Transformer 而非 BiLSTM，因为 VITS 需要并行计算
    - 相对位置编码提高对不同长度文本的泛化能力
    - 输出 μ 和 log σ 参数化一个对角高斯先验
    """

    def __init__(
        self,
        vocab_size,
        hidden_dim=192,
        ffn_dim=768,
        n_heads=2,
        n_layers=6,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        nn.init.normal_(self.emb.weight, std=0.02)

        self.pos_enc = RelativePositionalEncoding(hidden_dim)
        self.layers = nn.ModuleList([
            EncoderBlock(hidden_dim, n_heads, ffn_dim)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim * 2)  # → μ_p, log σ_p
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_ids, text_lengths=None):
        """
        Args:
            text_ids: (B, T_text) 字符 token ids
            text_lengths: (B,) 每个样本的实际长度

        Returns:
            x:     (B, hidden, T_text)  编码器输出特征
            m_p:   (B, hidden, T_text)  先验均值
            logs_p: (B, hidden, T_text) 先验 log 标准差
            mask:  (B, 1, T_text)       有效位置 mask
        """
        B, T = text_ids.shape
        device = text_ids.device

        # Embedding + Dropout
        x = self.emb(text_ids) * math.sqrt(self.hidden_dim)
        x = self.dropout(x)  # (B, T, hidden)

        # 构造 mask
        if text_lengths is not None:
            mask = torch.arange(T, device=device).unsqueeze(0) < text_lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).float()  # (B, 1, T)
        else:
            mask = torch.ones(B, 1, T, device=device)

        # 相对位置编码
        rel_pos = self.pos_enc(T)  # (T, T, hidden)

        # Transformer blocks
        for layer in self.layers:
            x = layer(x, rel_pos, mask)

        x = self.norm(x)
        x = x * mask.transpose(1, 2)  # (B, T, hidden)

        # 投影到先验参数
        stats = self.proj(x)  # (B, T, 2*hidden)
        m_p, logs_p = stats.chunk(2, dim=-1)

        # 转置为 (B, hidden, T) — 适配后续模块
        x = x.transpose(1, 2)
        m_p = m_p.transpose(1, 2)
        logs_p = logs_p.transpose(1, 2)

        return x, m_p, logs_p, mask


# --------------------------------------------------------
# 2. PosteriorEncoder — 频谱 → 后验分布参数
# --------------------------------------------------------

class WNGatedConvBlock(nn.Module):
    """
    WaveNet 风格的门控卷积块

    与标准残差卷积不同，门控机制让模型学习"信息闸门"：
    output = tanh(conv_a) * sigmoid(conv_b)

    这使得模型可以选择性地传递或阻断信息，比普通 ReLU 更有表达力。
    """

    def __init__(self, hidden_dim, kernel_size, dilation):
        super().__init__()
        self.dilated_conv = weight_norm(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size,
                      dilation=dilation, padding=get_padding(kernel_size, dilation))
        )
        self.res_proj = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.skip_proj = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, x):
        """
        Args: x (B, hidden, T)
        Returns: (residual_output, skip_output)
        """
        residual = x
        h = self.dilated_conv(x)
        h_a, h_b = h.chunk(2, dim=1)
        h = torch.tanh(h_a) * torch.sigmoid(h_b)  # 门控激活

        res = self.res_proj(h) + residual  # 残差输出
        skip = self.skip_proj(h)           # 跳跃连接
        return res, skip


class PosteriorEncoder(nn.Module):
    """
    后验编码器：线性频谱 → 后验分布参数 (μ_q, log σ_q)

    架构：
        Linear Spectrogram (B, spec_channels, T_spec)
            ↓
        Conv1d 投影到 hidden_dim
            ↓
        N × WaveNet 门控卷积块 (膨胀卷积，指数递增 dilation)
            ↓
        Conv1d → (μ_q, log σ_q)

    关键设计：
    - 仅在训练时使用（推理时不需要）
    - 输入是线性频谱（非 mel），保留更多频率细节
    - 膨胀卷积的 dilation 按 2^k 递增，感受野指数增长
    - 跳跃连接汇总所有层的输出，避免信息丢失
    """

    def __init__(
        self,
        spec_channels=513,
        hidden_dim=192,
        kernel_size=5,
        n_layers=16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pre_conv = nn.Conv1d(spec_channels, hidden_dim, 1)

        self.blocks = nn.ModuleList([
            WNGatedConvBlock(hidden_dim, kernel_size, dilation=2 ** (i % 4))
            for i in range(n_layers)
        ])

        self.post_conv = nn.Conv1d(hidden_dim, hidden_dim * 2, 1)

    def forward(self, spec, spec_lengths=None):
        """
        Args:
            spec: (B, spec_channels, T_spec)  线性频谱
            spec_lengths: (B,)

        Returns:
            z:    (B, hidden, T_spec)   采样的隐变量
            m_q:  (B, hidden, T_spec)   后验均值
            logs_q: (B, hidden, T_spec) 后验 log 标准差
            mask: (B, 1, T_spec)
        """
        B, _, T = spec.shape
        device = spec.device

        if spec_lengths is not None:
            mask = torch.arange(T, device=device).unsqueeze(0) < spec_lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).float()
        else:
            mask = torch.ones(B, 1, T, device=device)

        x = self.pre_conv(spec) * mask

        # WaveNet blocks with skip connections
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
            x = x * mask

        # 后验参数
        stats = self.post_conv(skip_sum * mask)
        m_q, logs_q = stats.chunk(2, dim=1)

        # 限制 logs_q 范围，防止数值不稳定
        logs_q = torch.clamp(logs_q, min=-10.0, max=10.0)

        # 重参数化采样: z = μ + σ * ε,  ε ~ N(0, I)
        eps = torch.randn_like(m_q) * mask
        z = m_q + torch.exp(logs_q) * eps

        return z, m_q, logs_q, mask


# --------------------------------------------------------
# 3. Flow — 归一化流 (可逆变换)
# --------------------------------------------------------

class AffineCouplingLayer(nn.Module):
    """
    仿射耦合层 (Dinh et al., 2017)

    归一化流的核心思想：通过一系列可逆变换，
    将简单分布（高斯）逐步变换为复杂分布。

    仿射耦合层的巧妙之处：
    1. 将输入分成两半 (x₁, x₂)
    2. x₁ 保持不变
    3. x₂ 被 x₁ 的函数变换: z₂ = (x₂ - t(x₁)) * exp(-s(x₁))
    4. 这样雅可比矩阵是三角的 → 行列式可以高效计算

    可逆性保证了：
    - 前向 (x → z): 训练时计算后验到先验的变换
    - 逆向 (z → x): 推理时从先验采样得到解码器输入
    """

    def __init__(self, channels, hidden_dim=256, kernel_size=5, n_layers=4):
        super().__init__()
        self.half = channels // 2

        # 神经网络：从 x₁ 预测仿射变换参数 (scale s, shift t)
        layers = [
            nn.Conv1d(self.half, hidden_dim, 1),
            nn.ReLU(),
        ]
        for i in range(n_layers - 1):
            layers.extend([
                weight_norm(nn.Conv1d(
                    hidden_dim, hidden_dim, kernel_size,
                    padding=get_padding(kernel_size, 1)
                )),
                nn.ReLU(),
            ])
        layers.append(nn.Conv1d(hidden_dim, self.half * 2, 1))
        self.net = nn.Sequential(*layers)

        # 初始化最后一层为 0 → 初始时变换接近恒等映射
        # 这使得训练初期 Flow 几乎不改变输入，逐步学习变换
        last_conv = self.net[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(self, x, mask=None, reverse=False):
        """
        Args:
            x: (B, channels, T)
            mask: (B, 1, T)
            reverse: True = 推理 (z → x), False = 训练 (x → z)

        Returns:
            y: 变换后的张量
            log_det: log |det(∂z/∂x)| — 对数行列式（KL 散度需要）
        """
        x1, x2 = x[:, :self.half], x[:, self.half:]

        h = self.net(x1)
        s, t = h[:, :self.half], h[:, self.half:]

        # 限制 scale 范围，防止数值爆炸
        s = torch.clamp(s, min=-5.0, max=5.0)

        if mask is not None:
            s = s * mask
            t = t * mask

        if not reverse:
            # 前向 (训练): x → z
            z2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=[1, 2])  # 对数行列式
        else:
            # 逆向 (推理): z → x
            z2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=[1, 2])

        y = torch.cat([x1, z2], dim=1)
        return y, log_det


class Flow(nn.Module):
    """
    归一化流 (Normalizing Flow)

    多个仿射耦合层的堆叠，交替反转通道顺序。
    每次反转确保所有维度都有机会被其他维度变换。

    为什么需要 Flow？
    - TextEncoder 输出的是高斯先验（简单分布）
    - 真实语音的隐空间可能非常复杂（多峰、弯曲）
    - Flow 提供了可学习的可逆变换，桥接简单分布和复杂分布
    - 类似"弯曲空间的坐标变换"

    架构：K 个 (AffineCoupling + Flip) 交替堆叠
    """

    def __init__(self, channels, hidden_dim=256, kernel_size=5, n_layers=4, n_flows=4):
        super().__init__()
        self.flows = nn.ModuleList([
            AffineCouplingLayer(channels, hidden_dim, kernel_size, n_layers)
            for _ in range(n_flows)
        ])

    def forward(self, x, mask=None, reverse=False):
        """
        Args:
            x: (B, channels, T)
            mask: (B, 1, T)
            reverse: True = 推理模式

        Returns:
            x: 变换后的张量
            log_det_sum: 所有层的 log |det| 之和
        """
        log_det_sum = 0

        if not reverse:
            flows = self.flows
        else:
            flows = reversed(self.flows)

        for flow in flows:
            x, log_det = flow(x, mask, reverse)
            log_det_sum = log_det_sum + log_det

            # 反转通道（最后一层不反转）
            if flow is not (self.flows[-1] if not reverse else self.flows[0]):
                x = torch.flip(x, dims=[1])

        return x, log_det_sum


# --------------------------------------------------------
# 4-5. HiFi-GAN Generator (Decoder)
# --------------------------------------------------------

class ResBlock(nn.Module):
    """
    多感受野融合残差块 (Multi-Receptive Field Fusion)

    HiFi-GAN 的核心设计：每个 ResBlock 包含 3 组膨胀卷积，
    dilation 分别为 1, 3, 5，覆盖不同尺度的时域模式。

    多个 ResBlock 的跳跃连接累加，融合不同尺度的特征。
    """

    def __init__(self, channels, kernel_size=7, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilations:
            self.convs1.append(
                weight_norm(nn.Conv1d(
                    channels, channels, kernel_size,
                    dilation=d, padding=get_padding(kernel_size, d)
                ))
            )
            self.convs2.append(
                weight_norm(nn.Conv1d(
                    channels, channels, kernel_size,
                    dilation=1, padding=get_padding(kernel_size, 1)
                ))
            )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            residual = x
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = c1(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = c2(x)
            x = x + residual
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class Generator(nn.Module):
    """
    HiFi-GAN 生成器：隐变量 z → 波形

    架构：
        z (B, hidden_dim, T_z)
            ↓  Conv1d 预处理
            ↓  8 × [ConvTranspose1d(上采样) + ResBlock(MRF)]
            ↓  Conv1d → tanh
        waveform (B, 1, T_wav)

    上采样倍率序列：[8, 8, 2, 2, 2, 2, 2, 2]
    总上采样倍率 = 8×8×2×2×2×2×2×2 = 2048
    但通常使用 4 个上采样层：[8, 8, 2, 2] → 总倍率 256

    hop_length = 256 意味着：
    每 1 帧隐变量 → 256 个音频采样点
    16kHz / 256 = 62.5 Hz 帧率

    为什么用 tanh 输出？
    - 音频波形通常在 [-1, 1] 范围内
    - tanh 自然限制输出范围，无需额外裁剪
    """

    def __init__(
        self,
        hidden_dim=192,
        upsample_rates=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(7, 11),
        resblock_dilations=((1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()

        self.pre_conv = weight_norm(nn.Conv1d(hidden_dim, 512, 7, padding=3))

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        ch = 512
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(
                    ch, ch // 2, k, stride=u,
                    padding=(k - u) // 2
                ))
            )
            ch = ch // 2
            for rk, rd in zip(resblock_kernel_sizes, resblock_dilations):
                self.resblocks.append(ResBlock(ch, rk, rd))

        self.post_conv = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

    def forward(self, z):
        """
        Args:
            z: (B, hidden_dim, T_z) 隐变量

        Returns:
            waveform: (B, 1, T_wav)
        """
        x = self.pre_conv(z)  # (B, 512, T_z)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)  # (B, ch, T_up)

            # 所有 ResBlock 的跳跃连接求和取平均
            xs = 0
            n_rb = len(self.resblocks) // len(self.ups)
            for j in range(n_rb):
                xs = xs + self.resblocks[i * n_rb + j](x)
            x = xs / n_rb

        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.post_conv(x)
        x = torch.tanh(x)
        return x  # (B, 1, T_wav)

    def remove_weight_norm(self):
        remove_weight_norm(self.pre_conv)
        for up in self.ups:
            remove_weight_norm(up)
        for rb in self.resblocks:
            rb.remove_weight_norm()
        remove_weight_norm(self.post_conv)


# --------------------------------------------------------
# 6. DurationPredictor — 时长预测器
# --------------------------------------------------------

class DurationPredictor(nn.Module):
    """
    确定性时长预测器

    预测每个文本 token 对应的帧数（时长）。
    用于推理时展开文本编码序列到音频时间轴。

    架构：3 层 Conv1d + LayerNorm + ReLU

    为什么需要时长预测？
    - 训练时：MAS 提供 ground truth 对齐 → 直接提取时长
    - 推理时：没有音频，不知道每段文本对应多长时间
    - DurationPredictor 学习从文本特征预测时长
    """

    def __init__(self, hidden_dim=192, kernel_size=3, n_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, hidden, T_text)
            mask: (B, 1, T_text)

        Returns:
            log_dur: (B, T_text) 对数时长
        """
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = x.transpose(1, 2)  # (B, T, hidden) for LayerNorm
            x = norm(x)
            x = x.transpose(1, 2)  # (B, hidden, T)
            x = F.relu(x)
            if mask is not None:
                x = x * mask

        # (B, hidden, T) → (B, T, hidden) → (B, T, 1) → (B, T)
        log_dur = self.proj(x.transpose(1, 2)).squeeze(-1)
        return log_dur


# --------------------------------------------------------
# 7. Discriminator — 多周期判别器
# --------------------------------------------------------

class PeriodDiscriminator(nn.Module):
    """
    单周期判别器

    将 1D 波形重塑为 2D (period × subsequence)，
    然后用 Conv2d 捕获周期性模式。

    为什么重塑？
    - 语音有天然的周期性（基频 F0）
    - period=2 捕获半周期模式
    - period=3 捕获三周期模式
    - 不同 period 看到不同尺度的结构
    """

    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), stride=1, padding=(2, 0))),
        ])
        self.post_conv = weight_norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(self, x):
        """
        Args: x (B, 1, T) 波形
        Returns: score (B, 1, T'), features [list]
        """
        B, C, T = x.shape

        # 填充到 period 的倍数，然后重塑为 2D
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            x = F.pad(x, (0, pad_len))
            T = T + pad_len

        x = x.view(B, C, T // self.period, self.period)  # (B, 1, T/p, p)

        features = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            features.append(x)

        x = self.post_conv(x)
        features.append(x)

        return x.flatten(1, -1), features


class Discriminator(nn.Module):
    """
    多周期判别器 (Multi-Period Discriminator, MPD)

    组合多个不同周期的判别器，从多个角度评估波形真实性。
    这是 HiFi-GAN 的关键创新之一。

    periods = [2, 3, 5, 7, 11] — 使用质数避免周期性重复
    """

    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(self, y_real, y_fake):
        """
        Args:
            y_real: (B, 1, T) 真实波形
            y_fake: (B, 1, T) 生成波形

        Returns:
            real_scores: list of (B, T')
            fake_scores: list of (B, T')
            real_features: list of feature lists
            fake_features: list of feature lists
        """
        real_scores, fake_scores = [], []
        real_features, fake_features = [], []

        for disc in self.discriminators:
            rs, rf = disc(y_real)
            fs, ff = disc(y_fake)
            real_scores.append(rs)
            fake_scores.append(fs)
            real_features.append(rf)
            fake_features.append(ff)

        return real_scores, fake_scores, real_features, fake_features


# --------------------------------------------------------
# 8. StochasticDurationPredictor — 随机时长预测器
# --------------------------------------------------------

class StochasticDurationPredictor(nn.Module):
    """
    随机时长预测器 (Stochastic Duration Predictor, SDP) — 简化版

    使用高斯建模时长的随机性。
    同一句话可以有不同的说话速度 → 时长有随机性。

    架构：
        text_features → Conv layers → (μ_dr, log σ_dr)
        训练: 从后验 N(μ_dr, σ_dr²) 采样 duration, 计算 log_prob
        推理: 从先验 N(0, I) 采样, 过 MLP 得到 duration

    这是 VITS 中"随机性"的来源 — 让合成语音不那么"机械"。

    注：完整版 SDP 使用归一化流建模时长分布，
    此处简化为直接高斯建模，保留核心思想。
    """

    def __init__(self, hidden_dim=192, kernel_size=3, n_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 文本特征提取
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim)])
        for _ in range(n_layers - 1):
            self.convs.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # 后验参数: (μ_dr, log σ_dr)
        self.proj_posterior = nn.Linear(hidden_dim, hidden_dim * 2)

        # 先验投影（推理时用）
        self.proj_prior = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None, duration=None, reverse=False):
        """
        Args:
            x: (B, hidden, T_text) 文本特征
            mask: (B, 1, T_text)
            duration: (B, T_text) ground truth 时长（训练时用）
            reverse: 推理模式

        Returns:
            训练: log_prob (B,) — 后验时长概率
            推理: duration_pred (B, T_text) — 预测时长
        """
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h)
            h = h.transpose(1, 2)
            h = norm(h)
            h = h.transpose(1, 2)
            h = F.relu(h)
            if mask is not None:
                h = h * mask

        if not reverse:
            # 训练: 后验 → 计算 log_prob
            stats = self.proj_posterior(h.transpose(1, 2))  # (B, T, 2*hidden)
            m_dr, logs_dr = stats.chunk(2, dim=-1)
            logs_dr = torch.clamp(logs_dr, min=-10.0, max=10.0)

            # duration: (B, T_text) → (B, T, 1)
            dur = duration.float().unsqueeze(-1)

            # 对数概率: log N(dur; μ_dr, σ_dr²)
            log_prob = -0.5 * (
                math.log(2 * math.pi)
                + 2 * logs_dr
                + (dur - m_dr) ** 2 * torch.exp(-2 * logs_dr)
            )
            # 在 hidden 和 T 维度取平均（不是求和，避免随维度增大）
            if mask is not None:
                log_prob = log_prob * mask.transpose(1, 2)
            log_prob = log_prob.mean(dim=[1, 2])

            return log_prob
        else:
            # 推理: 从先验采样
            # 加噪声到特征 → 投影到时长
            noise = torch.randn_like(h) * 0.667
            h_noisy = h + noise
            if mask is not None:
                h_noisy = h_noisy * mask

            dur_pred = self.proj_prior(h_noisy.transpose(1, 2)).squeeze(-1)
            # 映射为正整数时长
            dur_pred = torch.clamp(dur_pred.exp().round().long(), min=1)

            return dur_pred


# --------------------------------------------------------
# 形状测试
# --------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Ch05 VITS — Module Shape Tests")
    print("=" * 60)

    B, T_text, T_spec = 2, 15, 80
    hidden = 192
    spec_ch = 513
    vocab = 200

    # 1. TextEncoder
    print("\n1. TextEncoder")
    te = TextEncoder(vocab_size=vocab, hidden_dim=hidden)
    text_ids = torch.randint(0, vocab, (B, T_text))
    text_lens = torch.tensor([T_text, T_text - 3])
    x, m_p, logs_p, mask = te(text_ids, text_lens)
    print(f"   x:      {x.shape}")        # (B, hidden, T_text)
    print(f"   m_p:    {m_p.shape}")      # (B, hidden, T_text)
    print(f"   logs_p: {logs_p.shape}")   # (B, hidden, T_text)
    print(f"   mask:   {mask.shape}")     # (B, 1, T_text)

    # 2. PosteriorEncoder
    print("\n2. PosteriorEncoder")
    pe = PosteriorEncoder(spec_channels=spec_ch, hidden_dim=hidden, n_layers=8)
    spec = torch.randn(B, spec_ch, T_spec)
    spec_lens = torch.tensor([T_spec, T_spec - 5])
    z, m_q, logs_q, spec_mask = pe(spec, spec_lens)
    print(f"   z:      {z.shape}")        # (B, hidden, T_spec)
    print(f"   m_q:    {m_q.shape}")      # (B, hidden, T_spec)
    print(f"   logs_q: {logs_q.shape}")   # (B, hidden, T_spec)

    # 3. Flow
    print("\n3. Flow")
    flow = Flow(channels=hidden, hidden_dim=128, n_flows=2)
    z_out, log_det = flow(z, spec_mask)
    print(f"   z_out:   {z_out.shape}")    # (B, hidden, T_spec)
    print(f"   log_det: {log_det.shape}")  # (B,)
    z_inv, log_det_inv = flow(z_out, spec_mask, reverse=True)
    print(f"   z_inv:   {z_inv.shape}")    # 应 ≈ z
    print(f"   逆变换误差: {(z - z_inv).abs().max().item():.6f}")

    # 4. Generator
    print("\n4. Generator (HiFi-GAN)")
    gen = Generator(hidden_dim=hidden, upsample_rates=(8, 8, 2, 2))
    wav = gen(z)
    print(f"   wav: {wav.shape}")          # (B, 1, T_wav)
    print(f"   上采样倍率: {wav.shape[-1] / T_spec:.0f}x")

    # 5. DurationPredictor
    print("\n5. DurationPredictor")
    dp = DurationPredictor(hidden_dim=hidden)
    log_dur = dp(x, mask)
    print(f"   log_dur: {log_dur.shape}")  # (B, T_text)

    # 6. Discriminator
    print("\n6. Discriminator (MPD)")
    disc = Discriminator(periods=(2, 3, 5))
    y_real = torch.randn(B, 1, 8192)
    y_fake = torch.randn(B, 1, 8192)
    rs, fs, rf, ff = disc(y_real, y_fake)
    print(f"   判别器数量: {len(disc.discriminators)}")
    print(f"   real_scores: {len(rs)} 个, 每个 shape: {rs[0].shape}")

    # 参数量统计
    print("\n" + "=" * 60)
    total = sum(p.numel() for p in te.parameters())
    print(f"TextEncoder:       {total:>12,}")
    total = sum(p.numel() for p in pe.parameters())
    print(f"PosteriorEncoder:  {total:>12,}")
    total = sum(p.numel() for p in flow.parameters())
    print(f"Flow:              {total:>12,}")
    total = sum(p.numel() for p in gen.parameters())
    print(f"Generator:         {total:>12,}")
    total = sum(p.numel() for p in dp.parameters())
    print(f"DurationPredictor: {total:>12,}")
    total = sum(p.numel() for p in disc.parameters())
    print(f"Discriminator:     {total:>12,}")

    print("\nAll shape tests passed!")
