"""
Ch05: VITS — 完整模型

VITS = Conditional Variational Autoencoder with Adversarial Learning
       for End-to-End Text-to-Speech

Reference:
    Kim et al., 2021. "Conditional Variational Autoencoder with
    Adversarial Learning for End-to-End Text-to-Speech"

Architecture (训练时):
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Text ─→ TextEncoder ─→ (μ_p, σ_p) 先验分布        │
    │                            ↓                        │
    │  Wav ─→ Spectrogram ─→ PosteriorEncoder             │
    │              ↓           ↓                          │
    │           z_q ← (μ_q, σ_q) 后验分布                │
    │              ↓                                      │
    │           Flow (可逆变换)                            │
    │              ↓                                      │
    │           z_p (应该匹配先验分布)                     │
    │              ↓                                      │
    │         Generator ─→ Waveform                       │
    │              ↓                                      │
    │         Discriminator (对抗训练)                     │
    │                                                     │
    └─────────────────────────────────────────────────────┘

Architecture (推理时):
    Text ─→ TextEncoder ─→ 采样 z_p ─→ Flow⁻¹ ─→ Generator ─→ Wav

核心损失函数:
    L = L_reconstruction + L_KL + L_duration + L_adv
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import (
    TextEncoder,
    PosteriorEncoder,
    Flow,
    Generator,
    DurationPredictor,
    StochasticDurationPredictor,
    Discriminator,
)


# --------------------------------------------------------
# Monotonic Alignment Search (MAS)
# --------------------------------------------------------

def monotonic_alignment_search(log_p, x_mask, y_mask):
    """
    单调对齐搜索 (Monotonic Alignment Search)

    VITS 的关键算法：在不知道对齐的情况下，
    找到文本帧到音频帧的最优单调对齐。

    直觉：
    - 文本有 T_text 个 token
    - 音频有 T_spec 帧
    - 每个文本 token 至少对应 1 帧音频
    - 对齐必须单调（不能倒退）
    - 我们要找到使 log_p 之和最大的对齐路径

    这是一个动态规划问题：
    - 状态: dp[i][j] = 对齐到文本第 i 个 token、音频第 j 帧的最大概率
    - 转移: dp[i][j] = max(dp[i-1][j-1], dp[i][j-1]) + log_p[i][j]
                      （上一时刻要么消耗了一个 token，要么没有）

    Args:
        log_p: (B, T_text, T_spec) 每对 (text, spec) 的对数似然
        x_mask: (B, 1, T_text) 文本有效位置
        y_mask: (B, 1, T_spec) 频谱有效位置

    Returns:
        attn: (B, T_text, T_spec) 硬对齐矩阵 (每列只有一个 1)
    """
    B, T_text, T_spec = log_p.shape
    device = log_p.device

    # 获取每个样本的有效长度
    x_lengths = x_mask.squeeze(1).sum(dim=1).long()  # (B,)
    y_lengths = y_mask.squeeze(1).sum(dim=1).long()  # (B,)

    attn = torch.zeros(B, T_text, T_spec, device=device)

    for b in range(B):
        tx = x_lengths[b].item()
        ty = y_lengths[b].item()

        if tx == 0 or ty == 0:
            continue

        # 动态规划
        log_p_b = log_p[b, :tx, :ty]  # (tx, ty)

        # dp[i][j] = 对齐到第 i 个 text token、第 j 个 spec frame 的最大概率
        dp = torch.full((tx + 1, ty + 1), -1e9, device=device)
        dp[0][0] = 0

        # 回溯矩阵
        path = torch.zeros(tx + 1, ty + 1, dtype=torch.long, device=device)

        for j in range(1, ty + 1):
            for i in range(1, min(j + 1, tx + 1)):
                # 两种选择：
                # 1. 上一个帧对应同一个 token (不消耗 token)
                score_same = dp[i][j - 1]
                # 2. 上一个帧对应上一个 token (消耗一个 token)
                score_prev = dp[i - 1][j - 1]

                if score_same >= score_prev:
                    dp[i][j] = score_same + log_p_b[i - 1, j - 1]
                    path[i][j] = 0  # 同一个 token
                else:
                    dp[i][j] = score_prev + log_p_b[i - 1, j - 1]
                    path[i][j] = 1  # 前一个 token

        # 回溯路径
        i, j = tx, ty
        while i > 0 and j > 0:
            attn[b, i - 1, j - 1] = 1.0
            if path[i][j] == 1:
                i -= 1
            j -= 1

    return attn


def sequence_mask(length, max_length=None):
    """
    将长度向量转换为布尔 mask

    Args:
        length: (B,) 每个序列的实际长度
        max_length: 最大长度（默认为 length 中的最大值）

    Returns:
        mask: (B, max_length) bool
    """
    if max_length is None:
        max_length = length.max()
    ids = torch.arange(max_length, device=length.device)
    return ids.unsqueeze(0) < length.unsqueeze(1)


# --------------------------------------------------------
# VITS Generator
# --------------------------------------------------------

class VITS(nn.Module):
    """
    VITS 端到端 TTS 生成器

    将 VAE + Flow + GAN 三个组件整合为一个模型。

    训练时的数据流：
    1. Text → TextEncoder → (μ_p, σ_p) 先验
    2. Spec → PosteriorEncoder → z_q, (μ_q, σ_q) 后验
    3. z_q → Flow → z_p (应该接近先验)
    4. z_q → Generator → 波形
    5. 波形 → Discriminator → 对抗损失

    推理时的数据流：
    1. Text → TextEncoder → (μ_p, σ_p) → 采样 z_p
    2. z_p → Flow⁻¹ → z_q
    3. z_q → Generator → 波形

    参数量（默认配置）: ~24M (Generator) + ~25M (Discriminator)
    """

    def __init__(
        self,
        vocab_size=200,
        spec_channels=513,
        hidden_dim=192,
        ffn_dim=768,
        n_heads=2,
        n_enc_layers=6,
        n_post_layers=8,
        n_flow_layers=4,
        upsample_rates=(8, 8, 2, 2),
        use_sdp=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spec_channels = spec_channels
        self.use_sdp = use_sdp

        # 核心子模块
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            n_heads=n_heads,
            n_layers=n_enc_layers,
        )

        self.posterior_encoder = PosteriorEncoder(
            spec_channels=spec_channels,
            hidden_dim=hidden_dim,
            n_layers=n_post_layers,
        )

        self.flow = Flow(
            channels=hidden_dim,
            hidden_dim=hidden_dim,
            n_flows=n_flow_layers,
        )

        self.generator = Generator(
            hidden_dim=hidden_dim,
            upsample_rates=upsample_rates,
        )

        # 时长预测器
        self.duration_predictor = DurationPredictor(hidden_dim=hidden_dim)
        if use_sdp:
            self.stochastic_duration_predictor = StochasticDurationPredictor(
                hidden_dim=hidden_dim,
            )

        # 上采样倍率 → 用于计算波形长度
        self.upsample_factor = 1
        for r in upsample_rates:
            self.upsample_factor *= r

    def forward(
        self,
        text_ids,
        text_lengths,
        spec,
        spec_lengths,
    ):
        """
        训练前向传播

        Args:
            text_ids:     (B, T_text)     文本 token ids
            text_lengths: (B,)            文本长度
            spec:         (B, spec_ch, T_spec) 线性频谱
            spec_lengths: (B,)            频谱长度

        Returns:
            dict with all intermediate values for loss computation:
            - wav_hat: 生成波形
            - m_p, logs_p: 先验参数
            - m_q, logs_q: 后验参数
            - z_q, z_p: 隐变量
            - log_det: Flow 的对数行列式
            - attn: MAS 对齐矩阵
            - log_dur_pred: 时长预测
        """
        # 1. 文本编码 → 先验分布参数
        x, m_p, logs_p, text_mask = self.text_encoder(text_ids, text_lengths)

        # 2. 后验编码 → 后验分布参数 + 采样 z
        z_q, m_q, logs_q, spec_mask = self.posterior_encoder(spec, spec_lengths)

        # 3. Flow: z_q → z_p (后验到先验的变换)
        z_p, log_det = self.flow(z_q, spec_mask)

        # 4. Monotonic Alignment Search
        # 计算先验下的 log_p(z_p | text)
        # log N(z_p; m_p, σ_p²) = -0.5 * (log(2π) + 2*logσ + (z-μ)²/σ²)
        # z_p: (B, hidden, T_spec), m_p: (B, hidden, T_text)
        # 需要 log_p: (B, T_text, T_spec)
        logs_p_clamped = torch.clamp(logs_p, min=-10.0, max=10.0)
        # Expand: z_p → (B, hidden, 1, T_spec), m_p → (B, hidden, T_text, 1)
        z_p_exp = z_p.unsqueeze(2)                    # (B, hidden, 1, T_spec)
        m_p_exp = m_p.unsqueeze(3)                    # (B, hidden, T_text, 1)
        logs_p_exp = logs_p_clamped.unsqueeze(3)      # (B, hidden, T_text, 1)

        log_p_per_frame = -0.5 * (
            math.log(2 * math.pi)
            + 2 * logs_p_exp
            + (z_p_exp - m_p_exp) ** 2 * torch.exp(-2 * logs_p_exp)
        )
        # log_p_per_frame: (B, hidden, T_text, T_spec) → sum over hidden
        log_p = log_p_per_frame.sum(dim=1)  # (B, T_text, T_spec)

        # MAS: 找到最优单调对齐
        with torch.no_grad():
            attn = monotonic_alignment_search(log_p, text_mask, spec_mask)

        # 从对齐矩阵提取时长：每个文本 token 对应多少帧
        # attn: (B, T_text, T_spec) → 每行求和 = 该 token 的时长
        duration = attn.sum(dim=2)  # (B, T_text)

        # 对齐先验参数到 spec 帧空间
        # attn: (B, T_text, T_spec) — 每列只有一个 1
        # m_p: (B, hidden, T_text) → m_p_aligned: (B, hidden, T_spec)
        m_p_aligned = torch.matmul(m_p, attn)       # (B, hidden, T_spec)
        logs_p_aligned = torch.matmul(logs_p, attn)  # (B, hidden, T_spec)

        # 5. 时长预测器
        log_dur_pred = self.duration_predictor(x, text_mask)

        # SDP (Stochastic Duration Predictor)
        sdp_log_prob = None
        if self.use_sdp:
            # 扩展 duration 到与 spec 帧对齐
            sdp_log_prob = self.stochastic_duration_predictor(
                x, text_mask, duration=duration
            )

        # 6. 生成波形
        wav_hat = self.generator(z_q)

        return {
            "wav_hat": wav_hat,
            "m_p": m_p,
            "logs_p": logs_p,
            "m_p_aligned": m_p_aligned,
            "logs_p_aligned": logs_p_aligned,
            "m_q": m_q,
            "logs_q": logs_q,
            "z_q": z_q,
            "z_p": z_p,
            "log_det": log_det,
            "spec_mask": spec_mask,
            "text_mask": text_mask,
            "attn": attn,
            "duration": duration,
            "log_dur_pred": log_dur_pred,
            "sdp_log_prob": sdp_log_prob,
        }

    def infer(self, text_ids, text_lengths=None, noise_scale=0.667, length_scale=1.0):
        """
        推理：文本 → 波形

        Args:
            text_ids: (B, T_text)
            text_lengths: (B,) or None
            noise_scale: 采样噪声的缩放因子 (越小越确定性)
            length_scale: 语速控制 (1.0=正常, >1.0=慢, <1.0=快)

        Returns:
            wav: (B, 1, T_wav) 生成波形
            attn: (B, T_text, T_wav_z) 注意力对齐
        """
        # 1. 文本编码
        x, m_p, logs_p, text_mask = self.text_encoder(text_ids, text_lengths)

        # 2. 时长预测
        if self.use_sdp:
            # 使用随机时长预测器
            duration = self.stochastic_duration_predictor(
                x, text_mask, reverse=True
            )
        else:
            # 使用确定性时长预测器
            log_dur = self.duration_predictor(x, text_mask)
            duration = torch.clamp(
                torch.exp(log_dur) * text_mask.squeeze(1), min=1.0
            ).round().long()

        # 语速调整
        if length_scale != 1.0:
            duration = (duration.float() * length_scale).round().long().clamp(min=1)

        # 3. 根据时长展开文本编码序列
        # 将 (B, hidden, T_text) 按 duration 展开为 (B, hidden, T_total)
        z_p = self._expand_by_duration(m_p, logs_p, duration, noise_scale)
        # z_p: (B, hidden, T_total)

        # 4. 逆 Flow: z_p → z_q
        z_q, _ = self.flow(z_p, reverse=True)

        # 5. 生成波形
        wav = self.generator(z_q)

        return wav, duration

    def _expand_by_duration(self, m_p, logs_p, duration, noise_scale):
        """
        根据时长展开先验参数，并从中采样 z_p

        Args:
            m_p: (B, hidden, T_text)
            logs_p: (B, hidden, T_text)
            duration: (B, T_text) 每个 token 的帧数
            noise_scale: 采样噪声缩放

        Returns:
            z_p: (B, hidden, T_total)
        """
        B, hidden, T_text = m_p.shape
        device = m_p.device

        # 计算展开后的总长度
        total_length = duration.sum(dim=1).max().item()

        # 构造展开索引：[0,0,0,1,1,2,2,2,2,...]
        z_p = torch.zeros(B, hidden, total_length, device=device)

        for b in range(B):
            pos = 0
            for t in range(T_text):
                d = duration[b, t].item()
                if d <= 0 or pos + d > total_length:
                    continue
                # 复制该 token 的均值
                z_p[b, :, pos:pos + d] = m_p[b, :, t:t + 1]
                # 加噪声
                noise = torch.randn(hidden, d, device=device) * noise_scale
                z_p[b, :, pos:pos + d] += torch.exp(logs_p[b, :, t:t + 1]) * noise
                pos += d

        return z_p


# --------------------------------------------------------
# Loss Functions
# --------------------------------------------------------

class VITSLoss(nn.Module):
    """
    VITS 多损失函数

    L_gen = L_mel + L_KL + L_dur + L_adv_G + L_feat
    L_disc = L_adv_D

    各损失的含义：
    1. L_mel: 重建损失 — 生成波形的 Mel 应接近真实 Mel
    2. L_KL: KL 散度 — 后验(经 Flow 变换后)应接近先验
    3. L_dur: 时长损失 — 预测时长应接近 MAS 对齐时长
    4. L_adv_G: 生成器对抗损失 — 骗过判别器
    5. L_feat: 特征匹配损失 — 中间特征应接近真实
    6. L_adv_D: 判别器损失 — 区分真假
    """

    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, sample_rate=16000):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        # 预计算 Mel 滤波器
        self.register_buffer(
            "mel_basis",
            self._create_mel_basis(n_fft, n_mels, sample_rate),
        )

    def _create_mel_basis(self, n_fft, n_mels, sample_rate):
        """创建 Mel 滤波器组"""
        import librosa
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        return torch.from_numpy(mel_basis).float()

    def _spectral_loss(self, wav, wav_target):
        """
        频谱重建损失 (Mel-scale L1)

        将波形转换为 Mel 频谱后计算 L1 距离。
        相比直接在波形域计算 L1，Mel 域的 L1 更符合人耳感知。
        """
        # STFT
        window = torch.hann_window(self.win_length, device=wav.device)

        # 确保波形长度一致
        min_len = min(wav.shape[-1], wav_target.shape[-1])
        wav = wav[..., :min_len]
        wav_target = wav_target[..., :min_len]

        spec_gen = torch.stft(
            wav.squeeze(1), self.n_fft, self.hop_length,
            self.win_length, window, return_complex=True
        ).abs()

        spec_target = torch.stft(
            wav_target.squeeze(1), self.n_fft, self.hop_length,
            self.win_length, window, return_complex=True
        ).abs()

        # Mel scale
        mel_gen = torch.matmul(self.mel_basis, spec_gen)
        mel_target = torch.matmul(self.mel_basis, spec_target)

        # Log + clamp
        mel_gen = torch.log(torch.clamp(mel_gen, min=1e-5))
        mel_target = torch.log(torch.clamp(mel_target, min=1e-5))

        return F.l1_loss(mel_gen, mel_target)

    def kl_divergence(self, m_q, logs_q, m_p_aligned, logs_p_aligned, z_p, log_det, mask):
        """
        KL 散度 + Flow 对数行列式

        注意：m_p_aligned, logs_p_aligned 已经通过对齐矩阵从 text space
        扩展到 spec space，与 m_q, logs_q 在同一维度。

        KL(q(z|x) || p(z|c)) = E_q[log q(z|x) - log p(z|c)]

        对对角高斯：
        KL = Σ [logs_p - logs_q + (σ_q² + (μ_q - μ_p)²)/(2σ_p²) - 0.5]

        加上 Flow 的 log_det 修正。

        Args:
            m_q: (B, hidden, T_spec)
            logs_q: (B, hidden, T_spec)
            m_p_aligned: (B, hidden, T_spec) — 对齐到 spec 帧的先验均值
            logs_p_aligned: (B, hidden, T_spec) — 对齐到 spec 帧的先验 log σ
            z_p: (B, hidden, T_spec) — flow 变换后的后验采样
            log_det: (B,) — Flow 对数行列式
            mask: (B, 1, T_spec)

        Returns:
            kl_loss: scalar
        """
        logs_p_aligned = torch.clamp(logs_p_aligned, min=-10.0, max=10.0)

        # KL(q || p) = logs_p - logs_q + (exp(2*logs_q) + (m_q - m_p)^2) / (2*exp(2*logs_p)) - 0.5
        kl = logs_p_aligned - logs_q - 0.5
        kl = kl + 0.5 * ((m_q - m_p_aligned) ** 2 + torch.exp(2 * logs_q)) * torch.exp(-2 * logs_p_aligned)

        kl = kl * mask
        kl = kl.sum() / mask.sum()

        # 减去 Flow 的 log_det（Flow 变换的贡献）
        kl = kl - log_det.mean() / mask.sum()

        return kl

    def duration_loss(self, log_dur_pred, duration, mask):
        """
        时长预测损失

        用 MSE 在对数域比较预测时长和 MAS 对齐时长。
        在对数域计算是因为时长分布是偏斜的（大多为 1-3 帧，少数很长）。
        """
        # duration → log(duration+1) 避免 log(0)
        log_dur_target = torch.log(duration.float() + 1e-6)
        loss = F.mse_loss(log_dur_pred * mask.squeeze(1), log_dur_target * mask.squeeze(1))
        return loss

    def generator_adv_loss(self, fake_scores):
        """
        生成器对抗损失 (Least Squares GAN)

        让判别器把所有生成样本判为"真"：
        L_G = Σ (D(G(z)) - 1)²
        """
        loss = 0
        for score in fake_scores:
            loss = loss + torch.mean((score - 1.0) ** 2)
        return loss / len(fake_scores)

    def discriminator_loss(self, real_scores, fake_scores):
        """
        判别器损失 (Least Squares GAN)

        让判别器正确区分真假：
        L_D = Σ [(D(x) - 1)² + D(G(z))²]
        """
        loss = 0
        for rs, fs in zip(real_scores, fake_scores):
            loss = loss + torch.mean((rs - 1.0) ** 2) + torch.mean(fs ** 2)
        return loss / len(real_scores)

    def feature_matching_loss(self, real_features, fake_features):
        """
        特征匹配损失

        不仅要求最终判别结果，还要求中间层特征接近。
        这比纯对抗损失更稳定，因为中间特征更容易学习。
        """
        loss = 0
        n = 0
        for rf_list, ff_list in zip(real_features, fake_features):
            # 跳过最后一层（是判别分数，不是特征）
            for rf, ff in zip(rf_list[:-1], ff_list[:-1]):
                loss = loss + F.l1_loss(ff, rf.detach())
                n += 1
        return loss / max(n, 1)


# --------------------------------------------------------
# 形状测试
# --------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Ch05 VITS — Full Model Shape Test")
    print("=" * 60)

    B = 2
    vocab = 200
    spec_ch = 513
    hidden = 192

    # 构造测试数据
    T_text = 12
    T_spec = 40
    text_ids = torch.randint(0, vocab, (B, T_text))
    text_lengths = torch.tensor([T_text, T_text - 2])
    spec = torch.randn(B, spec_ch, T_spec)
    spec_lengths = torch.tensor([T_spec, T_spec - 5])

    # 初始化模型 (小配置，方便测试)
    model = VITS(
        vocab_size=vocab,
        spec_channels=spec_ch,
        hidden_dim=hidden,
        ffn_dim=384,
        n_heads=2,
        n_enc_layers=2,
        n_post_layers=4,
        n_flow_layers=2,
        upsample_rates=(8, 8, 2, 2),
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # 前向传播
    print("\n--- Forward pass ---")
    outputs = model(text_ids, text_lengths, spec, spec_lengths)
    print(f"wav_hat:        {outputs['wav_hat'].shape}")
    print(f"m_p:            {outputs['m_p'].shape}")
    print(f"z_q:            {outputs['z_q'].shape}")
    print(f"z_p:            {outputs['z_p'].shape}")
    print(f"log_det:        {outputs['log_det'].shape}")
    print(f"attn:           {outputs['attn'].shape}")
    print(f"duration:       {outputs['duration']}")
    print(f"log_dur_pred:   {outputs['log_dur_pred'].shape}")

    # 推理
    print("\n--- Inference ---")
    with torch.no_grad():
        wav, dur = model.infer(text_ids, text_lengths, noise_scale=0.667)
    print(f"Generated wav:  {wav.shape}")
    print(f"Duration:       {dur}")
    print(f"Audio length:   {wav.shape[-1]} samples")

    # 损失计算测试
    print("\n--- Loss computation ---")
    criterion = VITSLoss()

    # 构造目标波形（与生成波形等长）
    wav_target = torch.randn(B, 1, outputs['wav_hat'].shape[-1])

    mel_loss = criterion._spectral_loss(outputs['wav_hat'], wav_target)
    print(f"Mel loss:       {mel_loss.item():.4f}")

    kl_loss = criterion.kl_divergence(
        outputs['m_q'], outputs['logs_q'],
        outputs['m_p_aligned'], outputs['logs_p_aligned'],
        outputs['z_p'], outputs['log_det'],
        outputs['spec_mask'],
    )
    print(f"KL loss:        {kl_loss.item():.4f}")

    dur_loss = criterion.duration_loss(
        outputs['log_dur_pred'], outputs['duration'],
        outputs['text_mask'],
    )
    print(f"Duration loss:  {dur_loss.item():.4f}")

    print("\nAll tests passed!")
