"""
Ch05: VITS Training Script

训练 VITS 端到端 TTS 模型。

VITS 的训练比 Tacotron2 复杂得多，因为：
1. 生成器 (Generator) 和判别器 (Discriminator) 交替训练
2. 多损失函数：重建 + KL + 时长 + 对抗 + 特征匹配
3. 每个 batch 需要计算线性频谱（后验编码器输入）

Usage:
    python train.py \
        --data-dir ../../../data/processed \
        --epochs 50 \
        --batch-size 2

Note: VITS 显存需求大，batch_size=2 需要约 12GB 显存。
如果显存不足，可以用 --max-spec-len 300 截断频谱长度。
"""

import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import VITS, VITSLoss, sequence_mask


# --------------------------------------------------------
# 文本处理
# --------------------------------------------------------

class CharTokenizer:
    """字符级分词器"""

    def __init__(self, chars=None):
        if chars is None:
            chars = (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
                "，。！？、；：""''（）【】《》 "
            )
        self.chars = chars
        self.vocab = {c: i + 1 for i, c in enumerate(self.chars)}  # 0 = pad
        self.pad_id = 0
        self.vocab_size = len(self.vocab) + 1

    @classmethod
    def from_texts(cls, texts):
        unique_chars = set()
        for t in texts:
            unique_chars.update(t)
        chars = "".join(sorted(unique_chars))
        return cls(chars)

    def encode(self, text):
        return [self.vocab.get(c, self.pad_id) for c in text]


# --------------------------------------------------------
# 音频处理
# --------------------------------------------------------

def compute_linear_spec(wave, sr=16000, n_fft=1024, hop_length=256, win_length=1024):
    """
    计算线性频谱 (Linear Spectrogram)

    与 Mel 频谱不同，线性频谱保留所有频率 bin，
    为后验编码器提供更丰富的频率信息。

    Returns:
        spec: (n_fft//2 + 1, T) numpy array
    """
    window = np.hanning(win_length)
    n_freq = n_fft // 2 + 1

    # 补零
    pad_len = (n_fft - len(wave) % hop_length) % hop_length
    wave = np.pad(wave, (0, pad_len), mode="constant")

    n_frames = 1 + (len(wave) - n_fft) // hop_length
    if n_frames < 1:
        # 音频太短，补零到至少 1 帧
        wave = np.pad(wave, (0, n_fft - len(wave)), mode="constant")
        n_frames = 1

    spec = np.zeros((n_freq, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = wave[start:start + win_length]
        if len(frame) < win_length:
            frame = np.pad(frame, (0, win_length - len(frame)))
        fft_frame = np.fft.rfft(frame * window, n=n_fft)
        spec[:, i] = np.abs(fft_frame)

    # Log 压缩
    spec = np.log(spec + 1e-5)
    return spec


def compute_mel_spec(wave, sr=16000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80):
    """计算 Mel 频谱（用于损失函数）"""
    window = np.hanning(win_length)
    n_freq = n_fft // 2 + 1

    pad_len = (n_fft - len(wave) % hop_length) % hop_length
    wave = np.pad(wave, (0, pad_len), mode="constant")

    n_frames = 1 + (len(wave) - n_fft) // hop_length
    if n_frames < 1:
        wave = np.pad(wave, (0, n_fft - len(wave)), mode="constant")
        n_frames = 1

    spec = np.zeros((n_freq, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = wave[start:start + win_length]
        if len(frame) < win_length:
            frame = np.pad(frame, (0, win_length - len(frame)))
        fft_frame = np.fft.rfft(frame * window, n=n_fft)
        spec[:, i] = np.abs(fft_frame)

    # Mel 滤波
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700.0)
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595.0) - 1)

    fft_freqs = np.linspace(0, sr // 2, n_freq)
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sr // 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    mel_filter = np.zeros((n_mels, n_freq))
    for i in range(n_mels):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        up = (fft_freqs - left) / (center - left)
        down = (right - fft_freqs) / (right - center)
        mel_filter[i] = np.maximum(0, np.minimum(up, down))

    mel = mel_filter @ spec
    mel = np.log(mel + 1e-5)
    return mel


# --------------------------------------------------------
# Dataset
# --------------------------------------------------------

class VITSDataset(Dataset):
    """
    VITS 训练数据集

    每个样本包含：
    - text: 字符 token ids
    - spec: 线性频谱 (后验编码器输入)
    - wav: 原始波形 (判别器输入 + 重建目标)
    """

    def __init__(
        self,
        data_dir,
        tokenizer,
        max_spec_len=500,
        max_wav_sec=12,
        target_sr=16000,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_spec_len = max_spec_len
        self.max_wav_sec = max_wav_sec
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # 加载 manifest
        manifest_path = self.data_dir / "train.list"
        self.samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 4:
                    wav_rel = parts[0]
                    text = parts[3]
                    wav_path = self.data_dir / wav_rel
                    if wav_path.exists():
                        self.samples.append({
                            "wav_path": wav_path,
                            "text": text,
                        })

        # 按时长过滤
        filtered = []
        for s in self.samples:
            try:
                info = sf.info(s["wav_path"])
                if info.duration <= self.max_wav_sec and info.duration > 0.5:
                    filtered.append(s)
            except Exception:
                continue
        self.samples = filtered
        print(f"[dataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载音频
        wav, sr = sf.read(sample["wav_path"])
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sr)

        # 计算线性频谱
        spec = compute_linear_spec(wav, self.target_sr, self.n_fft, self.hop_length, self.win_length)

        # 截断
        if spec.shape[1] > self.max_spec_len:
            spec = spec[:, :self.max_spec_len]
            wav = wav[:self.max_spec_len * self.hop_length]

        # 文本编码
        text_ids = self.tokenizer.encode(sample["text"])

        return {
            "text": torch.LongTensor(text_ids),
            "spec": torch.FloatTensor(spec),
            "wav": torch.FloatTensor(wav),
        }


def collate_fn(batch):
    """批次整理（填充变长序列）"""
    texts = [b["text"] for b in batch]
    specs = [b["spec"] for b in batch]
    wavs = [b["wav"] for b in batch]

    # 填充 text
    text_lens = [len(t) for t in texts]
    max_text_len = max(text_lens)
    text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    for i, t in enumerate(texts):
        text_padded[i, :len(t)] = t

    # 填充 spec (B, spec_ch, T_spec)
    spec_ch = specs[0].shape[0]
    spec_lens = [s.shape[1] for s in specs]
    max_spec_len = max(spec_lens)
    spec_padded = torch.zeros(len(batch), spec_ch, max_spec_len)
    for i, s in enumerate(specs):
        spec_padded[i, :, :s.shape[1]] = s

    # 填充 wav (确保长度是 hop_length 的倍数)
    hop = 256
    wav_lens = [w.shape[0] for w in wavs]
    max_wav_len = max(wav_lens)
    # 向上取整到 hop_length 的倍数
    max_wav_len = ((max_wav_len + hop - 1) // hop) * hop
    wav_padded = torch.zeros(len(batch), max_wav_len)
    for i, w in enumerate(wavs):
        wav_padded[i, :w.shape[0]] = w

    return {
        "text": text_padded,
        "text_lengths": torch.LongTensor(text_lens),
        "spec": spec_padded,
        "spec_lengths": torch.LongTensor(spec_lens),
        "wav": wav_padded,
        "wav_lengths": torch.LongTensor(wav_lens),
    }


# --------------------------------------------------------
# Training
# --------------------------------------------------------

def train_step(
    model, disc, batch, criterion, opt_g, opt_d, device,
    c_mel=45.0, c_kl=1.0, c_dur=1.0, c_adv=1.0, c_feat=2.0,
):
    """
    VITS 单步训练

    训练流程：
    1. Generator forward → 生成波形 + 所有中间变量
    2. Discriminator(real, fake) → 对抗损失
    3. Generator backward (重建 + KL + 时长 + 对抗 + 特征匹配)
    4. Discriminator backward (区分真假)
    """
    text = batch["text"].to(device)
    text_lengths = batch["text_lengths"].to(device)
    spec = batch["spec"].to(device)
    spec_lengths = batch["spec_lengths"].to(device)
    wav_real = batch["wav"].to(device)

    # ==================== Generator ====================
    opt_g.zero_grad()

    # 前向传播
    outputs = model(text, text_lengths, spec, spec_lengths)
    wav_hat = outputs["wav_hat"].squeeze(1)  # (B, T_wav)

    # 对齐波形长度
    min_len = min(wav_hat.shape[-1], wav_real.shape[-1])
    wav_hat = wav_hat[:, :min_len]
    wav_real_aligned = wav_real[:, :min_len]

    # 重建损失 (Mel L1)
    loss_mel = criterion._spectral_loss(
        wav_hat.unsqueeze(1), wav_real_aligned.unsqueeze(1)
    ) * c_mel

    # KL 散度
    loss_kl = criterion.kl_divergence(
        outputs["m_q"], outputs["logs_q"],
        outputs["m_p_aligned"], outputs["logs_p_aligned"],
        outputs["z_p"], outputs["log_det"],
        outputs["spec_mask"],
    ) * c_kl

    # 时长损失
    loss_dur = criterion.duration_loss(
        outputs["log_dur_pred"], outputs["duration"],
        outputs["text_mask"],
    ) * c_dur

    # SDP 损失 (如果有)
    loss_sdp = torch.tensor(0.0, device=device)
    if outputs["sdp_log_prob"] is not None:
        loss_sdp = -outputs["sdp_log_prob"].mean() * 0.5

    # 对抗损失 + 特征匹配
    wav_hat_2d = wav_hat.unsqueeze(1)  # (B, 1, T)
    wav_real_2d = wav_real_aligned.unsqueeze(1)

    real_scores, fake_scores, real_features, fake_features = disc(wav_real_2d, wav_hat_2d)
    loss_adv_g = criterion.generator_adv_loss(fake_scores) * c_adv
    loss_feat = criterion.feature_matching_loss(real_features, fake_features) * c_feat

    # Generator 总损失
    loss_g = loss_mel + loss_kl + loss_dur + loss_sdp + loss_adv_g + loss_feat
    loss_g.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_g.step()

    # ==================== Discriminator ====================
    opt_d.zero_grad()

    # 重新前向（不计算梯度给 Generator）
    with torch.no_grad():
        outputs_d = model(text, text_lengths, spec, spec_lengths)
    wav_hat_d = outputs_d["wav_hat"].squeeze(1)[:, :min_len]
    wav_hat_d_2d = wav_hat_d.unsqueeze(1).detach()

    real_scores_d, fake_scores_d, _, _ = disc(wav_real_2d.detach(), wav_hat_d_2d)
    loss_adv_d = criterion.discriminator_loss(real_scores_d, fake_scores_d)
    loss_adv_d.backward()
    torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
    opt_d.step()

    return {
        "loss_g": loss_g.item(),
        "loss_mel": loss_mel.item(),
        "loss_kl": loss_kl.item(),
        "loss_dur": loss_dur.item(),
        "loss_sdp": loss_sdp.item(),
        "loss_adv_g": loss_adv_g.item(),
        "loss_feat": loss_feat.item(),
        "loss_adv_d": loss_adv_d.item(),
    }


def train_epoch(model, disc, dataloader, criterion, opt_g, opt_d, device, epoch):
    """训练一个 epoch"""
    model.train()
    disc.train()

    total_losses = {}
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        losses = train_step(model, disc, batch, criterion, opt_g, opt_d, device)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v
        n_batches += 1

        pbar.set_postfix({
            "G": f"{losses['loss_g']:.3f}",
            "mel": f"{losses['loss_mel']:.3f}",
            "kl": f"{losses['loss_kl']:.3f}",
            "D": f"{losses['loss_adv_d']:.3f}",
        })

    avg = {k: v / n_batches for k, v in total_losses.items()}
    return avg


def main():
    parser = argparse.ArgumentParser(description="Train VITS")
    parser.add_argument("--data-dir", type=str, default="../../../data/processed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr-g", type=float, default=2e-4, help="Generator learning rate")
    parser.add_argument("--lr-d", type=float, default=2e-4, help="Discriminator learning rate")
    parser.add_argument("--save-dir", type=str, default="../checkpoints")
    parser.add_argument("--max-spec-len", type=int, default=400, help="Max spec frames per sample")
    parser.add_argument("--max-wav-sec", type=float, default=10, help="Max audio seconds")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using {device}")

    # 数据集
    print("[data] Loading dataset...")
    # 先扫描文本建立 tokenizer
    manifest_path = Path(args.data_dir) / "train.list"
    all_texts = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 4:
                all_texts.append(parts[3])
    tokenizer = CharTokenizer.from_texts(all_texts)
    print(f"[data] Vocab size: {tokenizer.vocab_size}")

    dataset = VITSDataset(
        args.data_dir,
        tokenizer,
        max_spec_len=args.max_spec_len,
        max_wav_sec=args.max_wav_sec,
    )
    if len(dataset) == 0:
        print("[error] No data found. Run data/download_neko_1k.py first.")
        return

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )

    # 模型
    spec_ch = 513  # n_fft//2 + 1
    model = VITS(
        vocab_size=tokenizer.vocab_size,
        spec_channels=spec_ch,
        hidden_dim=args.hidden_dim,
        ffn_dim=args.hidden_dim * 4,
        n_heads=2,
        n_enc_layers=4,
        n_post_layers=6,
        n_flow_layers=4,
        upsample_rates=(8, 8, 2, 2),
    ).to(device)

    from modules import Discriminator
    disc = Discriminator(periods=(2, 3, 5, 7, 11)).to(device)

    model_params = sum(p.numel() for p in model.parameters())
    disc_params = sum(p.numel() for p in disc.parameters())
    print(f"[model] Generator: {model_params:,} params")
    print(f"[model] Discriminator: {disc_params:,} params")

    # 优化器
    opt_g = optim.AdamW(model.parameters(), lr=args.lr_g, betas=(0.8, 0.99))
    opt_d = optim.AdamW(disc.parameters(), lr=args.lr_d, betas=(0.8, 0.99))

    # 损失函数
    criterion = VITSLoss().to(device)

    # 保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # 保存 tokenizer
    with open(save_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump({"chars": tokenizer.chars}, f, ensure_ascii=False)

    # 损失日志
    log_path = save_dir / "loss_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,loss_g,loss_mel,loss_kl,loss_dur,loss_sdp,loss_adv_g,loss_feat,loss_adv_d\n")

    # 训练
    for epoch in range(1, args.epochs + 1):
        avg = train_epoch(model, disc, dataloader, criterion, opt_g, opt_d, device, epoch)

        print(f"\n[epoch {epoch}/{args.epochs}]")
        for k, v in avg.items():
            print(f"  {k}: {v:.4f}")

        with open(log_path, "a") as f:
            f.write(f"{epoch}," + ",".join(f"{avg[k]:.6f}" for k in
                    ["loss_g", "loss_mel", "loss_kl", "loss_dur", "loss_sdp", "loss_adv_g", "loss_feat", "loss_adv_d"]) + "\n")

        # 保存 checkpoint
        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_path = save_dir / f"vits_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "disc_state_dict": disc.state_dict(),
                "opt_g_state_dict": opt_g.state_dict(),
                "opt_d_state_dict": opt_d.state_dict(),
                "tokenizer_chars": tokenizer.chars,
                "hidden_dim": args.hidden_dim,
                "vocab_size": tokenizer.vocab_size,
            }, ckpt_path)
            print(f"  [save] {ckpt_path}")

    # 最终模型
    final_path = save_dir / "vits_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "tokenizer_chars": tokenizer.chars,
        "hidden_dim": args.hidden_dim,
        "vocab_size": tokenizer.vocab_size,
    }, final_path)
    print(f"\n[done] Final model: {final_path}")


if __name__ == "__main__":
    main()
