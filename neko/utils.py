"""
Neko Speech - 共享工具库
所有章节共用的音频处理和可视化工具
"""

import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path


def load_audio(path: str, sr: int = 16000) -> torch.Tensor:
    """加载音频文件，返回单声道波形

    Args:
        path: 音频文件路径
        sr: 目标采样率（默认 16kHz）

    Returns:
        waveform: [T] 一维张量
    """
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return torch.from_numpy(wav).float()


def save_audio(waveform: torch.Tensor, path: str, sr: int = 16000):
    """保存波形为音频文件

    Args:
        waveform: [T] 或 [1, T] 张量
        path: 输出路径
        sr: 采样率
    """
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)
    wav_np = waveform.cpu().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav_np, sr)


def mel_spectrogram(
    waveform: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """计算梅尔频谱

    Args:
        waveform: [T] 或 [B, T] 波形
        n_fft: FFT 窗口大小
        hop_length: 帧移
        win_length: 窗口长度
        n_mels: 梅尔滤波器数量
        sample_rate: 采样率

    Returns:
        mel: [B, n_mels, T] 或 [n_mels, T] 梅尔频谱
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # 加窗
    window = torch.hann_window(win_length, device=waveform.device)

    # STFT
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )

    # 幅度谱
    magnitude = torch.abs(stft)

    # Mel 滤波器
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
    )
    mel_basis = torch.from_numpy(mel_basis).float().to(waveform.device)

    # 应用 Mel 滤波
    mel = torch.matmul(mel_basis, magnitude)

    # 对数压缩
    mel = torch.log(torch.clamp(mel, min=1e-5))

    return mel


def griffin_lim(
    mel: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_iter: int = 32,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Griffin-Lim 相位恢复算法（基线声码器）

    Args:
        mel: [n_mels, T] 梅尔频谱
        n_fft, hop_length, win_length: STFT 参数
        n_iter: 迭代次数
        sample_rate: 采样率

    Returns:
        waveform: [T] 重建的波形
    """
    mel_np = mel.cpu().numpy()

    # 反 Mel 滤波（近似）
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=mel_np.shape[0],
    )
    mel_basis_inv = np.linalg.pinv(mel_basis)
    magnitude = np.maximum(0, np.dot(mel_basis_inv, np.exp(mel_np)))

    # Griffin-Lim
    wav = librosa.griffinlim(
        magnitude,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
    )

    return torch.from_numpy(wav).float()


def plot_mel(mel: torch.Tensor, save_path: str = None, title: str = "Mel Spectrogram"):
    """可视化梅尔频谱

    Args:
        mel: [n_mels, T] 梅尔频谱
        save_path: 保存路径（可选）
        title: 标题
    """
    import matplotlib.pyplot as plt

    mel_np = mel.cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(mel_np, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Channels')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int = 42):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
