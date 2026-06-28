"""
Neko Speech - 数据集加载器
统一的音频数据集加载接口
"""

import torch
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from .utils import load_audio, mel_spectrogram


class NekoDataset(Dataset):
    """猫娘语音数据集

    数据格式：manifest.list
    wav_path|speaker|language|text

    示例：
    wavs/000001.wav|neko|zh|你好，我是猫娘。
    """

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 16000,
        max_wav_length: int = 25 * 16000,  # 最长 25 秒
        compute_mel: bool = True,
    ):
        """
        Args:
            manifest_path: manifest.list 文件路径
            sample_rate: 采样率
            max_wav_length: 最大波形长度（采样点）
            compute_mel: 是否计算 Mel 频谱
        """
        self.sample_rate = sample_rate
        self.max_wav_length = max_wav_length
        self.compute_mel = compute_mel

        # 读取 manifest
        self.items = []
        manifest_path = Path(manifest_path)
        base_dir = manifest_path.parent

        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                if len(parts) >= 4:
                    wav_path, speaker, lang, text = parts[:4]
                    # 相对路径转绝对路径
                    if not Path(wav_path).is_absolute():
                        wav_path = str(base_dir / wav_path)
                    self.items.append({
                        'wav_path': wav_path,
                        'speaker': speaker,
                        'language': lang,
                        'text': text,
                    })

        # 构建字符表
        all_chars = set()
        for item in self.items:
            all_chars.update(item['text'])
        self.chars = sorted(list(all_chars))
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}

        print(f"Loaded {len(self.items)} samples, vocab size: {len(self.chars)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # 加载音频
        wav = load_audio(item['wav_path'], sr=self.sample_rate)

        # 截断
        if wav.shape[0] > self.max_wav_length:
            wav = wav[:self.max_wav_length]

        # 文本编码
        text_indices = [self.char2idx.get(c, 0) for c in item['text']]
        text_tensor = torch.tensor(text_indices, dtype=torch.long)

        result = {
            'wav': wav,
            'wav_length': wav.shape[0],
            'text': text_tensor,
            'text_length': text_tensor.shape[0],
            'speaker': item['speaker'],
            'language': item['language'],
            'raw_text': item['text'],
        }

        # 计算 Mel 频谱
        if self.compute_mel:
            mel = mel_spectrogram(wav, sample_rate=self.sample_rate)
            result['mel'] = mel
            result['mel_length'] = mel.shape[-1]

        return result

    def get_vocab_size(self):
        return len(self.chars)

    def get_char2idx(self):
        return self.char2idx.copy()


def collate_fn(batch):
    """批次数据整理函数（处理变长序列）"""

    # 找出最大长度
    max_wav_len = max(item['wav'].shape[0] for item in batch)
    max_text_len = max(item['text'].shape[0] for item in batch)

    # 填充
    wavs = torch.zeros(len(batch), max_wav_len)
    texts = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    wav_lengths = []
    text_lengths = []

    for i, item in enumerate(batch):
        wav_len = item['wav'].shape[0]
        text_len = item['text'].shape[0]

        wavs[i, :wav_len] = item['wav']
        texts[i, :text_len] = item['text']

        wav_lengths.append(wav_len)
        text_lengths.append(text_len)

    result = {
        'wav': wavs,
        'wav_length': torch.tensor(wav_lengths),
        'text': texts,
        'text_length': torch.tensor(text_lengths),
        'speaker': [item['speaker'] for item in batch],
        'language': [item['language'] for item in batch],
        'raw_text': [item['raw_text'] for item in batch],
    }

    # Mel 频谱（如果有）
    if 'mel' in batch[0]:
        max_mel_len = max(item['mel'].shape[-1] for item in batch)
        mels = torch.zeros(len(batch), batch[0]['mel'].shape[0], max_mel_len)
        mel_lengths = []

        for i, item in enumerate(batch):
            mel_len = item['mel'].shape[-1]
            mels[i, :, :mel_len] = item['mel']
            mel_lengths.append(mel_len)

        result['mel'] = mels
        result['mel_length'] = torch.tensor(mel_lengths)

    return result


def create_dataloader(
    manifest_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
):
    """创建 DataLoader

    Args:
        manifest_path: manifest.list 路径
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作线程数
        **kwargs: 传递给 NekoDataset 的其他参数

    Returns:
        DataLoader
    """
    dataset = NekoDataset(manifest_path, **kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return dataloader
