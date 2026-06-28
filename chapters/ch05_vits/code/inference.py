"""
Ch05: VITS Inference — 端到端文本到波形

VITS 推理不需要声码器！
Generator 直接从隐变量生成波形。

Usage:
    # 使用训练好的模型
    python inference.py \
        --checkpoint ../checkpoints/vits_final.pt \
        --text "你好，我是猫娘。" \
        --output ../outputs/vits_neko.wav

    # 快速测试（随机初始化，输出噪声）
    python inference.py \
        --text "测试一下" \
        --output ../outputs/vits_test.wav

    # 调整语速
    python inference.py \
        --checkpoint ../checkpoints/vits_final.pt \
        --text "你好，我是猫娘。" \
        --length-scale 1.2 \
        --output ../outputs/vits_slow.wav
"""

import argparse
import time

import numpy as np
import soundfile as sf
import torch

from model import VITS
from train import CharTokenizer


def synthesize(model, text, device, noise_scale=0.667, length_scale=1.0):
    """
    VITS 端到端合成：Text → Waveform

    数据流：
    1. Text → TextEncoder → (μ_p, σ_p) 先验参数
    2. 从 N(μ_p, σ_p²) 采样 z_p
    3. z_p → Flow⁻¹ → z_q (逆变换)
    4. z_q → Generator → waveform

    Args:
        model: VITS 模型
        text: 输入文本字符串
        device: torch device
        noise_scale: 采样噪声缩放（越小越确定性）
        length_scale: 语速（1.0=正常）

    Returns:
        waveform: numpy array
        duration: 预测时长
        time_ms: 合成耗时
    """
    # 编码文本
    tokenizer = model._tokenizer  # 存储在模型上的 tokenizer
    text_ids = tokenizer.encode(text)
    text_tensor = torch.LongTensor([text_ids]).to(device)
    text_lengths = torch.LongTensor([len(text_ids)]).to(device)

    model.eval()
    with torch.no_grad():
        t0 = time.time()
        wav, duration = model.infer(
            text_tensor, text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
        )
        time_ms = (time.time() - t0) * 1000

    waveform = wav.squeeze().cpu().numpy()
    return waveform, duration, time_ms


def main():
    parser = argparse.ArgumentParser(description="VITS Inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. If None, uses random model.")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, default="../outputs/vits_output.wav")
    parser.add_argument("--noise-scale", type=float, default=0.667,
                        help="Noise scale for sampling (smaller = more deterministic)")
    parser.add_argument("--length-scale", type=float, default=1.0,
                        help="Length scale (>1 = slower, <1 = faster)")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[device] {device}")

    # 加载模型
    if args.checkpoint:
        print(f"[load] Loading: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        tokenizer = CharTokenizer(ckpt.get("tokenizer_chars"))
        hidden_dim = ckpt.get("hidden_dim", 192)
        vocab_size = ckpt.get("vocab_size", tokenizer.vocab_size)

        model = VITS(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            ffn_dim=hidden_dim * 4,
            n_heads=2,
            n_enc_layers=4,
            n_post_layers=6,
            n_flow_layers=4,
            upsample_rates=(8, 8, 2, 2),
        ).to(device)

        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    else:
        print("[load] No checkpoint — using random model (expect noise!)")
        tokenizer = CharTokenizer.from_texts([args.text])
        model = VITS(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=192,
            ffn_dim=768,
            n_heads=2,
            n_enc_layers=4,
            n_post_layers=6,
            n_flow_layers=4,
            upsample_rates=(8, 8, 2, 2),
        ).to(device)

    # 把 tokenizer 附加到模型上，方便 synthesize 使用
    model._tokenizer = tokenizer

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Parameters: {total_params:,}")
    print(f"[input] Text: {args.text}")
    print(f"[input] Tokens: {tokenizer.encode(args.text)}")

    # 合成
    waveform, duration, time_ms = synthesize(
        model, args.text, device,
        noise_scale=args.noise_scale,
        length_scale=args.length_scale,
    )

    # 保存
    sf.write(args.output, waveform, args.sr)

    # 统计
    audio_dur = len(waveform) / args.sr
    rtf = (time_ms / 1000) / audio_dur if audio_dur > 0 else 0

    print("\n" + "=" * 55)
    print("VITS End-to-End TTS Results")
    print("=" * 55)
    print(f"  Output file    : {args.output}")
    print(f"  Audio duration : {audio_dur:.2f}s")
    print(f"  Synthesis time : {time_ms:.0f}ms")
    print(f"  RTF (real-time): {rtf:.3f}x")
    print(f"  Duration pred  : {duration[0].tolist()}")
    print(f"  Noise scale    : {args.noise_scale}")
    print(f"  Length scale   : {args.length_scale}")
    print("=" * 55)

    if rtf < 1.0:
        print("[note] RTF < 1.0 means faster than real-time!")
    else:
        print(f"[note] RTF > 1.0 means slower than real-time.")

    print(f"Done! Saved to {args.output}")


if __name__ == "__main__":
    main()
