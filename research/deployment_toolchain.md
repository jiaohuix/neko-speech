# 部署工具链研究笔记

> 研究日期: 2026-06-29
> 来源: MNN 框架 (/home/jhx/Projects/nlp/MNN/)

---

## 1. MNN 模型转换流水线

### 1.1 PyTorch → ONNX → MNN 三步走

```
[PyTorch .pth]
    │ torch.onnx.export()
    ▼
[ONNX .onnx]
    │ MNNConvert --modelFile xxx.onnx --MNNModel xxx.mnn
    ▼
[MNN .mnn]
    │ adb push xxx.mnn /sdcard/
    ▼
[Android App 推理]
```

### 1.2 ONNX 导出关键参数（来自 Bert-VITS2-MNN）

```python
torch.onnx.export(
    model,
    (input_ids, attention_mask, token_type_ids),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "hidden_state": {0: "batch_size", 1: "sequence_length"}
    },
    do_constant_folding=True,
    opset_version=16  # 关键：VITS 需要 opset 16+
)
```

### 1.3 MNN 转换命令

```bash
# 编译 MNNConvert 工具
cd /home/jhx/Projects/nlp/MNN
mkdir build && cd build
cmake .. && make -j$(nproc)

# 转换 ONNX → MNN
./tools/converter/MNNConvert \
    --modelFile model.onnx \
    --MNNModel model.mnn \
    --bizCode TTS \
    --weightQuantBits 8  # INT8 量化，减小体积
```

---

## 2. Bert-VITS2-MNN 端侧性能参考

| 项目 | 数值 |
|------|------|
| 测试机型 | Snapdragon 888 |
| 采样率 | 22050 Hz |
| 模型体积 | ≈ 29.7 MB (INT8) |
| 合成文本 | "RTX 5090 将于明年发布" (10字+1英文) |
| 合成耗时 | ≈ 1856 ms |
| 音频时长 | ≈ 5.20 s |
| **RTF** | **≈ 0.357** (实时!) |
| 吞吐 | ~2.8 s 音频/秒 |

### 6 个 MNN 子模型

```
bertvits2-jni/src/main/cpp/ 加载:
1. encoder.mnn    - TextEncoder
2. decoder.mnn    - HiFi-GAN Generator
3. flow.mnn       - Normalizing Flow
4. dp.mnn         - Duration Predictor
5. sdp.mnn        - Stochastic Duration Predictor
6. emb.mnn        - Embedding layers
```

---

## 3. Android 部署流水线

### 3.1 环境搭建 (WSL)

```bash
# 1. JDK
sudo apt install openjdk-17-jdk

# 2. Android cmdline-tools
mkdir -p ~/android-sdk/cmdline-tools
cd ~/android-sdk/cmdline-tools
wget https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
unzip commandlinetools-linux-*.zip
mv cmdline-tools latest

# 3. NDK
sdkmanager "ndk;25.2.9519653"

# 4. 连接手机 (WiFi 调试)
adb tcpip 5555
adb connect <phone_ip>:5555
```

### 3.2 编译 MNN Android 库

```bash
cd /home/jhx/Projects/nlp/MNN
./project/android/build_64.sh  # 生成 libMNN.so
```

### 3.3 编译 App

```bash
cd apps/Android/MoeAvatar/
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

---

## 4. sherpa-onnx 方案（备选）

sherpa-onnx 是另一个端侧 TTS 方案，更简单但灵活性较低。

### 安装
```bash
pip install sherpa-onnx
```

### 支持的模型格式
- VITS (端到端)
- Matcha-TTS
- 预训练模型直接从 HuggingFace 下载

### 使用
```python
import sherpa_onnx

tts = sherpa_onnx.OfflineTts(
    model_dir="vits-model/",
    rule_fsts="rule.fst"
)
tts.generate("你好世界", output="output.wav")
```

---

## 5. neko-speech 部署章节规划

### Ch11: 端侧部署 (计划)

#### 5.1 ONNX 导出
- PyTorch → ONNX 转换细节
- 动态轴设置
- opset 版本选择
- 验证转换正确性

#### 5.2 MNN 转换
- ONNX → MNN 转换
- INT8 量化
- 模型体积优化
- 基准测试

#### 5.3 Android 部署
- WSL 环境搭建
- MNN 编译
- App 集成
- 实时 TTS demo

#### 5.4 sherpa-onnx 部署
- 桌面端部署
- CPU 推理优化
- 流式 TTS

#### 5.5 性能对比表

| 模型 | 格式 | 体积 | CPU RTF | GPU RTF | Android RTF |
|------|------|------|---------|---------|-------------|
| Tacotron2 | PyTorch | 100MB | 3.0x | 0.3x | N/A |
| Tacotron2 | ONNX | 95MB | 1.5x | - | N/A |
| FastSpeech2 | PyTorch | 30MB | 0.05x | 0.01x | N/A |
| FastSpeech2 | ONNX | 28MB | 0.03x | - | 0.02x |
| VITS | PyTorch | 170MB | 1.0x | 0.1x | N/A |
| VITS | ONNX | 160MB | 0.5x | - | N/A |
| VITS | MNN(INT8) | 30MB | 0.3x | - | 0.35x |

---

## 6. 关键参考资料

- [Bert-VITS2-MNN](https://github.com/Voine/Bert-VITS2-MNN) - 完整 Android TTS 方案
- [MNN 框架](https://github.com/alibaba/MNN) - 阿里端侧推理引擎
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) - 端侧语音推理
- [MNN Android 部署指南](/home/jhx/Projects/nlp/MNN/docs/android-wsl-开发指南/) - 13篇完整教程

---

*下一步: 实现 neko-speech 各模型的 ONNX/MNN 转换脚本*
