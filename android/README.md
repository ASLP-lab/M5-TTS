# M5 Android 离线 Demo

本目录包含独立的 **Java Android 工程、Gradle wrapper、模型打包工具和文本正规化规则**。手机端使用 ONNX Runtime 执行语音克隆，不依赖 Python、不调用服务器生成接口。

这里有两条独立流程，而不是把 checkpoint 直接打进 APK：

- **Android 源码 + 兼容的音频 encoder → Gradle 构建 → APK**。
- **M5 checkpoint → ONNX 导出 → `package_model.py` → `.m5model` 外置模型包**。

使用时先构建并安装 APK，再在 App 中导入 `.m5model`。模型文件单独下载和打包，不包含在源码中。

## 目录结构

```text
android/
├── app/
│   ├── build.gradle
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/com/m5/offline/    # 原生界面、推理、音频和模型管理
│       └── res/                   # 图标和字符串
├── gradle/wrapper/                # wrapper JAR 与 Gradle 版本配置
├── text_normalization/rules.json
├── build.gradle
├── settings.gradle
├── gradle.properties
├── gradlew
├── gradlew.bat
├── package_model.py
└── README.md
```

| 主要源码 | 职责 |
| --- | --- |
| `MainActivity.java` | 导入模型、音色管理、文本输入、生成、播放和保存 |
| `M5Engine.java` | CPU FP32 推理、KV cache、分段生成及音频解码 |
| `Tokenizer.java` / `TextNormalizer.java` | BPE 文本编码及数字、日期等文本正规化 |
| `VoiceEncoder.java` / `VoiceStore.java` | 将参考录音编码成可复用音色、保存和导入音色 |
| `AndroidAudioReader.java` / `ReferenceAudio.java` | 音频读取、单声道/24 kHz 转换及预处理 |
| `ModelInstaller.java` / `ModelPackage.java` | 读取模型容器、校验和安装外置模型 |
| `NpyMatrix.java` / `WavWriter.java` | embedding 内存映射及 PCM16 WAV 写入 |

## 当前能力与限制

- 支持导入本地模型、选择参考音色、从参考录音和准确转写创建音色、停止任务和保存结果。
- 输出 **24 kHz、单声道 PCM16 WAV**；默认 CPU FP32、4 线程，可选 1–8 线程。
- 长文本按段生成，但**整个任务结束后才播放，不是流式播放**。输入上限 2000 字，每段最多 500 帧（约 40 秒）。
- 当前固定 `lang="Chinese"`、`speaker=None`，口音和 caption 为空；**未提供 Python 版的 `speaker_id=1` 精品说话人入口，也未开放六语言/两方言选择**。
- 这是单 `app` 模块示例工程，不是独立 AAR/Android SDK；没有自建 C++/JNI/CMake，native 推理库来自 ONNX Runtime AAR。

## 1. 构建环境

| 项目 | 要求 |
| --- | --- |
| JDK | 17 |
| Gradle / Android Gradle Plugin | 8.9 / 8.7.3，由工程固定 |
| Android SDK | compile/target 35；安装 platform 35 及 AGP 所需 Build Tools |
| Java 源码级别 | 11 |
| 手机系统 | Android 9+（minSdk 28） |
| CPU ABI | 仅 `arm64-v8a` |
| ONNX Runtime Android | 1.23.2，由 Maven 获取 |
| applicationId | `com.m5.offline` |

设置 `JAVA_HOME`、`ANDROID_HOME`，或通过 Android Studio 打开本目录并配置本机 SDK。`local.properties` 只供本机使用，不要提交。构建不需要 PyTorch、CUDA 或自有 NDK 工程；首次获取 Gradle、AGP 和 Maven 依赖需要联网。

## 2. 准备 APK 内置 encoder

APK 内置参考音频 encoder，用于用户在手机上创建音色；主 TTS 模型仍在外部 `.m5model` 中。

准备与本 App 匹配的 `codec_encoder.onnx`，在以下方式中任选一种：

1. 放到 `android/models/codec_encoder.onnx`（默认路径）。
2. 构建时使用 `-Pm5VoiceEncoder=/absolute/path/codec_encoder.onnx`。
3. 设置环境变量 `M5_VOICE_ENCODER=/absolute/path/codec_encoder.onnx`。

优先级为 Gradle 参数、环境变量、默认路径。相对路径以 `android/` 为基准；建议外部文件使用绝对路径。构建自动复制 encoder 和 `text_normalization/rules.json` 至 APK assets，不需要手工修改生成的 assets。

**codec 兼容性不能跳过：** 当前 Java 实现和 Gradle 固定校验原配 encoder 的 SHA256，`VoiceEncoder.java` 同时固定文件大小。Python 打包器也会核对模型的 `codec_identity.json`。重新导出同一种 codec 也可能得到不同文件哈希，不代表能直接替换。不要仅修改哈希绕过检查；迁移其他 encoder 需要同步验证输入/输出、音色 token、decoder 及身份配置，并重新构建和测试。

## 3. 构建 APK

从仓库根目录执行，先将示例 encoder 路径替换为真实文件：

```bash
cd android
./gradlew --no-daemon \
  -Pm5VoiceEncoder=/absolute/path/codec_encoder.onnx \
  :app:assembleDebug
```

如果已经使用默认路径或环境变量，可以省略 `-Pm5VoiceEncoder`。Windows 使用 `gradlew.bat`。

输出文件：`android/app/build/outputs/apk/debug/app-debug.apk`（相对于仓库根目录）。

这是使用本机 debug key 签名的测试 APK，不是正式发布版本。正式发布需自行配置安全的签名流程，并更新 `app/build.gradle` 中的版本；不同签名通常不能覆盖安装旧 APK。

## 4. 准备 `.m5model` 外置模型包

### 4.1 准备兼容的 ONNX 模型目录

先在仓库根目录安装 `pip install -r requirements.txt`，再下载完整 ONNX 模型：

```bash
hf download ASLP-lab/M5-TTS-50M-VoiceClone-ONNX --local-dir outputs/android_onnx
```

下面示例使用下载后的 `outputs/android_onnx`。自己的 checkpoint 可先通过 `onnx/export.py` 导出，见[ONNX 导出说明](../README_zh.md#导出-onnx)；不要将 `.safetensors` 直接传给安卓打包器。

当前 App 的 TTS 架构限定为 8 层 backbone、hidden size 512、2 个 KV heads、head dim 64、129992 文本词表和 16 个音频码本，音频词表为 2051。还必须使用兼容的 codec 和 FP32 embedding。只有满足这些约束的 ONNX 导出才能继续打包；Python ONNX 推理成功并不自动证明 Android 兼容。

打包输入需要：

```text
onnx_model/
├── backbone.onnx
├── depth_frame.onnx
├── codec_decoder_cached.onnx
├── codec_cached.json
├── text_embeddings.npy
├── audio_embeddings.npy
├── tokenizer.json
├── manifest.json
└── codec_identity.json
```

若要在电脑上创建参考音色，还需要完整 ONNX 运行目录中的 `codec_encoder.onnx`、`codec_decoder.onnx` 和 `checksums.json`。

### 4.2 准备授权参考音色

`package_model.py` 至少需要一个 `.npz` 参考音色，不会自动打包历史角色音色。已有配套 `.npz` 时可跳过此步。

在安装根目录 `requirements.txt` 的同一环境中，用 ONNX Python API 编码参考录音。下面示例保存音频 token 和转写，不是训练新音色：

```python
import hashlib
import json
from pathlib import Path
import numpy as np
from infer_onnx import M5Onnx, read_reference

model = M5Onnx("./outputs/android_onnx", device="cpu", local_files_only=True)
audio = read_reference("./assets/aslp_default_voice.wav")
codes = model.encode_reference(audio)
with (model.root / "codec_encoder.onnx").open("rb") as stream:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(block)
metadata = {
    "reference_text": "猪携带该疾病，然后通过蚊子传染给人类。",
    "encoder_sha256": digest.hexdigest(),
}
output = Path("outputs/android_voices/default_voice.npz")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("xb") as stream:
    np.savez_compressed(stream, codes=codes, metadata=json.dumps(metadata, ensure_ascii=False))
```

参考录音和转写必须匹配；替换为其他音色时使用本人或已获授权、允许对应用途的录音。音频编码须使用与 APK 相同的 encoder。

### 4.3 打包

若已有音色文件，打包本身只需要 NumPy：

```bash
python android/package_model.py \
  --model-dir ./outputs/android_onnx \
  --voices ./outputs/android_voices \
  --output ./outputs/android_package
```

`--model-dir`、`--voices` 必须显式指定；输出目录必须不存在，避免覆盖旧包。输出包括 `m5-v20-offline-model.m5model`、原始 ZIP、`bundle.json` 和打包元数据。文件名中的 `v20` 是保留的命名，不代表任意输入模型已经通过 v20 兼容或质量验证。

打包器将文本 tokenizer 转成 `tokenizer.bin`，音色转成 `voices.json`，保留源 checkpoint 哈希但不写入其内部机器路径。`.m5model` 是自定义文件头加 ZIP，包含 9 个运行文件及记录逐文件大小/SHA256 的 `bundle.json`，**不包含 encoder**。哈希是完整性检查，不是发行方数字签名；只导入可信模型。

## 5. 安装和测听

1. 在 Android 9+、arm64 手机上安装构建出的 APK。
2. 将 `.m5model` 复制到手机，不要提前解压。
3. 打开 App，点击“导入离线模型”，通过系统文件选择器选中模型包。
4. 选择包内参考音色，或导入 1–30 秒、20 MiB 以内的参考录音并填写准确转写，创建自己的音色。
5. 输入文本，生成完成后播放或保存 WAV。运行期间无需网络。

建议为安装和模型导入预留约 4 GB 存储空间；导入会暂存和解压文件，这不是最低运行内存保证。当前无内置录音/ASR，其他音频格式的可用性取决于手机 MediaCodec 支持。

## 发布应用

发布前在目标设备上测试模型导入、音频生成、取消任务和内存占用。正式 APK 需使用自己的签名配置，不要提交签名私钥或本机 SDK 配置。

模型和参考音色单独分发；确认所用权重、codec和音频允许对应的再分发用途。
