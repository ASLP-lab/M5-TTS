---
license: apache-2.0
pipeline_tag: feature-extraction
tags:
- m5-tts
- audio-tokenizer
- audio-codec
- pytorch
---

# M5-TTS-Audio-Tokenizer

English | [简体中文](README_zh.md)

[GitHub](https://github.com/ASLP-lab/M5-TTS) · [Demo](https://aslp-lab.github.io/M5-TTS-Demo-Page/) · [Space](https://huggingface.co/spaces/ASLP-lab/M5-TTS-Space) · [Models](https://huggingface.co/collections/ASLP-lab/m5-tts)

This repository provides the **Qwen 12Hz audio codec used by M5-TTS**, encoding reference recordings into discrete audio tokens and decoding generated tokens into speech. It is **not a standalone text-to-speech model** and is distinct from the text tokenizer bundled with M5 checkpoints.

| Property | Value |
| --- | --- |
| Audio sample rate | 24 kHz |
| Codec frame rate | 12.5 frames/s |
| Codebooks used by M5 | 16 |
| Codes layout through the M5 adapter | `(batch, frames, 16)` for a padded batch |

The [VoiceClone](https://huggingface.co/ASLP-lab/M5-TTS-50M-VoiceClone) and [CustomVoice](https://huggingface.co/ASLP-lab/M5-TTS-50M-CustomVoice) inference scripts download this codec automatically. You normally do not need to load it separately.

## Quickstart

For standalone audio encoding and reconstruction, use Python 3.10 and the M5 repository:

```bash
git clone https://github.com/ASLP-lab/M5-TTS.git
cd M5-TTS
conda create -n m5tts python=3.10 -y
conda activate m5tts
pip install -r requirements.txt
```

Run from the repository root; this CPU example does not require FlashAttention:

```python
import soundfile as sf
from m5.codec_qwen12hz import Qwen3TTSTokenizer

codec = Qwen3TTSTokenizer.from_pretrained(
    "ASLP-lab/M5-TTS-Audio-Tokenizer",
    device_map="cpu",
)

encoded = codec.encode("assets/aslp_default_voice.wav")
waveforms, sample_rate = codec.decode(encoded)
sf.write("reconstructed.wav", waveforms[0], sample_rate)
```

`encode()` also accepts NumPy waveforms with an explicit source sample rate, for example `codec.encode(audio, sr=24000)`. The adapter resamples input audio as needed; reconstruction is lossy. The TTS inference scripts apply their reference-audio preprocessing separately.

## License and Attribution

Apache-2.0. This repository uses the [Qwen3-TTS 12Hz tokenizer](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz). The M5 repository includes its minimal codec implementation with the [upstream license and modification notice](https://github.com/ASLP-lab/M5-TTS/tree/main/m5/codec_qwen12hz). No separate `qwen-tts` package installation is required.
