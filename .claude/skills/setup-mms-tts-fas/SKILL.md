---
name: setup-mms-tts-fas
description: Set up and run Meta's facebook/mms-tts-fas Persian TTS model (VITS via 🤗 transformers). Use when synthesizing Persian speech with the MMS single-speaker VITS model, or reproducing the benchmark sample for it.
---

# Setup: facebook/mms-tts-fas (Persian / فارسی)

- **Hugging Face:** https://huggingface.co/facebook/mms-tts-fas
- **Family:** Meta [MMS](https://huggingface.co/facebook/mms-tts) (Massively Multilingual Speech)
- **Architecture:** VITS (end-to-end, non-autoregressive)
- **Voice:** single speaker, ~male timbre
- **Sample rate:** 16 kHz
- **License:** CC-BY-NC 4.0 (non-commercial)

## Why it's the easy first pick

- Runs through plain `transformers` — **no `espeak-ng` / phonemizer binary**
  (MMS ships its own char/uroman tokenizer for Persian).
- **CPU-friendly:** RTF ≈ 0.2–0.3 on a 4-core CPU (faster than real-time).
- Model is small (~145 MB).

## Install

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
pip3 install transformers
```

That's it. No system packages needed.

## Run

```bash
python3 scripts/run_mms_tts_fas.py          # writes samples/mms-tts-fas/NN.wav
python3 scripts/make_demo.py mms-tts-fas     # stitches -> samples/mms-tts-fas/demo.wav
```

## Minimal usage snippet

```python
import torch, scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer

model = VitsModel.from_pretrained("facebook/mms-tts-fas")
tok = AutoTokenizer.from_pretrained("facebook/mms-tts-fas")
inputs = tok("سلام دنیا", return_tensors="pt")
with torch.no_grad():
    wav = model(**inputs).waveform.squeeze().numpy()
scipy.io.wavfile.write("out.wav", model.config.sampling_rate, wav)
```

## Quirks / tuning

- Control speed/expressiveness via `model.speaking_rate` and
  `model.noise_scale` before inference (defaults are fine).
- No built-in number/date normalization — digits like `۰۹۱۲...` are read
  digit-by-digit. Pre-normalize text for best results.
- Non-commercial license — fine for evaluation, not for shipping products.

## Status

✅ Working on CPU. Sample generated for benchmark sentences 01–05.
