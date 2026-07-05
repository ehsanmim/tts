---
name: setup-seyedali-mms
description: Set up and run SeyedAli/Persian-Speech-synthesis-MMS Persian TTS (VITS via 🤗 transformers). Use when synthesizing Persian speech with this MMS-style finetune or reproducing its benchmark sample. No espeak / no coqui venv needed.
---

# Setup: SeyedAli/Persian-Speech-synthesis-MMS (Persian / فارسی)

- **Hugging Face:** https://huggingface.co/SeyedAli/Persian-Speech-synthesis-MMS
- **Architecture:** VITS, MMS-style (ships `config.json` + `model.safetensors` + `vocab.json`)
- **Engine:** 🤗 transformers `VitsModel` — **same code path as `facebook/mms-tts-fas`**
- **Voice:** single speaker
- **Sample rate:** 16 kHz
- **CPU RTF:** ~0.18

## Why it's easy

It's a transformers-native VITS checkpoint, so it loads with `VitsModel` /
`AutoTokenizer` in the **main env** — no `espeak-ng`, no coqui venv. If
`mms-tts-fas` runs, this runs.

## Install

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
pip3 install transformers
```

## Run

```bash
python3 scripts/run_seyedali_mms.py          # -> docs/audio/seyedali-mms/NN.wav
python3 scripts/make_demo.py seyedali-mms
```

## Minimal usage snippet

```python
import torch, scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer
model = VitsModel.from_pretrained("SeyedAli/Persian-Speech-synthesis-MMS")
tok = AutoTokenizer.from_pretrained("SeyedAli/Persian-Speech-synthesis-MMS")
inp = tok("سلام دنیا", return_tensors="pt")
with torch.no_grad():
    wav = model(**inp).waveform.squeeze().numpy()
scipy.io.wavfile.write("out.wav", model.config.sampling_rate, wav)
```

## Notes

- Sibling id `SeyedAli/Persian-Speech-synthesis` (without `-MMS`) is **not
  reachable** (404 via the API) — use the `-MMS` repo.
- Same tuning knobs as MMS: `model.speaking_rate`, `model.noise_scale`.

## Status

✅ Working on CPU in the main transformers env. Sample generated for sentences 01–05.
