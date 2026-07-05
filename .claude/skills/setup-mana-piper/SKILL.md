---
name: setup-mana-piper
description: Set up and run MahtaFetrat/Mana-Persian-Piper (Piper VITS ONNX voice, trained on the ManaTTS corpus). Use for fast, clean Persian TTS or to reproduce its benchmark sample. CPU-trivial via piper-tts.
---

# Setup: MahtaFetrat/Mana-Persian-Piper (Persian / فارسی)

- **Hugging Face:** https://huggingface.co/MahtaFetrat/Mana-Persian-Piper
- **Architecture:** [Piper](https://github.com/OHF-Voice/piper1-gpl) (VITS) — ships a ready **ONNX** voice
- **Voice:** `fa_IR-mana-medium` (single speaker), trained on the curated **ManaTTS** corpus
- **Sample rate:** 22.05 kHz
- **CPU:** extremely fast — ~0.2 s/sentence (RTF ≈ 0.03)
- **License:** MIT

## Why include it

The ManaTTS corpus is a large, clean Persian dataset, so pronunciation is very
accurate. Piper is deterministic and fast — a strong, stable alternative to the
heavy Chatterbox cloning model. Best quality-per-compute in the set.

## Install

```bash
python3 -m venv .venvs/piper
.venvs/piper/bin/pip install piper-tts huggingface_hub
# espeak-ng is bundled with piper-tts (its own espeak-ng-data); system espeak-ng also works.
```

## Run

```bash
.venvs/piper/bin/python scripts/run_mana_piper.py   # -> docs/audio/mana-piper/NN.wav
python3 scripts/make_demo.py mana-piper
```

## Files used

- `fa_IR-mana-medium.onnx` — the voice model
- `fa_IR-mana-medium.onnx.json` — Piper config (phoneme map, sample rate, inference params)

## Minimal usage snippet

```python
import wave
from piper import PiperVoice
voice = PiperVoice.load("fa_IR-mana-medium.onnx", config_path="fa_IR-mana-medium.onnx.json")
with wave.open("out.wav", "wb") as w:
    voice.synthesize_wav("سلام دنیا", w)
```

## Notes

- The repo also ships the raw Lightning checkpoint (`ckpt/…`) if you want to
  re-export; the ONNX is all you need for inference.
- Tune speed/expressiveness via `piper.config.SynthesisConfig`
  (`length_scale`, `noise_scale`, `noise_w`).

## Status

✅ Working on CPU via `.venvs/piper`. Sample generated for sentences 01–05.
