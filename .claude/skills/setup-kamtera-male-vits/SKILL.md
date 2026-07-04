---
name: setup-kamtera-male-vits
description: Set up and run Kamtera/persian-tts-male-vits Persian TTS (Coqui VITS + espeak-ng). Use when synthesizing Persian speech with the Kamtera male VITS voice, or reproducing its benchmark sample. Requires the coqui venv.
---

# Setup: Kamtera/persian-tts-male-vits (Persian / فارسی)

- **Hugging Face:** https://huggingface.co/Kamtera/persian-tts-male-vits
- **Architecture:** VITS (Coqui format: `config.json` + `best_model_*.pth`)
- **Engine:** [coqui-tts](https://github.com/idiap/coqui-ai-TTS) + `espeak-ng` (lang `fa`)
- **Voice:** single male speaker
- **Sample rate:** 22.05 kHz
- **CPU RTF:** ~0.23

## Dependencies

Identical to the female model — reuses the shared **`.venvs/coqui`** venv.
See [`setup-kamtera-female-vits`](../setup-kamtera-female-vits/SKILL.md) for the
full install (espeak-ng, ffmpeg, coqui-tts, `transformers==4.57.1`, torchcodec).

## Run

```bash
.venvs/coqui/bin/python scripts/run_kamtera_male_vits.py   # -> docs/audio/kamtera-male-vits/NN.wav
python3 scripts/make_demo.py kamtera-male-vits
```

## Model file choice

Repo ships several checkpoints. We use **`best_model_98066.pth`** (highest step)
+ `config.json`. Alternatives: `best_model_91323.pth`, `checkpoint_108000.pth`.

## Status

✅ Working on CPU via `.venvs/coqui`. Sample generated for sentences 01–05.
