---
name: setup-karim-gptinformal-vits
description: Set up and run karim23657/persian-tts-female-GPTInformal-Persian-vits (Coqui VITS + espeak-ng), an informal/colloquial Persian female voice. Use when synthesizing conversational Persian speech or reproducing its benchmark sample. Requires the coqui venv.
---

# Setup: karim23657/persian-tts-female-GPTInformal-Persian-vits

- **Hugging Face:** https://huggingface.co/karim23657/persian-tts-female-GPTInformal-Persian-vits
- **Architecture:** VITS (Coqui format: `config.json` + `best_model_*.pth`)
- **Engine:** [coqui-tts](https://github.com/idiap/coqui-ai-TTS) + `espeak-ng` (lang `fa`)
- **Voice:** female, **informal / colloquial** style (trained on the GPTInformal-Persian dataset)
- **Sample rate:** 24 kHz
- **CPU RTF:** ~0.24

## Why include it

Distinct from the Kamtera voices: trained on **informal/conversational** Persian,
so it reads everyday text with a more casual tone. Same engine, so setup is free
if the Kamtera models are already working.

## Dependencies

Reuses the shared **`.venvs/coqui`** venv. See
[`setup-kamtera-female-vits`](../setup-kamtera-female-vits/SKILL.md) for the full
install (espeak-ng, ffmpeg, coqui-tts, `transformers==4.57.1`, torchcodec).

## Run

```bash
.venvs/coqui/bin/python scripts/run_karim_gptinformal_vits.py   # -> docs/audio/karim-gptinformal-vits/NN.wav
python3 scripts/make_demo.py karim-gptinformal-vits
```

## Model file choice

Repo ships several checkpoints. We use **`best_model_98066.pth`** (highest step)
+ `config.json`. Alternatives: `best_model_76602.pth`, `checkpoint_108000.pth`.
The repo also has `tests/*.wav` reference outputs from the author.

## Status

✅ Working on CPU via `.venvs/coqui`. Sample generated for sentences 01–05.
Related: the author (karim23657) also has `persian-tts-vits` and a
Persian-TTS demo Space aggregating several voices.
