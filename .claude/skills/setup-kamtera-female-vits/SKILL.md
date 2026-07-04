---
name: setup-kamtera-female-vits
description: Set up and run Kamtera/persian-tts-female-vits Persian TTS (Coqui VITS + espeak-ng). Use when synthesizing Persian speech with the Kamtera female VITS voice, or reproducing its benchmark sample. Requires the coqui venv.
---

# Setup: Kamtera/persian-tts-female-vits (Persian / فارسی)

- **Hugging Face:** https://huggingface.co/Kamtera/persian-tts-female-vits
- **Architecture:** VITS (Coqui format: `config.json` + `best_model_*.pth`)
- **Engine:** [coqui-tts](https://github.com/idiap/coqui-ai-TTS) + `espeak-ng` phonemizer (lang `fa`)
- **Voice:** single female speaker
- **Sample rate:** 24 kHz
- **CPU RTF:** ~0.24 (faster than real-time)

## The dependency dance (important)

Coqui pins strict versions, so it lives in **its own venv** (`.venvs/coqui`)
to avoid clobbering the transformers env used by `mms-tts-fas`.

```bash
# system phonemizer + audio IO
apt-get install -y espeak-ng ffmpeg

# isolated venv
python3 -m venv .venvs/coqui
.venvs/coqui/bin/pip install -U pip
.venvs/coqui/bin/pip install coqui-tts
.venvs/coqui/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# coqui-tts 0.27.5 imports symbols removed in transformers 5.x → pin:
.venvs/coqui/bin/pip install "transformers==4.57.1"

# torch >= 2.9 needs torchcodec for audio IO (needs ffmpeg present):
.venvs/coqui/bin/pip install torchcodec
```

### Gotchas hit during setup
- `ImportError: isin_mps_friendly` → transformers too new; pin `4.57.1`.
- `is_torchcodec_available` / `cannot import` → transformers too old; `4.57.1` is the sweet spot for coqui-tts 0.27.5.
- `torchcodec ... FFmpeg ... not found` → install system `ffmpeg`, then `pip install torchcodec`.

## Run

```bash
.venvs/coqui/bin/python scripts/run_kamtera_female_vits.py   # -> samples/kamtera-female-vits/NN.wav
python3 scripts/make_demo.py kamtera-female-vits             # stitches demo.wav (stdlib only)
```

## Model file choice

The repo ships several checkpoints. We use **`best_model_30824.pth`** (highest
training step) with `config.json`. Alternatives: `best_model_23604.pth`,
`checkpoint_48000.pth`.

## Minimal usage snippet

```python
from huggingface_hub import hf_hub_download
from TTS.utils.synthesizer import Synthesizer
ckpt = hf_hub_download("Kamtera/persian-tts-female-vits", "best_model_30824.pth")
cfg  = hf_hub_download("Kamtera/persian-tts-female-vits", "config.json")
syn = Synthesizer(tts_checkpoint=ckpt, tts_config_path=cfg, use_cuda=False)
syn.save_wav(syn.tts("سلام دنیا"), "out.wav")
```

## Status

✅ Working on CPU via `.venvs/coqui`. Sample generated for sentences 01–05.
Sibling models: `Kamtera/persian-tts-male-vits`, `Kamtera/persian-tts-female-glow_tts`
load the same way (swap REPO + checkpoint name).
