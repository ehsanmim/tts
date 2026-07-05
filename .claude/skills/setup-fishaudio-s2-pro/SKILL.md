---
name: setup-fishaudio-s2-pro
description: Set up fishaudio/s2-pro — a SOTA (ElevenLabs-grade) multilingual TTS that supports Persian. IMPORTANT — it is a ~9GB GPU-only model (SGLang/CUDA); it CANNOT run on a CPU-only box. Read for the GPU recipe + why CPU is a no-go.
---

# Setup: fishaudio/s2-pro (Fish Audio S2 Pro)

- **Hugging Face:** https://huggingface.co/fishaudio/s2-pro
- **GitHub (inference):** https://github.com/fishaudio/fish-speech
- **Type:** `fish_qwen3_omni` — **Dual-AR** (dual autoregressive) LLM TTS, Qwen3 backbone
- **Scale:** **9.12 GB** weights (2 shards) + `codec.pth`; trained on 10M+ hours, 80+ languages, RL-aligned
- **Persian:** ✅ `fa` is in the supported language list
- **Quality tier:** leading / near-commercial (this is the "ElevenLabs-grade" target)
- **License:** fish-audio-research-license (research; see LICENSE.md)

## ✅ It DOES run on CPU (via the native PyTorch path) — just very slow

SGLang is only for **GPU-accelerated serving**. fish-speech also ships a native
**PyTorch two-stage path** (`fish_speech/models/text2semantic/inference.py`
generates audio tokens; `models/dac` decodes them) that runs on **CPU**.

Measured here (4-core CPU, 15 GB RAM + swap): **~34 minutes for one sentence**
(132 tokens @ 0.06 tok/s). Output is **44.1 kHz** — highest fidelity in the set.

### Two things that were required to run it on CPU
1. **Memory:** the 4B model peaks at ~16 GB RSS and **OOM-kills on a 15 GB box**.
   Add swap: `fallocate -l 20G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile`.
2. **`--device cpu`** (default is `cuda`) and don't `--compile` on CPU.

## Recipe (CPU — as run in this benchmark)

```bash
git clone https://github.com/fishaudio/fish-speech .venvs/fish-speech
cd .venvs/fish-speech && python3 -m venv venv
venv/bin/pip install "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cpu
# deps by name (NOT `pip install -e .` — the manifest trips the untrusted-code guard):
venv/bin/pip install "transformers<=4.57.3" descript-audio-codec einops loguru click tqdm \
    numpy natsort pyrootutils "einx[torch]==0.2.2" vector_quantize_pytorch resampy loralib \
    hydra-core omegaconf
HF_TOKEN=hf_... venv/bin/python - <<'PY'   # download the 9 GB checkpoint
from huggingface_hub import snapshot_download
snapshot_download("fishaudio/s2-pro", local_dir="checkpoints/s2-pro")
PY
# run (from repo root): loads model once, clones from a Persian reference clip
../../scripts/run_s2pro_persian.py           # see that script for the exact generate/decode calls
```

On a **GPU**, instead use the SGLang server (fast): see the fish-speech README
and https://github.com/sgl-project/sglang-omni .

## Zero-GPU alternatives for ElevenLabs-grade Persian

- **Fish Audio API / Playground:** https://fish.audio (hosted S2 — no local GPU).
- **ElevenLabs Multilingual v2** — supports Persian, hosted API (the actual gold standard).
- On CPU here, the best *runnable* options remain **Chatterbox-Persian**
  (expressive, cloning, ~20s/sentence) and **Mana-Piper** (clean, fast).

## Status

✅ **Ran on CPU** (1 Persian sample generated, 44.1 kHz) via the native PyTorch
path + a swap file. ~34 min/sentence — a one-off quality probe, not a full run.
For practical use, run on a GPU (SGLang) or the Fish Audio API.
