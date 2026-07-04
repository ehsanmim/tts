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

## ⛔ Why it does NOT run in this benchmark's environment

This box is **CPU-only with ~7 GB free disk**. S2 Pro is designed for the
**SGLang** streaming engine, which is **CUDA/GPU-only** (CUDA graph replay,
paged KV cache, RadixAttention). There is no practical CPU inference path:
- The 9.12 GB download alone doesn't fit the free disk here.
- Even loaded on CPU, a ~4–5B dual-AR LLM generating audio tokens would be
  minutes+ per sentence, and the audio-codec decode + Dual-AR loop are custom to
  fish-speech (not a plain `transformers` `generate`).

So it is **documented, not benchmarked** here. Run it where there's a GPU.

## Recipe (on a CUDA GPU, ≥16 GB VRAM recommended)

```bash
git clone https://github.com/fishaudio/fish-speech
cd fish-speech
pip install -e .            # pulls torch (CUDA), sglang, etc.
huggingface-cli download fishaudio/s2-pro --local-dir checkpoints/s2-pro
# then follow the repo's S2 inference / SGLang server instructions, e.g.:
#   python -m tools.api_server --checkpoint checkpoints/s2-pro ...
# Persian: pass language / just feed Persian text; provide a reference wav for cloning.
```

## Zero-GPU alternatives for ElevenLabs-grade Persian

- **Fish Audio API / Playground:** https://fish.audio (hosted S2 — no local GPU).
- **ElevenLabs Multilingual v2** — supports Persian, hosted API (the actual gold standard).
- On CPU here, the best *runnable* options remain **Chatterbox-Persian**
  (expressive, cloning, ~20s/sentence) and **Mana-Piper** (clean, fast).

## Status

⛔ Not run — GPU/SGLang-only + 9 GB (exceeds free disk on this CPU environment).
Documented as the SOTA target; reproduce on a GPU or via the Fish Audio API.
