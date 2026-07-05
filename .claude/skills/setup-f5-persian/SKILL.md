---
name: setup-f5-persian
description: Set up and run Lumos675/F5_TTS_Persian (F5-TTS flow-matching finetune) for Persian. High-quality zero-shot voice cloning, CPU-runnable (~90s/sentence). Ships its own vocab. Requires the f5 venv.
---

# Setup: Lumos675/F5_TTS_Persian

- **Hugging Face:** https://huggingface.co/Lumos675/F5_TTS_Persian
- **Base:** [F5-TTS](https://github.com/SWivid/F5-TTS) v1 Base (`F5TTS_v1_Base`) — flow-matching DiT + vocos vocoder
- **Voice:** zero-shot cloning (needs a ~5–10s reference clip + its transcript)
- **Sample rate:** 24 kHz
- **CPU:** ~90 s/sentence (nfe_step=32) — much faster than s2-pro
- **License:** Apache-2.0
- **Ships:** `model_last.pt` (5.4 GB full training ckpt — f5-tts extracts EMA weights), `vocab.txt` (2609 char tokens — **complete**, unlike the XTTS finetune), `setting.json`

## Install

```bash
python3 -m venv .venvs/f5
.venvs/f5/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venvs/f5/bin/pip install f5-tts
```

### Gotcha — torchcodec/CUDA on CPU
`f5-tts` pulls a **CUDA `torchcodec`** that fails on CPU
(`OSError: libnvrtc.so.13`). Fix: drop torchcodec and pin the torch stack to
2.8 (which still has torchaudio's soundfile/ffmpeg backend):

```bash
.venvs/f5/bin/pip uninstall -y torchcodec
.venvs/f5/bin/pip install --force-reinstall "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cpu
apt-get install -y ffmpeg    # torchaudio.load backend
```

(The `f5-tts requires torchcodec` pip warning is harmless — audio loads via
torchaudio's StreamReader/ffmpeg.)

## Run

```bash
.venvs/f5/bin/python scripts/run_f5_persian.py        # -> docs/audio/f5-persian/NN.wav
python3 scripts/make_demo.py f5-persian
```

## Minimal usage snippet

```python
from huggingface_hub import hf_hub_download
from f5_tts.api import F5TTS
ckpt = hf_hub_download("Lumos675/F5_TTS_Persian", "model_last.pt")
vocab = hf_hub_download("Lumos675/F5_TTS_Persian", "vocab.txt")
f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab, device="cpu")
f5.infer(ref_file="ref_fa.wav", ref_text="<transcript of ref>",
         gen_text="سلام دنیا", file_wave="out.wav", nfe_step=32)
```

## Tuning (see scripts/run_f5_persian_tuned.py)

The tuned variant (`docs/audio/f5-persian-tuned/`) fixes the common Persian
mispronunciations:

1. **Text normalization (biggest win):** raw digits are the #1 mispronunciation
   source. `num2fawords` converts numbers → Persian words; phone numbers are read
   digit-by-digit, dates part-by-part. `pip install num2fawords`.
2. **`nfe_step` 32 → 48:** more flow-matching steps = steadier pronunciation
   (slower: ~2 min/sentence). Try 48–64 for quality, 16 for speed.
3. **`cfg_strength`** (2.0): raise for more text adherence; too high over-articulates.
4. **Reference clip + accurate `ref_text`** strongly drive voice + pronunciation —
   a clean, natural (ideally human) Persian reference helps most.
5. `speed`, `sway_sampling_coef` (-1) affect pacing/trajectory.

## Status

✅ Working on CPU via `.venvs/f5`. Full 5-sentence set generated. A leading open
naturalness option for Persian; flow-matching like ToucanTTS but a dedicated F5 finetune.
