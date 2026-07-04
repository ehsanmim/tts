---
name: setup-xtts-v2
description: Notes on coqui/XTTS-v2 for Persian TTS. IMPORTANT finding — base XTTS-v2 does NOT support Persian (fa). Read before attempting XTTS for Persian; use a fa finetune instead.
---

# Setup: coqui/XTTS-v2 — and why it does NOT work for Persian (base)

- **Hugging Face:** https://huggingface.co/coqui/XTTS-v2
- **Architecture:** autoregressive (GPT-style) + HiFi-GAN decoder, multilingual, voice cloning
- **Engine:** [coqui-tts](https://github.com/idiap/coqui-ai-TTS) (`.venvs/coqui`)
- **License:** Coqui Public Model License (CPML) — **non-commercial**

## ⛔ Key finding

Base XTTS-v2 **does not support Persian**. Calling it with `language="fa"` raises:

```
AssertionError: ❗ Language fa is not supported. Supported languages are
['en','es','fr','de','it','pt','pl','tr','ru','nl','cs','ar','zh-cn','hu','ko','ja','hi']
```

So despite XTTS-v2 appearing on the HF `fa` TTS list (it's tagged multilingual),
it **cannot synthesize Persian** out of the box. Do not waste CPU time on it for
`fa`.

## What to do instead (for a heavy "quality ceiling" Persian voice)

- **XTTS Persian finetunes** — community checkpoints that add `fa` to the
  language set (search HF for `xtts persian` / `xtts fa`). Load via
  `TTS.tts.models.xtts.Xtts` from the finetune's `config.json` + checkpoint,
  which lists `fa` in `config.languages`.
- **Chatterbox Persian** — `Thomcles/Chatterbox-TTS-Persian-Farsi`,
  `mazrba/Chatterbox-TTS-Persian-gguf` (heavy; slow on CPU).

## The (non-working for fa) invocation, for reference

```python
import os; os.environ["COQUI_TOS_AGREED"]="1"
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")   # 58 built-in speakers
# tts.tts_to_file(text=..., speaker="Daisy Studious", language="fa", ...)  # ← fails: fa unsupported
```

Downloads/loads fine (~1.8 GB); the failure is purely the language gate.

## Attempted finetune: alikhabazian/XTTS_Persian — also blocked

https://huggingface.co/alikhabazian/XTTS_Persian ships a 5.7 GB finetune whose
`config.json` **does** list `fa`. But it is **incomplete for inference**:

- The GPT text-vocab was expanded to **10120 tokens** (checkpoint
  `gpt.text_embedding` = `[10120, 1024]`), but the repo **does not include the
  matching `vocab.json` tokenizer** — only `config.json` + `best_model.pth`.
- With the base coqui vocab (6681 tokens) the shapes mismatch
  (`size mismatch for gpt.text_embedding.weight: [10120] vs [6681]`), and even if
  forced, Persian text would tokenize against the wrong vocab → garbled output.
- README is empty (`license: mit` only) — no usage/tokenizer info.

Conclusion: **not reproducible as published.** A usable XTTS-fa needs the repo to
also publish its extended tokenizer. `scripts/run_xtts_persian.py` records the
exact (blocked) loading path for the next attempt.

## Status

❌ Base XTTS-v2: `fa` unsupported.
❌ alikhabazian/XTTS_Persian finetune: incomplete upload (missing tokenizer).
Both documented so we don't retry blindly.
