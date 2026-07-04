---
name: setup-chatterbox-persian
description: Set up Thomcles/Chatterbox-TTS-Persian-Farsi (Chatterbox fa finetune, voice cloning). GATED repo — needs a Hugging Face token with granted access. Read before attempting; documents the access + assembly steps.
---

# Setup: Thomcles/Chatterbox-TTS-Persian-Farsi

- **Hugging Face:** https://huggingface.co/Thomcles/Chatterbox-TTS-Persian-Farsi
- **Base model:** [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox)
- **Type:** Chatterbox (Llama-backbone T3 + S3Gen), expressive, voice cloning
- **License:** CC-BY-NC 4.0 (non-commercial)
- **Ships:** only `t3_fa.safetensors` (the Farsi-finetuned **T3** text-to-token stage)

## ⛔ Blocker 1 — gated repo

The repo is **gated**. Anonymous download fails:

```
GatedRepoError: 401 ... Cannot access gated repo ...
```

To unblock:
1. On the HF model page, **request/accept access** (logged in).
2. Create a read token: https://huggingface.co/settings/tokens
3. Expose it to the session: `export HF_TOKEN=hf_xxx` (or `huggingface-cli login`).

## Assembly (once access is granted)

The finetune is **only the T3 stage** — you need the rest of Chatterbox from the
base repo, then swap in `t3_fa.safetensors`:

```bash
.venvs/coqui/bin/pip install chatterbox-tts    # or resemble's package
```

```python
from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device="cpu")   # pulls base ResembleAI/chatterbox
# replace the T3 weights with the Farsi finetune:
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
t3 = load_file(hf_hub_download("Thomcles/Chatterbox-TTS-Persian-Farsi", "t3_fa.safetensors"))
model.t3.load_state_dict(t3, strict=False)
wav = model.generate("سلام دنیا", audio_prompt_path="ref_fa.wav")   # needs a Persian reference clip
```

(Exact API depends on the installed `chatterbox-tts` version — verify attribute
names for the T3 submodule.)

## ⚠️ Blocker 2 — CPU speed

Chatterbox is a ~0.5B Llama-backbone autoregressive model. On this **CPU-only**
box expect **minutes per sentence**. Realistic only with a GPU.

## Status

⏳ Not run — **gated** (no HF token in this environment) + very slow on CPU.
Resume once an HF token with access is provided.
