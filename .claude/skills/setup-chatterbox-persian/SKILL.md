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

## Assembly (VERIFIED WORKING) ✅

The finetune is **only the T3 stage** of the **multilingual** Chatterbox
(`text_emb` = 2454 tokens → `ChatterboxMultilingualTTS`, not the English one).
Load the base multilingual model, then swap in `t3_fa.safetensors`.

```bash
python3 -m venv .venvs/chatterbox
.venvs/chatterbox/bin/pip install chatterbox-tts
```

Run (needs the token in the env, **never** commit it):

```bash
HF_TOKEN=hf_... .venvs/chatterbox/bin/python scripts/run_chatterbox_persian.py [--smoke]
python3 scripts/make_demo.py chatterbox-persian
```

### Two gotchas that had to be patched (see `scripts/run_chatterbox_persian.py`)

1. **Language gate:** `generate()` validates `language_id` against
   `SUPPORTED_LANGUAGES`, which has no `fa`. Add it before use:
   `chatterbox.mtl_tts.SUPPORTED_LANGUAGES["fa"] = "Persian"`.
2. **Chinese segmenter download:** `MTLTokenizer.__init__` eagerly builds a
   `ChineseCangjieConverter` (pkuseg) that downloads a model and **fails a hash
   check behind the proxy**. It's only used for `zh`, so stub it:
   `chatterbox.models.tokenizers.tokenizer.ChineseCangjieConverter = <noop>`.

T3 loads with **0 missing / 0 unexpected** keys. `[fa]` isn't a single token (it
splits into subtokens), but that matches how the finetune was trained, so output
is fine. A **Persian reference clip** is required for voice conditioning — we
reuse `docs/audio/kamtera-female-vits/02.wav`.

## CPU speed

Chatterbox is a ~0.5B Llama-backbone autoregressive model, but in practice it
ran at **~20 s per sentence** on this 4-core CPU (not minutes) — usable for a
small benchmark, still far slower than the VITS models (which are <1 s).

## Status

✅ **Working** with an HF token. Sample generated for sentences 01–05 (voice
cloned from a Persian reference clip). The only heavy/cloning model in the set
that actually produces Persian.
