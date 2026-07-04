# Candidate Persian TTS Models

Source: [HF TTS + language=fa, sorted by trending](https://huggingface.co/models?pipeline_tag=text-to-speech&language=fa&sort=trending)

Legend: ✅ done & working · 🚧 in progress · ⏳ queued · ⚠️ blocked/heavy · ❌ failed

## Recommended starting set (CPU-friendly first)

| # | Model | Type | Engine | Voice | CPU? | Status |
|---|-------|------|--------|-------|------|--------|
| 1 | [`facebook/mms-tts-fas`](https://huggingface.co/facebook/mms-tts-fas) | VITS | 🤗 transformers | single (male-ish) | ✅ fast | ✅ |
| 2 | [`Kamtera/persian-tts-female-vits`](https://huggingface.co/Kamtera/persian-tts-female-vits) | VITS | Coqui TTS | female | ✅ fast | ✅ |
| 3 | [`Kamtera/persian-tts-male-vits`](https://huggingface.co/Kamtera/persian-tts-male-vits) | VITS | Coqui TTS | male | ✅ fast | ⏳ |
| 4 | [`Kamtera/persian-tts-female-glow_tts`](https://huggingface.co/Kamtera/persian-tts-female-glow_tts) | GlowTTS | Coqui TTS | female | ✅ ok | ⏳ |
| 5 | [`SeyedAli/Persian-Speech-synthesis`](https://huggingface.co/SeyedAli/Persian-Speech-synthesis) | VITS | Coqui TTS | female | ✅ ok | ⏳ |

## Stretch goals (heavier — slow on CPU, better on GPU)

| # | Model | Type | Why heavy | Status |
|---|-------|------|-----------|--------|
| 6 | [`coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2) | Autoregressive + voice cloning; `fa` supported | ~1.8 GB, slow on CPU | ⚠️ |
| 7 | [`facebook/mms-tts` multilingual](https://huggingface.co/facebook/mms-tts) | large multilingual | big | ⚠️ |

## Why this order

- **VITS models run in real-time or faster on CPU** and need no phonemizer
  binary in the MMS case → lowest-friction first wins.
- **Coqui models** (Kamtera, SeyedAli) need the `TTS`/`coqui-tts` library and
  `espeak-ng` for phonemization — one extra system dependency.
- **XTTS-v2** gives the most natural voice + cloning, but is autoregressive and
  painfully slow without a GPU; we keep it last.

## Evaluation criteria

For each model we note:
- **Naturalness** — does it sound human?
- **Pronunciation** — correct Persian phonemes, ezafe, numbers, dates?
- **Prosody** — sentence melody / stress.
- **Speed** — real-time factor (RTF) on this CPU.
- **Setup friction** — how hard was it to install/run.

Results table (filled as we go):

| Model | Natural | Pronun. | Prosody | RTF (CPU) | Notes |
|-------|---------|---------|---------|-----------|-------|
| mms-tts-fas | tbd (your ear) | reads digits one-by-one | tbd | 0.2–0.3 | zero-friction setup, no espeak |
| kamtera-female-vits | tbd (your ear) | espeak fa phonemizer | tbd | 0.24 | female voice, 24 kHz, isolated venv |
