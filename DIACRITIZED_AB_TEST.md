# Plan — A/B test: plain text vs. diacritized front-end output

**Goal:** measure how much the Persian TTS front-end (`scripts/fa_diacritics.py`)
improves each model, by synthesizing the **same five sentences twice** — once
from the raw unvoweled text, once from the diacritized text — and comparing them
side by side.

This is meant to be handed to Claude on the web (claude.ai/code), which has more
compute than the CPU-only box these models were first set up on.

## The two inputs

| File | Contents |
|------|----------|
| [`sample_texts/sentences.txt`](sample_texts/sentences.txt) | original, unvoweled ("none-marked") text |
| [`sample_texts/sentences_diacritized.txt`](sample_texts/sentences_diacritized.txt) | same sentences after the **Full TTS front-end**: digits→words, Ezafe kasra (abreza/persian-ezafe-albert), word-internal harakat (KaamelDict), homograph fixes (صِفْر not صَفَر, نُه not نَه) |

Both files use the same `NN<TAB>text` format and the same sentence ids (01–05),
so any model can read either one with the existing `read_sentences()`.

The diacritized text was produced by:

```bash
# CPU-only Windows example (bin/ is Scripts/ on Windows)
.venvs/diacritics/Scripts/python.exe -c "import sys; sys.path.insert(0,'scripts'); \
from fa_diacritics import Frontend, read_sentences; fe=Frontend(); \
[print(f'{i}\t{fe(t)}') for i,t in read_sentences('sample_texts/sentences.txt')]"
```

(Set `PYTHONUTF8=1` on Windows so Persian prints correctly.)

## What we expect to learn

Not every engine will benefit — that is the point of the test:

- **espeak-ng models** (Kamtera female/male, karim GPTInformal, Mana-Piper,
  Toucan) re-phonemize the text with their own G2P. They may **ignore or even
  choke on** the added harakat. Worth confirming empirically — if quality drops,
  that model should keep the plain input.
- **Char/byte-vocab models** (F5-TTS, Chatterbox, MMS/SeyedAli) consume the
  characters more directly, so explicit harakat should most help Ezafe and
  homograph disambiguation. F5 and Chatterbox already ship `-ezafe` variants
  that do this internally — this test extends the same idea to **every** model
  on identical text.

## Steps for the web run

For each model in [`models.json`](models.json) that we want to A/B test:

1. **Synthesize the diacritized input.** Point the model's
   `scripts/run_<model>.py` at `sentences_diacritized.txt` instead of the
   hardcoded `sentences.txt`. Two clean options:
   - add a `SENT_FILE = os.environ.get("SENT_FILE", <default>)` override to the
     script (preferred — non-destructive, reusable), then run with
     `SENT_FILE=sample_texts/sentences_diacritized.txt`, **or**
   - copy the run script to `run_<model>_diac.py` with the path swapped.
2. **Write the audio to a distinct folder** so both versions survive:
   `docs/audio/<model>-diac/01.wav … 05.wav` (the plain audio stays in
   `docs/audio/<model>/`).
3. **Register the diacritized variant** in `models.json` as a new entry:
   - `id`: `<model>-diac`
   - `name`: `<name> — diacritized input`
   - reuse the same `hf_url` / `engine` / `sample_rate`
   - `notes`: "same model, fed `sentences_diacritized.txt` (Full TTS front-end
     output) instead of raw text — A/B against `<model>`."
4. **Rebuild the comparison site** so the plain and `-diac` rows sit next to each
   other:
   ```bash
   python3 scripts/build_site.py            # references audio files
   python3 scripts/build_site.py --embed    # standalone, audio inlined
   ```

### Priority order (fastest / most informative first)

1. `mana-piper` — ~0.2 s/sentence, instant A/B.
2. `mms-tts-fas`, `seyedali-mms` — fast, char-ish vocab, good harakat probe.
3. `kamtera-female-vits`, `kamtera-male-vits`, `karim-gptinformal-vits` —
   espeak; confirms whether harakat helps or hurts espeak G2P.
4. `toucan-persian` — SOTA CPU quality, ~5–8 s/sentence.
5. `f5-persian` / `chatterbox-persian` — heavy; compare the **plain** variant
   against the diacritized input here (note: `f5-persian-ezafe` and
   `chatterbox-persian-ezafe` already apply this front-end internally, so they
   are the existing reference point for "with front-end").

## Status (first 5 models completed)

| Model | Plain audio | Diacritized audio | Notes |
|-------|------------|-------------------|-------|
| `mana-piper` | ✅ `docs/audio/mana-piper/` | ✅ `docs/audio/mana-piper-diac/` | espeak-ng re-phonemizes — test whether harakat helps/hurts |
| `mms-tts-fas` | ✅ `docs/audio/mms-tts-fas/` | ✅ `docs/audio/mms-tts-fas-diac/` | char-level tokenizer — should benefit from explicit harakat |
| `seyedali-mms` | ✅ `docs/audio/seyedali-mms/` | ✅ `docs/audio/seyedali-mms-diac/` | char-level tokenizer — same code path as mms-tts-fas |
| `kamtera-female-vits` | ✅ `docs/audio/kamtera-female-vits/` | ✅ `docs/audio/kamtera-female-vits-diac/` | espeak-ng re-phonemizes |
| `kamtera-male-vits` | ✅ `docs/audio/kamtera-male-vits/` | ✅ `docs/audio/kamtera-male-vits-diac/` | espeak-ng re-phonemizes |

The comparison page (`docs/index.html`) now shows each model as a **PLAIN / DIAC** pair
(blue row vs green row) for every sentence, so the A/B comparison is one click.

## Evaluation

Judge each plain vs. `-diac` pair on the criteria already in
[`MODELS.md`](MODELS.md):

- **Pronunciation** — Ezafe (ساعتِ سه), homographs (صِفْر/نُه), numbers & the
  date/phone in sentence 05. This is where the front-end should win.
- **Naturalness / prosody** — did the added marks improve or distort melody?
- **Regressions** — any model that sounds worse on diacritized text (likely some
  espeak ones) → keep it on plain input and record why.

Fill the results into `MODELS.md`, noting for each model whether the diacritized
front-end **helped, hurt, or was neutral**, so we end with a clear rule for which
engines should be fed front-end output.
