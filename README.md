# Persian TTS Quality Benchmark

A hands-on evaluation of Persian (فارسی) text-to-speech models from the
[Hugging Face TTS leaderboard](https://huggingface.co/models?pipeline_tag=text-to-speech&language=fa&sort=trending).

The goal: set up each candidate model, synthesize the **same Persian sample
sentences**, and compare output quality (naturalness, pronunciation, prosody,
speed) so we can pick the best voice.

## Repository layout

```
tts/
├── README.md              # this file
├── MODELS.md              # candidate models, status, and notes
├── models.json            # registry the comparison page is built from
├── sample_texts/          # the Persian sentences every model must speak
│   └── sentences.txt
├── scripts/               # synthesis + site tooling
│   ├── run_<model>.py     # one synthesis script per model
│   ├── make_demo.py       # stitch a model's clips into demo.wav
│   └── build_site.py      # generate docs/index.html
├── docs/                  # the GitHub Pages comparison site
│   ├── index.html         # side-by-side player (built from models.json)
│   └── audio/<model>/*.wav
└── .claude/skills/
    └── setup-<model>/     # reproducible setup notes per model
        └── SKILL.md
```

## How each model is added

For every model we evaluate, we produce four things:

1. **`scripts/run_<model>.py`** — a self-contained synthesis script.
2. **`docs/audio/<model>/`** — `.wav` output for each sample sentence.
3. **an entry in `models.json`** — so it appears on the comparison page.
4. **`.claude/skills/setup-<model>/SKILL.md`** — install steps, the Hugging
   Face link, dependencies, quirks, and CPU/GPU notes so the setup can be
   reproduced next time without re-discovering everything.

## Comparison site (GitHub Pages)

The side-by-side player lives at [`docs/index.html`](docs/index.html) and is
generated from `models.json` + the audio under `docs/audio/`:

```bash
python3 scripts/build_site.py            # small page, references audio files (for Pages)
python3 scripts/build_site.py --embed    # standalone page, audio inlined as base64
```

To publish: **Settings → Pages → Deploy from branch → `main` / `/docs`**.

## Environment (this run)

| Resource | Value |
|----------|-------|
| Compute  | CPU only (no GPU) |
| Cores    | 4 |
| RAM      | 15 GB |
| Disk free| ~31 GB |
| Python   | 3.11 |

**Implication:** we prioritize lightweight **VITS-based** models that run
comfortably on CPU. Heavy autoregressive / voice-cloning models (XTTS-v2,
StyleTTS2, Bark) are included as stretch goals but will be slow.

## Sample sentences

See [`sample_texts/sentences.txt`](sample_texts/sentences.txt). The same text
is fed to every model for a fair comparison.

## Status

See [`MODELS.md`](MODELS.md) for the live checklist of which models are set up,
working, or blocked.
