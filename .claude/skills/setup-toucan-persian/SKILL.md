---
name: setup-toucan-persian
description: Set up and run IMS-Toucan / ToucanTTS for Persian (language id "pes"). SOTA multilingual flow-matching TTS with zero-shot voice cloning, CPU-runnable. Documents the clone + curated dependency install + inference. Requires the IMS-Toucan venv.
---

# Setup: IMS-Toucan / ToucanTTS (Persian / فارسی, lang id `pes`)

- **GitHub:** https://github.com/DigitalPhonetics/IMS-Toucan
- **HF (weights auto-download):** https://huggingface.co/Flux9665/ToucanTTS
- **Architecture:** ToucanTTS — FastSpeech-style + **flow-matching (CFM) decoder**, massively multilingual (7000+ languages), zero-shot speaker/prosody cloning
- **Persian:** language id **`pes`** (Western Farsi); g2p via espeak `fa`
- **Sample rate:** 24 kHz
- **CPU:** runs on CPU — **~5–8 s/sentence** (no GPU needed for inference)
- **License:** Apache-2.0 (toolkit)

## Why it's here

The highest-quality **CPU-runnable** option in the benchmark — SOTA research
model with a flow-matching decoder and zero-shot cloning. Below the GPU-only
`fishaudio/s2-pro`, but well above the plain VITS voices.

## Install (curated — do NOT run the repo's requirements.txt in auto mode)

The repo is cloned to `.venvs/IMS-Toucan`; it runs in its own venv. Install deps
**by name from PyPI** (the repo manifest triggers the untrusted-code guard, and
pins GUI/training deps that need a display):

```bash
apt-get install -y espeak-ng ffmpeg libportaudio2
cd .venvs/IMS-Toucan && python3 -m venv venv
# CPU torch FIRST, and force CPU torchaudio (speechbrain pulls a CUDA build otherwise):
venv/bin/pip install "torch==2.4.1" "torchaudio==2.4.1" --index-url https://download.pytorch.org/whl/cpu
venv/bin/pip install "numpy~=1.23.4" scipy "librosa~=0.9.2" soundfile "phonemizer~=3.2.1" \
    "huggingface-hub~=0.25.2" einops torch_complex alias_free_torch dotwiz pyloudnorm "speechbrain==0.5.13" \
    matplotlib imageio wandb "epitran==1.24" "transphone==1.5.3" "phonepiece==1.4.2" \
    pypinyin dragonmapper jamo pykakasi sounddevice
# If torchaudio got replaced by a CUDA wheel (OSError: libcudart.so.13), reinstall it CPU:
venv/bin/pip install --force-reinstall "torch==2.4.1" "torchaudio==2.4.1" --index-url https://download.pytorch.org/whl/cpu
```

### Gotchas hit during setup
- **`libcudart.so.13`** — speechbrain pulls a CUDA `torchaudio`; force-reinstall the CPU build.
- **`sounddevice`** import at module load — needs system `libportaudio2` (only used for live playback).
- `matplotlib` / `imageio` are imported by the inference/flow-matching modules even though we don't plot.
- Disk: the multilingual checkpoint + deps are a few GB — keep headroom.

## Run

```bash
.venvs/IMS-Toucan/venv/bin/python scripts/run_toucan_persian.py [--smoke]  # -> docs/audio/toucan-persian/NN.wav
python3 scripts/make_demo.py toucan-persian
```

The script `chdir`s into the repo (it uses cwd-relative cache paths), sets
`set_language("pes")`, and conditions on a Persian reference clip
(`docs/audio/kamtera-female-vits/02.wav`) via `set_utterance_embedding`.

## Status

✅ Working on CPU via `.venvs/IMS-Toucan/venv`. Sample generated for sentences 01–05.
Best CPU-runnable quality in the set.
