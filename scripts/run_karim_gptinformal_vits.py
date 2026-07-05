#!/usr/bin/env python3
"""Synthesize the Persian benchmark sentences with
karim23657/persian-tts-female-GPTInformal-Persian-vits.

Model:  https://huggingface.co/karim23657/persian-tts-female-GPTInformal-Persian-vits
Engine: Coqui TTS (coqui-tts) + espeak-ng phonemizer (fa)
Run with the coqui venv:  .venvs/coqui/bin/python scripts/run_karim_gptinformal_vits.py
Output: docs/audio/karim-gptinformal-vits/NN.wav
"""
import os
import re
import time
from huggingface_hub import hf_hub_download
from TTS.utils.synthesizer import Synthesizer

REPO = "karim23657/persian-tts-female-GPTInformal-Persian-vits"
CKPT = "best_model_98066.pth"   # highest-step checkpoint in the repo
CFG = "config.json"
FOLDER = "karim-gptinformal-vits"

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_FILE = os.path.join(HERE, "sample_texts", "sentences.txt")
OUT_DIR = os.path.join(HERE, "docs", "audio", FOLDER)


def read_sentences(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = re.match(r"^(\S+)\s+(.*)$", line)
            if m:
                items.append((m.group(1), m.group(2)))
    return items


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading {REPO} ...")
    ckpt = hf_hub_download(REPO, CKPT)
    cfg = hf_hub_download(REPO, CFG)
    print("Loading synthesizer ...")
    syn = Synthesizer(tts_checkpoint=ckpt, tts_config_path=cfg, use_cuda=False)
    sr = syn.tts_config.audio["sample_rate"]

    for sid, text in read_sentences(SENT_FILE):
        t0 = time.time()
        wav = syn.tts(text)
        dur = len(wav) / sr
        rtf = (time.time() - t0) / max(dur, 1e-6)
        out = os.path.join(OUT_DIR, f"{sid}.wav")
        syn.save_wav(wav, out)
        print(f"  [{sid}] {dur:5.2f}s audio  RTF={rtf:.2f}  -> {out}")

    print("Done. Wrote WAVs to", OUT_DIR)


if __name__ == "__main__":
    main()
