#!/usr/bin/env python3
"""Synthesize the Persian benchmark sentences with MahtaFetrat/Mana-Persian-Piper.

Model:  https://huggingface.co/MahtaFetrat/Mana-Persian-Piper  (Piper VITS, ONNX)
        Trained on the curated ManaTTS Persian corpus. Voice: fa_IR-mana-medium.
Engine: piper-tts (onnxruntime) + bundled espeak-ng phonemizer.
Run with the piper venv:  .venvs/piper/bin/python scripts/run_mana_piper.py
Output: docs/audio/mana-piper/NN.wav
"""
import os
import re
import time
import wave
from huggingface_hub import hf_hub_download
from piper import PiperVoice

REPO = "MahtaFetrat/Mana-Persian-Piper"
FOLDER = "mana-piper"
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
    onnx = hf_hub_download(REPO, "fa_IR-mana-medium.onnx")
    cfg = hf_hub_download(REPO, "fa_IR-mana-medium.onnx.json")
    print("Loading Piper voice ...")
    voice = PiperVoice.load(onnx, config_path=cfg, use_cuda=False)

    for sid, text in read_sentences(SENT_FILE):
        out = os.path.join(OUT_DIR, f"{sid}.wav")
        t0 = time.time()
        with wave.open(out, "wb") as w:
            voice.synthesize_wav(text, w)
        print(f"  [{sid}] {time.time()-t0:5.2f}s  -> {out}")

    print("Done. Wrote WAVs to", OUT_DIR)


if __name__ == "__main__":
    main()
