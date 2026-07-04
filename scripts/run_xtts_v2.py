#!/usr/bin/env python3
"""Synthesize the Persian benchmark sentences with coqui/XTTS-v2.

Model:  https://huggingface.co/coqui/XTTS-v2  (autoregressive, multilingual, voice cloning)
Engine: Coqui TTS (coqui-tts). Uses a built-in studio speaker + language="fa".
Run with the coqui venv:  .venvs/coqui/bin/python scripts/run_xtts_v2.py
Output: docs/audio/xtts-v2/NN.wav

NOTE: XTTS is autoregressive — expect ~30-60 s per sentence on CPU.
License: Coqui Public Model License (CPML), non-commercial.
"""
import os
os.environ["COQUI_TOS_AGREED"] = "1"
import re
import time
from TTS.api import TTS

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
SPEAKER = "Daisy Studious"   # built-in studio speaker (female)
FOLDER = "xtts-v2"

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
    print(f"Loading {MODEL} ...")
    tts = TTS(MODEL)
    for sid, text in read_sentences(SENT_FILE):
        out = os.path.join(OUT_DIR, f"{sid}.wav")
        t0 = time.time()
        tts.tts_to_file(text=text, speaker=SPEAKER, language="fa", file_path=out)
        print(f"  [{sid}] took {time.time()-t0:5.1f}s  -> {out}")
    print("Done. Wrote WAVs to", OUT_DIR)


if __name__ == "__main__":
    main()
