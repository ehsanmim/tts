#!/usr/bin/env python3
"""Synthesize the diacritized benchmark sentences with IMS-Toucan / ToucanTTS.

A/B pair for run_toucan_persian.py: same model, diacritized input
(sentences_diacritized.txt — Full TTS front-end: digits→words, Ezafe kasra, harakat).
Run:    .venvs/IMS-Toucan/venv/bin/python scripts/run_toucan_persian_diac.py [--smoke]
Output: docs/audio/toucan-persian-diac/NN.wav
"""
import os
import re
import sys
import time

TTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.join(TTS_ROOT, ".venvs", "IMS-Toucan")
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

SENT_FILE = os.environ.get(
    "SENT_FILE",
    os.path.join(TTS_ROOT, "sample_texts", "sentences_diacritized.txt"),
)
OUT_DIR = os.path.join(TTS_ROOT, "docs", "audio", "toucan-persian-diac")
REF_WAV = os.path.join(TTS_ROOT, "docs", "audio", "kamtera-female-vits", "02.wav")
SMOKE = "--smoke" in sys.argv[1:]


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
    from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
    print("Loading ToucanTTS ...")
    tts = ToucanTTSInterface(device="cpu")
    tts.set_language("pes")
    if os.path.exists(REF_WAV):
        tts.set_utterance_embedding(REF_WAV)

    sents = read_sentences(SENT_FILE)
    if SMOKE:
        sents = sents[:1]
    for sid, text in sents:
        out = os.path.join(OUT_DIR, f"{sid}.wav")
        t0 = time.time()
        tts.read_to_file(text_list=[text], file_location=out, prosody_creativity=0.0)
        print(f"  [{sid}] {time.time()-t0:5.1f}s  -> {out}")

    print("Done. Wrote WAVs to", OUT_DIR)


if __name__ == "__main__":
    main()
