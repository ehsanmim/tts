#!/usr/bin/env python3
"""Synthesize the Persian benchmark sentences with IMS-Toucan / ToucanTTS.

Model:  https://github.com/DigitalPhonetics/IMS-Toucan  (massively multilingual, 7000+ langs)
        Persian language id = "pes" (Western Farsi); flow-matching decoder; zero-shot cloning.
Engine: the IMS-Toucan repo (cloned to .venvs/IMS-Toucan) + its venv.
Run:    .venvs/IMS-Toucan/venv/bin/python scripts/run_toucan_persian.py [--smoke]
Output: docs/audio/toucan-persian/NN.wav

NOTE: downloads the default multilingual checkpoint on first run. CPU inference is
slow-ish (seconds per sentence). Reference voice conditions the timbre.
"""
import os
import re
import sys
import time

TTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.join(TTS_ROOT, ".venvs", "IMS-Toucan")
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)  # the repo uses cwd-relative model/cache paths

SENT_FILE = os.path.join(TTS_ROOT, "sample_texts", "sentences.txt")
OUT_DIR = os.path.join(TTS_ROOT, "docs", "audio", "toucan-persian")
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
    print("Loading ToucanTTS (downloads default multilingual model on first run) ...")
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
