#!/usr/bin/env python3
"""Synthesize the diacritized benchmark sentences with SeyedAli/Persian-Speech-synthesis-MMS.

A/B pair for run_seyedali_mms.py: same model, diacritized input
(sentences_diacritized.txt — Full TTS front-end: digits→words, Ezafe kasra, harakat).
Output: docs/audio/seyedali-mms-diac/NN.wav
"""
import os
import re
import time
import wave
import numpy as np
import torch
from transformers import VitsModel, AutoTokenizer

MODEL_ID = "SeyedAli/Persian-Speech-synthesis-MMS"
FOLDER = "seyedali-mms-diac"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_FILE = os.environ.get(
    "SENT_FILE",
    os.path.join(HERE, "sample_texts", "sentences_diacritized.txt"),
)
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


def write_wav(path, audio, sr):
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading {MODEL_ID} ...")
    model = VitsModel.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()
    sr = model.config.sampling_rate

    for sid, text in read_sentences(SENT_FILE):
        inputs = tokenizer(text, return_tensors="pt")
        t0 = time.time()
        with torch.no_grad():
            audio = model(**inputs).waveform.squeeze().cpu().numpy()
        dur = len(audio) / sr
        rtf = (time.time() - t0) / max(dur, 1e-6)
        out = os.path.join(OUT_DIR, f"{sid}.wav")
        write_wav(out, audio, sr)
        print(f"  [{sid}] {dur:5.2f}s audio  RTF={rtf:.2f}  -> {out}")

    print("Done. Wrote WAVs to", OUT_DIR)


if __name__ == "__main__":
    main()
