#!/usr/bin/env python3
"""Concatenate a model's per-sentence WAVs into one demo.wav (with gaps).

Usage: python3 scripts/make_demo.py <model-folder-name>
       e.g. python3 scripts/make_demo.py mms-tts-fas
"""
import os
import sys
import glob
import wave
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32767.0
    return data, sr


def main():
    name = sys.argv[1]
    d = os.path.join(HERE, "docs", "audio", name)
    files = sorted(glob.glob(os.path.join(d, "[0-9]*.wav")))
    if not files:
        print("no per-sentence wavs in", d)
        sys.exit(1)
    chunks, sr = [], None
    for f in files:
        data, s = read_wav(f)
        sr = s
        chunks.append(data)
        chunks.append(np.zeros(int(0.6 * sr), dtype=np.float32))  # 0.6s gap
    audio = np.concatenate(chunks)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    out = os.path.join(d, "demo.wav")
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    print("wrote", out, f"({len(audio)/sr:.1f}s @ {sr} Hz)")


if __name__ == "__main__":
    main()
