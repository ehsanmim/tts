#!/usr/bin/env python3
"""Tuned run of Lumos675/F5_TTS_Persian — better Persian pronunciation.

Two changes vs scripts/run_f5_persian.py:
  1. Persian text normalization (the biggest fix): digits -> Persian words,
     phone numbers read digit-by-digit, dates read part-by-part. Raw digits are
     the #1 source of mispronunciation.
  2. More sampling steps (nfe_step 32 -> 48) + slightly higher guidance for
     stability/adherence.

Run:    .venvs/f5/bin/python scripts/run_f5_persian_tuned.py [N]
Output: docs/audio/f5-persian-tuned/NN.wav
"""
import os
import re
import sys
import time
from huggingface_hub import hf_hub_download
from num2fawords import words

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_FILE = os.path.join(HERE, "sample_texts", "sentences.txt")
OUT_DIR = os.path.join(HERE, "docs", "audio", "f5-persian-tuned")
REF_WAV = os.path.join(HERE, "docs", "audio", "kamtera-female-vits", "02.wav")
REF_ID = "02"
NFE = 48
CFG = 2.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 999

# Persian/Arabic-indic digits -> ASCII
_DIGIT_MAP = {}
for base in (0x06F0, 0x0660):  # Persian, Arabic-Indic
    for i in range(10):
        _DIGIT_MAP[base + i] = str(i)


def normalize_fa(text: str) -> str:
    text = text.translate(_DIGIT_MAP)
    # dates / slash-joined numbers: read each part as a cardinal
    text = re.sub(r"\d+(?:/\d+)+",
                  lambda m: " ".join(words(int(p)) for p in m.group().split("/")),
                  text)
    # long digit runs (phone numbers) -> digit by digit
    text = re.sub(r"\d{5,}",
                  lambda m: " ".join(words(int(d)) for d in m.group()), text)
    # remaining numbers -> cardinal words
    text = re.sub(r"\d+", lambda m: words(int(m.group())), text)
    return text


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
    from f5_tts.api import F5TTS
    ckpt = hf_hub_download("Lumos675/F5_TTS_Persian", "model_last.pt")
    vocab = hf_hub_download("Lumos675/F5_TTS_Persian", "vocab.txt")
    print(f"Loading F5-TTS Persian (CPU, nfe={NFE}, cfg={CFG}) ...")
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab, device="cpu")

    sents = read_sentences(SENT_FILE)
    ref_text = normalize_fa(dict(sents)[REF_ID])

    for sid, text in sents[:N]:
        norm = normalize_fa(text)
        if norm != text:
            print(f"  [{sid}] normalized: {norm}")
        out = os.path.join(OUT_DIR, f"{sid}.wav")
        t0 = time.time()
        f5.infer(ref_file=REF_WAV, ref_text=ref_text, gen_text=norm,
                 file_wave=out, nfe_step=NFE, cfg_strength=CFG, seed=42)
        print(f"  [{sid}] {time.time()-t0:5.1f}s  -> {out}")

    print("Done. Wrote WAVs to", OUT_DIR)


if __name__ == "__main__":
    main()
