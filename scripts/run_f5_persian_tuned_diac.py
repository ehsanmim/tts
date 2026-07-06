#!/usr/bin/env python3
"""Synthesize the diacritized benchmark sentences with Lumos675/F5_TTS_Persian (tuned).

A/B pair for run_f5_persian_tuned.py: same model and tuned settings (nfe=48, cfg=2.0),
diacritized input (sentences_diacritized.txt — Full TTS front-end output — digits already
converted to words, so no normalize_fa needed).
Run:    .venvs/f5/bin/python scripts/run_f5_persian_tuned_diac.py [N]
Output: docs/audio/f5-persian-tuned-diac/NN.wav
"""
import os
import re
import sys
import time
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_FILE = os.environ.get(
    "SENT_FILE",
    os.path.join(HERE, "sample_texts", "sentences_diacritized.txt"),
)
OUT_DIR = os.path.join(HERE, "docs", "audio", "f5-persian-tuned-diac")
REF_WAV = os.path.join(HERE, "docs", "audio", "kamtera-female-vits", "02.wav")
REF_ID = "02"
NFE = 48
CFG = 2.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 999


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
    ref_text = dict(sents)[REF_ID]

    for sid, text in sents[:N]:
        out = os.path.join(OUT_DIR, f"{sid}.wav")
        t0 = time.time()
        f5.infer(ref_file=REF_WAV, ref_text=ref_text, gen_text=text,
                 file_wave=out, nfe_step=NFE, cfg_strength=CFG, seed=42)
        print(f"  [{sid}] {time.time()-t0:5.1f}s  -> {out}")

    print("Done. Wrote WAVs to", OUT_DIR)


if __name__ == "__main__":
    main()
