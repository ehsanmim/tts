#!/usr/bin/env python3
"""Synthesize Persian benchmark sentences with Lumos675/F5_TTS_Persian.

Model:  https://huggingface.co/Lumos675/F5_TTS_Persian  (F5-TTS v1 Base finetune)
Engine: f5-tts (flow-matching DiT + vocos vocoder). Zero-shot voice cloning.
Run:    .venvs/f5/bin/python scripts/run_f5_persian.py [N]
Output: docs/audio/f5-persian/NN.wav

Ships its own char vocab (2609 tokens) — the finetune IS complete. CPU-runnable
(minutes/sentence). Voice cloned from a Persian reference clip. License: Apache-2.0.
"""
import os
import re
import sys
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_FILE = os.path.join(HERE, "sample_texts", "sentences.txt")
OUT_DIR = os.path.join(HERE, "docs", "audio", "f5-persian")
REF_WAV = os.path.join(HERE, "docs", "audio", "kamtera-female-vits", "02.wav")
REF_ID = "02"
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
    print("Loading F5-TTS Persian (CPU) ...")
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab, device="cpu")

    sents = read_sentences(SENT_FILE)
    ref_text = dict(sents)[REF_ID]

    import time
    for sid, text in sents[:N]:
        out = os.path.join(OUT_DIR, f"{sid}.wav")
        t0 = time.time()
        f5.infer(ref_file=REF_WAV, ref_text=ref_text, gen_text=text,
                 file_wave=out, nfe_step=32, seed=42)
        print(f"  [{sid}] {time.time()-t0:5.1f}s  -> {out}")

    print("Done. Wrote WAVs to", OUT_DIR)


if __name__ == "__main__":
    main()
