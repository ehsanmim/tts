#!/usr/bin/env python3
"""F5-TTS Persian — tuned + Ezafe + KaamelDict word diacritics.

Text front-end (see fa_diacritics.py) before synthesis:
  1. normalize_fa    — digits/dates/phones → Persian words (num2fawords)
  2. apply_ezafe     — adds kasra ِ to Ezafe-bearing words via
                       abreza/persian-ezafe-albert (F1 98.7%)
  3. diacritize_sentence — adds fatha/kasra/damma/sukun to each word
                       via KaamelDict phoneme lookup (116k entries)
  4. F5-TTS infer    — nfe_step=48, cfg_strength=2.0

The combined pipeline gives F5-TTS explicit phonetic cues for both the
Ezafe linking vowel and word-internal short vowels (e.g. مَتْن vs مَتَن).

Run:    .venvs/f5/bin/python scripts/run_f5_persian_ezafe.py [N]
Output: docs/audio/f5-persian-ezafe/NN.wav
"""
import os
import sys
import time

from huggingface_hub import hf_hub_download

from fa_diacritics import Frontend, read_sentences

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_FILE = os.path.join(HERE, "sample_texts", "sentences.txt")
OUT_DIR = os.path.join(HERE, "docs", "audio", "f5-persian-ezafe")
REF_WAV = os.path.join(HERE, "docs", "audio", "kamtera-female-vits", "02.wav")
REF_ID = "02"
NFE = 48
CFG = 2.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 999


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    frontend = Frontend()

    from f5_tts.api import F5TTS
    ckpt  = hf_hub_download("Lumos675/F5_TTS_Persian", "model_last.pt")
    vocab = hf_hub_download("Lumos675/F5_TTS_Persian", "vocab.txt")
    print(f"Loading F5-TTS Persian (CPU, nfe={NFE}, cfg={CFG}) ...")
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab, device="cpu")

    sents = read_sentences(SENT_FILE)
    ref_text = frontend(dict(sents)[REF_ID])

    for sid, text in sents[:N]:
        final = frontend(text)
        if final != text:
            print(f"  [{sid}] → {final}")
        out = os.path.join(OUT_DIR, f"{sid}.wav")
        t0 = time.time()
        f5.infer(ref_file=REF_WAV, ref_text=ref_text, gen_text=final,
                 file_wave=out, nfe_step=NFE, cfg_strength=CFG, seed=42)
        print(f"  [{sid}] {time.time()-t0:5.1f}s  -> {out}")

    print("Done. Wrote WAVs to", OUT_DIR)


if __name__ == "__main__":
    main()
