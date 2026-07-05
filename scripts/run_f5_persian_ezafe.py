#!/usr/bin/env python3
"""F5-TTS Persian — tuned + Ezafe diacritization.

Three-stage text front-end before synthesis:
  1. normalize_fa  — digits/dates/phones → Persian words (num2fawords)
  2. apply_ezafe   — adds kasra ِ to Ezafe-bearing words via
                     abreza/persian-ezafe-albert (F1 98.7%)
  3. F5-TTS infer  — nfe_step=48, cfg_strength=2.0 (same as tuned)

Kasra in the input text gives F5-TTS an explicit phonetic cue for the
linking vowel, potentially improving Ezafe pronunciation.

Run:    .venvs/f5/bin/python scripts/run_f5_persian_ezafe.py [N]
Output: docs/audio/f5-persian-ezafe/NN.wav
"""
import os
import re
import sys
import time

import torch
from huggingface_hub import hf_hub_download
from num2fawords import words
from transformers import AutoModelForTokenClassification, AutoTokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_FILE = os.path.join(HERE, "sample_texts", "sentences.txt")
OUT_DIR = os.path.join(HERE, "docs", "audio", "f5-persian-ezafe")
REF_WAV = os.path.join(HERE, "docs", "audio", "kamtera-female-vits", "02.wav")
REF_ID = "02"
NFE = 48
CFG = 2.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 999

KASRA = "ِ"  # Arabic kasra U+0650

# Persian/Arabic-Indic digits → ASCII
_DIGIT_MAP = {}
for _base in (0x06F0, 0x0660):
    for _i in range(10):
        _DIGIT_MAP[_base + _i] = str(_i)


def normalize_fa(text: str) -> str:
    text = text.translate(_DIGIT_MAP)
    text = re.sub(r"\d+(?:/\d+)+",
                  lambda m: " ".join(words(int(p)) for p in m.group().split("/")),
                  text)
    text = re.sub(r"\d{5,}",
                  lambda m: " ".join(words(int(d)) for d in m.group()),
                  text)
    text = re.sub(r"\d+", lambda m: words(int(m.group())), text)
    return text


def load_albert():
    print("Loading abreza/persian-ezafe-albert ...")
    tok = AutoTokenizer.from_pretrained("abreza/persian-ezafe-albert")
    model = AutoModelForTokenClassification.from_pretrained("abreza/persian-ezafe-albert")
    model.eval()
    return tok, model, model.config.id2label


def apply_ezafe(tok, model, id2label, text: str) -> str:
    token_list = text.split()
    if not token_list:
        return text
    enc = tok(token_list, is_split_into_words=True,
              return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    preds = logits.argmax(-1).tolist()
    word_ids = enc.word_ids()

    word_labels: dict[int, str] = {}
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid not in word_labels:
            word_labels[wid] = id2label[preds[pos]]

    result = []
    for i, w in enumerate(token_list):
        if word_labels.get(i) == "NEEDS_EZAFE":
            result.append(w + KASRA)
        else:
            result.append(w)
    return " ".join(result)


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

    albert_tok, albert_model, id2label = load_albert()

    from f5_tts.api import F5TTS
    ckpt = hf_hub_download("Lumos675/F5_TTS_Persian", "model_last.pt")
    vocab = hf_hub_download("Lumos675/F5_TTS_Persian", "vocab.txt")
    print(f"Loading F5-TTS Persian (CPU, nfe={NFE}, cfg={CFG}) ...")
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab, device="cpu")

    sents = read_sentences(SENT_FILE)
    ref_raw = dict(sents)[REF_ID]
    ref_text = apply_ezafe(albert_tok, albert_model, id2label, normalize_fa(ref_raw))

    for sid, text in sents[:N]:
        normed = normalize_fa(text)
        final = apply_ezafe(albert_tok, albert_model, id2label, normed)
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
