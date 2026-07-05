#!/usr/bin/env python3
"""Shared Persian text front-end: digits→words, Ezafe, word-internal harakat.

Three stages, applied in order before feeding text to a TTS model:
  1. normalize_fa    — digits/dates/phones → Persian words (num2fawords)
  2. apply_ezafe     — adds kasra ِ to Ezafe-bearing words via
                       abreza/persian-ezafe-albert (F1 98.7%)
  3. diacritize_sentence — adds fatha/kasra/damma/sukun to each word
                       via KaamelDict phoneme lookup (116k entries)

Used by run_f5_persian_ezafe.py and run_chatterbox_persian_ezafe.py.
"""
import ast
import csv
import re

import torch
from huggingface_hub import hf_hub_download
from num2fawords import words
from transformers import AutoModelForTokenClassification, AutoTokenizer

KASRA  = "ِ"   # U+0650
FATHA  = "َ"   # U+064E
DAMMA  = "ُ"   # U+064F
SUKUN  = "ْ"   # U+0652

# Persian/Arabic-Indic digits → ASCII
_DIGIT_MAP = {}
for _base in (0x06F0, 0x0660):
    for _i in range(10):
        _DIGIT_MAP[_base + _i] = str(_i)

# KaamelDict phoneme notation
SHORT_V = {"a": FATHA, "e": KASRA, "o": DAMMA}
LONG_V  = set("Aiu")
CONS_PH = set("bptsjcChxdzrZSGfqkglmnhyv?")  # C = ch (KaamelDict uses both cases)
# Maps Persian consonant chars to their primary phoneme symbol in KaamelDict
CHAR_PH = {
    "ب": "b", "پ": "p", "ت": "t", "ث": "s", "ج": "j", "چ": "c",
    "ح": "h", "خ": "x", "د": "d", "ذ": "z", "ر": "r", "ز": "z",
    "ژ": "Z", "س": "s", "ش": "S", "ص": "s", "ض": "z", "ط": "t",
    "ظ": "z", "ع": "?", "غ": "G", "ف": "f", "ق": "q", "ک": "k",
    "گ": "g", "ل": "l", "م": "m", "ن": "n", "ه": "h",
}
# Zero-width chars and tatweel to skip silently
_SKIP = {"‌", "‍", "ـ"}
# Punctuation to strip when looking up a word
_PUNCT = set("،.؟!()[]؛,:»«")
# KaamelDict's first pronunciation variant is the wrong homograph for these
# digit-derived words: صفر lists safar (the month) before sefr (zero), and
# نه lists nah (no) before noh (nine). Applied to every matching token, which
# is right for text coming out of normalize_fa's digit spelling.
PH_OVERRIDES = {
    "صفر": "sefr",    # صِفْر zero, not صَفَر the month Safar
    "نه":  "noh",     # نُه nine, not نَه no
    "ساعت": "sA?at",  # ساعَت — the stored variant sAat drops the glottal stop
}


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


def load_kaameldict() -> dict:
    print("Loading KaamelDict ...")
    path = hf_hub_download("MahtaFetrat/KaamelDict", "KaamelDict.csv",
                           repo_type="dataset")
    lookup: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            phones = ast.literal_eval(row["phoneme"])
            if not phones:
                continue
            seq = phones[0] if isinstance(phones[0], (list, tuple)) else phones
            if not seq:
                continue
            lookup[row["grapheme"].strip()] = "".join(seq)
    return lookup


def _add_diacritics(word: str, ph_str: str) -> str:
    """Add fatha/kasra/damma/sukun to a single Persian word given its KaamelDict phoneme string."""
    phs = list(ph_str)
    pi = 0
    result: list[str] = []

    def peek():
        return phs[pi] if pi < len(phs) else None

    def consume():
        nonlocal pi
        pi += 1

    def after_cons():
        """After appending a consonant char, check next phoneme and add vowel diacritic or sukun."""
        n = peek()
        if n in SHORT_V:
            result.append(SHORT_V[n])
            consume()
        elif n in LONG_V or n is None:
            pass  # long vowel follows (letter will handle it) or word ends
        else:
            result.append(SUKUN)

    for ch in word:
        if ch in _SKIP:
            result.append(ch)
            continue
        n = peek()
        if n is None:
            result.append(ch)
            continue

        if ch == "آ":
            result.append(ch)
            if peek() == "?":
                consume()
            if peek() == "A":
                consume()
            continue

        if ch == "ا":
            if n == "?":
                result.append(ch)
                consume()
                after_cons()
            elif n == "A":
                result.append(ch)
                consume()
            else:
                result.append(ch)
            continue

        if ch == "و":
            if n == "v":
                result.append(ch)
                consume()
                after_cons()
            elif n == "u":
                result.append(ch)
                consume()
            else:
                result.append(ch)
            continue

        if ch in "یي":
            if n == "y":
                result.append(ch)
                consume()
                after_cons()
            elif n == "i":
                result.append(ch)
                consume()
                # Glide/geminate y after i belongs to this letter too
                # (e.g. کیفیت keyfiyyat, خیابان xiyAbAn)
                while peek() == "y":
                    consume()
            else:
                result.append(ch)
            continue

        if ch in CHAR_PH:
            # Skip any non-consonant phonemes (shouldn't normally happen mid-word)
            while pi < len(phs) and phs[pi] not in CONS_PH:
                pi += 1
            if pi < len(phs):
                consume()
            # A geminate (شدّه) consonant is one letter with a doubled phoneme
            # (e.g. ملی melli, مهم mohemm) — consume the double
            while peek() == CHAR_PH[ch]:
                consume()
            result.append(ch)
            after_cons()
            continue

        # Unknown char (diacritics already present, punctuation inside word, etc.)
        result.append(ch)

    return "".join(result)


def diacritize_sentence(text: str, lookup: dict) -> str:
    """Add word-internal diacritics to each token using KaamelDict lookup."""
    out = []
    for tok in text.split():
        # Strip leading/trailing punctuation for lookup
        key = tok
        prefix = ""
        suffix = ""
        while key and key[0] in _PUNCT:
            prefix += key[0]
            key = key[1:]
        while key and key[-1] in _PUNCT:
            suffix = key[-1] + suffix
            key = key[:-1]

        # An Ezafe kasra appended by apply_ezafe would break the dictionary
        # lookup; strip it for the lookup and re-append it after.
        ezafe = ""
        if key.endswith(KASRA):
            ezafe = KASRA
            key = key[:-1]

        # Try with and without zero-width non-joiner
        ph = (PH_OVERRIDES.get(key) or lookup.get(key)
              or lookup.get(key.replace("‌", "").replace("‍", "")))
        if ph:
            diac = _add_diacritics(key, ph)
            # Some KaamelDict entries bake the Ezafe vowel into the
            # pronunciation (برای barAye) — don't add the kasra twice
            if diac.endswith(KASRA):
                ezafe = ""
            out.append(prefix + diac + ezafe + suffix)
        else:
            out.append(tok)
    return " ".join(out)


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


class Frontend:
    """Loads all three stages once and applies them to sentences."""

    def __init__(self):
        self.albert_tok, self.albert_model, self.id2label = load_albert()
        self.kaamel = load_kaameldict()

    def __call__(self, text: str) -> str:
        normed = normalize_fa(text)
        ezafed = apply_ezafe(self.albert_tok, self.albert_model,
                             self.id2label, normed)
        return diacritize_sentence(ezafed, self.kaamel)
