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

KASRA  = "ِ"   # U+0650
FATHA  = "َ"   # U+064E
DAMMA  = "ُ"   # U+064F
SUKUN  = "ْ"   # U+0652
SHADDA = "ّ"   # U+0651

# Persian/Arabic-Indic digits → ASCII
_DIGIT_MAP = {}
for _base in (0x06F0, 0x0660):
    for _i in range(10):
        _DIGIT_MAP[_base + _i] = str(_i)

# KaamelDict phoneme notation
SHORT_V = {"a": FATHA, "e": KASRA, "o": DAMMA}
LONG_V  = set("Aiu")
CONS_PH = set("bptsjchxdzrZSGfqkglmnhyv?")
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

# KaamelDict lists several readings per grapheme and we take the first,
# which is wrong for numeral homographs: صفر defaults to safar (the month)
# instead of sefr (zero), نه to na (no) instead of noh (nine). Applied only
# to tokens produced by digit expansion, so negation نه stays untouched.
NUM_PH_OVERRIDES = {"صفر": "sefr", "نه": "noh"}


def normalize_fa(text: str) -> tuple[str, frozenset]:
    """Expand digits to Persian words. Returns (text, set of words produced
    from digits) so numeral homographs can be diacritized correctly."""
    num_tokens: set[str] = set()

    def conv(n: str) -> str:
        w = words(int(n))
        num_tokens.update(w.split())
        return w

    text = text.translate(_DIGIT_MAP)
    text = re.sub(r"\d+(?:/\d+)+",
                  lambda m: " ".join(conv(p) for p in m.group().split("/")),
                  text)
    text = re.sub(r"\d{5,}",
                  lambda m: " ".join(conv(d) for d in m.group()),
                  text)
    text = re.sub(r"\d+", lambda m: conv(m.group()), text)
    return text, frozenset(num_tokens)


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
                # a single ی can spell /iy/ (کیفیت keyfiyat) — absorb the y
                if peek() == "y":
                    consume()
                    after_cons()
            else:
                result.append(ch)
            continue

        if ch in CHAR_PH:
            # Skip any non-consonant phonemes (shouldn't normally happen mid-word)
            while pi < len(phs) and phs[pi] not in CONS_PH:
                pi += 1
            result.append(ch)
            if pi < len(phs):
                cur = phs[pi]
                consume()
                # gemination (ملی melli) is written with shadda, not a doubled letter
                if peek() == cur:
                    consume()
                    result.append(SHADDA)
            after_cons()
            continue

        # Unknown char (diacritics already present, punctuation inside word, etc.)
        result.append(ch)

    return "".join(result)


def diacritize_sentence(text: str, lookup: dict,
                        num_tokens: frozenset = frozenset()) -> str:
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

        # An Ezafe kasra appended by apply_ezafe would break the lookup —
        # strip it, diacritize the bare word, put it back
        ezafe = ""
        if key.endswith(KASRA):
            ezafe = KASRA
            key = key[:-1]

        if key in num_tokens and key in NUM_PH_OVERRIDES:
            ph = NUM_PH_OVERRIDES[key]
        else:
            # Try with and without zero-width non-joiner
            ph = lookup.get(key) or lookup.get(key.replace("‌", "").replace("‍", ""))
        if ph:
            dia = _add_diacritics(key, ph)
            # dictionary readings like barAye already end in the ezafe vowel —
            # don't double the kasra
            if ezafe and dia.endswith(KASRA):
                ezafe = ""
            out.append(prefix + dia + ezafe + suffix)
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    frontend = Frontend()

    from f5_tts.api import F5TTS
    ckpt  = hf_hub_download("Lumos675/F5_TTS_Persian", "model_last.pt")
    vocab = hf_hub_download("Lumos675/F5_TTS_Persian", "vocab.txt")
    print(f"Loading F5-TTS Persian (CPU, nfe={NFE}, cfg={CFG}) ...")
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab, device="cpu")

    sents = read_sentences(SENT_FILE)
    ref_raw   = dict(sents)[REF_ID]
    ref_normed, ref_nums = normalize_fa(ref_raw)
    ref_ezafe   = apply_ezafe(albert_tok, albert_model, id2label, ref_normed)
    ref_text    = diacritize_sentence(ref_ezafe, kaamel, ref_nums)

    for sid, text in sents[:N]:
        normed, nums = normalize_fa(text)
        ezafed = apply_ezafe(albert_tok, albert_model, id2label, normed)
        final  = diacritize_sentence(ezafed, kaamel, nums)
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
