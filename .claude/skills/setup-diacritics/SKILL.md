---
name: setup-diacritics
description: Persian diacritization benchmark — dadmatools Ezafe (kasreh) + KaamelDict G2P annotation. Generates docs/diacritics.html.
---

# Persian Diacritization Benchmark

## State of the art (2025)

Full Arabic-script diacritization (adding all harakat to Persian text) has **no
off-the-shelf open-source tool**. Unlike Arabic (CATT, Sadeed, Fine-Tashkeel),
Persian almost never uses short vowels in everyday writing, so labeled training
data is scarce.

**Available tools:**

| Tool | What it adds | Type | Dialect |
|------|-------------|------|---------|
| dadmatools kasreh | Ezafe kasra ِ (linking /-e/) only | Neural (XLM-R) | Tehran Persian |
| KaamelDict | G2P phoneme string per word (Latin) | Dictionary (116k words) | Standard Persian |

## Install (`.venvs/diacritics`)

```bash
python3 -m venv .venvs/diacritics
.venvs/diacritics/bin/pip install dadmatools torch --index-url https://download.pytorch.org/whl/cpu
.venvs/diacritics/bin/pip install huggingface_hub tokenizers transformers sentencepiece gdown scikit-learn datasets
```

dadmatools downloads its models to `~/.cache/dadmatools/` on first run (~5 MB).
KaamelDict is downloaded from HF to the HF cache on first run (~11 MB).

## Run

```bash
.venvs/diacritics/bin/python scripts/run_diacritics.py
# -> docs/diacritics.html
```

## Tool 1: dadmatools kasreh

```python
from dadmatools.pipeline.language import Pipeline
p = Pipeline('kasreh')
doc = p('کتاب‌خانه ملی ایران')
for token in doc:
    if token._.kasreh and token._.kasreh != 'O':
        print(token.text + 'ِ')   # Ezafe detected
```

Labels: `S-kasreh` = Ezafe on this word; `O` = no Ezafe.

## Tool 2: KaamelDict lookup

```python
import csv, ast
from huggingface_hub import hf_hub_download
path = hf_hub_download('MahtaFetrat/KaamelDict', 'KaamelDict.csv', repo_type='dataset')
lookup = {}
with open(path, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        phones = ast.literal_eval(row['phoneme'])
        seq = phones[0] if isinstance(phones[0], (list, tuple)) else phones
        lookup[row['grapheme'].strip()] = ''.join(seq)

print(lookup['سلام'])    # salAm
print(lookup['تهران'])   # tehrAn
```

**Phoneme notation:** `A`=long ā, `a`=short a, `e`=short e, `o`=short o,
`i`=long i, `u`=long u, `S`=sh, `Z`=zh, `c`=ch, `G`=gh, `x`=kh, `?`=ʔ (glottal).

## Why no full Arabic-script diacritization exists for Persian

1. Persian writing almost never uses harakat (short vowels) outside
   dictionaries and children's books — so labeled training data is rare.
2. Arabic tashkeel models (CATT, Sadeed, Fine-Tashkeel) are trained on Arabic
   only; Persian and Arabic differ significantly in morphology and phonology.
3. The closest future path: fine-tune a seq2seq model (ByT5 or mT5) on
   automatically generated diacritized Persian text (e.g., from Persian
   dictionaries or romanized corpora).
