---
name: setup-diacritics
description: Persian diacritization benchmark — dadmatools kasreh + abreza/persian-ezafe-albert (best, F1 98.7%) + KaamelDict G2P. Generates docs/diacritics.html.
---

# Persian Diacritization Benchmark

## State of the art (2025)

Full Arabic-script diacritization (adding all harakat) has **no off-the-shelf
open-source tool** for Persian. The best available option covers **Ezafe** —
the linking vowel /-e/ that is Persian's single most impactful diacritic omission.

**Available tools (ranked):**

| Tool | What it adds | Type | F1 | Dialect |
|------|-------------|------|----|---------|
| **abreza/persian-ezafe-albert** | Ezafe kasra ِ | Neural (ALBERT-fa) | **98.7%** | Tehran |
| dadmatools kasreh | Ezafe kasra ِ | Neural (XLM-R) | lower (misses ساعت سه etc.) | Tehran |
| KaamelDict | G2P phoneme string per word (Latin) | Dictionary 116k | — | Standard Persian |

## Install (`.venvs/diacritics`)

```bash
python3 -m venv .venvs/diacritics
.venvs/diacritics/bin/pip install dadmatools torch --index-url https://download.pytorch.org/whl/cpu
.venvs/diacritics/bin/pip install huggingface_hub tokenizers transformers sentencepiece gdown scikit-learn datasets
```

Models download on first run:
- dadmatools: `cache/dadmatools/` (~5 MB)
- albert: HF cache (~45 MB)
- KaamelDict: HF cache (~11 MB)

## Run

```bash
.venvs/diacritics/bin/python scripts/run_diacritics.py
# -> docs/diacritics.html
```

## Best tool: abreza/persian-ezafe-albert

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tok = AutoTokenizer.from_pretrained('abreza/persian-ezafe-albert')
model = AutoModelForTokenClassification.from_pretrained('abreza/persian-ezafe-albert')
model.eval()
id2label = model.config.id2label
KASRA = 'ِ'

def apply_ezafe(text):
    words = text.split()
    enc = tok(words, is_split_into_words=True, return_tensors='pt', truncation=True)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    preds = logits.argmax(-1).tolist()
    word_ids = enc.word_ids()
    word_labels = {}
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid not in word_labels:
            word_labels[wid] = id2label[preds[pos]]
    return ' '.join(w + KASRA if word_labels.get(i) == 'NEEDS_EZAFE' else w
                    for i, w in enumerate(words))

print(apply_ezafe('ساعت سه بعدازظهر'))   # ساعتِ سهِ بعدازظهر
print(apply_ezafe('آسمان آبی پرواز'))     # آسمانِ آبی پرواز
```

## Tool: dadmatools kasreh

```python
from dadmatools.pipeline.language import Pipeline
p = Pipeline('kasreh')
doc = p('کتاب‌خانه ملی ایران')
for token in doc:
    if token._.kasreh and token._.kasreh != 'O':
        print(token.text + 'ِ')
```

Labels: `S-kasreh` = Ezafe detected; `O` = no Ezafe.
**Known miss:** numeral-time phrases like `ساعت سه` → use albert instead.

## Tool: KaamelDict lookup

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

**Phoneme notation:** `A`=ā, `a`=short a, `e`=short e, `o`=short o,
`S`=sh, `Z`=zh, `c`=ch, `G`=gh, `x`=kh, `?`=ʔ.

## Why no full harakat exists for Persian

1. Persian writing almost never uses harakat outside dictionaries and
   children's books → labeled training data is very scarce.
2. Arabic tashkeel models (CATT, Sadeed, Fine-Tashkeel) are Arabic-only.
3. Future path: fine-tune ByT5/mT5 on diacritized Persian dictionary data.
