#!/usr/bin/env python3
"""Persian diacritization benchmark.

Tools:
  1. dadmatools kasreh — Ezafe detection via XLM-RoBERTa; lower recall
     (misses numeral-time constructions like ساعت سه).
  2. abreza/persian-ezafe-albert — ALBERT-fa fine-tuned on HomoRich dataset;
     F1 98.73%, catches ساعتِ سه and many more Ezafe constructions.
  3. KaamelDict (MahtaFetrat/KaamelDict, HF dataset) — 116k-word G2P phoneme
     dictionary; word-level lookup returns the pronunciation in compact Latin
     notation (e.g. سلام → salAm). Output is Latin phonemes, not Arabic
     diacritics — shown as ruby annotations on the page.
  4. Full TTS front-end (fa_diacritics.py) — digits→words (num2fawords),
     ALBERT Ezafe kasra, then KaamelDict-driven word-internal harakat
     (fatha/kasra/damma/sukun) with homograph overrides (صِفْر not صَفَر,
     نُه not نَه). This is the exact text fed to the F5/Chatterbox ezafe
     variants on the TTS benchmark page.

Run:    .venvs/diacritics/bin/python scripts/run_diacritics.py
Output: docs/diacritics.html
"""
import ast
import csv
import os
import re

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_FILE = os.path.join(HERE, "sample_texts", "sentences.txt")
OUT_HTML = os.path.join(HERE, "docs", "diacritics.html")

KASRA = "ِ"  # Arabic kasra
HARAKAT = {"ِ", "َ", "ُ", "ْ"}  # kasra, fatha, damma, sukun


# ── helpers ──────────────────────────────────────────────────────────────────

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


def load_kaameldict():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("MahtaFetrat/KaamelDict", "KaamelDict.csv", repo_type="dataset")
    lookup = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = row["grapheme"].strip()
            p = row["phoneme"].strip()
            try:
                phones = ast.literal_eval(p)
                if phones:
                    seq = phones[0] if isinstance(phones[0], (list, tuple)) else phones
                    lookup[g] = "".join(seq)
            except Exception:
                pass
    return lookup


def apply_kasreh_dadma(pipeline, text):
    """dadmatools kasreh: append kasra to Ezafe-bearing tokens."""
    doc = pipeline(text)
    parts = []
    for token in doc:
        kval = token._.kasreh
        if kval and kval != "O":
            parts.append(token.text + KASRA)
        else:
            parts.append(token.text)
    return " ".join(parts)


def apply_albert_ezafe(model, tokenizer, id2label, text):
    """abreza/persian-ezafe-albert: word-level Ezafe prediction."""
    words = text.split()
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    preds = logits.argmax(-1).tolist()
    word_ids = enc.word_ids()

    word_labels = {}
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid not in word_labels:
            word_labels[wid] = id2label[preds[pos]]

    result = []
    for i, w in enumerate(words):
        if word_labels.get(i) == "NEEDS_EZAFE":
            result.append(w + KASRA)
        else:
            result.append(w)
    return " ".join(result)


def annotate_kaameldict(lookup, text):
    """Return list of (word, phoneme_or_None) for each whitespace token."""
    tokens = text.split()
    result = []
    for tok in tokens:
        key = tok.strip("،.؟!()[]،؛")
        result.append((tok, lookup.get(key)))
    return result


# ── HTML ─────────────────────────────────────────────────────────────────────

CSS = """
:root {
  --bg:#ffffff; --bg2:#f8f8f8; --border:#e0e0e0; --text:#1a1a1a;
  --muted:#666; --accent:#2563eb; --tag-bg:#eef2ff; --tag:#3730a3;
  --ruby-c:#b45309; --kasra-c:#dc2626; --badge-bg:#f0fdf4; --badge:#166534;
  --best-bg:#fefce8; --best-border:#ca8a04;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#111827; --bg2:#1f2937; --border:#374151; --text:#f3f4f6;
    --muted:#9ca3af; --accent:#60a5fa; --tag-bg:#1e3a5f; --tag:#93c5fd;
    --ruby-c:#fbbf24; --kasra-c:#f87171; --badge-bg:#14532d; --badge:#86efac;
    --best-bg:#1c1a07; --best-border:#ca8a04;
  }
}
:root[data-theme="dark"] {
  --bg:#111827; --bg2:#1f2937; --border:#374151; --text:#f3f4f6;
  --muted:#9ca3af; --accent:#60a5fa; --tag-bg:#1e3a5f; --tag:#93c5fd;
  --ruby-c:#fbbf24; --kasra-c:#f87171; --badge-bg:#14532d; --badge:#86efac;
  --best-bg:#1c1a07; --best-border:#ca8a04;
}
:root[data-theme="light"] {
  --bg:#ffffff; --bg2:#f8f8f8; --border:#e0e0e0; --text:#1a1a1a;
  --muted:#666; --accent:#2563eb; --tag-bg:#eef2ff; --tag:#3730a3;
  --ruby-c:#b45309; --kasra-c:#dc2626; --badge-bg:#f0fdf4; --badge:#166534;
  --best-bg:#fefce8; --best-border:#ca8a04;
}
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:system-ui,sans-serif;
       margin:0; padding:24px; line-height:1.6; }
h1 { font-size:1.6rem; margin-bottom:4px; }
.subtitle { color:var(--muted); margin-bottom:28px; font-size:0.95rem; }
.tools-grid { display:grid; grid-template-columns:1fr 1fr; gap:20px;
              margin-bottom:40px; }
@media (max-width:640px)  { .tools-grid { grid-template-columns:1fr; } }
.tool-card { background:var(--bg2); border:1px solid var(--border);
             border-radius:10px; padding:20px; }
.tool-card.best { border-color:var(--best-border); background:var(--best-bg); }
.tool-header { display:flex; align-items:center; gap:8px; margin-bottom:12px; flex-wrap:wrap; }
.tool-name { font-weight:700; font-size:1rem; }
.tag  { background:var(--tag-bg);  color:var(--tag);   font-size:0.7rem;
        padding:2px 7px; border-radius:12px; font-weight:600; }
.badge { background:var(--badge-bg); color:var(--badge); font-size:0.7rem;
         padding:2px 7px; border-radius:12px; font-weight:600; }
.best-badge { background:var(--best-border); color:#fff; font-size:0.7rem;
              padding:2px 7px; border-radius:12px; font-weight:700; }
.tool-desc { color:var(--muted); font-size:0.82rem; margin-bottom:14px; }
.sent-block { margin-bottom:16px; border-bottom:1px solid var(--border); padding-bottom:12px; }
.sent-block:last-child { border-bottom:none; margin-bottom:0; padding-bottom:0; }
.sent-id { font-size:0.73rem; color:var(--muted); margin-bottom:3px; }
.input-text { color:var(--muted); font-size:0.82rem; direction:rtl; text-align:right;
              margin-bottom:5px; font-family:'Tahoma','Segoe UI',serif; }
.output-text { direction:rtl; text-align:right; font-size:1.05rem; line-height:2.2;
               font-family:'Tahoma','Segoe UI',serif; }
.kasra { color:var(--kasra-c); }
ruby { ruby-position:under; }
rt { font-size:0.58em; color:var(--ruby-c); font-family:monospace; }
.gap-note { background:var(--tag-bg); border:1px solid var(--border); border-radius:8px;
            padding:14px 18px; margin-bottom:28px; font-size:0.88rem; }
.gap-note strong { color:var(--accent); }
.legend { font-size:0.8rem; color:var(--muted); margin-bottom:20px; }
.legend span { margin-right:14px; }
"""

JS = """
(function(){
  var s = localStorage.getItem('theme');
  if (s) document.documentElement.setAttribute('data-theme', s);
  window.toggleTheme = function() {
    var t = document.documentElement.getAttribute('data-theme');
    var n = (t === 'dark') ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', n);
    localStorage.setItem('theme', n);
  };
})();
"""


def escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_kasra_html(text):
    parts = []
    for ch in text:
        if ch == KASRA:
            parts.append(f'<span class="kasra">{ch}</span>')
        else:
            parts.append(escape(ch))
    return "".join(parts)


def render_harakat_html(text):
    parts = []
    for ch in text:
        if ch in HARAKAT:
            parts.append(f'<span class="kasra">{ch}</span>')
        else:
            parts.append(escape(ch))
    return "".join(parts)


def render_kaamel_html(annotated):
    parts = []
    for word, phoneme in annotated:
        if phoneme:
            parts.append(f"<ruby>{escape(word)}<rt>{escape(phoneme)}</rt></ruby>")
        else:
            parts.append(escape(word))
    return " ".join(parts)


def build_html(sentences, dadma_out, albert_out, kaamel_out, pipeline_out):
    def sent_blocks(outputs, render_fn):
        html = ""
        for (sid, orig), out in zip(sentences, outputs):
            extra = ""
            if isinstance(out, list):  # kaameldict
                coverage = sum(1 for _, p in out if p)
                extra = f" &nbsp;·&nbsp; {coverage}/{len(out)} words"
            html += f"""
        <div class="sent-block">
          <div class="sent-id">#{sid}{extra}</div>
          <div class="input-text">{escape(orig)}</div>
          <div class="output-text">{render_fn(out)}</div>
        </div>"""
        return html

    card1 = f"""
    <div class="tool-card">
      <div class="tool-header">
        <span class="tool-name">dadmatools kasreh</span>
        <span class="tag">Neural</span>
      </div>
      <div class="tool-desc">
        XLM-RoBERTa token classifier.  Detects some Ezafe constructions but has
        notable gaps — misses numeral-time phrases like
        <span style="direction:rtl">ساعت سه</span> and other cases where
        the noun-phrase context is atypical.
      </div>
      {sent_blocks(dadma_out, render_kasra_html)}
    </div>"""

    card2 = f"""
    <div class="tool-card best">
      <div class="tool-header">
        <span class="tool-name">abreza/persian-ezafe-albert</span>
        <span class="tag">Neural</span>
        <span class="best-badge">★ best Ezafe</span>
        <span class="badge">F1 98.7%</span>
      </div>
      <div class="tool-desc">
        ALBERT-fa fine-tuned on the HomoRich dataset (HomographDataset +
        GE2PE corpus).  Token-level binary classifier
        (NEEDS_EZAFE&nbsp;/&nbsp;NO_EZAFE).  Correctly handles
        <span style="direction:rtl">ساعتِ سه</span>,
        <span style="direction:rtl">گفتارِ فارسی</span>,
        <span style="direction:rtl">آسمانِ آبی</span> and many more.
        Kasra added in
        <span class="kasra" style="color:var(--kasra-c)">red</span>.
      </div>
      {sent_blocks(albert_out, render_kasra_html)}
    </div>"""

    card3 = f"""
    <div class="tool-card">
      <div class="tool-header">
        <span class="tool-name">KaamelDict G2P</span>
        <span class="tag">Dictionary</span>
        <span class="badge">116k entries</span>
      </div>
      <div class="tool-desc">
        Word-level lookup in MahtaFetrat/KaamelDict.
        Shows phonemic pronunciation in compact Latin notation under each word
        (<code>A</code>=ā, <code>a</code>=a, <code>e</code>=e,
        <code>o</code>=o, <code>S</code>=sh, <code>?</code>=ʔ).
        This is a pronunciation guide — not Arabic-script diacritics.
        Full harakat restoration for Persian has no off-the-shelf tool yet.
      </div>
      {sent_blocks(kaamel_out, render_kaamel_html)}
    </div>"""

    card4 = f"""
    <div class="tool-card best">
      <div class="tool-header">
        <span class="tool-name">Full TTS front-end</span>
        <span class="tag">Pipeline</span>
        <span class="best-badge">★ fed to TTS</span>
      </div>
      <div class="tool-desc">
        <code>fa_diacritics.py</code> — the three stages combined:
        digits→words (num2fawords), ALBERT Ezafe kasra, then KaamelDict-driven
        word-internal harakat (fatha/kasra/damma/sukun) with homograph
        overrides for digit-derived words
        (<span style="direction:rtl">صِفْر</span> zero not
        <span style="direction:rtl">صَفَر</span> the month,
        <span style="direction:rtl">نُه</span> nine not
        <span style="direction:rtl">نَه</span> no).
        This exact text drives the f5-persian-ezafe and
        chatterbox-persian-ezafe voices on the
        <a href="index.html" style="color:var(--accent)">TTS benchmark</a>.
        All added marks in
        <span class="kasra" style="color:var(--kasra-c)">red</span>.
      </div>
      {sent_blocks(pipeline_out, render_harakat_html)}
    </div>"""

    return f"""<title>Persian Diacritization — Benchmark</title>
<script>{JS}</script>
<style>{CSS}</style>
<body>
<h1>Persian Diacritization — Benchmark</h1>
<p class="subtitle">
  Text-to-text: unvoweled Persian → output with diacritics / pronunciation annotation
  &nbsp;·&nbsp;
  <button onclick="toggleTheme()" style="border:1px solid var(--border);background:var(--bg2);
    color:var(--text);padding:3px 10px;border-radius:6px;cursor:pointer;font-size:0.8rem">
    toggle theme
  </button>
</p>

<div class="gap-note">
  <strong>State of the art:</strong>
  Unlike Arabic (CATT, Sadeed, Fine-Tashkeel), <em>full harakat restoration for Persian
  does not yet exist as an open-source tool</em>.  The tools here cover
  <strong>Ezafe</strong> — the linking vowel&nbsp;/-e/ that is Persian writing's
  single most impactful omission for TTS.
  <strong>abreza/persian-ezafe-albert</strong> (F1&nbsp;98.7%) is the current
  best option; dadmatools' kasreh has lower recall.
  KaamelDict adds word-level G2P phoneme pronunciation.
  Combining ALBERT Ezafe with KaamelDict-driven harakat placement
  (<strong>Full TTS front-end</strong> card) gets close to full
  diacritization for dictionary words.
</div>

<div class="legend">
  <span><span class="kasra" style="color:var(--kasra-c)">ِ</span> = kasra/Ezafe added</span>
  <span style="color:var(--ruby-c)">phoneme</span> = KaamelDict G2P (Latin, shown under word)</span>
</div>

<div class="tools-grid">
  {card1}
  {card2}
  {card3}
  {card4}
</div>

<p style="font-size:0.8rem;color:var(--muted)">
  <a href="https://github.com/Dadmatech/DadmaTools" style="color:var(--accent)">dadmatools</a> ·
  <a href="https://huggingface.co/abreza/persian-ezafe-albert" style="color:var(--accent)">abreza/persian-ezafe-albert</a>
  (ALBERT-fa, HomoRich) ·
  <a href="https://huggingface.co/datasets/MahtaFetrat/KaamelDict" style="color:var(--accent)">MahtaFetrat/KaamelDict</a> ·
  <a href="index.html" style="color:var(--accent)">← Persian TTS benchmark</a>
</p>
</body>"""


def main():
    sentences = read_sentences(SENT_FILE)

    print("Loading dadmatools kasreh ...")
    from dadmatools.pipeline.language import Pipeline
    dadma_pipeline = Pipeline("kasreh")

    print("Loading abreza/persian-ezafe-albert ...")
    albert_tok = AutoTokenizer.from_pretrained("abreza/persian-ezafe-albert")
    albert_model = AutoModelForTokenClassification.from_pretrained("abreza/persian-ezafe-albert")
    albert_model.eval()
    id2label = albert_model.config.id2label

    print("Loading KaamelDict ...")
    lookup = load_kaameldict()

    print("Loading full TTS front-end ...")
    from fa_diacritics import Frontend
    frontend = Frontend()

    dadma_out, albert_out, kaamel_out, pipeline_out = [], [], [], []

    for sid, text in sentences:
        print(f"  [{sid}] ...")
        dadma_out.append(apply_kasreh_dadma(dadma_pipeline, text))
        albert_out.append(apply_albert_ezafe(albert_model, albert_tok, id2label, text))
        kaamel_out.append(annotate_kaameldict(lookup, text))
        pipeline_out.append(frontend(text))

    html = build_html(sentences, dadma_out, albert_out, kaamel_out, pipeline_out)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Done → {OUT_HTML}")


if __name__ == "__main__":
    main()
