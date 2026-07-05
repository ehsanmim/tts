#!/usr/bin/env python3
"""Persian diacritization benchmark.

Tools:
  1. dadmatools kasreh — Ezafe (linking -e) detection; adds kasra ِ where the
     Ezafe construction is predicted (e.g. کتاب → کتابِ before ملی).
  2. KaamelDict (MahtaFetrat/KaamelDict, HF dataset) — 116k-word G2P phoneme
     dictionary; word-level lookup returns the pronunciation in compact Latin
     notation (e.g. سلام → salAm).  Output is Latin phonemes, not Arabic
     diacritics — shown as ruby annotations on the page.

Run:    .venvs/diacritics/bin/python scripts/run_diacritics.py
Output: docs/diacritics.html
"""
import ast
import csv
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_FILE = os.path.join(HERE, "sample_texts", "sentences.txt")
OUT_HTML = os.path.join(HERE, "docs", "diacritics.html")

KASRA = "ِ"  # Arabic kasra  ِ


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


def apply_kasreh(pipeline, text):
    """Run dadmatools kasreh on text; append kasra to Ezafe-bearing tokens."""
    doc = pipeline(text)
    parts = []
    for token in doc:
        kval = token._.kasreh
        if kval and kval != "O":
            parts.append(token.text + KASRA)
        else:
            parts.append(token.text)
    return " ".join(parts)


def annotate_kaameldict(lookup, text):
    """Return list of (word, phoneme_or_None) for each whitespace token."""
    tokens = text.split()
    result = []
    for tok in tokens:
        # strip punctuation for lookup key
        key = tok.strip("،.؟!()[]،؛")
        result.append((tok, lookup.get(key)))
    return result


# ── HTML generation ───────────────────────────────────────────────────────────

CSS = """
:root {
  --bg: #ffffff;
  --bg2: #f8f8f8;
  --border: #e0e0e0;
  --text: #1a1a1a;
  --muted: #666;
  --accent: #2563eb;
  --tag-bg: #eef2ff;
  --tag: #3730a3;
  --ruby-c: #b45309;
  --kasra-c: #dc2626;
  --badge-bg: #f0fdf4;
  --badge: #166534;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111827;
    --bg2: #1f2937;
    --border: #374151;
    --text: #f3f4f6;
    --muted: #9ca3af;
    --accent: #60a5fa;
    --tag-bg: #1e3a5f;
    --tag: #93c5fd;
    --ruby-c: #fbbf24;
    --kasra-c: #f87171;
    --badge-bg: #14532d;
    --badge: #86efac;
  }
}
:root[data-theme="dark"] {
  --bg: #111827; --bg2: #1f2937; --border: #374151; --text: #f3f4f6;
  --muted: #9ca3af; --accent: #60a5fa; --tag-bg: #1e3a5f; --tag: #93c5fd;
  --ruby-c: #fbbf24; --kasra-c: #f87171; --badge-bg: #14532d; --badge: #86efac;
}
:root[data-theme="light"] {
  --bg: #ffffff; --bg2: #f8f8f8; --border: #e0e0e0; --text: #1a1a1a;
  --muted: #666; --accent: #2563eb; --tag-bg: #eef2ff; --tag: #3730a3;
  --ruby-c: #b45309; --kasra-c: #dc2626; --badge-bg: #f0fdf4; --badge: #166534;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif;
       margin: 0; padding: 24px; line-height: 1.6; }
h1 { font-size: 1.6rem; margin-bottom: 4px; }
.subtitle { color: var(--muted); margin-bottom: 28px; font-size: 0.95rem; }
.tools-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px;
              margin-bottom: 40px; }
@media (max-width: 768px) { .tools-grid { grid-template-columns: 1fr; } }
.tool-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px;
             padding: 20px; }
.tool-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.tool-name { font-weight: 700; font-size: 1.05rem; }
.tag { background: var(--tag-bg); color: var(--tag); font-size: 0.72rem;
       padding: 2px 8px; border-radius: 12px; font-weight: 600; }
.badge { background: var(--badge-bg); color: var(--badge); font-size: 0.72rem;
         padding: 2px 8px; border-radius: 12px; font-weight: 600; }
.tool-desc { color: var(--muted); font-size: 0.85rem; margin-bottom: 16px; }
.sent-block { margin-bottom: 18px; border-bottom: 1px solid var(--border); padding-bottom: 14px; }
.sent-block:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.sent-id { font-size: 0.75rem; color: var(--muted); margin-bottom: 4px; }
.input-text { color: var(--muted); font-size: 0.85rem; direction: rtl; text-align: right;
              margin-bottom: 6px; font-family: 'Tahoma', 'Segoe UI', serif; }
.output-text { direction: rtl; text-align: right; font-size: 1.05rem; line-height: 2;
               font-family: 'Tahoma', 'Segoe UI', serif; }
.kasra { color: var(--kasra-c); }
ruby { ruby-position: under; }
rt { font-size: 0.6em; color: var(--ruby-c); font-family: monospace; }
.gap-note { background: var(--tag-bg); border: 1px solid var(--border); border-radius: 8px;
            padding: 16px 20px; margin-bottom: 32px; font-size: 0.9rem; }
.gap-note strong { color: var(--accent); }
.legend { font-size: 0.82rem; color: var(--muted); margin-bottom: 24px; }
.legend span { margin-right: 16px; }
.section-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 16px; color: var(--text); }
"""

JS = """
(function(){
  var stored = localStorage.getItem('theme');
  if (stored) document.documentElement.setAttribute('data-theme', stored);
  window.toggleTheme = function() {
    var t = document.documentElement.getAttribute('data-theme');
    var next = (t === 'dark') ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
  };
})();
"""

def escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_kasreh_html(text):
    """Mark added kasra chars in red so they stand out."""
    parts = []
    for ch in text:
        if ch == KASRA:
            parts.append(f'<span class="kasra">{ch}</span>')
        else:
            parts.append(escape(ch))
    return "".join(parts)


def render_kaameldict_html(annotated):
    """Return HTML with ruby annotations for each word."""
    parts = []
    for word, phoneme in annotated:
        if phoneme:
            parts.append(f"<ruby>{escape(word)}<rt>{escape(phoneme)}</rt></ruby>")
        else:
            parts.append(escape(word))
    return " ".join(parts)


def build_html(sentences, kasreh_outputs, kaamel_outputs):
    tool_cards = ""

    # Tool 1: dadmatools kasreh
    blocks = ""
    for (sid, orig), kasreh in zip(sentences, kasreh_outputs):
        blocks += f"""
        <div class="sent-block">
          <div class="sent-id">#{sid}</div>
          <div class="input-text">{escape(orig)}</div>
          <div class="output-text">{render_kasreh_html(kasreh)}</div>
        </div>"""
    tool_cards += f"""
    <div class="tool-card">
      <div class="tool-header">
        <span class="tool-name">dadmatools — Ezafe (kasreh)</span>
        <span class="tag">Neural</span>
        <span class="badge">Persian</span>
      </div>
      <div class="tool-desc">
        Detects Ezafe constructions (e.g. کتاب<span style="color:var(--kasra-c)">ِ</span> من)
        and adds kasra&nbsp;ِ to the preceding noun/adjective.
        Ezafe is the linking vowel&nbsp;/-e/ that connects nouns to their modifiers in Persian —
        the single most common diacritic omission.
        Only kasra is added; all other short vowels remain unmarked.
      </div>
      {blocks}
    </div>"""

    # Tool 2: KaamelDict G2P phoneme annotation
    blocks = ""
    for (sid, orig), annotated in zip(sentences, kaamel_outputs):
        coverage = sum(1 for _, p in annotated if p)
        total = len(annotated)
        blocks += f"""
        <div class="sent-block">
          <div class="sent-id">#{sid} &nbsp;·&nbsp; {coverage}/{total} words in dictionary</div>
          <div class="input-text">{escape(orig)}</div>
          <div class="output-text">{render_kaameldict_html(annotated)}</div>
        </div>"""
    tool_cards += f"""
    <div class="tool-card">
      <div class="tool-header">
        <span class="tool-name">KaamelDict — G2P phoneme annotation</span>
        <span class="tag">Dictionary</span>
        <span class="badge">116k entries</span>
      </div>
      <div class="tool-desc">
        Word-level lookup in MahtaFetrat/KaamelDict (116,598 entries).
        Shows the phonemic pronunciation in compact Latin notation under each word
        (notation: <code>A</code>=long&nbsp;ā, <code>a</code>=short&nbsp;a,
        <code>e</code>=short&nbsp;e, <code>o</code>=short&nbsp;o,
        <code>S</code>=sh, <code>?</code>=ʔ).
        This is a pronunciation guide, not Arabic-script diacritics —
        full harakat restoration for Persian has no off-the-shelf tool yet.
      </div>
      {blocks}
    </div>"""

    return f"""<title>Persian Diacritization — Benchmark</title>
<script>{JS}</script>
<style>{CSS}</style>
<body>
<h1>Persian Diacritization — Benchmark</h1>
<p class="subtitle">
  Text-to-text: unvoweled Persian input → output with diacritics / pronunciation annotation
  &nbsp;·&nbsp;
  <button onclick="toggleTheme()" style="border:1px solid var(--border);background:var(--bg2);
    color:var(--text);padding:3px 10px;border-radius:6px;cursor:pointer;font-size:0.8rem">
    toggle theme
  </button>
</p>

<div class="gap-note">
  <strong>State of the art:</strong>
  Unlike Arabic (CATT, Sadeed, Fine-Tashkeel), <em>dedicated full-diacritization for Persian
  does not yet exist as an open-source tool</em>. Persian writing almost never uses short vowels
  outside dictionaries and children's books, so labeled training data is scarce.
  The two tools below cover what is currently available:
  (1)&nbsp;Ezafe detection adds the single most common diacritic — the linking&nbsp;/-e/;
  (2)&nbsp;KaamelDict provides word-level phoneme pronunciation from a 116k-word dictionary.
</div>

<div class="legend">
  <span><span class="kasra" style="color:var(--kasra-c)">ِ</span> = kasra added by Ezafe detector</span>
  <span style="color:var(--ruby-c)">phoneme</span> = KaamelDict G2P annotation (Latin, shown under word)</span>
</div>

<div class="tools-grid">
  {tool_cards}
</div>

<p style="font-size:0.8rem;color:var(--muted)">
  Tools: <a href="https://github.com/Dadmatech/DadmaTools" style="color:var(--accent)">dadmatools</a>
  (XLM-RoBERTa kasreh classifier) ·
  <a href="https://huggingface.co/datasets/MahtaFetrat/KaamelDict" style="color:var(--accent)">MahtaFetrat/KaamelDict</a>
  (G2P dictionary) · Sample sentences from
  <a href="index.html" style="color:var(--accent)">Persian TTS benchmark</a>
</p>
</body>"""


def main():
    sentences = read_sentences(SENT_FILE)

    print("Loading dadmatools kasreh pipeline ...")
    from dadmatools.pipeline.language import Pipeline
    kas_pipeline = Pipeline("kasreh")

    print("Loading KaamelDict ...")
    lookup = load_kaameldict()

    kasreh_outputs = []
    kaamel_outputs = []

    for sid, text in sentences:
        print(f"  [{sid}] processing ...")
        kasreh_outputs.append(apply_kasreh(kas_pipeline, text))
        kaamel_outputs.append(annotate_kaameldict(lookup, text))

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    html = build_html(sentences, kasreh_outputs, kaamel_outputs)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Done → {OUT_HTML}")


if __name__ == "__main__":
    main()
