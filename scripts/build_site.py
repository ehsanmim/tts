#!/usr/bin/env python3
"""Build a self-contained comparison page: docs/index.html.

Scans samples/<model-id>/NN.wav, embeds each clip as a base64 data URI, and
lays out a per-sentence comparison grid so every model can be A/B'd on the
same Persian text. Fully self-contained (no external assets) so it works both
as a GitHub Pages site and as a standalone file.

Usage: python3 scripts/build_site.py
"""
import os
import re
import sys
import json
import base64
import html

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(HERE, "docs")
# Default: reference audio by relative path (real files under docs/audio/ —
# small HTML, ideal for GitHub Pages). With --embed: inline base64 data URIs
# into a single self-contained file (ideal for sending/opening standalone).
EMBED = "--embed" in sys.argv[1:]


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


def data_uri(path):
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode("ascii")
    return f"data:audio/wav;base64,{b}"


def audio_tag(abs_path, rel_path):
    if not os.path.exists(abs_path):
        return '<span class="missing">—</span>'
    src = data_uri(abs_path) if EMBED else rel_path
    return f'<audio controls preload="none" src="{src}"></audio>'


def main():
    reg = json.load(open(os.path.join(HERE, "models.json"), encoding="utf-8"))
    sentences = read_sentences(os.path.join(HERE, reg["sample_text_file"]))
    models = reg["models"]
    os.makedirs(DOCS, exist_ok=True)

    # Per-model info cards
    cards = []
    for m in models:
        cards.append(f"""
      <div class="card">
        <h3><a href="{html.escape(m['hf_url'])}" target="_blank" rel="noopener">{html.escape(m['name'])}</a></h3>
        <dl>
          <dt>Engine</dt><dd>{html.escape(m['engine'])}</dd>
          <dt>Voice</dt><dd>{html.escape(m['voice'])}</dd>
          <dt>Sample rate</dt><dd>{m.get('sample_rate','?')} Hz</dd>
          <dt>CPU RTF</dt><dd>{html.escape(str(m.get('cpu_rtf','?')))}</dd>
        </dl>
        <p class="notes">{html.escape(m.get('notes',''))}</p>
      </div>""")

    # Comparison grid: one block per sentence, all models side by side
    blocks = []
    for sid, text in sentences:
        rows = []
        for m in models:
            wav = os.path.join(DOCS, "audio", m["id"], f"{sid}.wav")
            rel = f"audio/{m['id']}/{sid}.wav"
            rows.append(f"""
          <div class="model-row">
            <span class="model-name">{html.escape(m['name'])}</span>
            {audio_tag(wav, rel)}
          </div>""")
        blocks.append(f"""
      <section class="sentence">
        <div class="sid">#{html.escape(sid)}</div>
        <p class="fa" dir="rtl" lang="fa">{html.escape(text)}</p>
        <div class="rows">{''.join(rows)}</div>
      </section>""")

    page = TEMPLATE.format(
        model_count=len(models),
        sentence_count=len(sentences),
        cards="".join(cards),
        blocks="".join(blocks),
    )
    out = os.path.join(DOCS, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    size_mb = os.path.getsize(out) / 1e6
    print(f"wrote {out}  ({size_mb:.2f} MB, {len(models)} models, {len(sentences)} sentences)")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Persian TTS Quality Comparison</title>
<style>
  :root {{
    --bg:#0f1216; --panel:#181d24; --panel2:#1f2630; --text:#e6edf3;
    --muted:#8b98a5; --accent:#5eb0ef; --border:#2a323d;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{ --bg:#f6f8fa; --panel:#fff; --panel2:#f0f3f6; --text:#1f2328;
      --muted:#59636e; --accent:#0969da; --border:#d0d7de; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    background:var(--bg); color:var(--text); line-height:1.5; }}
  header {{ padding:2rem 1.25rem 1rem; max-width:1000px; margin:0 auto; }}
  h1 {{ margin:0 0 .35rem; font-size:1.7rem; }}
  .sub {{ color:var(--muted); }}
  main {{ max-width:1000px; margin:0 auto; padding:0 1.25rem 4rem; }}
  h2 {{ font-size:1rem; text-transform:uppercase; letter-spacing:.06em;
    color:var(--muted); margin:2rem 0 .75rem; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr));
    gap:.75rem; }}
  .card {{ background:var(--panel); border:1px solid var(--border);
    border-radius:10px; padding:1rem; }}
  .card h3 {{ margin:0 0 .5rem; font-size:1rem; }}
  .card a {{ color:var(--accent); text-decoration:none; }}
  .card a:hover {{ text-decoration:underline; }}
  dl {{ display:grid; grid-template-columns:auto 1fr; gap:.15rem .6rem; margin:.5rem 0; }}
  dt {{ color:var(--muted); font-size:.82rem; }}
  dd {{ margin:0; font-size:.82rem; }}
  .notes {{ color:var(--muted); font-size:.8rem; margin:.5rem 0 0; }}
  .sentence {{ background:var(--panel); border:1px solid var(--border);
    border-radius:10px; padding:1rem 1.1rem; margin-bottom:1rem; }}
  .sid {{ color:var(--muted); font-size:.8rem; font-weight:600; }}
  .fa {{ font-size:1.35rem; margin:.35rem 0 .9rem; }}
  .rows {{ display:grid; gap:.5rem; }}
  .model-row {{ display:grid; grid-template-columns:200px 1fr; align-items:center;
    gap:.75rem; background:var(--panel2); border-radius:8px; padding:.5rem .7rem; }}
  .model-name {{ font-size:.85rem; color:var(--muted); }}
  audio {{ width:100%; height:36px; }}
  .missing {{ color:var(--muted); }}
  @media (max-width:640px) {{
    .model-row {{ grid-template-columns:1fr; }}
  }}
  footer {{ max-width:1000px; margin:0 auto; padding:1rem 1.25rem 3rem;
    color:var(--muted); font-size:.8rem; }}
  a {{ color:var(--accent); }}
</style>
</head>
<body>
<header>
  <h1>Persian TTS Quality Comparison</h1>
  <p class="sub">{model_count} model(s) · {sentence_count} sentences · same text, side by side.
  Source list: <a href="https://huggingface.co/models?pipeline_tag=text-to-speech&language=fa&sort=trending" target="_blank" rel="noopener">Hugging Face fa TTS</a></p>
</header>
<main>
  <h2>Models</h2>
  <div class="cards">{cards}</div>
  <h2>Side-by-side samples</h2>
  {blocks}
</main>
<footer>
  Generated by <code>scripts/build_site.py</code>. Audio is embedded (base64) so
  this page is fully self-contained.
</footer>
</body>
</html>
"""


if __name__ == "__main__":
    main()
