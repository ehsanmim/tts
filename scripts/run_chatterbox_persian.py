#!/usr/bin/env python3
"""Synthesize the Persian benchmark sentences with Thomcles/Chatterbox-TTS-Persian-Farsi.

Model:  https://huggingface.co/Thomcles/Chatterbox-TTS-Persian-Farsi  (GATED)
        A Farsi finetune of the **T3** stage of ResembleAI/chatterbox (multilingual).
Engine: chatterbox-tts (ChatterboxMultilingualTTS) — base weights from
        ResembleAI/chatterbox with the Farsi T3 swapped in.
Auth:   needs HF_TOKEN with access to the gated repo.
Run:    HF_TOKEN=hf_... .venvs/chatterbox/bin/python scripts/run_chatterbox_persian.py [--smoke]
Output: docs/audio/chatterbox-persian/NN.wav

NOTE: autoregressive ~0.5B model — SLOW on CPU (minutes/sentence). License CC-BY-NC 4.0.
"""
import os
import re
import sys
import time
import torch
import torchaudio
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import chatterbox.mtl_tts as mtl
import chatterbox.models.tokenizers.tokenizer as _tk

# The multilingual tokenizer eagerly builds a Chinese pkuseg segmenter (network
# download, hash check fails behind the proxy). We only synthesize Persian, and
# the Cangjie path is used only for language_id=='zh', so stub it out.
class _NoopCangjie:
    def __init__(self, *a, **k):
        pass
    def __call__(self, txt):
        return txt
_tk.ChineseCangjieConverter = _NoopCangjie

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# The base multilingual model doesn't list Persian; the finetune adds it. Open the gate.
mtl.SUPPORTED_LANGUAGES["fa"] = "Persian"

FT_REPO = "Thomcles/Chatterbox-TTS-Persian-Farsi"
FOLDER = "chatterbox-persian"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_FILE = os.path.join(HERE, "sample_texts", "sentences.txt")
OUT_DIR = os.path.join(HERE, "docs", "audio", FOLDER)
REF_WAV = os.path.join(HERE, "docs", "audio", "kamtera-female-vits", "02.wav")
SMOKE = "--smoke" in sys.argv[1:]


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
    print("Loading base ChatterboxMultilingual (CPU) ...")
    model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")

    # Sanity: does the multilingual tokenizer know the [fa] language token?
    fa_ids = model.tokenizer.tokenizer.encode("[fa]").ids
    print("  '[fa]' -> token ids:", fa_ids, "(single id => real token)")

    print("Swapping in Farsi-finetuned T3 ...")
    ft_path = hf_hub_download(FT_REPO, "t3_fa.safetensors")
    state = load_file(ft_path)
    missing, unexpected = model.t3.load_state_dict(state, strict=False)
    print(f"  T3 load: {len(missing)} missing, {len(unexpected)} unexpected keys")

    sents = read_sentences(SENT_FILE)
    if SMOKE:
        sents = sents[:1]
    for sid, text in sents:
        t0 = time.time()
        wav = model.generate(text, language_id="fa", audio_prompt_path=REF_WAV,
                             cfg_weight=0.5, exaggeration=0.5, temperature=0.8)
        dst = os.path.join(OUT_DIR, f"{sid}.wav")
        torchaudio.save(dst, wav.cpu() if hasattr(wav, "cpu") else torch.tensor(wav),
                        model.sr)
        print(f"  [{sid}] {time.time()-t0:5.1f}s  -> {dst}")

    print("Done. Wrote WAVs to", OUT_DIR)


if __name__ == "__main__":
    main()
