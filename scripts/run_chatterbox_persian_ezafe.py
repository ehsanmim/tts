#!/usr/bin/env python3
"""Chatterbox-Persian with the diacritized text front-end + expressive settings.

Same model assembly as run_chatterbox_persian.py (base multilingual Chatterbox
with the gated Thomcles Farsi T3 swapped in), but the input text goes through
the shared front-end (fa_diacritics.py): digits→words, Ezafe kasra, and
KaamelDict word-internal harakat. exaggeration is raised 0.5 → 0.7 for a more
emotive delivery — the reason to reach for Chatterbox over the VITS models.

Auth:   needs HF_TOKEN with access to the gated finetune repo.
Run:    HF_TOKEN=hf_... .venvs/chatterbox/bin/python scripts/run_chatterbox_persian_ezafe.py [--smoke]
Output: docs/audio/chatterbox-persian-ezafe/NN.wav
"""
import os
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

from fa_diacritics import Frontend, read_sentences

# The base multilingual model doesn't list Persian; the finetune adds it. Open the gate.
mtl.SUPPORTED_LANGUAGES["fa"] = "Persian"

FT_REPO = "Thomcles/Chatterbox-TTS-Persian-Farsi"
FOLDER = "chatterbox-persian-ezafe"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_FILE = os.path.join(HERE, "sample_texts", "sentences.txt")
OUT_DIR = os.path.join(HERE, "docs", "audio", FOLDER)
REF_WAV = os.path.join(HERE, "docs", "audio", "kamtera-female-vits", "02.wav")
SMOKE = "--smoke" in sys.argv[1:]
EXAGGERATION = 0.7   # 0.5 = neutral; higher = more emotive
CFG_WEIGHT = 0.5
TEMPERATURE = 0.8


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    frontend = Frontend()

    print("Loading base ChatterboxMultilingual (CPU) ...")
    model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")

    print("Swapping in Farsi-finetuned T3 ...")
    ft_path = hf_hub_download(FT_REPO, "t3_fa.safetensors")
    state = load_file(ft_path)
    missing, unexpected = model.t3.load_state_dict(state, strict=False)
    print(f"  T3 load: {len(missing)} missing, {len(unexpected)} unexpected keys")

    sents = read_sentences(SENT_FILE)
    if SMOKE:
        sents = sents[:1]
    for sid, text in sents:
        final = frontend(text)
        if final != text:
            print(f"  [{sid}] → {final}")
        t0 = time.time()
        wav = model.generate(final, language_id="fa", audio_prompt_path=REF_WAV,
                             cfg_weight=CFG_WEIGHT, exaggeration=EXAGGERATION,
                             temperature=TEMPERATURE)
        dst = os.path.join(OUT_DIR, f"{sid}.wav")
        torchaudio.save(dst, wav.cpu() if hasattr(wav, "cpu") else torch.tensor(wav),
                        model.sr)
        print(f"  [{sid}] {time.time()-t0:5.1f}s  -> {dst}")

    print("Done. Wrote WAVs to", OUT_DIR)


if __name__ == "__main__":
    main()
