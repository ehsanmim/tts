#!/usr/bin/env python3
"""Synthesize Persian benchmark sentences with fishaudio/s2-pro on CPU.

Model:  https://huggingface.co/fishaudio/s2-pro  (Dual-AR LLM TTS, 4B, ~9GB)
Engine: fish-speech native PyTorch path (NOT SGLang) — runs on CPU, just slow.
Repo:   cloned to .venvs/fish-speech ; checkpoint in .venvs/fish-speech/checkpoints/s2-pro
Run:    .venvs/fish-speech/venv/bin/python scripts/run_s2pro_persian.py [N]
        (N = number of sentences to do, default 1 for a quick quality probe)
Output: docs/audio/s2-pro/NN.wav

WARNING: 4B autoregressive model on CPU — minutes per sentence. Voice is cloned
from a Persian reference clip. License: fish-audio-research-license.
"""
import os
import re
import sys
import time
from pathlib import Path

TTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.join(TTS_ROOT, ".venvs", "fish-speech")
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

import numpy as np
import torch
import soundfile as sf
from loguru import logger

from fish_speech.models.text2semantic.inference import (
    init_model, load_codec_model, encode_audio, generate_long, decode_to_audio,
)

CKPT = Path("checkpoints/s2-pro")
DEVICE = "cpu"
PRECISION = torch.bfloat16
SENT_FILE = os.path.join(TTS_ROOT, "sample_texts", "sentences.txt")
OUT_DIR = os.path.join(TTS_ROOT, "docs", "audio", "s2-pro")
REF_WAV = os.path.join(TTS_ROOT, "docs", "audio", "kamtera-female-vits", "02.wav")
REF_ID = "02"  # the reference clip's sentence id (its transcript is the prompt text)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1


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
    sents = read_sentences(SENT_FILE)
    ref_text = dict(sents)[REF_ID]

    logger.info("Loading s2-pro (4B) on CPU — this takes a while ...")
    t0 = time.time()
    model, decode_one_token = init_model(CKPT, DEVICE, PRECISION, compile=False)
    with torch.device(DEVICE):
        model.setup_caches(max_batch_size=1, max_seq_len=model.config.max_seq_len,
                           dtype=next(model.parameters()).dtype)
    codec = load_codec_model(CKPT / "codec.pth", DEVICE, PRECISION)
    logger.info(f"Model+codec loaded in {time.time()-t0:.0f}s")

    # Encode the Persian reference clip once for voice cloning.
    ref_tokens = encode_audio(REF_WAV, codec, DEVICE).cpu()

    for sid, text in sents[:N]:
        t1 = time.time()
        torch.manual_seed(42)
        gen = generate_long(
            model=model, device=DEVICE, decode_one_token=decode_one_token,
            text=text, num_samples=1, max_new_tokens=0, top_p=0.9, top_k=30,
            temperature=1.0, compile=False, iterative_prompt=True, chunk_length=300,
            prompt_text=[ref_text], prompt_tokens=[ref_tokens],
        )
        codes = []
        for r in gen:
            if r.action == "sample":
                codes.append(r.codes)
            elif r.action == "next":
                break
        merged = torch.cat(codes, dim=1)
        audio = decode_to_audio(merged.to(DEVICE), codec)
        out = os.path.join(OUT_DIR, f"{sid}.wav")
        sf.write(out, audio.cpu().float().numpy(), codec.sample_rate)
        logger.info(f"[{sid}] {time.time()-t1:.0f}s -> {out}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
