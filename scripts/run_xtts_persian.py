#!/usr/bin/env python3
"""Synthesize the Persian benchmark sentences with alikhabazian/XTTS_Persian.

Model:  https://huggingface.co/alikhabazian/XTTS_Persian  (XTTS-v2 finetune, adds `fa`)
Engine: Coqui TTS (coqui-tts), loaded via TTS.tts.models.xtts.Xtts.
        Autoregressive voice-cloning model — needs a reference speaker clip.
Run with the coqui venv:  .venvs/coqui/bin/python scripts/run_xtts_persian.py
Output: docs/audio/xtts-persian/NN.wav  (24 kHz)

NOTE: SLOW on CPU (tens of seconds per sentence). The finetune is the full 5.7 GB
model; base XTTS support files (vocab, mel_stats) come from coqui/XTTS-v2.
"""
import os
os.environ["COQUI_TOS_AGREED"] = "1"
import re
import time
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

REPO = "alikhabazian/XTTS_Persian"
FOLDER = "xtts-persian"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_FILE = os.path.join(HERE, "sample_texts", "sentences.txt")
OUT_DIR = os.path.join(HERE, "docs", "audio", FOLDER)
# A Persian reference voice for conditioning (reuse an already-generated sample).
REF_WAV = os.path.join(HERE, "docs", "audio", "kamtera-female-vits", "02.wav")


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
    print("Downloading finetune (config + 5.7 GB checkpoint) ...")
    cfg_path = hf_hub_download(REPO, "config.json")
    ckpt_path = hf_hub_download(REPO, "best_model.pth")
    vocab = hf_hub_download("coqui/XTTS-v2", "vocab.json")
    mel = hf_hub_download("coqui/XTTS-v2", "mel_stats.pth")
    dvae = hf_hub_download("coqui/XTTS-v2", "dvae.pth")

    config = XttsConfig()
    config.load_json(cfg_path)
    # Point support-file paths at the real cached base files.
    config.model_args.tokenizer_file = vocab
    config.model_args.mel_norm_file = mel
    config.model_args.dvae_checkpoint = dvae
    config.model_args.xtts_checkpoint = ckpt_path
    # Finetune expanded the GPT text vocab to 10120 tokens — build the GPT to
    # match so the checkpoint's embedding/head shapes line up.
    config.model_args.gpt_number_text_tokens = 10120

    print("Loading XTTS finetune (CPU) ...")
    model = Xtts.init_from_config(config)
    # strict=False: the gpt_inference.* KV-cache submodule is built lazily.
    model.load_checkpoint(config, checkpoint_path=ckpt_path, vocab_path=vocab,
                          use_deepspeed=False, strict=False)
    model.eval()

    print("Extracting conditioning latents from reference voice ...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[REF_WAV])

    for sid, text in read_sentences(SENT_FILE):
        t0 = time.time()
        out = model.inference(text, "fa", gpt_cond_latent, speaker_embedding,
                              temperature=0.7)
        wav = torch.tensor(out["wav"]).unsqueeze(0)
        dst = os.path.join(OUT_DIR, f"{sid}.wav")
        torchaudio.save(dst, wav, 24000)
        print(f"  [{sid}] took {time.time()-t0:5.1f}s  -> {dst}")

    print("Done. Wrote WAVs to", OUT_DIR)


if __name__ == "__main__":
    main()
