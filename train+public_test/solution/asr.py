"""Transcribe every user turn of a split with a Whisper-family model and (on train) report WER against audio_text.

    python solution/asr.py --split train --model vinai/PhoWhisper-large --limit 300 --out out/asr_train.jsonl
Output lines: {"id", "final": str, "history": [str, ...]} — the format run.py --text expects.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def norm_text(s):
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return s.split()


def wer(ref, hyp):
    r, h = norm_text(ref), norm_text(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def read_wav(p):
    with wave.open(str(p)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, p
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def jobs_for(item):
    """[(key, wav_path, ref_text_or_None)] for every user turn: history turns first, final turn last."""
    out = []
    k = 0
    for h in item["history"]:
        if h["role"] == "user":
            out.append((("h", k), h["audio"], h.get("audio_text")))
            k += 1
    out.append((("f", 0), item["audio"], item.get("audio_text")))
    return out


def fold_ids(items, fold, folds, holdout=200, seed=0):
    """Ids of train fold `fold` of `folds`, drawn from the items outside the run.py/sft_data.py holdout."""
    held = {d["id"] for d in random.Random(seed).sample(items, holdout)} if holdout else set()
    rest = [d["id"] for d in items if d["id"] not in held]
    return {i for k, i in enumerate(rest) if k % folds == fold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="vinai/PhoWhisper-large")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final-only", action="store_true", help="only the final turn (faster WER check)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fold", type=int, help="transcribe only this train fold (outside the holdout), see --folds")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from transformers import pipeline

    items = [json.loads(l) for l in open(ROOT / a.split / f"{a.split}.jsonl", encoding="utf-8") if l.strip()]
    if a.limit:
        items = random.Random(a.seed).sample(items, a.limit)
    if a.fold is not None:
        keep = fold_ids(items, a.fold, a.folds)
        items = [d for d in items if d["id"] in keep]
    jobs = []
    for it in items:
        for key, path, ref in jobs_for(it):
            if a.final_only and key[0] == "h":
                continue
            jobs.append((it["id"], key, ROOT / a.split / path, ref))

    try:
        asr = pipeline("automatic-speech-recognition", model=a.model, dtype=torch.float16, device=0)
    except TypeError:  # transformers < 4.56
        asr = pipeline("automatic-speech-recognition", model=a.model, torch_dtype=torch.float16, device=0)
    gen_kwargs = {"language": "vi", "task": "transcribe", "num_beams": 1, "do_sample": False}
    t0 = time.time()
    audio = [{"raw": read_wav(p), "sampling_rate": 16000} for _, _, p, _ in jobs]
    secs = sum(len(x["raw"]) for x in audio) / 16000  # before the pipeline: it pops "raw" from each dict
    hyps = [o["text"].strip() for o in asr(audio, batch_size=a.batch, generate_kwargs=gen_kwargs)]
    dt = time.time() - t0

    res = {}
    errs = toks = 0
    for (qid, key, _, ref), hyp in zip(jobs, hyps):
        r = res.setdefault(qid, {"id": qid, "final": "", "history": []})
        if key[0] == "f":
            r["final"] = hyp
        else:
            r["history"].append(hyp)
        if ref:
            e, n = wer(ref, hyp)
            errs, toks = errs + e, toks + n
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in res.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(jobs)} clips, {secs / 60:.1f} min audio in {dt:.0f}s (RTF {dt / secs:.3f})")
    if toks:
        print(f"WER {100 * errs / toks:.2f}%  ({a.model})")


if __name__ == "__main__":
    main()
