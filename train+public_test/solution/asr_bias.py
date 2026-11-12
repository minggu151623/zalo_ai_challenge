"""Context-biased Whisper decoding: the final user turn is transcribed with a decoder prompt built from the sample's own
inputs (earlier assistant replies and names in earlier tool results), so names and terms already in the conversation
(contacts, notes, products, "Andrew Scott", "Muado"...) are spelled the way the agent needs them.

    python solution/asr_bias.py --lora out/asr_lora --hist out/asr_hold_ft.jsonl --limit 200 --out out/asr_hold_bias.jsonl
--hist gives the earlier user turns as already transcribed (never the gold text), --no-bias decodes the same way without a
prompt for an A/B, and train items report WER against audio_text.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from asr import ROOT, read_wav, wer

KEYS = ("name", "title", "query", "label", "location", "venue", "content", "subject", "item", "product", "symbol")


def bias_text(item, max_chars=400):
    """Names and short strings from earlier assistant turns, most recent first, plus the last assistant reply."""
    names, last_reply = [], ""

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in KEYS and isinstance(v, str) and 1 < len(v) <= 40:
                    names.append(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for h in reversed(item.get("history") or []):
        if h["role"] != "assistant":
            continue
        if not last_reply and h.get("text"):
            last_reply = h["text"]
        walk(h.get("tool_results") or [])
        for c in h.get("tool_calls") or []:
            walk(c.get("arguments") or {})
    seen, out = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    text = ", ".join(out)
    text = text[:max_chars].rsplit(",", 1)[0] if len(text) > max_chars else text
    return (text + ". " if text else "") + last_reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="vinai/PhoWhisper-large")
    ap.add_argument("--lora")
    ap.add_argument("--hist", help="asr.py output whose history transcripts are kept as they are")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bias", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32", help="float16 on CUDA; MPS gives NaNs in float16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    items = [json.loads(l) for l in open(ROOT / a.split / f"{a.split}.jsonl", encoding="utf-8") if l.strip()]
    if a.limit:
        items = random.Random(a.seed).sample(items, a.limit)
    hist = {d["id"]: d for d in map(json.loads, open(a.hist, encoding="utf-8"))} if a.hist else {}
    proc = WhisperProcessor.from_pretrained(a.model)
    dt = getattr(torch, a.dtype)
    model = WhisperForConditionalGeneration.from_pretrained(a.model, dtype=dt)
    if a.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.lora).merge_and_unload()
    model = model.to(a.device).eval()
    errs = toks = 0
    t0 = time.time()
    with open(a.out, "w", encoding="utf-8") as f:
        for k, it in enumerate(items):
            feats = proc(read_wav(ROOT / a.split / it["audio"]), sampling_rate=16000, return_tensors="pt").input_features
            kw = {"language": "vi", "task": "transcribe", "num_beams": 1, "do_sample": False}
            b = "" if a.no_bias else bias_text(it)
            if b:
                kw["prompt_ids"] = torch.tensor(proc.get_prompt_ids(b)[-200:]).to(a.device)
            with torch.no_grad():
                ids = model.generate(feats.to(a.device, dt), **kw)
            hyp = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if b and hyp.startswith(b.strip()[:20]):  # some versions return the prompt too
                hyp = hyp[len(b.strip()):].strip()
            f.write(json.dumps({"id": it["id"], "final": hyp, "history": (hist.get(it["id"]) or {}).get("history", []),
                                "bias": b}, ensure_ascii=False) + "\n")
            if it.get("audio_text"):
                e, n = wer(it["audio_text"], hyp)
                errs, toks = errs + e, toks + n
            if k % 20 == 0:
                print(f"{k}/{len(items)} {time.time() - t0:.0f}s WER so far {100 * errs / max(toks, 1):.2f}%", flush=True)
    print(f"final-turn WER {100 * errs / max(toks, 1):.2f}% over {len(items)} ({'no bias' if a.no_bias else 'biased'})")


if __name__ == "__main__":
    main()
