"""Local scorer for train: replays gold label.tool_calls to get the gold state_hash, then scores a submission
with the official formula (state-hash gate, then F1 over matched calls; fuzzy args by token-F1 > 0.6)."""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(split):
    sys.path.insert(0, str(ROOT / split))
    from zaic_tools import ToolEnv
    return ToolEnv


def fuzzy_params(catalog):
    return {t["name"]: {k for k, v in t["parameters"].get("properties", {}).items() if v.get("x-match") == "fuzzy"}
            for a in catalog["apps"] for t in a["tools"]}


def token_f1(a, b):
    ta = unicodedata.normalize("NFC", str(a).lower()).split()
    tb = unicodedata.normalize("NFC", str(b).lower()).split()
    if not ta and not tb:
        return 1.0
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def norm(v):
    return " ".join(v.lower().split()) if isinstance(v, str) else v


def call_match(sub, gold, fz):
    if sub.get("name") != gold["name"]:
        return False
    sa, ga = sub.get("arguments") or {}, gold.get("arguments") or {}
    if set(sa) != set(ga):
        return False
    fuzzy = fz.get(gold["name"], set())
    for k, gv in ga.items():
        sv = sa[k]
        if k in fuzzy:
            if token_f1(sv, gv) <= 0.6:
                return False
        elif k == "params" and isinstance(gv, list) and isinstance(sv, list):
            if sorted(sv) != sorted(gv):
                return False
        elif norm(sv) != norm(gv):
            return False
    return True


def sample_score(sub_calls, sub_hash, gold_calls, gold_hash, fz, ok_flags=None):
    if not gold_calls:
        return float(not sub_calls)
    if sub_hash != gold_hash:
        return 0.0
    used = [False] * len(gold_calls)
    m = 0
    for c in sub_calls:
        for j, g in enumerate(gold_calls):
            if not used[j] and call_match(c, g, fz):
                used[j] = True
                m += 1
                break
    if m == 0:
        return 0.0
    u = len(sub_calls) - m
    p, r = m / (m + u), m / len(gold_calls)
    return 2 * p * r / (p + r)


def gold_hashes(split="train"):
    ToolEnv = load_env(split)
    out = {}
    for line in open(ROOT / split / f"{split}.jsonl", encoding="utf-8"):
        d = json.loads(line)
        env = ToolEnv(split, d["id"], root=ROOT)
        for c in d["label"]["tool_calls"]:
            env.call(c["name"], c["arguments"])
        out[d["id"]] = (d["label"]["tool_calls"], env.get_state_hash(), d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--split", default="train")
    ap.add_argument("--details", help="write per-sample scores to this jsonl")
    a = ap.parse_args()
    fz = fuzzy_params(json.load(open(ROOT / a.split / f"{a.split}_tools_catalog.json", encoding="utf-8")))
    gold = gold_hashes(a.split)
    subs = [json.loads(l) for l in open(a.submission, encoding="utf-8") if l.strip()]
    scores, cats = [], Counter()
    det = open(a.details, "w", encoding="utf-8") if a.details else None
    for s in subs:
        gc, gh, d = gold[s["id"]]
        sc = sample_score(s["tool_calls"], s["state_hash"], gc, gh, fz)
        scores.append(sc)
        kind = "empty" if not gc else "ask" if any(c["name"] == "sys.ask_missing_info" for c in gc) else "exec"
        cats[kind, "n"] += 1
        cats[kind, "s"] += sc
        cats[kind, "hash_ok"] += (s["state_hash"] == gh) if gc else (not s["tool_calls"])
        if det:
            det.write(json.dumps({"id": s["id"], "score": sc, "hash_ok": s["state_hash"] == gh, "kind": kind}, ensure_ascii=False) + "\n")
    print(f"samples {len(scores)}  score {100 * sum(scores) / max(1, len(scores)):.2f}")
    for k in ("exec", "ask", "empty"):
        n = cats[k, "n"]
        if n:
            print(f"  {k:5s} n={n:4d}  score={100 * cats[k, 's'] / n:6.2f}  hash_ok={100 * cats[k, 'hash_ok'] / n:6.2f}%")


if __name__ == "__main__":
    main()
