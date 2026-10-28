"""Turn train trajectories into chat SFT samples with exactly the prompt the agent sees at inference.

    python solution/sft_data.py --out out/sft_train.jsonl [--asr out/asr_train.jsonl] [--holdout-seed 0 --holdout 200]
Each line: {"id", "messages": [...], "tools": [...]} — one assistant tool call per step, gold tool_results as tool turns,
then a short closing reply. --asr adds a second copy of each sample whose user turns are ASR transcripts.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLOSE_DONE = "Dạ, em đã làm xong ạ."
CLOSE_ASK = "Dạ, anh/chị cho em thêm thông tin để em làm tiếp nhé."
CLOSE_NONE = "Dạ, việc này em chưa làm được trên các ứng dụng hiện có ạ."


def episode(item, tools, user_hist, user_final):
    msgs = [{"role": "system", "content": agent.system_prompt(item)}] + agent.history_messages(item, user_hist) + \
           [{"role": "user", "content": user_final}]
    calls, results = item["label"]["tool_calls"], item["label"]["tool_results"]
    for c, r in zip(calls, results):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [agent._tc(c["name"], c["arguments"])]})
        msgs.append({"role": "tool", "content": agent._clip(r, agent.RESULT_CHARS)})
    if not calls:
        close = CLOSE_NONE
    elif calls[-1]["name"] == "sys.ask_missing_info":
        close = None  # the agent stops right after ask_missing_info
    else:
        close = CLOSE_DONE
    if close:
        msgs.append({"role": "assistant", "content": close})
    return {"id": item["id"], "messages": msgs, "tools": tools}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr")
    ap.add_argument("--extra", help="train-format samples from synth.py (apps seen only in public_test's catalog)")
    ap.add_argument("--holdout", type=int, default=200, help="samples kept out for evaluation (same draw as run.py --limit)")
    ap.add_argument("--holdout-seed", type=int, default=0)
    a = ap.parse_args()

    cat = json.load(open(ROOT / "train/train_tools_catalog.json", encoding="utf-8"))
    ctools = {t["name"]: t for app in cat["apps"] for t in app["tools"]}
    if a.extra:  # synthetic samples use tools of apps that only public_test's catalog describes
        pub = json.load(open(ROOT / "public_test/public_test_tools_catalog.json", encoding="utf-8"))
        ctools = {**{t["name"]: t for app in pub["apps"] for t in app["tools"]}, **ctools}
    items = [json.loads(l) for l in open(ROOT / "train/train.jsonl", encoding="utf-8")]
    held = {d["id"] for d in random.Random(a.holdout_seed).sample(items, a.holdout)} if a.holdout else set()
    asr = {d["id"]: d for d in map(json.loads, open(a.asr, encoding="utf-8"))} if a.asr else {}

    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            if it["id"] in held:
                continue
            tools = agent.build_tools(it, ctools)
            hist = [h["audio_text"] for h in it["history"] if h["role"] == "user"]
            f.write(json.dumps(episode(it, tools, hist, it["audio_text"]), ensure_ascii=False) + "\n")
            n += 1
            if it["id"] in asr:
                r = asr[it["id"]]
                f.write(json.dumps(episode(it, tools, r["history"], r["final"]), ensure_ascii=False) + "\n")
                n += 1
        if a.extra:
            for line in open(a.extra, encoding="utf-8"):
                it = json.loads(line)
                f.write(json.dumps(episode(it, agent.build_tools(it, ctools), [], it["audio_text"]), ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} SFT samples ({len(held)} held out) -> {a.out}")


if __name__ == "__main__":
    main()
