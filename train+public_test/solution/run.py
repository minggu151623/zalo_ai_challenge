"""Run the agent over a split and write submission.jsonl plus traces.

    python solution/run.py --split train --backend vllm --model Qwen/Qwen3-4B-Instruct-2507 --limit 200 --out out/
    python solution/run.py --split train --backend oracle --limit 50 --out out/    # plumbing check: replays gold calls
Transcripts: --text gold (train audio_text) or --text asr.jsonl ({"id", "final", "history": [...]} per line).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_items(split, limit, seed, ids=None):
    items = [json.loads(l) for l in open(ROOT / split / f"{split}.jsonl", encoding="utf-8") if l.strip()]
    if ids:
        keep = set(ids)
        items = [d for d in items if d["id"] in keep]
    if limit and limit < len(items):
        items = random.Random(seed).sample(items, limit)
    return items


def transcripts(item, text_src, asr):
    if text_src == "gold":
        return [h["audio_text"] for h in item["history"] if h["role"] == "user"], item["audio_text"]
    a = asr[item["id"]]
    return a["history"], a["final"]


def oracle_backend(items):
    """Emits the gold calls one per round, then a closing sentence."""
    gold = {d["id"]: d["label"]["tool_calls"] for d in items}
    pos = {}

    def generate(batch, _tools):
        outs = []
        for msgs in batch:
            qid = msgs[0]["_id"]
            i = pos.get(qid, 0)
            pos[qid] = i + 1
            g = gold[qid]
            outs.append(f'<tool_call>\n{json.dumps(g[i], ensure_ascii=False)}\n</tool_call>' if i < len(g) else "Xong.")
        return outs
    return generate


def vllm_backend(model, max_model_len, gpu_util, dtype, tp=1, force_call=0.0, first_log=None):
    """force_call > 0: when a reply to a user turn has no tool call but the model gave <tool_call> at least this
    probability as its first token, the reply is regenerated starting with <tool_call> (curbs false refusals).
    first_log: file to append {"id", "p_tc", "forced"} for those replies, to calibrate force_call."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_util, dtype=dtype,
              enable_prefix_caching=True, seed=0, tensor_parallel_size=tp)
    tok = llm.get_tokenizer()
    tc_id = tok.convert_tokens_to_ids("<tool_call>")
    sp = SamplingParams(temperature=0.0, max_tokens=768, seed=0, logprobs=20)
    flog = open(first_log, "a", encoding="utf-8") if first_log else None

    def generate(batch, tools):
        prompts = []
        for msgs, t in zip(batch, tools):
            clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in msgs]
            p = tok.apply_chat_template(clean, tools=t, add_generation_prompt=True, tokenize=False)
            ids = tok(p).input_ids
            if len(ids) > max_model_len - 800:
                p = tok.decode(ids[-(max_model_len - 800):])  # crude guard; real budget control is in agent.py
            prompts.append(p)
        outs = llm.generate(prompts, sp, use_tqdm=False)
        texts = [o.outputs[0].text for o in outs]
        redo = []
        for k, (o, msgs) in enumerate(zip(outs, batch)):
            if msgs[-1]["role"] != "user" or "<tool_call>" in texts[k] or not o.outputs[0].logprobs:
                continue
            lp = o.outputs[0].logprobs[0].get(tc_id)
            p_tc = math.exp(lp.logprob) if lp is not None else 0.0
            forced = bool(force_call) and p_tc >= force_call
            if forced:
                redo.append(k)
            if flog:
                flog.write(json.dumps({"id": msgs[0].get("_id"), "p_tc": p_tc, "forced": forced}) + "\n")
                flog.flush()
        if redo:
            more = llm.generate([prompts[k] + "<tool_call>" for k in redo], sp, use_tqdm=False)
            for k, o in zip(redo, more):
                texts[k] = "<tool_call>" + o.outputs[0].text
        return texts
    return generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "oracle"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--text", default="gold")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (2 on Kaggle 2xT4)")
    ap.add_argument("--wave", type=int, default=50, help="episodes per wave; each finished episode is written at once")
    ap.add_argument("--force-call", type=float, default=0.0, help="see vllm_backend; 0 = off")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / a.split))
    from zaic_tools import ToolEnv

    catalog = json.load(open(ROOT / a.split / f"{a.split}_tools_catalog.json", encoding="utf-8"))
    ctools = {t["name"]: t for app in catalog["apps"] for t in app["tools"]}
    asr = {}
    if a.text != "gold":
        asr = {d["id"]: d for d in map(json.loads, open(a.text, encoding="utf-8"))}
    items = load_items(a.split, a.limit, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sub_path, trace_path = out / "submission.jsonl", out / "traces.jsonl"
    # resume: samples already written are kept; an episode cut off mid-run is redone from its initial state
    done_ids = {json.loads(l)["id"] for l in open(sub_path, encoding="utf-8") if l.strip()} if sub_path.exists() else set()
    if done_ids:
        print(f"resume: {len(done_ids)} samples already in {sub_path}, skipping them", flush=True)
    items = [it for it in items if it["id"] not in done_ids]

    states = []
    for it in items:
        env = ToolEnv(a.split, it["id"], root=ROOT)  # exactly one ToolEnv per sample
        hist, final = transcripts(it, a.text, asr)
        s = agent.AgentState(it, env, agent.build_tools(it, ctools), hist, final)
        s.messages[0]["_id"] = it["id"]
        states.append(s)

    if not states:
        print("nothing to do", flush=True)
        return
    gen = oracle_backend(items) if a.backend == "oracle" else vllm_backend(
        a.model, a.max_model_len, a.gpu_util, a.dtype, a.tp, a.force_call, str(out / "first_token.jsonl"))
    t0 = time.time()
    n_done = [0]

    with open(sub_path, "a", encoding="utf-8") as fs, open(trace_path, "a", encoding="utf-8") as ft:
        def on_done(s):
            fs.write(json.dumps({"id": s.item["id"], "tool_calls": s.env.get_tool_calls(), "state_hash": s.env.get_state_hash()},
                                ensure_ascii=False) + "\n")
            ft.write(json.dumps({"id": s.item["id"], "messages": s.messages[1:]}, ensure_ascii=False) + "\n")
            fs.flush()
            ft.flush()
            n_done[0] += 1

        def log(msg):
            print(f"[{time.time() - t0:6.0f}s] {msg} | written {len(done_ids) + n_done[0]}/{len(done_ids) + len(states)}", flush=True)

        agent.run_agents(states, gen, log=log, on_done=on_done, wave=a.wave)
    dt = time.time() - t0
    print(f"{len(states)} samples in {dt:.0f}s ({dt / len(states):.2f}s/sample batched) -> {out}", flush=True)


if __name__ == "__main__":
    main()
