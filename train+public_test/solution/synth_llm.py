"""LLM-authored episodes for the apps that only public_test's catalog has (44% of public samples call one of them, and
the template samples of synth.py cover them with simple, uniform requests).

    # 1) author: Qwen3-14B writes a spoken request + the gold calls, given a real train sample as the style example and a
    #    generated app state (synth.py's generators). Calls are executed with zaic_tools' Simulator; failures are dropped.
    python solution/synth_llm.py author --n 3000 --split-dir synv --model Qwen/Qwen3-14B-AWQ
    # 2) solve: an independent agent run on the written requests (run.py --split synv, same 14B)
    python solution/run.py --split synv --compact --model Qwen/Qwen3-14B-AWQ --out out/synv_solve
    python solution/score.py out/synv_solve/submission.jsonl --split synv --details out/synv_solve/details.jsonl
    # 3) keep the episodes where the solver reached the author's final state with the same calls (score 1)
    python solution/synth_llm.py keep --split-dir synv --details out/synv_solve/details.jsonl --dev 300 --out out/synth_llm.jsonl
synv/ is a split laid out like train/ (synv.jsonl, synv_environment/<id>/state.json, synv_tools_catalog.json, zaic_tools),
so run.py and score.py work on it unchanged; `keep --dev N` also writes split synvdev/ (N kept episodes, never trained on)
to measure models on unseen apps. Nothing is read from public_test samples or environments: only its tool catalog.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402
import synth  # noqa: E402
from synth import Ctx, GENS, ROOT, Simulator  # noqa: E402

COND = re.compile(r"\bnếu\b|\bthì\b|không thì|chưa .* thì|còn .* thì|hết .* thì", re.I)
KINDS = [  # (name, weight, instruction)
    ("direct", 30, "a direct request on the new app (one or more actions); read first only when an ID or value must be looked up"),
    ("conditional", 25, "a conditional request: the user wants something done only if a fact holds (price, stock, time, "
                        "status, balance, weather...). The calls first read the fact; then do the action ONLY if the condition "
                        "holds in the data shown above. Make the condition hold or fail about equally often."),
    ("multi", 20, "a request that combines the new app with one other installed app (e.g. then text someone, set a reminder, "
                  "add to calendar, pay), in the order the user says"),
    ("ask", 15, "a request for a write action on the new app where a REQUIRED parameter is missing and cannot be inferred "
                "(e.g. no quantity, no date, several matches). The calls are the reads needed (if any), then "
                "sys.ask_missing_info with tool_name = the write tool and params = the missing parameter names; nothing after it"),
    ("info", 10, "a question answered by read tools of the new app only (no write)"),
]
INSTR = """Write ONE new request that the user says next, mainly about the app "{app}" ({desc}), and the exact tool calls a perfect assistant makes for it.
Kind of request: {kind}.
Rules:
- Spoken Vietnamese like the example: natural, casual, may use regional words, fillers and spoken numbers ("hai trăm", "tám giờ rưỡi"); 1-2 sentences; it may refer to the earlier conversation.
- Only the tools listed above. Every argument comes from the request, the context, or an earlier call's result. An ID (…_id) may only be used after a call in YOUR list (or in the earlier conversation) returned it: e.g. first ecommerce.search with a query taken from the user's words, then ecommerce.add_to_cart with a product_id from that result. Use only IDs that exist in the data dump.
- The data dump above is exactly what the read tools will return: use it to decide conditions and pick IDs.
- Dates YYYY-MM-DD and times ISO 8601 with +07:00, computed from the current time. Integers without separators. Optional parameters only when the user says them.
- Do not call a tool that is not needed; do not repeat a call.
Answer with JSON only: {{"request": "...", "calls": [{{"name": "...", "arguments": {{...}}}}, ...]}}"""


def load_all(holdout=200):
    items = [json.loads(l) for l in open(ROOT / "train/train.jsonl", encoding="utf-8")]
    held = {d["id"] for d in random.Random(0).sample(items, holdout)}
    base = [d for d in items if d["id"] not in held]
    envs = {d["id"]: json.load(open(ROOT / "train/train_environment" / d["id"] / "state.json", encoding="utf-8")) for d in base}
    pub = json.load(open(ROOT / "public_test/public_test_tools_catalog.json", encoding="utf-8"))
    trn = json.load(open(ROOT / "train/train_tools_catalog.json", encoding="utf-8"))
    seen = {x["id"] for x in pub["apps"]}
    catalog = {**pub, "apps": pub["apps"] + [x for x in trn["apps"] if x["id"] not in seen]}
    return base, envs, catalog


def example_kind(d):
    calls = d["label"]["tool_calls"]
    if not calls:
        return None
    if calls[-1]["name"] == "sys.ask_missing_info":
        return "ask"
    if COND.search(d["audio_text"]):
        return "conditional"
    if len({c["name"].split(".")[0] for c in calls}) >= 2:
        return "multi"
    return "direct"


def show_example(d):
    out = []
    for h in d.get("history") or []:
        if h["role"] == "user":
            out.append(f"(earlier) User: {h['audio_text']}")
    out.append(f"User: {d['audio_text']}")
    for c, r in zip(d["label"]["tool_calls"], d["label"]["tool_results"]):
        out.append("call " + json.dumps({"name": c["name"], "arguments": c["arguments"]}, ensure_ascii=False))
        out.append("  result " + agent._clip(r, 300))
    return "\n".join(out)


def candidate(i, rng, base, envs, catalog, ctools, pools, app):
    b = rng.choice(base)
    item = {"id": f"synv_{i:05d}", "audio": None, "history": [], "context": copy.deepcopy(b["context"])}
    if rng.random() < 0.5:  # the base's state is the one after its history
        item["history"] = copy.deepcopy(b.get("history") or [])
    states = {a: copy.deepcopy(s) for a, s in envs[b["id"]].items() if a != "_meta"}
    now = datetime.fromisoformat(item["context"]["current_time"])
    st, _ = GENS[app](Ctx(rng, now))
    states[app] = st
    tools = [t["name"] for a in catalog["apps"] if a["id"] == app for t in a["tools"]]
    item["context"]["installed_apps"][app] = {"tools": tools}
    kind = rng.choices([k[0] for k in KINDS], [k[1] for k in KINDS])[0]
    ex = rng.choice(pools.get(kind) or pools["direct"])
    meta = next(a for a in catalog["apps"] if a["id"] == app)
    lines = [agent.system_prompt(item), "", "# Tools", *[agent.compact_tool(ctools[n]) for n in
             dict.fromkeys(t for a in item["context"]["installed_apps"].values() for t in a["tools"]) if n in ctools],
             agent.compact_tool(ctools["sys.ask_missing_info"]) if "sys.ask_missing_info" in ctools else "",
             "", "# Data in the apps (what read tools return)",
             f"{app}: {agent._clip(st, 3500)}"]
    for a in rng.sample([a for a in states if a != app], min(4, len(states) - 1)):
        lines.append(f"{a}: {agent._clip(states[a], 700)}")
    if item["history"]:
        lines.append("\n# Earlier conversation (already done)")
        for h in item["history"]:
            if h["role"] == "user":
                lines.append(f"User: {h['audio_text']}")
            else:
                for c in h.get("tool_calls") or []:
                    lines.append("Assistant calls " + json.dumps(c, ensure_ascii=False))
                if h.get("text"):
                    lines.append(f"Assistant: {h['text']}")
    lines += ["\n# Example of a real request with its gold calls (other apps; copy its style and difficulty, not its content)",
              show_example(ex), "\n# Task",
              INSTR.format(app=app, desc=f"{meta.get('name_vi', app)}: {meta.get('description', '')}",
                           kind=next(k[2] for k in KINDS if k[0] == kind))]
    return {"item": item, "states": states, "kind": kind, "app": app, "prompt": "\n".join(lines)}


def parse(text):
    text = text.split("</think>")[-1]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    req, calls = obj.get("request"), obj.get("calls")
    if not isinstance(req, str) or not req.strip() or not isinstance(calls, list):
        return None
    out = []
    for c in calls:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str) or not isinstance(c.get("arguments", {}), dict):
            return None
        out.append({"name": c["name"], "arguments": c.get("arguments") or {}})
    return req.strip(), out


def execute(cand, req, calls, catalog):
    """Run the calls; None when one fails, uses an ID nobody read, or does not fit the requested kind."""
    item, states, app = cand["item"], cand["states"], cand["app"]
    allowed = {t for a in item["context"]["installed_apps"].values() for t in a["tools"]} | {"sys.ask_missing_info"}
    if not calls or any(c["name"] not in allowed for c in calls):
        return None
    if not any(c["name"].split(".")[0] == app for c in calls) and not (
            calls[-1]["name"] == "sys.ask_missing_info" and str(calls[-1]["arguments"].get("tool_name", "")).startswith(app + ".")):
        return None
    asks = [k for k, c in enumerate(calls) if c["name"] == "sys.ask_missing_info"]
    if asks and asks[0] != len(calls) - 1:
        return None
    sim = Simulator({"context": {"current_time": item["context"]["current_time"],  # as zaic_tools' Session builds it
                                 "user_profile": {"city": (item["context"].get("user_profile") or {}).get("city")},
                                 "installed_apps": {a: {"tools": v["tools"], "state": states.get(a, {})}
                                                    for a, v in item["context"]["installed_apps"].items()}}}, catalog)
    seen = json.dumps([h.get("tool_results") for h in item["history"]], ensure_ascii=False)
    results = []
    for c in calls:
        if any(k.endswith("_id") and isinstance(v, str) and f'"{v}"' not in seen for k, v in c["arguments"].items()):
            return None
        if not sim.validate(c["name"], c["arguments"]).get("ok"):
            return None
        r = sim.call(c["name"], c["arguments"])
        if not r.get("ok") or r.get("done") is False:
            return None
        results.append(r)
        seen += json.dumps(r, ensure_ascii=False)
    it = copy.deepcopy(item)
    it["audio_text"] = req
    it["label"] = {"tool_calls": calls, "tool_results": results}
    it["synth_app"], it["synth_kind"] = app, cand["kind"]
    return it


def write_split(d, name, items, states, catalog):
    d.mkdir(parents=True, exist_ok=True)
    env = d / f"{name}_environment"
    env.mkdir(exist_ok=True)
    with open(d / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({k: v for k, v in it.items()}, ensure_ascii=False) + "\n")
            meta = {"question_id": it["id"], "current_time": it["context"]["current_time"],
                    "city": (it["context"].get("user_profile") or {}).get("city"),
                    "tools": sorted({t for a in it["context"]["installed_apps"].values() for t in a["tools"]})}
            (env / it["id"]).mkdir(exist_ok=True)
            json.dump({"_meta": meta, **states[it["id"]]}, open(env / it["id"] / "state.json", "w", encoding="utf-8"),
                      ensure_ascii=False)
    json.dump(catalog, open(d / f"{name}_tools_catalog.json", "w", encoding="utf-8"), ensure_ascii=False)
    if not (d / "zaic_tools").exists():
        shutil.copytree(ROOT / "public_test/zaic_tools", d / "zaic_tools")


def author(a):
    base, envs, catalog = load_all()
    ctools = {t["name"]: t for x in catalog["apps"] for t in x["tools"]}
    pools = {}
    for d in base:
        k = example_kind(d)
        if k:
            pools.setdefault(k, []).append(d)
    rng = random.Random(a.seed)
    apps = sorted(GENS)
    cands = [candidate(i, rng, base, envs, catalog, ctools, pools, apps[i % len(apps)]) for i in range(a.start, a.start + a.n)]
    print(f"{len(cands)} candidates; kinds", {k[0]: sum(c['kind'] == k[0] for c in cands) for k in KINDS}, flush=True)
    if a.dump:
        for c in cands[:a.dump]:
            print(c["prompt"], "\n" + "=" * 80)
        return
    from vllm import LLM, SamplingParams
    llm = LLM(a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util,
              enable_prefix_caching=True, seed=0, dtype="half")
    tok = llm.get_tokenizer()
    kw = {"enable_thinking": False} if "Qwen3-" in a.model and "2507" not in a.model else {}
    prompts = [tok.apply_chat_template([{"role": "user", "content": c["prompt"]}], add_generation_prompt=True,
                                       tokenize=False, **kw) for c in cands]
    keep = [i for i, p in enumerate(prompts) if len(tok(p).input_ids) < a.max_model_len - 700]
    outs = llm.generate([prompts[i] for i in keep], SamplingParams(temperature=0.8, top_p=0.95, max_tokens=600, seed=a.seed))
    items, states, why = [], {}, {"parse": 0, "exec": 0, "long": len(cands) - len(keep)}
    for i, o in zip(keep, outs):
        p = parse(o.outputs[0].text)
        if not p:
            why["parse"] += 1
            continue
        it = execute(cands[i], p[0], p[1], catalog)
        if not it:
            why["exec"] += 1
            continue
        items.append(it)
        states[it["id"]] = cands[i]["states"]
    write_split(ROOT / a.split_dir, Path(a.split_dir).name, items, states, catalog)
    print(f"{len(items)} episodes -> {a.split_dir}; dropped {why}; kinds",
          {k[0]: sum(x['synth_kind'] == k[0] for x in items) for k in KINDS}, flush=True)

JUDGE = """You are checking one training example for a Vietnamese voice assistant that operates apps through tool calls.
Above: the context, the tools, the data in the apps, the earlier conversation, the user's new request, and the calls proposed
as the gold answer with their real results. The example is CORRECT only if ALL hold:
1. Every call is needed for the request; nothing extra (no unrequested alerts, reads whose result is not needed, repeats).
2. Every argument value is said by the user, in the context/earlier conversation, or taken from an earlier result; none is invented or guessed (stops, routes, amounts, times, places, quantities).
3. Conditions in the request are checked with a read call and decided correctly from its result.
4. Nothing the user asked for is left undone, unless a required value is missing, in which case sys.ask_missing_info names the right tool and parameter(s).
5. Dates/times match what the user said, relative to the current time.
Answer with one word: CORRECT or WRONG."""


def judge_prompt(it, states, ctools):
    used = {c["name"].split(".")[0] for c in it["label"]["tool_calls"]} | {it["synth_app"]}
    lines = [agent.system_prompt(it), "", "# Tools"]
    lines += [agent.compact_tool(ctools[n]) for a in sorted(used) for n in (it["context"]["installed_apps"].get(a) or {}).get("tools", []) if n in ctools]
    lines += ["", "# Data in the apps"] + [f"{a}: {agent._clip(states.get(a, {}), 2500)}" for a in sorted(used) if a in states]
    if it["history"]:
        lines.append("\n# Earlier conversation")
        for h in it["history"]:
            if h["role"] == "user":
                lines.append(f"User: {h['audio_text']}")
            else:
                lines += ["Assistant calls " + json.dumps(c, ensure_ascii=False) for c in h.get("tool_calls") or []]
                if h.get("text"):
                    lines.append(f"Assistant: {h['text']}")
    lines += ["\n# New request", f"User: {it['audio_text']}", "\n# Proposed gold calls"]
    for c, r in zip(it["label"]["tool_calls"], it["label"]["tool_results"]):
        lines += ["call " + json.dumps(c, ensure_ascii=False), "  result " + agent._clip(r, 600)]
    return "\n".join(lines + ["", JUDGE])


def judge(a):
    import math
    d = ROOT / a.split_dir
    name = d.name
    items = [json.loads(l) for l in open(d / f"{name}.jsonl", encoding="utf-8")][a.shard::a.nshard]
    catalog = json.load(open(d / f"{name}_tools_catalog.json", encoding="utf-8"))
    ctools = {t["name"]: t for x in catalog["apps"] for t in x["tools"]}
    states = {it["id"]: json.load(open(d / f"{name}_environment" / it["id"] / "state.json", encoding="utf-8")) for it in items}
    from vllm import LLM, SamplingParams
    llm = LLM(a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util, enable_prefix_caching=True,
              seed=0, dtype="half")
    tok = llm.get_tokenizer()
    kw = {"enable_thinking": False} if "Qwen3-" in a.model and "2507" not in a.model else {}
    prompts = [tok.apply_chat_template([{"role": "user", "content": judge_prompt(it, states[it["id"]], ctools)}],
                                       add_generation_prompt=True, tokenize=False, **kw) for it in items]
    ok_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("CORRECT", " CORRECT")}
    bad_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("WRONG", " WRONG")}
    keep_i = [i for i, p in enumerate(prompts) if len(tok(p).input_ids) < a.max_model_len - 8]
    outs = llm.generate([prompts[i] for i in keep_i], SamplingParams(temperature=0, max_tokens=1, logprobs=20))
    with open(a.out, "w", encoding="utf-8") as f:
        for i, o in zip(keep_i, outs):
            lp = o.outputs[0].logprobs[0]
            pc = sum(math.exp(v.logprob) for k, v in lp.items() if k in ok_ids)
            pw = sum(math.exp(v.logprob) for k, v in lp.items() if k in bad_ids)
            f.write(json.dumps({"id": items[i]["id"], "p": pc / max(pc + pw, 1e-9)}) + "\n")
    print(f"judged {len(keep_i)} of {len(items)} -> {a.out}", flush=True)


def keep(a):
    d = ROOT / a.split_dir
    name = d.name
    items = {it["id"]: it for it in map(json.loads, open(d / f"{name}.jsonl", encoding="utf-8"))}
    if a.judge:
        ok = [x["id"] for x in map(json.loads, open(a.judge, encoding="utf-8")) if x["p"] >= a.min_p and x["id"] in items]
    else:
        ok = [x["id"] for x in map(json.loads, open(a.details, encoding="utf-8")) if x["score"] >= 1.0 and x["id"] in items]
    random.Random(0).shuffle(ok)
    dev, train = ok[:a.dev], ok[a.dev:]
    with open(a.out, "w", encoding="utf-8") as f:
        for i in sorted(train):
            f.write(json.dumps(items[i], ensure_ascii=False) + "\n")
    if dev:
        catalog = json.load(open(d / f"{name}_tools_catalog.json", encoding="utf-8"))
        st = {i: {k: v for k, v in json.load(open(d / f"{name}_environment" / i / "state.json", encoding="utf-8")).items()
                  if k != "_meta"} for i in dev}
        write_split(d.parent / f"{name}dev", f"{name}dev", [items[i] for i in sorted(dev)], st, catalog)
    print(f"{len(ok)} of {len(items)} episodes agreed with the solver: {len(train)} -> {a.out}, {len(dev)} -> {name}dev/")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("author")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", type=int, default=0, help="first episode number (ids synv_<n>)")
    p.add_argument("--split-dir", default="synv", help="written under the data root, next to train/")
    p.add_argument("--model", default="Qwen/Qwen3-14B-AWQ")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-util", type=float, default=0.9)
    p.add_argument("--dump", type=int, default=0)
    q = sub.add_parser("keep")
    q.add_argument("--split-dir", default="synv")
    q.add_argument("--details", help="score.py details of a solver run (keep score 1)")
    q.add_argument("--judge", help="judge output (keep p >= --min-p) instead of --details")
    q.add_argument("--min-p", type=float, default=0.8)
    q.add_argument("--dev", type=int, default=300)
    q.add_argument("--out", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--split-dir", default="synv")
    j.add_argument("--model", default="Qwen/Qwen3-14B-AWQ")
    j.add_argument("--max-model-len", type=int, default=8192)
    j.add_argument("--gpu-util", type=float, default=0.92)
    j.add_argument("--shard", type=int, default=0)
    j.add_argument("--nshard", type=int, default=1)
    j.add_argument("--out", required=True)
    a = ap.parse_args()
    {"author": author, "keep": keep, "judge": judge}[a.cmd](a)


if __name__ == "__main__":
    main()
