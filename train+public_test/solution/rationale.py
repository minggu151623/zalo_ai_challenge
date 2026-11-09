"""Annotate SFT episodes with a short reasoning line before every assistant step, derived from the gold trajectory.

    python solution/rationale.py --data out/sft_text.jsonl --dump 2            # print annotation prompts, no GPU
    python solution/rationale.py --data out/sft_text.jsonl --out out/rationale.jsonl --model Qwen/Qwen3-14B-AWQ --tp 2
Output line: {"id", "steps": [str, ...]}: one rationale per assistant message after the final user turn, each written
from a prompt that shows only what the agent knows before that step (so it cannot leak later results). sft_data.py
--rationale puts each one in front of that message's content, so the agent learns to reason briefly before acting.
"""
from __future__ import annotations

import argparse
import json
import os
import re

INSTRUCTIONS = """Write the private reasoning the agent has right BEFORE taking the next action shown above, i.e. why that action is the right next step given only the information above (instructions, context, history, request, and results of the steps already taken).
- 1 or 2 short sentences, at most 35 words, in English; keep Vietnamese names, quoted words and values verbatim.
- After the first step, do not restate the whole request: focus on what this step needs and why.
- Be concrete and decisive: what the user wants; where each argument value comes from (quote the user's words or an earlier result; spell out date/time arithmetic, e.g. "thứ Sáu tuần sau" = Friday 2026-11-20, "8 giờ kém 15 tối" = 19:45); for a condition, check it against the data already shown and state the outcome; before sys.ask_missing_info, name the required parameter that is missing or ambiguous and why; when stopping without a tool call, say why nothing (more) should be done.
- You do not know the result of the next action yet. Do not write the tool call itself.
Answer with only the reasoning text."""


def _call_str(tc):
    f = tc["function"]
    return json.dumps({"name": f["name"], "arguments": f["arguments"]}, ensure_ascii=False)


def steps_of(msgs):
    last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    return last_user, [i for i in range(last_user + 1, len(msgs)) if msgs[i]["role"] == "assistant"]


def _trim_tools(sys_text, msgs):
    """Keep only the tool lines of apps this episode calls (plus sys.*): prompts are ~2x shorter. Episodes without any
    call keep every tool, since their reasoning is about what the installed apps cannot do."""
    apps = {tc["function"]["name"].split(".")[0] for m in msgs for tc in m.get("tool_calls") or []}
    if not apps or "\n# Tools\n" not in sys_text:
        return sys_text
    head, tools = sys_text.split("\n# Tools\n", 1)
    lines = tools.split("\n")
    keep = [l for l in lines if not re.match(r"^[a-z_]+\.[a-z_]+\(", l) or l.split(".")[0] in apps | {"sys"}]
    return head + "\n# Tools (only the apps relevant here)\n" + "\n".join(keep)


def build_prompts(sample):
    """One prompt per target assistant message; each shows only what the agent knows before that step."""
    msgs = sample["messages"]
    last_user, idx = steps_of(msgs)
    out = ["# Agent instructions and context", _trim_tools(msgs[0]["content"], msgs)]
    if sample.get("tools"):  # non-compact samples hand JSON schemas to the chat template instead
        out += ["# Tools"] + [json.dumps(t["function"], ensure_ascii=False) for t in sample["tools"]]
    hist = msgs[1:last_user]
    if hist:
        out.append("\n# Earlier conversation")
        for m in hist:
            if m["role"] == "user":
                out.append(f"User: {m['content']}")
            elif m["role"] == "tool":
                out.append(f"  result: {m['content']}")
            elif m.get("tool_calls"):
                out += [f"Assistant calls: {_call_str(tc)}" for tc in m["tool_calls"]]
            elif m.get("content"):
                out.append(f"Assistant: {m['content']}")
    out += ["\n# Current request", f"User: {msgs[last_user]['content']}", "\n# Steps already taken for this request"]
    head, done, prompts, n = "\n".join(out), [], [], 0
    for i in idx:
        m = msgs[i]
        if m.get("tool_calls"):
            nxt = "call " + " + ".join(_call_str(tc) for tc in m["tool_calls"])
        elif done:
            nxt = "stop calling tools and reply briefly to the user"
        else:
            nxt = "do not call any tool; just reply briefly"
        prompts.append("\n".join([head] + (done or ["(none)"]) + ["\n# Next action", nxt, "\n# Task", INSTRUCTIONS]))
        if m.get("tool_calls"):
            n += 1
            done.append(f"{n}. {nxt}")
            j = i + 1
            while j < len(msgs) and msgs[j]["role"] == "tool":
                done.append(f"   result: {msgs[j]['content'][:1000]}")
                j += 1
    return prompts


def clean(text):
    text = " ".join(text.split("</think>")[-1].split()).strip().strip('"')
    if not text or "<tool_call>" in text or '"arguments"' in text:
        return None
    words = text.split()
    return " ".join(words[:60])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="sft_data.py output (text transcripts; one annotation per id)")
    ap.add_argument("--out")
    ap.add_argument("--model", default="Qwen/Qwen3-14B-AWQ")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-util", type=float, default=0.9)
    ap.add_argument("--dump", type=int, default=0, help="print this many prompts and exit")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0, help="this process takes samples[shard::nshard]")
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=100, help="samples per generate call; output is appended per chunk")
    a = ap.parse_args()

    samples, seen = [], set()
    for line in open(a.data, encoding="utf-8"):
        s = json.loads(line)
        if s["id"] not in seen:  # ASR copies share the id and the trajectory
            seen.add(s["id"])
            samples.append(s)
    if a.limit:
        samples = samples[:a.limit]
    samples = samples[a.shard::a.nshard]
    done_ids = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")} if a.out and os.path.exists(a.out) else set()
    samples = [x for x in samples if x["id"] not in done_ids]
    per = [build_prompts(x) for x in samples]
    print(f"{len(samples)} samples to do ({len(done_ids)} already in {a.out}), {sum(map(len, per))} step prompts",
          flush=True)
    if a.dump:
        for p in [p for ps in per for p in ps][:a.dump]:
            print(p, "\n" + "=" * 80)
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util,
              enable_prefix_caching=True, seed=0, dtype="half")
    kw = {"enable_thinking": False} if "Qwen3-" in a.model and "2507" not in a.model else {}

    def chat(p):
        t = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, **kw)
        return t if len(tok(t, add_special_tokens=False)["input_ids"]) < a.max_model_len - 200 else None

    total = 0
    for c0 in range(0, len(samples), a.chunk):  # results are appended per chunk, so a killed run keeps its progress
        flat = [(si, k, chat(p)) for si in range(c0, min(c0 + a.chunk, len(samples))) for k, p in enumerate(per[si])]
        result = [None] * len(flat)
        for attempt, temp in enumerate((0.0, 0.7, 0.7)):
            todo = [i for i in range(len(flat)) if result[i] is None and flat[i][2] is not None]
            if not todo:
                break
            sp = SamplingParams(temperature=temp, top_p=0.95 if temp else 1.0, seed=attempt, max_tokens=100)
            outs = llm.generate([flat[i][2] for i in todo], sp, use_tqdm=False)
            for i, o in zip(todo, outs):
                result[i] = clean(o.outputs[0].text)
        by = {}
        for (si, k, _), r in zip(flat, result):
            by.setdefault(si, [None] * len(per[si]))[k] = r
        with open(a.out, "a", encoding="utf-8") as f:
            for si, steps in sorted(by.items()):
                if all(steps):
                    f.write(json.dumps({"id": samples[si]["id"], "steps": steps}, ensure_ascii=False) + "\n")
                    total += 1
        print(f"samples {min(c0 + a.chunk, len(samples))}/{len(samples)} done, {total} fully annotated", flush=True)

if __name__ == "__main__":
    main()
