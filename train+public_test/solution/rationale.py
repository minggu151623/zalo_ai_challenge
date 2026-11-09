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


def build_prompts(sample):
    """One prompt per target assistant message; each shows only what the agent knows before that step."""
    msgs = sample["messages"]
    last_user, idx = steps_of(msgs)
    out = ["# Agent instructions and context", msgs[0]["content"]]
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
                done.append(f"   result: {msgs[j]['content'][:1500]}")
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
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-util", type=float, default=0.9)
    ap.add_argument("--dump", type=int, default=0, help="print this many prompts and exit")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    samples, seen = [], set()
    for line in open(a.data, encoding="utf-8"):
        s = json.loads(line)
        if s["id"] not in seen:  # ASR copies share the id and the trajectory
            seen.add(s["id"])
            samples.append(s)
    if a.limit:
        samples = samples[:a.limit]
    per = [build_prompts(s) for s in samples]
    flat = [(si, k, p) for si, ps in enumerate(per) for k, p in enumerate(ps)]
    print(f"{len(samples)} samples, {len(flat)} step prompts", flush=True)
    if a.dump:
        for _, _, p in flat[:a.dump]:
            print(p, "\n" + "=" * 80)
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util,
              enable_prefix_caching=True, seed=0, dtype="half")
    texts = []
    for _, _, p in flat:
        kw = {"enable_thinking": False} if "Qwen3-" in a.model and "2507" not in a.model else {}
        t = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, **kw)
        ids = tok(t, add_special_tokens=False)["input_ids"]
        texts.append(t if len(ids) < a.max_model_len - 1200 else None)
    result = [None] * len(flat)
    for attempt, temp in enumerate((0.0, 0.7, 0.7)):
        todo = [i for i in range(len(flat)) if result[i] is None and texts[i] is not None]
        if not todo:
            break
        sp = SamplingParams(temperature=temp, top_p=0.95 if temp else 1.0, seed=attempt, max_tokens=120)
        outs = llm.generate([texts[i] for i in todo], sp)
        for i, o in zip(todo, outs):
            result[i] = clean(o.outputs[0].text)
        print(f"attempt {attempt}: {sum(r is not None for r in result)}/{len(flat)} steps annotated", flush=True)
    by = {}
    for (si, k, _), r in zip(flat, result):
        by.setdefault(si, [None] * len(per[si]))[k] = r
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for si, steps in sorted(by.items()):
            if all(steps):
                f.write(json.dumps({"id": samples[si]["id"], "steps": steps}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n}/{len(samples)} samples fully annotated -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
