"""LoRA SFT of the agent LLM on sft_data.py output; loss only on the assistant turns after the final user message.

    torchrun --nproc_per_node 2 solution/train_sft.py --data out/sft_train.jsonl --out out/lora
    python solution/train_sft.py --data out/sft_train.jsonl --check   # tokenise + mask report, no GPU needed
    python solution/train_sft.py --merge out/lora --out out/merged     # fold the adapter into the base weights
Prompts are rendered with the same chat template (and tools) vLLM uses in run.py, so train and inference text match.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def render(tok, msgs, tools, gen=False):
    return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=gen, tokenize=False)


def encode(tok, sample, max_len):
    """-> (input_ids, labels) or None. Targets: every assistant message after the last user turn."""
    msgs, tools = sample["messages"], sample["tools"]
    last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    full = render(tok, msgs, tools)
    spans = []
    for k in range(last_user + 1, len(msgs)):
        if msgs[k]["role"] != "assistant":
            continue
        pre, cur = render(tok, msgs[:k], tools, gen=True), render(tok, msgs[:k + 1], tools)
        if not (cur.startswith(pre) and full.startswith(cur)):
            return None  # template is not prefix-stable here; never seen with Qwen3-2507 but keep it safe
        end = cur.rstrip("\n")
        spans.append((len(pre), len(end)))  # assistant body up to and including <|im_end|>
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    if len(ids) > max_len or not spans:
        return None
    labels = [-100] * len(ids)
    for t, (a, b) in enumerate(enc["offset_mapping"]):
        if any(s <= a and b <= e for s, e in spans) and b > a:
            labels[t] = ids[t]
    return ids, labels


def load(tok, path, max_len):
    data, skipped = [], 0
    for line in open(path, encoding="utf-8"):
        r = encode(tok, json.loads(line), max_len)
        if r is None:
            skipped += 1
        else:
            data.append(r)
    return data, skipped


def check(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    data, skipped = load(tok, a.data, a.max_len)
    lens = sorted(len(x[0]) for x in data)
    tgt = sum(sum(l != -100 for l in y) for _, y in data)
    print(f"{len(data)} samples, {skipped} skipped; tokens total {sum(lens)} max {lens[-1]} p50 {lens[len(lens) // 2]}; "
          f"target tokens {tgt}")
    ids, labels = data[0]
    print("--- target text of sample 0 ---")
    print(tok.decode([i for i, l in zip(ids, labels) if l != -100]))


def train(a):
    import torch
    import torch.distributed as dist
    import transformers.integrations.sdpa_attention as sdpa_mod
    from peft import LoraConfig, get_peft_model
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # T4 has no flash kernel and the memory-efficient one rejects enable_gqa, so SDPA would fall back to the math
    # kernel and materialise a seq x seq matrix per head (8 GB at 8K tokens). Repeat KV heads instead and pin the kernel.
    sdpa_mod.use_gqa_in_sdpa = lambda *_a, **_k: False
    attn_ctx = lambda: sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION])  # noqa: E731

    ddp = "LOCAL_RANK" in os.environ
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if ddp else (0, 1)
    dev = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(dev)
    if ddp:
        dist.init_process_group("nccl")
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)

    tok = AutoTokenizer.from_pretrained(a.model)
    data, skipped = load(tok, a.data, a.max_len)
    log(f"{len(data)} samples ({skipped} skipped), world {world}")

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, attn_implementation="sdpa").cuda(dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=TARGETS,
                                             task_type="CAUSAL_LM"))
    for p in model.parameters():  # adapters in fp32, base frozen in fp16 (T4 has no bf16)
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler()
    base = model.base_model.model  # Qwen3ForCausalLM

    per_rank = math.ceil(len(data) / world)
    steps = math.ceil(per_rank * a.epochs / a.accum)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    log(f"{steps} optimizer steps, accum {a.accum}, effective batch {a.accum * world}")

    step, t0, run_loss, run_n = 0, time.time(), 0.0, 0
    out_of_time = False
    model.train()
    for ep in range(a.epochs):
        order = list(range(len(data)))
        random.Random(a.seed + ep).shuffle(order)
        if ep == 0:  # longest samples first on every rank, so an OOM shows up in the first minute
            longest = sorted(order, key=lambda j: -len(data[j][0]))[:world]
            order = longest + [j for j in order if j not in longest]
        order = (order + order[:per_rank * world - len(order)])[rank::world]
        for i, idx in enumerate(order):
            ids, labels = data[idx]
            x = torch.tensor([ids], device=dev)
            y = torch.tensor(labels[1:] + [-100], device=dev)
            pos = (y != -100).nonzero().squeeze(1)
            with torch.autocast("cuda", dtype=torch.float16), attn_ctx():
                h = base.model(input_ids=x).last_hidden_state[0, pos]  # logits only where there is a target
                logits = base.lm_head(h).float()
            loss = torch.nn.functional.cross_entropy(logits, y[pos])
            with attn_ctx():  # gradient checkpointing re-runs attention inside backward
                scaler.scale(loss / a.accum).backward()
            if step == 0 and i < 2:
                print(f"rank {rank} len {len(ids)}: peak mem {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB", flush=True)
            run_loss, run_n = run_loss + loss.item(), run_n + 1
            if (i + 1) % a.accum == 0 or i + 1 == len(order):
                if ddp:  # manual all-reduce of the adapter grads (cheaper than wrapping a gradient-checkpointed PEFT model)
                    for p in params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad)
                            p.grad /= world
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 5 == 0 or step == steps:
                    el = time.time() - t0
                    log(f"ep {ep + 1} step {step}/{steps} loss {run_loss / run_n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                        f"{el / 60:.1f} min, eta {el / step * (steps - step) / 60:.0f} min")
                    run_loss, run_n = 0.0, 0
                if a.max_minutes:  # every rank must agree, or the next all-reduce hangs
                    flag = torch.tensor([float(time.time() - t0 > 60 * a.max_minutes)], device=dev)
                    if ddp:
                        dist.all_reduce(flag)
                    out_of_time = flag.item() > 0
                if out_of_time:
                    break
        if rank == 0:
            model.save_pretrained(a.out)
            tok.save_pretrained(a.out)
            log(f"saved adapter after epoch {ep + 1}{' (time budget hit)' if out_of_time else ''} -> {a.out}")
        if out_of_time:
            break
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def merge(a):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, a.merge).merge_and_unload()
    model.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
    print(f"merged {a.merge} into {a.model} -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data")
    ap.add_argument("--out", default="out/lora")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8, help="samples per rank per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop and save once training has run this long")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--merge", help="adapter dir to merge into --model")
    a = ap.parse_args()
    if a.merge:
        merge(a)
    elif a.check:
        check(a)
    else:
        train(a)


if __name__ == "__main__":
    main()
