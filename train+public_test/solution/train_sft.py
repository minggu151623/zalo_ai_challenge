"""LoRA SFT of the agent LLM on sft_data.py output; loss only on the assistant turns after the final user message.

    torchrun --nproc_per_node 2 solution/train_sft.py --data out/sft_train.jsonl --out out/lora
    python solution/train_sft.py --data out/sft_train.jsonl --check   # tokenise + mask report, no GPU needed
    python solution/train_sft.py --merge out/lora --out out/merged     # fold the adapter into the base weights
    python solution/train_sft.py --data out/sft_train.jsonl --qlora --max-len 8192 --resume   # one 6-8 GB GPU (RTX 30xx+)
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
    """-> (input_ids, labels) or None. Targets: every assistant message after the last user turn. A sample over max_len
    loses its oldest history exchanges (user turn up to the next user turn) until it fits."""
    msgs = sample["messages"]
    while True:
        r = _encode(tok, msgs, sample["tools"], max_len)
        users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        if r != "long" or len(users) < 2:
            return None if r == "long" else r
        msgs = msgs[:users[0]] + msgs[users[1]:]


def _encode(tok, msgs, tools, max_len):
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
    if not spans:
        return None
    if len(ids) > max_len:
        return "long"
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

    bf16 = torch.cuda.get_device_capability(dev)[0] >= 8  # Ampere+; T4 (7.5) has no bf16
    half = torch.bfloat16 if bf16 else torch.float16
    if a.qlora:  # base in 4-bit NF4 so a 4B model fits in ~3 GB; the adapter still merges into the fp16 base
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=half)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=half, attn_implementation="sdpa", quantization_config=q,
                                                     device_map={"": dev})
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=half, attn_implementation="sdpa").cuda(dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=TARGETS,
                                             task_type="CAUSAL_LM"))
    if a.init:  # continue training a finished adapter (same rank) on new data
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        res = set_peft_model_state_dict(model, load_file(os.path.join(a.init, "adapter_model.safetensors")))
        log(f"init from {a.init}, unexpected keys {len(res.unexpected_keys)}")
    for p in model.parameters():  # adapters in fp32, base frozen in fp16/bf16
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler(enabled=not bf16)  # bf16 needs no loss scaling
    base = model.base_model.model  # Qwen3ForCausalLM

    per_rank = math.ceil(len(data) / world)
    steps = math.ceil(per_rank * a.epochs / a.accum)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    log(f"{steps} optimizer steps, accum {a.accum}, effective batch {a.accum * world}")

    step, t0, run_loss, run_n = 0, time.time(), 0.0, 0
    out_of_time = False
    ckpt, start = os.path.join(a.out, "ckpt.pt"), (0, 0)
    if a.resume and os.path.exists(ckpt):  # a long single-GPU run must survive a crash or a reboot
        from peft import set_peft_model_state_dict
        st = torch.load(ckpt, map_location=f"cuda:{dev}", weights_only=False)
        set_peft_model_state_dict(model, st["adapter"])
        for p in params:
            p.data = p.data.float()
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        step, start = st["step"], (st["ep"], st["i"])
        log(f"resumed at step {step} (epoch {start[0] + 1}, sample {start[1]})")

    def save_ckpt(ep, i):
        from peft import get_peft_model_state_dict
        os.makedirs(a.out, exist_ok=True)
        # Adam moments in bf16 (load_state_dict casts them back to fp32): a third smaller to copy off a cloud VM
        st = opt.state_dict()
        st["state"] = {k: {n: v.bfloat16() if torch.is_tensor(v) and v.numel() > 1 else v for n, v in s.items()}
                       for k, s in st["state"].items()}
        torch.save({"adapter": get_peft_model_state_dict(model), "opt": st, "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "ep": ep, "i": i}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)
    model.train()
    for ep in range(start[0], a.epochs):
        order = list(range(len(data)))
        random.Random(a.seed + ep).shuffle(order)
        if ep == 0:  # longest samples first on every rank, so an OOM shows up in the first minute
            longest = sorted(order, key=lambda j: -len(data[j][0]))[:world]
            order = longest + [j for j in order if j not in longest]
        order = (order + order[:per_rank * world - len(order)])[rank::world]
        for i, idx in enumerate(order):
            if ep == start[0] and i < start[1]:
                continue  # done before the resume
            ids, labels = data[idx]
            x = torch.tensor([ids], device=dev)
            y = torch.tensor(labels[1:] + [-100], device=dev)
            pos = (y != -100).nonzero().squeeze(1)
            with torch.autocast("cuda", dtype=half), attn_ctx():
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
                if step % 5 == 0 or step == steps or step <= 3:
                    el = time.time() - t0
                    log(f"ep {ep + 1} step {step}/{steps} loss {run_loss / run_n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                        f"{el / 60:.1f} min, eta {el / step * (steps - step) / 60:.0f} min")
                    run_loss, run_n = 0.0, 0
                if a.resume and rank == 0 and step % a.ckpt_every == 0:
                    save_ckpt(ep, i + 1)
                if a.max_minutes:  # every rank must agree, or the next all-reduce hangs
                    flag = torch.tensor([float(time.time() - t0 > 60 * a.max_minutes)], device=dev)
                    if ddp:
                        dist.all_reduce(flag)
                    out_of_time = flag.item() > 0
                if a.max_steps and step >= a.max_steps:
                    out_of_time = True
                if out_of_time:
                    break
        if rank == 0:
            for d in [a.out] + ([] if out_of_time else [f"{a.out}/epoch{ep + 1}"]):  # epochN kept for scoring each
                model.save_pretrained(d)
                tok.save_pretrained(d)
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
    ap.add_argument("--qlora", action="store_true", help="4-bit base (bitsandbytes), for a single small GPU")
    ap.add_argument("--resume", action="store_true", help="checkpoint to <out>/ckpt.pt and continue from it if present")
    ap.add_argument("--ckpt-every", type=int, default=10, help="optimizer steps between checkpoints (with --resume)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (speed benchmark)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--merge", help="adapter dir to merge into --model")
    ap.add_argument("--init", help="adapter dir to start training from instead of a fresh LoRA")
    a = ap.parse_args()
    if a.merge:
        merge(a)
    elif a.check:
        check(a)
    else:
        train(a)


if __name__ == "__main__":
    main()
