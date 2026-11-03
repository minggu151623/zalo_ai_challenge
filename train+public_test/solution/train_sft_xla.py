"""LoRA SFT on a TPU VM (Kaggle TPU v5e-8) with PyTorch/XLA: same data, targets and adapter format as train_sft.py.

    PJRT_DEVICE=TPU python solution/train_sft_xla.py --data out/sft_train.jsonl --out out/lora
    PJRT_DEVICE=TPU python solution/train_sft_xla.py --data out/sft_train.jsonl --out /tmp/lora --limit 64 --max-steps 4
One process per TPU chip (data parallel, LoRA grads all-reduced). XLA compiles a graph per input shape, so a sample is
padded to a length bucket and its target positions to a target bucket: a handful of compiles instead of one per sample.
Adapters go to <out>/epoch<N> and load like train_sft.py's (train_sft.py --merge <out>/epochN on any machine).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sft import MODEL, TARGETS, load  # noqa: E402

LEN_BUCKETS = [2048, 4096, 6144, 8192, 12288]
TGT_BUCKETS = [256, 512, 1024, 2048]


def bucket(n, sizes):
    return next((s for s in sizes if n <= s), None)


def xla_flash(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **_):
    """transformers attention hook -> Pallas flash attention. Causal only: padding sits at the end, so real tokens
    never attend to it, and the padded positions carry no target."""
    import torch
    from torch_xla.experimental.custom_kernel import flash_attention
    n = query.shape[1] // key.shape[1]  # GQA: repeat KV heads, same layout as transformers' repeat_kv
    q, k, v = (t.to(torch.bfloat16) for t in (query, key.repeat_interleave(n, 1), value.repeat_interleave(n, 1)))
    out = flash_attention(q, k, v, causal=True, sm_scale=scaling)
    return out.transpose(1, 2).contiguous(), None


def worker(_index, a):
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    if a.xla_cache:  # compiled graphs on disk: a resumed run skips the ~70 s compile per shape
        xr.initialize_cache(a.xla_cache, readonly=False)
    rank, world = xr.global_ordinal(), xr.world_size()
    dev = torch_xla.device()
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)

    tok = AutoTokenizer.from_pretrained(a.model)
    data, skipped = load(tok, a.data, a.max_len)
    fits = [d for d in data if bucket(sum(l != -100 for l in d[1]), TGT_BUCKETS)]
    if a.limit:  # smoke test: the longest samples (worst-case memory) plus a random rest
        by_len = sorted(range(len(fits)), key=lambda j: -len(fits[j][0]))
        rest = random.Random(a.seed).sample(by_len[world:], min(a.limit - world, len(by_len) - world))
        fits = [fits[j] for j in by_len[:world] + rest]
    log(f"{len(fits)} samples ({skipped} skipped, {len(data) - len(fits)} over the target bucket), world {world}")
    data = fits

    AttentionInterface.register("xla_flash", xla_flash)
    # every rank loads at once (8 bf16 copies are ~64 GB; the v5e-8 host has 377 GB). A per-rank load with a
    # rendezvous between ranks hung on Kaggle with no error, so each rank reports its own progress here.
    print(f"rank {rank}: loading {a.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="xla_flash")
    model = model.to(dev)
    print(f"rank {rank}: model on {dev}", flush=True)
    model.gradient_checkpointing_enable()
    # torch's checkpoint gets undone by XLA's CSE (the recompute is merged with the forward); torch_xla's adds a barrier
    for m in model.modules():
        if hasattr(m, "_gradient_checkpointing_func"):
            m._gradient_checkpointing_func = xla_checkpoint
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=a.dropout, target_modules=TARGETS,
                                             task_type="CAUSAL_LM"))
    for p in model.parameters():  # adapters in fp32 (peft casts the input to their dtype), base frozen in bf16
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    base = model.base_model.model

    accum = max(1, a.batch // world)
    per_rank = math.ceil(len(data) / world)
    steps = math.ceil(per_rank * a.epochs / accum)
    if a.max_steps:
        steps = min(steps, a.max_steps)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    log(f"{steps} optimizer steps, accum {accum}, effective batch {accum * world}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def micro(ids, labels):
        y = labels[1:] + [-100]
        pos = [t for t, v in enumerate(y) if v != -100]
        L, P = bucket(len(ids), LEN_BUCKETS), bucket(len(pos), TGT_BUCKETS)
        x = torch.tensor([ids + [pad] * (L - len(ids))]).to(dev)
        idx = torch.tensor(pos + [0] * (P - len(pos))).to(dev)
        tgt = torch.tensor([y[t] for t in pos] + [0] * (P - len(pos))).to(dev)
        w = torch.tensor([1.0 / len(pos)] * len(pos) + [0.0] * (P - len(pos))).to(dev)  # mean over real targets
        h = base.model(input_ids=x).last_hidden_state[0].index_select(0, idx)
        ce = torch.nn.functional.cross_entropy(base.lm_head(h).float(), tgt, reduction="none")
        loss = (ce * w).sum()
        (loss / accum).backward()
        xm.mark_step()
        return loss.detach(), (L, P)

    step, t0, run_loss, run_n = 0, time.time(), None, 0
    shapes, done = set(), False
    ckpt, start = os.path.join(a.out, "ckpt.pt"), (0, 0)
    if a.resume and os.path.exists(ckpt):  # a Colab VM can be reclaimed mid-run
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, {k: v.to(dev) for k, v in st["adapter"].items()})
        for p in params:
            p.data = p.data.float()
        opt.load_state_dict(st["opt"])  # moves the state to the params' device
        sched.load_state_dict(st["sched"])
        step, start = st["step"], (st["ep"], st["i"])
        log(f"resumed at step {step} (epoch {start[0] + 1}, sample {start[1]})")
    step0 = step

    def cpu(x, dtype=None):
        if isinstance(x, torch.Tensor):
            return x.cpu() if dtype is None or x.numel() == 1 else x.to(dtype).cpu()
        if isinstance(x, dict):
            return {k: cpu(v, dtype) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(cpu(v, dtype) for v in x)
        return x

    def save_ckpt(ep, i):
        # Adam moments in bf16 (load_state_dict casts them back to fp32): a third smaller to copy off a Colab VM
        os.makedirs(a.out, exist_ok=True)
        torch.save({"adapter": cpu(get_peft_model_state_dict(model)), "opt": cpu(opt.state_dict(), torch.bfloat16),
                    "sched": sched.state_dict(), "step": step, "ep": ep, "i": i}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    model.train()
    for ep in range(start[0], a.epochs):
        order = list(range(len(data)))
        random.Random(a.seed + ep).shuffle(order)
        if ep == 0:  # longest samples first on every rank, so an out-of-memory shows up at once
            longest = sorted(order, key=lambda j: -len(data[j][0]))[:world]
            order = longest + [j for j in order if j not in longest]
        order = (order + order[:per_rank * world - len(order)])[rank::world]
        for i, j in enumerate(order):
            if ep == start[0] and i < start[1]:
                continue  # done before the resume
            ts = time.time()
            loss, shape = micro(*data[j])
            run_loss, run_n = loss if run_loss is None else run_loss + loss, run_n + 1
            if shape not in shapes:  # first time this shape is seen: the step time is mostly compilation
                shapes.add(shape)
                xm.wait_device_ops()
                log(f"new shape {shape}: {time.time() - ts:.0f}s; memory {xm.get_memory_info(dev)}")
            if (i + 1) % accum and i + 1 != len(order):
                continue
            xm.all_reduce(xm.REDUCE_SUM, [p.grad for p in params if p.grad is not None], scale=1.0 / world)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            xm.mark_step()
            step += 1
            if step % 5 == 0 or step == steps or step <= 3:
                el = time.time() - t0
                log(f"ep {ep + 1} step {step}/{steps} loss {run_loss.item() / run_n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{el / 60:.1f} min, eta {el / max(1, step - step0) * (steps - step) / 60:.0f} min")
                run_loss, run_n = None, 0
            if a.resume and rank == 0 and step % a.ckpt_every == 0:
                save_ckpt(ep, i + 1)
            over = a.max_minutes and time.time() - t0 > 60 * a.max_minutes
            if xm.mesh_reduce("stop", bool(over) or step >= steps, max):  # every rank must leave together
                done = True
                break
        if rank == 0:  # one adapter per epoch, so a GPU job can score each of them
            sd = {k: v.cpu() for k, v in get_peft_model_state_dict(model).items()}
            path = f"{a.out}/epoch{ep + 1}"
            model.save_pretrained(path, state_dict=sd)
            tok.save_pretrained(path)
            log(f"saved adapter after epoch {ep + 1} (step {step}) -> {path}")
        xm.rendezvous(f"saved{ep}")
        if done:
            break
    log(f"done: {step} steps in {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="out/lora")
    ap.add_argument("--max-len", type=int, default=LEN_BUCKETS[-1])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16, help="samples per optimizer step over all chips")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout; 0 on a 16 GB chip: XLA keeps every "
                                                               "layer's dropout mask for the backward (~7 GB at 8192)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop and save once training has run this long")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="train on this many samples only (smoke test)")
    ap.add_argument("--resume", action="store_true", help="checkpoint to <out>/ckpt.pt and continue from it if present")
    ap.add_argument("--xla-cache", help="persistent compilation cache directory")
    ap.add_argument("--ckpt-every", type=int, default=20, help="optimizer steps between checkpoints (with --resume)")
    ap.add_argument("--single", action="store_true", help="one chip, no spawn (Colab v5e-1: its preset 2x2 host "
                                                          "bounds make xmp.spawn expect 4 workers)")
    a = ap.parse_args()
    if a.single:
        worker(0, a)
        return
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(worker, args=(a,))


if __name__ == "__main__":
    main()
