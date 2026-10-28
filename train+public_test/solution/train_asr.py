"""LoRA fine-tune of a Whisper-family ASR model on train audio (user turns of every sample outside the holdout).

    torchrun --nproc_per_node 2 solution/train_asr.py --out out/asr_lora
    python solution/train_asr.py --merge out/asr_lora --out out/asr_merged     # then: asr.py --model out/asr_merged
Targets are the train audio_text strings as written (case, punctuation, digits), so the output matches that style.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr import fold_ids, jobs_for, read_wav  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = "vinai/PhoWhisper-large"
TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def clips(holdout, seed, skip_fold=None, folds=2):
    """Training clips: every user turn outside the holdout, and outside fold `skip_fold` when given (cross-fold ASR)."""
    items = [json.loads(l) for l in open(ROOT / "train/train.jsonl", encoding="utf-8")]
    held = {d["id"] for d in random.Random(seed).sample(items, holdout)} if holdout else set()
    if skip_fold is not None:
        held |= fold_ids(items, skip_fold, folds, holdout, seed)
    out = []
    for it in items:
        if it["id"] in held:
            continue
        for _, path, ref in jobs_for(it):
            if ref:
                out.append((ROOT / "train" / path, ref))
    return out


def train(a):
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    ddp = "LOCAL_RANK" in os.environ
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if ddp else (0, 1)
    dev = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(dev)
    if ddp:
        dist.init_process_group("nccl")
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)

    proc = WhisperProcessor.from_pretrained(a.model)
    proc.tokenizer.set_prefix_tokens(language="vi", task="transcribe")
    data = clips(a.holdout, a.seed, a.skip_fold, a.folds)
    log(f"{len(data)} training clips (holdout {a.holdout} samples excluded, skip fold {a.skip_fold}), world {world}")

    model = WhisperForConditionalGeneration.from_pretrained(a.model, dtype=torch.float16).cuda(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=TARGETS))
    for p in model.parameters():  # adapters in fp32, base frozen in fp16 (T4 has no bf16)
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler()
    sot = model.config.decoder_start_token_id

    def batch(pairs):
        feats = proc.feature_extractor([read_wav(p) for p, _ in pairs], sampling_rate=16000, return_tensors="pt").input_features
        lab = [proc.tokenizer(t).input_ids for _, t in pairs]
        lab = [l[1:] if l and l[0] == sot else l for l in lab]  # the model prepends decoder_start itself
        n = max(map(len, lab))
        lab = torch.tensor([l + [-100] * (n - len(l)) for l in lab])
        return feats.to(dev, torch.float16), lab.to(dev)

    per_rank = math.ceil(len(data) / world)
    per_step = a.bs * a.accum
    steps = math.ceil(per_rank / per_step) * a.epochs
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    log(f"{steps} optimizer steps, effective batch {per_step * world}")

    step, t0, run_loss, run_n = 0, time.time(), 0.0, 0
    model.train()
    for ep in range(a.epochs):
        order = list(range(len(data)))
        random.Random(a.seed + ep).shuffle(order)
        order = (order + order[:per_rank * world - len(order)])[rank::world]
        chunks = [order[i:i + a.bs] for i in range(0, len(order), a.bs)]
        for ci, ch in enumerate(chunks):
            x, y = batch([data[j] for j in ch])
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_features=x, labels=y).loss
            scaler.scale(loss / a.accum).backward()
            run_loss, run_n = run_loss + loss.item(), run_n + 1
            if (ci + 1) % a.accum == 0 or ci + 1 == len(chunks):
                if ddp:
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
                if step % 10 == 0 or step == steps:
                    el = time.time() - t0
                    log(f"ep {ep + 1} step {step}/{steps} loss {run_loss / run_n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                        f"{el / 60:.1f} min, eta {el / step * (steps - step) / 60:.0f} min")
                    run_loss, run_n = 0.0, 0
        if rank == 0:
            model.save_pretrained(a.out)
            proc.save_pretrained(a.out)
            log(f"saved adapter after epoch {ep + 1} -> {a.out}")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def merge(a):
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    model = WhisperForConditionalGeneration.from_pretrained(a.model, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, a.merge).merge_and_unload()
    model.generation_config.forced_decoder_ids = None  # asr.py passes language/task explicitly
    model.save_pretrained(a.out, safe_serialization=True)
    WhisperProcessor.from_pretrained(a.model).save_pretrained(a.out)
    print(f"merged {a.merge} into {a.model} -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="out/asr_lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8, help="clips per forward pass per rank")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-fold", type=int, help="also leave out this train fold, to transcribe it later (asr.py --fold)")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--merge", help="adapter dir to merge into --model")
    a = ap.parse_args()
    merge(a) if a.merge else train(a)


if __name__ == "__main__":
    main()
