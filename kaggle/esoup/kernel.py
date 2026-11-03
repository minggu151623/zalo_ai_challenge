# Kaggle job (2xT4): esoup = exact LoRA delta average of two consecutive epochs of the sftc chain, sftc3 (holdout 64.21,
# public 50.88) and sftc4 (holdout 61.89), taken from their kernel outputs: ranks are concatenated so the merged delta is
# 0.5*B3@A3 + 0.5*B4@A4 at the same alpha/r. --compact holdout first; public_test only if it beats sftc3 by a margin.
import glob
import os
import shutil
import subprocess
import sys

W = "/tmp/zac"
OUT = "/kaggle/working"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True).returncode


def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not hits:
        sys.exit(f"missing input: {pattern}")
    return hits[0]


if sh("nvidia-smi -L") != 0:
    sys.exit("no GPU attached (quota spent or accelerator not set)")
os.makedirs(W, exist_ok=True)
sh(f"tar xzf {find('zac_text.tgz')} -C {W} 2>/dev/null")
shutil.copytree(os.path.dirname(find("run.py")), f"{W}/solution", dirs_exist_ok=True)
asr_hold, asr_pub = find("asr_hold_ft.jsonl"), find("asr_public_ft.jsonl")
sh('pip install -q -U "transformers>=4.56,<5" peft 2>&1 | tail -2')
sh("pip uninstall -y -q torchao")


def soup(srcs, out):
    import json
    import torch
    from safetensors.torch import load_file, save_file
    ws = [1 / len(srcs)] * len(srcs)
    ts = [load_file(f"{s}/adapter_model.safetensors") for s in srcs]
    cfgs = [json.load(open(f"{s}/adapter_config.json")) for s in srcs]
    assert all(t.keys() == ts[0].keys() for t in ts) and len({(c["r"], c["lora_alpha"]) for c in cfgs}) == 1
    z = {}
    for k in ts[0]:
        if "lora_A" in k:
            z[k] = torch.cat([t[k] for t in ts], 0)
        elif "lora_B" in k:
            z[k] = torch.cat([w * t[k].float() for w, t in zip(ws, ts)], 1).to(ts[0][k].dtype)
        else:
            sys.exit(f"unexpected key {k}")
    shutil.copytree(srcs[-1], out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("ckpt.pt", "epoch*"))
    cfg = dict(cfgs[0], r=cfgs[0]["r"] * len(srcs), lora_alpha=cfgs[0]["lora_alpha"] * len(srcs))
    json.dump(cfg, open(f"{out}/adapter_config.json", "w"), indent=2)
    save_file(z, f"{out}/adapter_model.safetensors")
    print("soup of", srcs, "-> rank", cfg["r"], flush=True)


srcs = [os.path.dirname(find(f"{k}/lora/adapter_model.safetensors")) for k in ("zac-sftc3", "zac-sftc4")]
lora = f"{W}/soup"
soup(srcs, lora)
if sh(f"cd {W} && python solution/train_sft.py --merge {lora} --out {W}/merged") != 0:
    sys.exit("merge failed")

sh("pip install -q vllm 2>&1 | tail -3")
sh('TV=$(python -c "import torch;print(torch.__version__.split(\'+\')[0])"); CU=$(python -c "import torch;print(\'cu\'+torch.version.cuda.replace(\'.\',\'\'))"); '
   'echo "torch $TV $CU"; pip install -q "torchaudio==$TV" --index-url https://download.pytorch.org/whl/$CU 2>&1 | tail -2 '
   '|| pip uninstall -y torchaudio')
if sh('python -c "from vllm import LLM; print(\'vllm ok\')"') != 0:
    sh("pip uninstall -y torchaudio")
    if sh('python -c "from vllm import LLM; print(\'vllm ok without torchaudio\')"') != 0:
        sys.exit("vllm import failed")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def run(args, out):
    base = f"cd {W} && python -u solution/run.py --compact --model {W}/merged --wave 50 {args} --out {out}"
    rc = sh(f"{base} --tp 2")
    return rc if rc == 0 else sh(f"{base} --tp 1 --gpu-util 0.92")  # resumes finished samples


E = f"{OUT}/sft200_asr"
run(f"--split train --limit 200 --seed 0 --text {asr_hold}", E)
sh(f"cd {W} && python solution/score.py {E}/submission.jsonl --details {E}/details.jsonl | tee {E}/score.txt")
hold = float(open(f"{E}/score.txt").read().split("score")[1].split()[0])
if hold < 63.5:  # sftc3 64.21; holdout noise is about 1 pt
    sys.exit(0)
P = f"{OUT}/public"
rc = run(f"--split public_test --text {asr_pub}", P)
sh(f"wc -l {P}/submission.jsonl; cd {W}/public_test && python -m zaic_tools --split public_test --root .. --validate {P}/submission.jsonl | tail -5")
sys.exit(rc)
