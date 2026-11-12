# Kaggle job (2xT4): cothide = the cot1b model (short reasoning before every tool call) evaluated with
# run.py --hide-reasoning, i.e. earlier steps' reasoning is blanked in the prompt. cot1b lost to sftc3 on the holdout
# (gold 66.38 vs 71.59, ASR 56.26 vs 64.21) mainly because greedy decoding copied the previous step's reasoning
# verbatim (47% of consecutive steps) and then acted on the stale text. Gold text first, then ASR text.
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
shutil.copytree(os.path.dirname(find("rationale.py")), f"{W}/solution", dirs_exist_ok=True)
sh(f"grep -c hide_reasoning {W}/solution/run.py")
asr_hold = find("asr_hold_ft.jsonl")
lora = os.path.dirname(glob.glob("/kaggle/input/**/lora/adapter_model.safetensors", recursive=True)[0])
print("cot adapter:", lora, flush=True)

sh('pip install -q -U "transformers>=4.56,<5" peft 2>&1 | tail -2')
sh("pip uninstall -y -q torchao")
if sh(f"cd {W} && python solution/train_sft.py --merge {lora} --out {W}/merged") != 0:
    sys.exit("merge failed")
sh("pip install -q vllm 2>&1 | tail -3")
if sh('python -c "from vllm import LLM; print(\'vllm ok\')"') != 0:
    sh("pip uninstall -y torchaudio")
    if sh('python -c "from vllm import LLM; print(\'vllm ok without torchaudio\')"') != 0:
        sys.exit("vllm import failed")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def run(args, out):
    base = f"cd {W} && python -u solution/run.py --compact --hide-reasoning --model {W}/merged --wave 50 {args} --out {out}"
    rc = sh(f"{base} --tp 2")
    return rc if rc == 0 else sh(f"{base} --tp 1 --gpu-util 0.92")


for name, text in (("hide200_gold", "gold"), ("hide200_asr", asr_hold)):
    E = f"{OUT}/{name}"
    run(f"--split train --limit 200 --seed 0 --text {text}", E)
    sh(f"cd {W} && python solution/score.py {E}/submission.jsonl --details {E}/details.jsonl | tee {E}/score.txt")
