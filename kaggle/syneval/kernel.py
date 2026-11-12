# Kaggle job (2xT4): sftc3 (huangvu1/zac-sftc3) on split synvdev (300 judge-kept LLM-authored episodes on the 11 apps
# that only public_test has; kernel output huangvu1/zac-synj). Merge first, then install vLLM (vLLM needs transformers 5,
# the merge needs <5).
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
sh(f"tar xzf {find('synvdev.tgz')} -C {W}")
shutil.copytree(os.path.dirname(find("synth_llm.py")), f"{W}/solution", dirs_exist_ok=True)
lora = os.path.dirname(glob.glob("/kaggle/input/**/lora/adapter_model.safetensors", recursive=True)[0])
print("adapter:", lora, flush=True)
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
E = f"{OUT}/sftc3_synvdev"
base = f"cd {W} && python -u solution/run.py --split synvdev --compact --model {W}/merged --wave 100 --out {E}"
if sh(f"{base} --tp 2") != 0:
    sh(f"{base} --tp 1 --gpu-util 0.92")
sh(f"cd {W} && python solution/score.py {E}/submission.jsonl --split synvdev --details {E}/details.jsonl | tee {E}/score.txt")
