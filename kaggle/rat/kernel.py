# Kaggle job (2xT4): rationales for short chain-of-thought SFT. Qwen3-14B-AWQ (development-time data generator only;
# never part of the submitted system) writes one or two sentences of reasoning for every gold step of the SFT data
# (train minus the 200 holdout + 1100 synth), one prompt per step showing only what the agent knows before it.
# One vLLM instance per T4 (data parallel, half the samples each); results are appended per chunk.
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
sh("pip install -q vllm 2>&1 | tail -3")
sh('TV=$(python -c "import torch;print(torch.__version__.split(\'+\')[0])"); CU=$(python -c "import torch;print(\'cu\'+torch.version.cuda.replace(\'.\',\'\'))"); '
   'echo "torch $TV $CU"; pip install -q "torchaudio==$TV" --index-url https://download.pytorch.org/whl/$CU 2>&1 | tail -2 '
   '|| pip uninstall -y torchaudio')
if sh('python -c "from vllm import LLM; print(\'vllm ok\')"') != 0:
    sh("pip uninstall -y torchaudio")
    if sh('python -c "from vllm import LLM; print(\'vllm ok without torchaudio\')"') != 0:
        sys.exit("vllm import failed")
sh("python -c \"from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-14B-AWQ')\" 2>&1 | tail -1")

sh(f"cd {W} && python solution/synth.py --n 1100 --out {W}/synth.jsonl")
sh(f"cd {W} && python solution/sft_data.py --compact --text-only --extra {W}/synth.jsonl --out {W}/sft_text.jsonl")
base = (f"cd {W} && python -u solution/rationale.py --data {W}/sft_text.jsonl --model Qwen/Qwen3-14B-AWQ "
        f"--gpu-util 0.92 --max-model-len 6144 --nshard 2")
procs = [subprocess.Popen(f"CUDA_VISIBLE_DEVICES={k} {base} --shard {k} --out {OUT}/rat{k}.jsonl 2>&1 | "
                          f"grep -vE 'INFO|WARNING|it/s]' | sed 's/^/[gpu{k}] /'", shell=True) for k in range(2)]
for p in procs:
    p.wait()
sh(f"cat {OUT}/rat0.jsonl {OUT}/rat1.jsonl > {OUT}/rationale.jsonl; wc -l {OUT}/rationale.jsonl; "
   f"head -c 4000 {OUT}/rationale.jsonl")
