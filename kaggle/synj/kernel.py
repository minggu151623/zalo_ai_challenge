# Kaggle job (2xT4): filter the 2578 LLM-authored unseen-app episodes (kernel minggu151624/zac-synv; states rebuilt
# from the same seeds, shipped as synv.tgz) with Qwen3-14B-AWQ as a one-token judge (development-time only), keep
# p(CORRECT) >= 0.8: 300 of them become split synvdev, the rest synth_llm.jsonl. Then sftc3 (huangvu1/zac-sftc3) on
# synvdev, i.e. how well the current best model handles apps it never saw in train.
import glob
import os
import shutil
import subprocess
import sys

W = "/tmp/zac"
OUT = "/kaggle/working"
M = "Qwen/Qwen3-14B-AWQ"


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
sh(f"tar xzf {find('synv.tgz')} -C {W}")
shutil.copytree(os.path.dirname(find("synth_llm.py")), f"{W}/solution", dirs_exist_ok=True)
sh("pip install -q vllm 2>&1 | tail -3")
if sh('python -c "from vllm import LLM; print(\'vllm ok\')"') != 0:
    sh("pip uninstall -y torchaudio")
    if sh('python -c "from vllm import LLM; print(\'vllm ok without torchaudio\')"') != 0:
        sys.exit("vllm import failed")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
sh(f"python -c \"from huggingface_hub import snapshot_download as s; s('{M}')\" 2>&1 | tail -1")

procs = [subprocess.Popen(f"export CUDA_VISIBLE_DEVICES={k}; cd {W} && python -u solution/synth_llm.py judge --split-dir synv "
                          f"--model {M} --max-model-len 6144 --shard {k} --nshard 2 --out {OUT}/judge{k}.jsonl 2>&1 | "
                          f"grep -vE 'INFO|WARNING|it/s]' | sed 's/^/[gpu{k}] /'", shell=True) for k in range(2)]
for p in procs:
    p.wait()
sh(f"cat {OUT}/judge0.jsonl {OUT}/judge1.jsonl > {OUT}/judge.jsonl; wc -l {OUT}/judge.jsonl")
sh(f"python -c \"import json;p=[json.loads(l)['p'] for l in open('{OUT}/judge.jsonl')];"
   f"print({{t:sum(x>=t for x in p) for t in (0.5,0.8,0.9,0.95)}})\"")
sh(f"cd {W} && python solution/synth_llm.py keep --split-dir synv --judge {OUT}/judge.jsonl --min-p 0.8 --dev 300 "
   f"--out {OUT}/synth_llm.jsonl && tar czf {OUT}/synvdev.tgz -C {W} synvdev")

lora = os.path.dirname(glob.glob("/kaggle/input/**/lora/adapter_model.safetensors", recursive=True)[0])
print("adapter:", lora, flush=True)
sh('pip install -q -U "transformers>=4.56,<5" peft 2>&1 | tail -2')
sh("pip uninstall -y -q torchao")
if sh(f"cd {W} && python solution/train_sft.py --merge {lora} --out {W}/merged") != 0:
    sys.exit("merge failed")
E = f"{OUT}/sftc3_synvdev"
base = f"cd {W} && python -u solution/run.py --split synvdev --compact --model {W}/merged --wave 100 --out {E}"
if sh(f"{base} --tp 2") != 0:
    sh(f"{base} --tp 1 --gpu-util 0.92")
sh(f"cd {W} && python solution/score.py {E}/submission.jsonl --split synvdev --details {E}/details.jsonl | tee {E}/score.txt")
