# Kaggle job (2xT4): LLM-authored training episodes for the 11 apps that only public_test's catalog has (sftc3's public
# run calls one of them in 281 of 634 samples; synth.py covers them only with simple templates).
# 1) Qwen3-14B-AWQ (development-time data generator only, never part of the submitted system) writes a spoken request
#    plus gold calls per episode (synth_llm.py author), one vLLM instance per T4; failing calls are dropped.
# 2) The same 14B solves every request independently as an agent (run.py, no thinking), one shard per T4.
# 3) Episodes where solver and author end in the same state with the same calls are kept: 300 become split synvdev
#    (evaluation on unseen apps), the rest synth_llm.jsonl (training).
# 4) Baseline: sftf3 (kernel output minggu151624/zac-sftf3) on synvdev.
import glob
import os
import shutil
import subprocess
import sys

W = "/tmp/zac"
OUT = "/kaggle/working"
N = 1700  # authored per GPU
M = "Qwen/Qwen3-14B-AWQ"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True).returncode


def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not hits:
        sys.exit(f"missing input: {pattern}")
    return hits[0]


def par(cmds):
    """One command per GPU, in parallel."""
    procs = [subprocess.Popen(f"export CUDA_VISIBLE_DEVICES={k}; {c} 2>&1 | grep -vE 'INFO|WARNING|it/s]' | "
                              f"sed 's/^/[gpu{k}] /'", shell=True) for k, c in enumerate(cmds)]
    for p in procs:
        p.wait()


if sh("nvidia-smi -L") != 0:
    sys.exit("no GPU attached (quota spent or accelerator not set)")
os.makedirs(W, exist_ok=True)
sh(f"tar xzf {find('zac_text.tgz')} -C {W} 2>/dev/null")
shutil.copytree(os.path.dirname(find("synth_llm.py")), f"{W}/solution", dirs_exist_ok=True)
sh("pip install -q vllm 2>&1 | tail -3")
if sh('python -c "from vllm import LLM; print(\'vllm ok\')"') != 0:
    sh("pip uninstall -y torchaudio")
    if sh('python -c "from vllm import LLM; print(\'vllm ok without torchaudio\')"') != 0:
        sys.exit("vllm import failed")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
sh(f"python -c \"from huggingface_hub import snapshot_download as s; s('{M}')\" 2>&1 | tail -1")

# 1) author
par([f"cd {W} && python -u solution/synth_llm.py author --n {N} --start {k * N} --seed {k} --split-dir synv{k} "
     f"--model {M} --max-model-len 6144 --gpu-util 0.92" for k in range(2)])
sh(f"mkdir -p {W}/synv/synv_environment && cat {W}/synv0/synv0.jsonl {W}/synv1/synv1.jsonl > {W}/synv/synv.jsonl && "
   f"cp -r {W}/synv0/synv0_environment/* {W}/synv1/synv1_environment/* {W}/synv/synv_environment/ && "
   f"cp {W}/synv0/synv0_tools_catalog.json {W}/synv/synv_tools_catalog.json && cp -r {W}/synv0/zaic_tools {W}/synv/ && "
   f"wc -l {W}/synv/synv.jsonl && cp {W}/synv/synv.jsonl {OUT}/synv_authored.jsonl")

# 2) solve
par([f"cd {W} && python -u solution/run.py --split synv --compact --model {M} --max-model-len 6144 --gpu-util 0.92 "
     f"--wave 400 --shard {k} --nshard 2 --out {W}/solve{k}" for k in range(2)])
S = f"{OUT}/synv_solve"
sh(f"mkdir -p {S} && cat {W}/solve0/submission.jsonl {W}/solve1/submission.jsonl > {S}/submission.jsonl && "
   f"cat {W}/solve0/traces.jsonl {W}/solve1/traces.jsonl > {S}/traces.jsonl")
sh(f"cd {W} && python solution/score.py {S}/submission.jsonl --split synv --details {S}/details.jsonl | tee {S}/score.txt")

# 3) keep
sh(f"cd {W} && python solution/synth_llm.py keep --split-dir synv --details {S}/details.jsonl --dev 300 "
   f"--out {OUT}/synth_llm.jsonl && tar czf {OUT}/synvdev.tgz -C {W} synvdev && wc -l {OUT}/synth_llm.jsonl")
sh(f"python - <<'PY'\nimport json,collections\nc=collections.Counter((d['synth_app'],d['synth_kind']) for d in map(json.loads,open('{OUT}/synth_llm.jsonl')))\n"
   f"print(sorted(collections.Counter(a for a,k in c.elements()).items()));print(sorted(collections.Counter(k for a,k in c.elements()).items()))\nPY")

# 4) baseline sftf3 on synvdev
lora = os.path.dirname(glob.glob("/kaggle/input/**/lora/adapter_model.safetensors", recursive=True)[0])
print("baseline adapter:", lora, flush=True)
sh('pip install -q -U "transformers>=4.56,<5" peft 2>&1 | tail -2')
sh("pip uninstall -y -q torchao")
if sh(f"cd {W} && python solution/train_sft.py --merge {lora} --out {W}/merged") == 0:
    E = f"{OUT}/sftf3_synvdev"
    rc = sh(f"cd {W} && python -u solution/run.py --split synvdev --compact --model {W}/merged --wave 100 --tp 2 --out {E}")
    if rc != 0:
        sh(f"cd {W} && python -u solution/run.py --split synvdev --compact --model {W}/merged --wave 100 --tp 1 --out {E}")
    sh(f"cd {W} && python solution/score.py {E}/submission.jsonl --split synvdev --details {E}/details.jsonl | tee {E}/score.txt")
