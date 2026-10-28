# Kaggle job: baseline LifeMate agent on 200 train samples (2xT4, vLLM tensor parallel).
import glob
import os
import shutil
import subprocess
import sys

W = "/tmp/zac"          # data + code (kept out of /kaggle/working so outputs stay small)
OUT = "/kaggle/working/base200"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True).returncode


os.makedirs(W, exist_ok=True)
tgz = glob.glob("/kaggle/input/**/zac_text.tgz", recursive=True)
if tgz:
    sh(f"tar xzf {tgz[0]} -C {W}")
else:  # Kaggle may have unpacked the archive on upload
    jl = glob.glob("/kaggle/input/**/train/train.jsonl", recursive=True)
    shutil.copytree(os.path.dirname(os.path.dirname(jl[0])), W, dirs_exist_ok=True)
code = os.path.dirname(glob.glob("/kaggle/input/**/run.py", recursive=True)[0])
shutil.copytree(code, f"{W}/solution", dirs_exist_ok=True)
sh(f"ls {W} {W}/solution; nvidia-smi --query-gpu=name,memory.total --format=csv")

sh("nvidia-smi | head -4")
sh("pip install -q vllm 2>&1 | tail -3")
# vllm upgrades torch (cu130) but Kaggle's preinstalled torchaudio is built for another CUDA, and vllm imports it
sh('TV=$(python -c "import torch;print(torch.__version__.split(\'+\')[0])"); CU=$(python -c "import torch;print(\'cu\'+torch.version.cuda.replace(\'.\',\'\'))"); '
   'echo "torch $TV $CU"; pip install -q "torchaudio==$TV" --index-url https://download.pytorch.org/whl/$CU 2>&1 | tail -2 '
   '|| pip uninstall -y torchaudio')
if sh('python -c "import torch; from vllm import LLM; print(\'vllm ok\', torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"') != 0:
    sh("pip uninstall -y torchaudio")
    if sh('python -c "from vllm import LLM; print(\'vllm ok without torchaudio\')"') != 0:
        sys.exit("vllm import failed")

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
base = f"cd {W} && python -u solution/run.py --split train --limit 200 --seed 0 --wave 50 --out {OUT}"
rc = sh(f"{base} --tp 2")
if rc != 0:
    print(f"tp=2 failed (rc={rc}); retrying with tp=1, resuming finished samples", flush=True)
    rc = sh(f"{base} --tp 1 --gpu-util 0.92")
sh(f"cd {W} && python solution/score.py {OUT}/submission.jsonl --details {OUT}/details.jsonl | tee {OUT}/score.txt")
sys.exit(rc)
