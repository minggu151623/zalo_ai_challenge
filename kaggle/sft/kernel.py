# Kaggle job: LoRA SFT of Qwen3-4B-Instruct-2507 on train (minus the 200 holdout), then vLLM eval on that holdout (2xT4).
import glob
import os
import shutil
import subprocess
import sys

W = "/tmp/zac"
OUT = "/kaggle/working"
TRAIN_MIN = 420  # leave room in the 12 h session for merge + eval


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True).returncode


os.makedirs(W, exist_ok=True)
tgz = glob.glob("/kaggle/input/**/zac_text.tgz", recursive=True)
sh(f"tar xzf {tgz[0]} -C {W} 2>/dev/null")
code = os.path.dirname(glob.glob("/kaggle/input/**/train_sft.py", recursive=True)[0])
shutil.copytree(code, f"{W}/solution", dirs_exist_ok=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")

# 1. train with Kaggle's own torch (vllm swaps torch later, so it is installed only after training)
sh('pip install -q -U "transformers>=4.56" peft 2>&1 | tail -2')
sh("pip uninstall -y -q torchao")  # Kaggle's torchao 0.10 makes peft refuse to build LoRA layers; it is not needed
sh('python -c "import torch, transformers, peft; print(\'torch\', torch.__version__, \'tf\', transformers.__version__, \'peft\', peft.__version__)"')
sh(f"cd {W} && python solution/sft_data.py --out {W}/sft_train.jsonl")
rc = sh(f"cd {W} && torchrun --nproc_per_node 2 solution/train_sft.py --data {W}/sft_train.jsonl --out {OUT}/lora "
        f"--max-minutes {TRAIN_MIN}")
if not os.path.exists(f"{OUT}/lora/adapter_config.json"):
    sys.exit(f"training failed (rc={rc}), no adapter saved")
if sh(f"cd {W} && python solution/train_sft.py --merge {OUT}/lora --out {W}/merged") != 0:
    sys.exit("merge failed")

# 2. eval with vLLM, same recipe as the baseline kernel
sh("pip install -q vllm 2>&1 | tail -3")
sh('TV=$(python -c "import torch;print(torch.__version__.split(\'+\')[0])"); CU=$(python -c "import torch;print(\'cu\'+torch.version.cuda.replace(\'.\',\'\'))"); '
   'echo "torch $TV $CU"; pip install -q "torchaudio==$TV" --index-url https://download.pytorch.org/whl/$CU 2>&1 | tail -2 '
   '|| pip uninstall -y torchaudio')
if sh('python -c "from vllm import LLM; print(\'vllm ok\')"') != 0:
    sh("pip uninstall -y torchaudio")
    if sh('python -c "from vllm import LLM; print(\'vllm ok without torchaudio\')"') != 0:
        sys.exit("vllm import failed")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
E = f"{OUT}/sft200"
base = f"cd {W} && python -u solution/run.py --split train --limit 200 --seed 0 --wave 50 --model {W}/merged --out {E}"
rc = sh(f"{base} --tp 2")
if rc != 0:
    rc = sh(f"{base} --tp 1 --gpu-util 0.92")
sh(f"cd {W} && python solution/score.py {E}/submission.jsonl --details {E}/details.jsonl | tee {E}/score.txt")
sys.exit(rc)
