# Kaggle job (2xT4): think0 = zero-shot Qwen3-4B-Thinking-2507 vs zero-shot Qwen3-4B-Instruct-2507 on the 200-sample
# holdout with gold transcripts (JSON tool schemas, no LoRA). Tells whether reasoning before the call fixes the
# conditional / date errors that cap the SFT model (sftc3: 71.6 gold, 64.2 ASR) before building a thinking SFT.
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


sh("pip install -q vllm 2>&1 | tail -3")
sh('TV=$(python -c "import torch;print(torch.__version__.split(\'+\')[0])"); CU=$(python -c "import torch;print(\'cu\'+torch.version.cuda.replace(\'.\',\'\'))"); '
   'echo "torch $TV $CU"; pip install -q "torchaudio==$TV" --index-url https://download.pytorch.org/whl/$CU 2>&1 | tail -2 '
   '|| pip uninstall -y torchaudio')
if sh('python -c "from vllm import LLM; print(\'vllm ok\')"') != 0:
    sh("pip uninstall -y torchaudio")
    if sh('python -c "from vllm import LLM; print(\'vllm ok without torchaudio\')"') != 0:
        sys.exit("vllm import failed")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def run(model, args, out):
    base = f"cd {W} && python -u solution/run.py --model {model} --wave 100 {args} --out {out}"
    rc = sh(f"{base} --tp 2")
    return rc if rc == 0 else sh(f"{base} --tp 1 --gpu-util 0.92")  # resumes finished samples


HOLD = "--split train --limit 200 --seed 0 --text gold"
for name, model, args in (
        ("think_gold", "Qwen/Qwen3-4B-Thinking-2507", "--max-tokens 6144 --temp 0.6 --max-model-len 24576"),
        ("instruct_gold", "Qwen/Qwen3-4B-Instruct-2507", "")):
    E = f"{OUT}/{name}"
    run(model, f"{HOLD} {args}", E)
    sh(f"cd {W} && python solution/score.py {E}/submission.jsonl --details {E}/details.jsonl | tee {E}/score.txt")
