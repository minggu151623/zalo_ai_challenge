# Kaggle job: SFT agent (LoRA from zac-sft) on fine-tuned-ASR transcripts (from zac-asrft):
# 200-sample train holdout (scored) and all of public_test (submission.jsonl).
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


os.makedirs(W, exist_ok=True)
sh(f"tar xzf {find('zac_text.tgz')} -C {W} 2>/dev/null")
shutil.copytree(os.path.dirname(find("run.py")), f"{W}/solution", dirs_exist_ok=True)
lora = os.path.dirname(find("lora/adapter_config.json"))
asr_hold, asr_pub = find("asr_hold_ft.jsonl"), find("asr_public_ft.jsonl")
print("inputs:", lora, asr_hold, asr_pub, flush=True)

sh('pip install -q -U "transformers>=4.56" peft 2>&1 | tail -2')
sh("pip uninstall -y -q torchao")
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
    base = f"cd {W} && python -u solution/run.py --model {W}/merged --wave 50 {args} --out {out}"
    rc = sh(f"{base} --tp 2")
    return rc if rc == 0 else sh(f"{base} --tp 1 --gpu-util 0.92")  # resumes finished samples


E = f"{OUT}/sft200_asr"
run(f"--split train --limit 200 --seed 0 --text {asr_hold}", E)
sh(f"cd {W} && python solution/score.py {E}/submission.jsonl --details {E}/details.jsonl | tee {E}/score.txt")
P = f"{OUT}/public"
rc = run(f"--split public_test --text {asr_pub}", P)
sh(f"wc -l {P}/submission.jsonl; cd {W}/public_test && python -m zaic_tools --split public_test --root .. --validate {P}/submission.jsonl | tail -5")
sys.exit(rc)
