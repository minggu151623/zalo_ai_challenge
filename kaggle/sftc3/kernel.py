# Kaggle job (2xT4): sftc3 = a third epoch. Starts from sftc2's adapter (kernel output huangvu1/zac-sftc2; sftc2 =
# sftc + epoch 2 at lr 5e-5, holdout 53.5 -> 59.1) and runs the same data once more in a new order at lr 3e-5,
# --compact. Then public_test and the 200-sample holdout, both with --compact.
import glob
import os
import shutil
import subprocess
import sys

W = "/tmp/zac"
OUT = "/kaggle/working"
TRAIN_MIN = 500  # 12 h session: setup ~15 min, then merge + public + holdout need ~2 h


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

asr_cv = find("asr_train_cv.jsonl")
init = os.path.dirname(glob.glob("/kaggle/input/**/lora/adapter_model.safetensors", recursive=True)[0])
print("init adapter:", init, flush=True)
sh(f"cd {W} && python solution/synth.py --n 1100 --out {W}/synth.jsonl")
sh(f"cd {W} && python solution/sft_data.py --compact --asr {asr_cv} --extra {W}/synth.jsonl --out {W}/sft_train.jsonl")
lora = f"{OUT}/lora"
sh(f"cd {W} && torchrun --nproc_per_node 2 solution/train_sft.py --data {W}/sft_train.jsonl --out {lora} "
   f"--init {init} --lr 3e-5 --seed 2 --epochs 1 --max-len 6144 --resume --ckpt-every 20 --max-minutes {TRAIN_MIN}")
if not os.path.exists(f"{lora}/adapter_config.json"):
    sys.exit("training failed")
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


P = f"{OUT}/public"
rc = run(f"--split public_test --text {asr_pub}", P)
sh(f"wc -l {P}/submission.jsonl; cd {W}/public_test && python -m zaic_tools --split public_test --root .. --validate {P}/submission.jsonl | tail -5")
E = f"{OUT}/sft200_asr"
run(f"--split train --limit 200 --seed 0 --text {asr_hold}", E)
sh(f"cd {W} && python solution/score.py {E}/submission.jsonl --details {E}/details.jsonl | tee {E}/score.txt")
sys.exit(rc)
