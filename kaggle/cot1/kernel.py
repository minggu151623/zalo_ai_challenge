# Kaggle job (2xT4): cot1 = short chain-of-thought SFT. Qwen3-14B-AWQ (kernel zac-rat) wrote a one- or two-sentence reasoning line
# for every gold step (rationale.py: each prompt shows only what the agent knows before that step), sft_data.py puts it
# in front of the step's tool call, and sftc3's adapter (kernel output huangvu1/zac-sftc3, holdout 64.21) trains one
# more epoch on that data at lr 5e-5 so the agent reasons briefly before every action. Then the 200-sample holdout
# with ASR text (compare to sftc3 64.21) and with gold text (sftc3 71.59).
import glob
import os
import shutil
import subprocess
import sys

W = "/tmp/zac"
OUT = "/kaggle/working"
TRAIN_MIN = 540  # 12 h session: setup ~20 min, merge + vllm + two holdout runs ~2 h


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True).returncode


def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not hits:
        sys.exit(f"missing input: {pattern}")
    return hits[0]


def vllm_setup():
    sh("pip install -q vllm 2>&1 | tail -3")
    sh('TV=$(python -c "import torch;print(torch.__version__.split(\'+\')[0])"); CU=$(python -c "import torch;print(\'cu\'+torch.version.cuda.replace(\'.\',\'\'))"); '
       'echo "torch $TV $CU"; pip install -q "torchaudio==$TV" --index-url https://download.pytorch.org/whl/$CU 2>&1 | tail -2 '
       '|| pip uninstall -y torchaudio')
    if sh('python -c "from vllm import LLM; print(\'vllm ok\')"') != 0:
        sh("pip uninstall -y torchaudio")
        if sh('python -c "from vllm import LLM; print(\'vllm ok without torchaudio\')"') != 0:
            sys.exit("vllm import failed")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


if sh("nvidia-smi -L") != 0:
    sys.exit("no GPU attached (quota spent or accelerator not set)")
os.makedirs(W, exist_ok=True)
sh(f"tar xzf {find('zac_text.tgz')} -C {W} 2>/dev/null")
shutil.copytree(os.path.dirname(find("rationale.py")), f"{W}/solution", dirs_exist_ok=True)
asr_hold, asr_cv = find("asr_hold_ft.jsonl"), find("asr_train_cv.jsonl")
init = os.path.dirname(glob.glob("/kaggle/input/**/lora/adapter_model.safetensors", recursive=True)[0])
print("init adapter:", init, flush=True)

# 1) rationales from kernel huangvu1/zac-rat (Qwen3-14B-AWQ, data generation only)
RAT = find("rationale.jsonl")
sh(f"wc -l {RAT}")
sh(f"cd {W} && python solution/synth.py --n 1100 --out {W}/synth.jsonl")

# 2) CoT SFT, one epoch from sftc3
sh('pip install -q -U "transformers>=4.56,<5" peft 2>&1 | tail -2')
sh("pip uninstall -y -q torchao")
sh(f"cd {W} && python solution/sft_data.py --compact --asr {asr_cv} --extra {W}/synth.jsonl --rationale {RAT} "
   f"--out {W}/sft_cot.jsonl")
sh(f"cp {W}/sft_cot.jsonl {OUT}/")
lora = f"{OUT}/lora"
sh(f"cd {W} && torchrun --nproc_per_node 2 solution/train_sft.py --data {W}/sft_cot.jsonl --out {lora} "
   f"--init {init} --lr 5e-5 --seed 4 --epochs 1 --max-len 6144 --resume --ckpt-every 20 --max-minutes {TRAIN_MIN}")
if not os.path.exists(f"{lora}/adapter_config.json"):
    sys.exit("training failed")
if sh(f"cd {W} && python solution/train_sft.py --merge {lora} --out {W}/merged") != 0:
    sys.exit("merge failed")
vllm_setup()


# 3) holdout: ASR text, then gold text
def run(args, out):
    base = f"cd {W} && python -u solution/run.py --compact --model {W}/merged --wave 50 {args} --out {out}"
    rc = sh(f"{base} --tp 2")
    return rc if rc == 0 else sh(f"{base} --tp 1 --gpu-util 0.92")  # resumes finished samples


for name, text in (("cot200_asr", asr_hold), ("cot200_gold", "gold")):
    E = f"{OUT}/{name}"
    run(f"--split train --limit 200 --seed 0 --text {text}", E)
    sh(f"cd {W} && python solution/score.py {E}/submission.jsonl --details {E}/details.jsonl | tee {E}/score.txt")
