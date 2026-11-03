# Kaggle job (2xT4): LoRA SFT on the v3 mix (gold + cross-fold-ASR copies of train, holdout excluded, + 1100 synthetic
# samples with multi-turn and follow-ups), 1 epoch at max-len 8192. Training only: 235 steps at ~2.2 min (sft2's pace)
# is ~8.7 h, which leaves no room for evaluation in a 12 h session; zac-eval3 merges and scores the adapter.
# --resume keeps <out>/ckpt.pt, so a run cut by the time budget can be continued from this kernel's output.
import glob
import os
import shutil
import subprocess
import sys

W = "/tmp/zac"
OUT = "/kaggle/working"
TRAIN_MIN = 640  # 12 h session: setup ~15 min, saving a few minutes


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
shutil.copytree(os.path.dirname(find("train_sft.py")), f"{W}/solution", dirs_exist_ok=True)
asr_cv = find("asr_train_cv.jsonl")
print("inputs:", asr_cv, flush=True)

sh('pip install -q -U "transformers>=4.56,<5" peft 2>&1 | tail -2')
sh("pip uninstall -y -q torchao")  # Kaggle's torchao 0.10 makes peft refuse to build LoRA layers
sh(f"cd {W} && python solution/synth.py --n 1100 --out {W}/synth.jsonl")
sh(f"cd {W} && python solution/sft_data.py --asr {asr_cv} --extra {W}/synth.jsonl --out {W}/sft_train.jsonl")
rc = sh(f"cd {W} && torchrun --nproc_per_node 2 solution/train_sft.py --data {W}/sft_train.jsonl --out {OUT}/lora "
        f"--epochs 1 --max-len 8192 --resume --ckpt-every 20 --max-minutes {TRAIN_MIN}")
sh(f"ls -la {OUT}/lora {OUT}/lora/epoch1 2>&1 | head -20")
sys.exit(0 if os.path.exists(f"{OUT}/lora/adapter_config.json") else 1)
