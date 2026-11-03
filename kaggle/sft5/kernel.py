# Kaggle job: sft5 = sft3's adapter (zac-sft3-lora; best public 45.80) trained further at lr 5e-5 on real data it never
# saw: the 200 holdout samples (gold text + their ASR copies) + 300 replayed train/ASR/synth samples; then public_test.
# sft4 (sft3 + 600 synth) lost 1.6 public points while gaining on holdout, so this run adds real data instead of synth
# and the holdout is spent on training (no holdout score).
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

init = os.path.dirname(find("zac-sft3-lora/**/adapter_model.safetensors"))
asr_cv = find("asr_train_cv.jsonl")
sh(f"cat {asr_cv} {asr_hold} > {W}/asr_all.jsonl")
sh(f"cd {W} && python solution/synth.py --n 1100 --out {W}/synth.jsonl")
sh(f"cd {W} && python solution/sft_data.py --asr {asr_cv} --extra {W}/synth.jsonl --out {W}/rest.jsonl")
sh(f"cd {W} && python solution/sft_data.py --asr {W}/asr_all.jsonl --holdout 0 --out {W}/all.jsonl")
import random
rest = open(f"{W}/rest.jsonl", encoding="utf-8").readlines()
seen = set(rest)
held = [l for l in open(f"{W}/all.jsonl", encoding="utf-8") if l not in seen]
mix = held + random.Random(5).sample(rest, 300)
random.Random(5).shuffle(mix)
open(f"{W}/sft5.jsonl", "w", encoding="utf-8").writelines(mix)
print(f"sft5 data: {len(held)} holdout + 300 replay", flush=True)
lora = f"{OUT}/lora5"
sh(f"cd {W} && torchrun --nproc_per_node 2 solution/train_sft.py --data {W}/sft5.jsonl --out {lora} --init {init} "
   f"--epochs 1 --lr 5e-5 --max-len 8192 --resume --ckpt-every 20 --max-minutes 230")
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
    base = f"cd {W} && python -u solution/run.py --model {W}/merged --wave 50 {args} --out {out}"
    rc = sh(f"{base} --tp 2")
    return rc if rc == 0 else sh(f"{base} --tp 1 --gpu-util 0.92")  # resumes finished samples


P = f"{OUT}/public"
rc = run(f"--split public_test --text {asr_pub}", P)
sh(f"wc -l {P}/submission.jsonl; cd {W}/public_test && python -m zaic_tools --split public_test --root .. --validate {P}/submission.jsonl | tail -5")
sys.exit(rc)
