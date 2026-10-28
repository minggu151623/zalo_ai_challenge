# Kaggle job: LoRA fine-tune PhoWhisper-large on train audio (holdout excluded), then transcribe the 200-sample
# holdout (WER vs audio_text) and all of public_test with the merged model, one T4 each.
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


os.makedirs(W, exist_ok=True)
sh(f"tar xzf {glob.glob('/kaggle/input/**/zac_text.tgz', recursive=True)[0]} -C {W} 2>/dev/null")
audio = glob.glob("/kaggle/input/**/zac_audio.tar", recursive=True)
if audio:
    sh(f"tar xf {audio[0]} -C {W}")
else:  # Kaggle unpacks archives on upload
    for sp in ("train", "public_test"):
        shutil.copytree(glob.glob(f"/kaggle/input/**/{sp}/audio", recursive=True)[0], f"{W}/{sp}/audio", dirs_exist_ok=True)
sh(f"ls {W}/train/audio | wc -l; ls {W}/public_test/audio | wc -l")
code = os.path.dirname(glob.glob("/kaggle/input/**/train_asr.py", recursive=True)[0])
shutil.copytree(code, f"{W}/solution", dirs_exist_ok=True)
sh('pip install -q -U "transformers>=4.56" peft 2>&1 | tail -2')
sh("pip uninstall -y -q torchao")  # Kaggle's torchao 0.10 makes peft refuse to build LoRA layers
sh('python -c "import torch, transformers, peft; print(\'torch\', torch.__version__, \'tf\', transformers.__version__, \'peft\', peft.__version__)"')

rc = sh(f"cd {W} && torchrun --nproc_per_node 2 solution/train_asr.py --out {OUT}/asr_lora")
if not os.path.exists(f"{OUT}/asr_lora/adapter_config.json"):
    sys.exit(f"training failed (rc={rc})")
if sh(f"cd {W} && python solution/train_asr.py --merge {OUT}/asr_lora --out {W}/asr_merged") != 0:
    sys.exit("merge failed")

jobs = [f"python -u solution/asr.py --split train --limit 200 --seed 0 --model {W}/asr_merged --out {OUT}/asr_hold_ft.jsonl",
        f"python -u solution/asr.py --split public_test --model {W}/asr_merged --out {OUT}/asr_public_ft.jsonl"]
procs = [subprocess.Popen(f"cd {W} && CUDA_VISIBLE_DEVICES={g} {c} > {OUT}/log_asr{g}.txt 2>&1", shell=True)
         for g, c in enumerate(jobs)]
rcs = [p.wait() for p in procs]
sh(f"grep -ahE 'clips|WER|Error' {OUT}/log_asr*.txt")
print("rcs", rcs)
sys.exit(max(rcs))
