# Kaggle job: cross-fold ASR transcripts of train (outside the 200 holdout) for noise-matched SFT data.
# GPU k fine-tunes PhoWhisper-large without fold k, then transcribes fold k, so every transcript is from a model
# that never heard that audio (like test time).
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
shutil.copytree(os.path.dirname(glob.glob("/kaggle/input/**/train_asr.py", recursive=True)[0]), f"{W}/solution", dirs_exist_ok=True)
sh(f"ls {W}/train/audio | wc -l")
sh('pip install -q -U "transformers>=4.56" peft 2>&1 | tail -2')
sh("pip uninstall -y -q torchao")  # Kaggle's torchao 0.10 makes peft refuse to build LoRA layers

procs = []
for k in (0, 1):
    cmd = (f"cd {W} && export CUDA_VISIBLE_DEVICES={k} && "
           f"python -u solution/train_asr.py --skip-fold {k} --accum 2 --out {W}/asr_lora_cv{k} && "
           f"python solution/train_asr.py --merge {W}/asr_lora_cv{k} --out {W}/asr_cv{k} && "
           f"python -u solution/asr.py --split train --fold {k} --model {W}/asr_cv{k} --out {OUT}/asr_train_cv{k}.jsonl")
    print(f"$ {cmd}", flush=True)
    procs.append(subprocess.Popen(f"{cmd} > {OUT}/log_cv{k}.txt 2>&1", shell=True))
rcs = [p.wait() for p in procs]
sh(f"grep -ahE 'training clips|step [0-9]+0/|clips,|WER|Error' {OUT}/log_cv*.txt | tail -12")
sh(f"cat {OUT}/asr_train_cv0.jsonl {OUT}/asr_train_cv1.jsonl > {OUT}/asr_train_cv.jsonl; wc -l {OUT}/asr_train_cv.jsonl")
print("rcs", rcs)
sys.exit(max(rcs))
