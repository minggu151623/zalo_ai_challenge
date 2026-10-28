# Kaggle job: transcribe the 200-sample train holdout (WER vs audio_text) and all of public_test with two ASR models,
# one model per T4, so the agent can be scored on ASR text and a public_test submission can be built.
import glob
import os
import shutil
import subprocess

W = "/tmp/zac"
OUT = "/kaggle/working"
MODELS = {"pho": "vinai/PhoWhisper-large", "turbo": "openai/whisper-large-v3-turbo"}


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True).returncode


os.makedirs(W, exist_ok=True)
sh(f"tar xzf {glob.glob('/kaggle/input/**/zac_text.tgz', recursive=True)[0]} -C {W} 2>/dev/null")
audio = glob.glob("/kaggle/input/**/zac_audio.tar", recursive=True)
if audio:
    sh(f"tar xf {audio[0]} -C {W}")
else:  # Kaggle may have unpacked the archive on upload
    for sp in ("train", "public_test"):
        src = glob.glob(f"/kaggle/input/**/{sp}/audio", recursive=True)[0]
        shutil.copytree(src, f"{W}/{sp}/audio", dirs_exist_ok=True)
code = os.path.dirname(glob.glob("/kaggle/input/**/asr.py", recursive=True)[0])
shutil.copytree(code, f"{W}/solution", dirs_exist_ok=True)
sh(f"ls {W}/train/audio | wc -l; ls {W}/public_test/audio | wc -l; nvidia-smi --query-gpu=name --format=csv")
sh('python -c "import torch, transformers; print(\'torch\', torch.__version__, \'tf\', transformers.__version__)"')

# each GPU runs one model over holdout then public_test
procs = []
for gpu, (tag, model) in enumerate(MODELS.items()):
    cmd = (f"cd {W} && export CUDA_VISIBLE_DEVICES={gpu} && "
           f"python -u solution/asr.py --split train --limit 200 --seed 0 --model {model} --out {OUT}/asr_hold_{tag}.jsonl "
           f"> {OUT}/log_{tag}.txt 2>&1 && "
           f"python -u solution/asr.py --split public_test --model {model} --out {OUT}/asr_public_{tag}.jsonl "
           f">> {OUT}/log_{tag}.txt 2>&1")
    print(f"$ {cmd}", flush=True)
    procs.append(subprocess.Popen(cmd, shell=True))
rcs = [p.wait() for p in procs]
for tag in MODELS:
    sh(f"grep -aE 'clips|WER|Error|error' {OUT}/log_{tag}.txt | tail -6")
print("rcs", rcs)
