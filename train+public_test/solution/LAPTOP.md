# Agent SFT on one small GPU (RTX 3060 6 GB laptop)

QLoRA: the base model is loaded in 4-bit, only the LoRA adapter trains. The adapter loads like the 2×T4 one and is
scored on Kaggle (6 GB is too small for vLLM with the 4B model).

## Setup (Windows PowerShell or Linux, Python 3.10–3.12, NVIDIA driver ≥ 550)

```
python -m venv .venv
.venv\Scripts\activate            # Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.56" peft bitsandbytes accelerate
```

About 12 GB of disk: 8 GB for the model download (first run), plus the venv.
Turn off sleep while plugged in, keep the laptop plugged in and raised, and use the vendor's battery-charge limit if it has one.

## 1. Benchmark (about 10 minutes)

The 2 longest samples go first, so the peak-memory line is the worst case.

```
python solution/train_sft.py --data sft_train_v3.jsonl --out lora_bench --qlora --max-len 8192 --accum 4 --max-steps 3
```

Send back the lines with `peak mem` and `step 3/...` (they show memory and speed).
If it runs out of memory, retry with `--max-len 7168`.

## 2. Full run

```
python solution/train_sft.py --data sft_train_v3.jsonl --out lora_v3 --qlora --max-len 8192 --resume
```

With `--resume` a checkpoint is written every 10 optimizer steps to `lora_v3/ckpt.pt`. After a crash, reboot or
Ctrl+C, run the same command again and it continues from there. The adapter is saved after each epoch to `lora_v3/`
(`adapter_config.json`, `adapter_model.safetensors`); send those two files back.
