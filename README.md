# zalo_ai_challenge

This repo is created to serve the Zalo AI Challenge 2026, track **LifeMate** (Vietnamese voice request → tool calls).

The competition dataset is not included. Put the official `train/` and `public_test/` folders in `train+public_test/` next to `solution/`.

## Pipeline

- **ASR:** `vinai/PhoWhisper-large`, LoRA fine-tuned on train audio (`solution/train_asr.py`, `solution/asr.py`).
- **Agent:** `Qwen/Qwen3-4B-Instruct-2507`, LoRA SFT on gold tool-call episodes plus synthetic samples for apps that are not in train (`solution/sft_data.py`, `solution/synth.py`, `solution/train_sft.py`). It runs with vLLM over `zaic_tools.ToolEnv` (`solution/agent.py`, `solution/run.py`).
- **Scoring:** `solution/score.py` reimplements the official metric (state-hash gate, then F1 over calls).

`kaggle/` holds the Kaggle kernels (2×T4) that run each stage.

## Results (200-sample train holdout)

| Setup | Score |
|---|---|
| Base Qwen3-4B, gold text | 23.65 |
| SFT, gold text | 63.03 |
| SFT, fine-tuned ASR text | 55.21 |

ASR WER on the holdout: 27.3% → 8.4% after LoRA.
