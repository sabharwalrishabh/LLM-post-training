# Tiny Reasoning Models (SFT to GRPO)

This repository contains the codebase for training and evaluating a tiny language model (**Qwen-2.5-0.5B and Qwen-2.5-0.5B-Instruct**) on mathematical reasoning tasks (**GSM8K**). 

The pipeline consists of three stages:
1. **Zero-Shot Evaluation:** Establishing a baseline.
2. **Supervised Fine-Tuning (SFT):** Teaching the model the reasoning format.
3. **Reinforcement Learning (GRPO):** Optimizing the model using verifiable rewards without a critic model.

## 🚀 Running the Pipeline

### 1. Zero-shot evaluation

```
python evaluation/main.py \
--model_signature $MODEL_SIGNATURE \
--output_path ./outputs/$MODEL_SIGNATURE-zero-shot
```

### 2. SFT

```
python finetuning/main.py \
--model_signature $MODEL_SIGNATURE \
--output_path ./checkpoints/$MODEL_SIGNATURE-sft \
--wandb_token <YOUR_WANDB_API_KEY> \
```

### 3. GRPO

```
python grpo/main.py \
--model_signature $MODEL_SIGNATURE \
--adapter_path ./checkpoints/$MODEL_SIGNATURE-sft \
--output_path ./checkpoints/$MODEL_SIGNATURE-sft_grpo \
--wandb_token <YOUR_WANDB_API_KEY>
```