python evaluation/main.py \
--model_signature Qwen/Qwen2.5-0.5B-Instruct \
--output_path ./outputs/Qwen/Qwen2.5-0.5B-Instruct-zero-shot

python evaluation/main.py \
--model_signature Qwen/Qwen2.5-0.5B \
--output_path ./outputs/Qwen/Qwen2.5-0.5B-zero-shot

# # Question-4 (i)
python finetuning/main.py \
--model_signature Qwen/Qwen2.5-0.5B-Instruct \
--output_path ./checkpoints/Qwen/Qwen2.5-0.5B-Instruct-sft \
--wandb_token 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 \

python finetuning/main.py \
--model_signature Qwen/Qwen2.5-0.5B \
--output_path ./checkpoints/Qwen/Qwen/Qwen2.5-0.5B-sft \
--wandb_token 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 \

# # # Question-4 (ii)
python evaluation/main.py \
--model_signature Qwen/Qwen2.5-0.5B-Instruct \
--sft_adapter_path ./checkpoints/Qwen/Qwen2.5-0.5B-Instruct-sft \
--output_path ./outputs/Qwen/Qwen2.5-0.5B-Instruct-SFT

python evaluation/main.py \
--model_signature Qwen/Qwen2.5-0.5B \
--sft_adapter_path ./checkpoints/Qwen/Qwen/Qwen2.5-0.5B-sft \
--output_path ./outputs/Qwen/Qwen2.5-0.5B-SFT

# # Question - 4
python grpo/main.py \
--model_signature Qwen/Qwen2.5-0.5B-Instruct \
--adapter_path ./checkpoints/Qwen/Qwen2.5-0.5B-Instruct-sft \
--output_path ./checkpoints/Qwen/Qwen2.5-0.5B-Instruct-sft_grpo \
--wandb_token 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 

python evaluation/main.py \
--model_signature Qwen/Qwen2.5-0.5B-Instruct \
--sft_adapter_path ./checkpoints/Qwen/Qwen2.5-0.5B-Instruct-sft \
--grpo_adapter_path ./checkpoints/Qwen/Qwen2.5-0.5B-Instruct-sft_grpo/checkpoint-125 \
--output_path ./outputs/Qwen/Qwen2.5-0.5B-Instruct-sft_grpo
