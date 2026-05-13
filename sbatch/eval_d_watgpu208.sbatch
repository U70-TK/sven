#!/bin/bash
#SBATCH --job-name=sven-eval-d
#SBATCH --partition=ALL
#SBATCH --nodelist=watgpu208
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/u901/t577wang/sven/sbatch/logs/%x-%j.out
#SBATCH --error=/u901/t577wang/sven/sbatch/logs/%x-%j.err

# Group D (watgpu208): qwen3-8b.
# (deepseek-coder-6.7b is handled separately by eval_deepseek_redo_watgpu208.sbatch
# after the tokenizer fix; previous run died mid-deepseek-6.7b due to wipe.)

export SVEN_ROOT=/u901/t577wang/sven
source "${SVEN_ROOT}/.venv/bin/activate"
source "${SVEN_ROOT}/sbatch/_eval_lib.sh"

nvidia-smi || true

eval_model qwen3-8b-prefix

echo "=== [$(date)] DONE group D ==="
