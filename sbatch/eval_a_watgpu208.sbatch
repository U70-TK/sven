#!/bin/bash
#SBATCH --job-name=sven-eval-a
#SBATCH --partition=ALL
#SBATCH --nodelist=watgpu208
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/u901/t577wang/sven/sbatch/logs/%x-%j.out
#SBATCH --error=/u901/t577wang/sven/sbatch/logs/%x-%j.err

# Group A (small models): deepseek-coder-1.3b, phi-2, qwen2.5-coder-3b

export SVEN_ROOT=/u901/t577wang/sven
source "${SVEN_ROOT}/.venv/bin/activate"
source "${SVEN_ROOT}/sbatch/_eval_lib.sh"

nvidia-smi || true

eval_model deepseek-coder-1.3b-prefix
eval_model phi-2-prefix
eval_model qwen2.5-coder-3b-prefix

echo "=== [$(date)] DONE group A ==="
