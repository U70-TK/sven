#!/bin/bash
#SBATCH --job-name=sven-eval-c
#SBATCH --partition=ALL
#SBATCH --nodelist=watgpu208
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/u901/t577wang/sven/sbatch/logs/%x-%j.out
#SBATCH --error=/u901/t577wang/sven/sbatch/logs/%x-%j.err

# Group C (7b trio): llama2-7b, codellama-7b, qwen2.5-coder-7b

export SVEN_ROOT=/u901/t577wang/sven
source "${SVEN_ROOT}/.venv/bin/activate"
source "${SVEN_ROOT}/sbatch/_eval_lib.sh"

nvidia-smi || true

eval_model llama2-7b-prefix
eval_model codellama-7b-prefix
eval_model qwen2.5-coder-7b-prefix

echo "=== [$(date)] DONE group C ==="
