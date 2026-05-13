#!/bin/bash
#SBATCH --job-name=sven-eval-b
#SBATCH --partition=ALL
#SBATCH --nodelist=watgpu208
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/u901/t577wang/sven/sbatch/logs/%x-%j.out
#SBATCH --error=/u901/t577wang/sven/sbatch/logs/%x-%j.err

# Group B (medium): qwen3-4b, mistral-7b

export SVEN_ROOT=/u901/t577wang/sven
source "${SVEN_ROOT}/.venv/bin/activate"
source "${SVEN_ROOT}/sbatch/_eval_lib.sh"

nvidia-smi || true

eval_model qwen3-4b-prefix
eval_model mistral-7b-prefix

echo "=== [$(date)] DONE group B ==="
