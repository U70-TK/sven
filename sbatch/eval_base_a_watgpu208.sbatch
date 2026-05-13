#!/bin/bash
#SBATCH --job-name=sven-base-a
#SBATCH --partition=ALL
#SBATCH --nodelist=watgpu208
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/u901/t577wang/sven/sbatch/logs/%x-%j.out
#SBATCH --error=/u901/t577wang/sven/sbatch/logs/%x-%j.err

# Base-LM (no-prefix) sec_eval generation, group A.
# Models: deepseek-coder-1.3b, phi-2, qwen2.5-coder-3b, qwen3-4b, mistral-7b, codellama-7b.
# CodeQL phase is intentionally split off; rerun separately in codeql_env.

set -e

export SVEN_ROOT=/u901/t577wang/sven
source "${SVEN_ROOT}/.venv/bin/activate"

nvidia-smi || true

MODELS=(
    deepseek-coder-1.3b
    phi-2
    qwen2.5-coder-3b
    qwen3-4b
    mistral-7b
    codellama-7b
)

cd "${SVEN_ROOT}/scripts"
for m in "${MODELS[@]}"; do
    out="${m}-base"
    echo "=== [$(date)] base-LM gen: ${out} ==="
    bash eval.sh "${out}" trained --base --gen_only
done

echo "=== [$(date)] DONE base group A ==="
