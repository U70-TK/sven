#!/bin/bash
#SBATCH --job-name=cosec-codeql
#SBATCH --partition=ALL
#SBATCH --nodelist=watgpu1008
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# CPU-only CodeQL pass over all 10 prefix models' generated samples.
# Uses codeql_env (Python 3.8) so CodeQL 2.11.1's Python extractor autobuild
# finds a compatible interpreter on PATH. No model loads here, so no GPU is used
# (the --gres line above is just to satisfy this node's scheduling).

set -e

export SVEN_ROOT=/u901/t577wang/sven
cd "${SVEN_ROOT}"
mkdir -p logs/slurm

source "${SVEN_ROOT}/codeql_env/bin/activate"

echo "=== [$(date)] node=$(hostname) python=$(python3 --version) ==="
echo "=== codeql: $(${SVEN_ROOT}/codeql/codeql --version | head -1) ==="

MODELS=(
    codellama-7b-prefix
    deepseek-coder-1.3b-prefix
    deepseek-coder-6.7b-prefix
    llama2-7b-prefix
    mistral-7b-prefix
    phi-2-prefix
    qwen2.5-coder-3b-prefix
    qwen2.5-coder-7b-prefix
    qwen3-4b-prefix
    qwen3-8b-prefix
)

for m in "${MODELS[@]}"; do
    echo
    echo "=== [$(date)] codeql_only: ${m} ==="
    if [ ! -d "${SVEN_ROOT}/experiments/sec_eval/${m}/trained" ]; then
        echo "  (no generated samples at experiments/sec_eval/${m}/trained -- skipping)"
        continue
    fi
    ( cd "${SVEN_ROOT}/scripts" && bash eval.sh "${m}" trained --codeql_only )
done

echo
echo "=== [$(date)] DONE all codeql scans ==="
