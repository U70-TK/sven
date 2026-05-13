#!/bin/bash
#SBATCH --job-name=sven-base-01
#SBATCH --partition=ALL
#SBATCH --nodelist=watgpu308
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/u901/t577wang/sven/sbatch/logs/%x-%j.out
#SBATCH --error=/u901/t577wang/sven/sbatch/logs/%x-%j.err

# Base-LM sec_eval gen: small-1 pair (deepseek-coder-1.3b, phi-2).
set -e
export SVEN_ROOT=/u901/t577wang/sven
source "${SVEN_ROOT}/.venv/bin/activate"
nvidia-smi || true
cd "${SVEN_ROOT}/scripts"

bash eval.sh deepseek-coder-1.3b-base trained --base --gen_only
bash eval.sh phi-2-base               trained --base --gen_only

echo "=== [$(date)] DONE base-01 ==="
