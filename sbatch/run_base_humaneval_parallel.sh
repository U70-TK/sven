#!/bin/bash
# Base-LM HumanEval dispatcher for an 8-GPU node (no Slurm).
# 10 base models share a queue across 8 GPU workers; each worker runs
# human_eval_gen.py + human_eval_exec.py for a model and picks the next.
#
# Usage:   bash sbatch/run_base_humaneval_parallel.sh
# Output:  sbatch/logs/base_humaneval/humaneval-<model>-lm.log
#
# Ctrl-C kills all workers cleanly.

set -u

SVEN_ROOT=/u901/t577wang/sven
LOG_DIR="${SVEN_ROOT}/sbatch/logs/base_humaneval"
mkdir -p "${LOG_DIR}"

source "${SVEN_ROOT}/.venv/bin/activate"

echo "=== [$(date)] GPU summary ==="
nvidia-smi --query-gpu=index,name,memory.used --format=csv

MODELS=(
    deepseek-coder-1.3b
    phi-2
    qwen2.5-coder-3b
    qwen3-4b
    mistral-7b
    codellama-7b
    llama2-7b
    qwen2.5-coder-7b
    qwen3-8b
    deepseek-coder-6.7b
)

GPUS=(0 1 2 3 4 5 6 7)

# Shared work queue
QFILE=$(mktemp)
for m in "${MODELS[@]}"; do echo "$m" >> "${QFILE}"; done
LOCK="${QFILE}.lock"
touch "${LOCK}"

PIDS=()
cleanup() {
    echo; echo "=== [$(date)] signal received, killing workers ==="
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait
    rm -f "${QFILE}" "${LOCK}"
    exit 130
}
trap cleanup INT TERM

worker() {
    local gpu="$1"
    cd "${SVEN_ROOT}/scripts"
    while :; do
        # Atomic dequeue
        local m
        m=$(flock -x "${LOCK}" bash -c '
            line=$(head -n1 "$1") || true
            [ -z "$line" ] && exit 1
            sed -i "1d" "$1"
            echo "$line"
        ' _ "${QFILE}") || break
        [ -z "$m" ] && break

        local out="human-eval-${m}-lm"
        local log="${LOG_DIR}/${out}.log"
        echo "[gpu ${gpu}] [$(date +%H:%M:%S)] -> ${m}"

        (
            echo "=== [$(date)] gen ${out} on GPU ${gpu} ==="
            export CUDA_VISIBLE_DEVICES="${gpu}"
            python human_eval_gen.py \
                --model_type lm \
                --model_dir "${m}" \
                --output_name "${out}"
            echo "=== [$(date)] exec ${out} ==="
            python human_eval_exec.py --output_name "${out}"
            echo "=== [$(date)] done ${out} ==="
        ) > "${log}" 2>&1

        if [ $? -ne 0 ]; then
            echo "[gpu ${gpu}] FAIL ${m}  (see ${log})"
        else
            echo "[gpu ${gpu}] OK   ${m}"
        fi
    done
}

for g in "${GPUS[@]}"; do
    worker "$g" &
    PIDS+=("$!")
done

wait
rm -f "${QFILE}" "${LOCK}"

deactivate

# ---- pass@k tabulation -------------------------------------------------------
source "${SVEN_ROOT}/codeql_env/bin/activate"
echo
echo "=== [$(date)] base-LM HumanEval pass@k ==="
for m in "${MODELS[@]}"; do
    out="human-eval-${m}-lm"
    echo
    echo "=== ${out} ==="
    python "${SVEN_ROOT}/scripts/print_results.py" \
        --eval_type human_eval \
        --eval_dir "${SVEN_ROOT}/experiments/human_eval/${out}" 2>/dev/null || echo "  (failed to tabulate -- check ${LOG_DIR}/${out}.log)"
done

echo
echo "=== [$(date)] DONE ==="
