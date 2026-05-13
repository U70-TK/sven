#!/bin/bash
# Parallel base-LM sec_eval dispatcher for an 8-GPU node (no Slurm).
# Launches 8 background workers, one per GPU (CUDA_VISIBLE_DEVICES=0..7).
# Each worker writes to a dedicated log under sbatch/logs/base_parallel/.
#
# Usage:   bash sbatch/run_base_parallel.sh
# Output:  sbatch/logs/base_parallel/base-<NN>-gpu<N>.{out,err}
#
# Ctrl-C kills all workers via the trap.

set -u

SVEN_ROOT=/u901/t577wang/sven
LOG_DIR="${SVEN_ROOT}/sbatch/logs/base_parallel"
mkdir -p "${LOG_DIR}"

# Activate venv once; child processes inherit it.
source "${SVEN_ROOT}/.venv/bin/activate"

# Each entry is "GPU_INDEX | space-separated-model-names" (matches the 8 sbatch files).
JOBS=(
    "0|deepseek-coder-1.3b phi-2"
    "1|qwen2.5-coder-3b qwen3-4b"
    "2|mistral-7b"
    "3|codellama-7b"
    "4|llama2-7b"
    "5|qwen2.5-coder-7b"
    "6|qwen3-8b"
    "7|deepseek-coder-6.7b"
)

PIDS=()
cleanup() {
    echo
    echo "=== [$(date)] received signal, killing workers ==="
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait
    exit 130
}
trap cleanup INT TERM

cd "${SVEN_ROOT}/scripts"

for entry in "${JOBS[@]}"; do
    gpu="${entry%%|*}"
    models="${entry#*|}"
    idx=$(printf "%02d" "$((gpu+1))")
    out="${LOG_DIR}/base-${idx}-gpu${gpu}.out"
    err="${LOG_DIR}/base-${idx}-gpu${gpu}.err"

    (
        echo "=== [$(date)] worker base-${idx} starting on GPU ${gpu} ==="
        echo "models: ${models}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i "${gpu}" 2>/dev/null || true
        for m in ${models}; do
            echo "--- [$(date)] gen ${m}-base on GPU ${gpu} ---"
            bash eval.sh "${m}-base" trained --base --gen_only
        done
        echo "=== [$(date)] worker base-${idx} done ==="
    ) >"${out}" 2>"${err}" &

    pid=$!
    PIDS+=("$pid")
    echo "spawned base-${idx} (gpu=${gpu}, pid=${pid})  -> ${out}"
done

echo
echo "=== [$(date)] all 8 workers spawned; waiting for completion ==="
echo "Tail any log to monitor progress, e.g.:"
echo "  tail -f ${LOG_DIR}/base-01-gpu0.out"
echo

# Track failures
fail=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "WARN: pid ${pid} exited non-zero"
        fail=$((fail+1))
    fi
done

echo
if [ "$fail" -eq 0 ]; then
    echo "=== [$(date)] ALL 8 workers completed successfully ==="
else
    echo "=== [$(date)] DONE, but ${fail} worker(s) failed; inspect *.err ==="
fi
exit "$fail"
