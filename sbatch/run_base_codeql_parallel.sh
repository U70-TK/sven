#!/bin/bash
# Parallel CodeQL scan for all base-LM sec_eval outputs (no GPU required).
# Spawns N background workers (default 4); each worker processes models from a
# shared queue. CodeQL is CPU-bound, so set N near the number of CPU cores you
# have but not so high that disk I/O thrashes.
#
# Usage:
#   bash sbatch/run_base_codeql_parallel.sh         # 4 workers
#   N=8 bash sbatch/run_base_codeql_parallel.sh     # 8 workers
#
# Output: sbatch/logs/base_codeql/codeql-<model>.{out,err}

set -u

SVEN_ROOT=/u901/t577wang/sven
LOG_DIR="${SVEN_ROOT}/sbatch/logs/base_codeql"
mkdir -p "${LOG_DIR}"

N="${N:-4}"

# Use the Py3.8 env (CodeQL 2.11.1 autobuild needs python3 <= 3.8 on PATH).
source "${SVEN_ROOT}/codeql_env/bin/activate"

echo "codeql: $(${SVEN_ROOT}/codeql/codeql --version | head -1)"
echo "python: $(python3 --version)"
echo "workers: ${N}"
echo

MODELS=(
    deepseek-coder-1.3b-base
    phi-2-base
    qwen2.5-coder-3b-base
    qwen3-4b-base
    mistral-7b-base
    codellama-7b-base
    llama2-7b-base
    qwen2.5-coder-7b-base
    qwen3-8b-base
    deepseek-coder-6.7b-base
)

# Shared work queue via a FIFO + lock-free index file.
QFILE=$(mktemp)
for m in "${MODELS[@]}"; do echo "$m" >> "${QFILE}"; done

PIDS=()
cleanup() {
    echo; echo "=== [$(date)] signal received, killing workers ==="
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait
    rm -f "${QFILE}"
    exit 130
}
trap cleanup INT TERM

cd "${SVEN_ROOT}/scripts"

worker() {
    local wid="$1"
    while :; do
        # atomic dequeue: lock + read first line + delete it
        local m
        m=$(flock -x "${QFILE}.lock" bash -c '
            line=$(head -n1 "$1") || true
            [ -z "$line" ] && exit 1
            sed -i "1d" "$1"
            echo "$line"
        ' _ "${QFILE}") || break
        if [ -z "$m" ]; then break; fi

        local out="${LOG_DIR}/codeql-${m}.out"
        local err="${LOG_DIR}/codeql-${m}.err"
        echo "[worker ${wid}] [$(date +%H:%M:%S)] -> ${m}"

        if [ ! -d "${SVEN_ROOT}/experiments/sec_eval/${m}/trained" ]; then
            echo "  (no generated samples for ${m}; skipping)" >"${out}"
            continue
        fi

        (
            echo "=== [$(date)] codeql_only: ${m} ==="
            bash eval.sh "${m}" trained --codeql_only
            echo "=== [$(date)] done: ${m} ==="
        ) >"${out}" 2>"${err}"

        if [ $? -ne 0 ]; then
            echo "[worker ${wid}] FAIL ${m}  (see ${err})"
        else
            echo "[worker ${wid}] OK   ${m}"
        fi
    done
}

touch "${QFILE}.lock"
for ((i=1; i<=N; i++)); do
    worker "$i" &
    PIDS+=("$!")
done

wait
rm -f "${QFILE}" "${QFILE}.lock"

echo
echo "=== [$(date)] all CodeQL scans finished ==="
echo "Per-model output: ${LOG_DIR}/codeql-<model>.out"
