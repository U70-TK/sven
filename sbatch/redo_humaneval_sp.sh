#!/bin/bash
# Re-run HumanEval for the 3 SentencePiece-tokenizer models after the
# evaler.process_completions decoding-boundary fix.
#
# Usage:  bash sbatch/redo_humaneval_sp.sh
# Spawns 6 parallel workers (one per GPU 0..5), then tabulates pass@k.
set -u

SVEN_ROOT=/u901/t577wang/sven
LOG_DIR="${SVEN_ROOT}/sbatch/logs/humaneval_redo"
mkdir -p "${LOG_DIR}"

cd "${SVEN_ROOT}"

echo "=== [$(date)] checking GPU availability ==="
nvidia-smi --query-gpu=index,name,memory.used --format=csv

# ---- gen + exec (parallel across 6 GPUs) -------------------------------------
source "${SVEN_ROOT}/.venv/bin/activate"
cd "${SVEN_ROOT}/scripts"

# GPU 5 is in use by someone else; use 0,1,2,3,4,6.
GPUS=(0 1 2 3 4 6)
PIDS=()
i=0
for m in codellama-7b llama2-7b mistral-7b; do
    CKPT="${SVEN_ROOT}/trained/${m}-prefix/checkpoint-last"
    for ctrl in sec vul; do
        GPU="${GPUS[$i]}"
        OUT="human-eval-${m}-prefix-${ctrl}"
        LOG="${LOG_DIR}/${OUT}.log"
        (
            echo "=== [$(date)] gen ${OUT} on GPU ${GPU} ==="
            export CUDA_VISIBLE_DEVICES="${GPU}"
            python human_eval_gen.py \
                --model_type prefix \
                --model_dir "${CKPT}" \
                --control "${ctrl}" \
                --output_name "${OUT}"
            echo "=== [$(date)] exec ${OUT} ==="
            python human_eval_exec.py --output_name "${OUT}"
            echo "=== [$(date)] done ${OUT} ==="
        ) > "${LOG}" 2>&1 &
        pid=$!
        PIDS+=("${pid}")
        echo "spawned ${OUT} on GPU ${GPU} (pid=${pid}) -> ${LOG}"
        i=$((i+1))
    done
done

echo
echo "=== [$(date)] all 6 workers spawned; waiting ==="
fail=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo "WARN: pid ${pid} exited non-zero"
        fail=$((fail+1))
    fi
done

if [ "${fail}" -gt 0 ]; then
    echo "=== ${fail} worker(s) failed; inspect ${LOG_DIR}/*.log ==="
    exit "${fail}"
fi

deactivate

# ---- tabulate pass@k (codeql_env has all needed deps) ------------------------
source "${SVEN_ROOT}/codeql_env/bin/activate"

echo
echo "=== [$(date)] pass@k for the 3 redone models ==="
for m in codellama-7b llama2-7b mistral-7b; do
    for ctrl in sec vul; do
        OUT="human-eval-${m}-prefix-${ctrl}"
        echo
        echo "=== ${OUT} ==="
        python "${SVEN_ROOT}/scripts/print_results.py" \
            --eval_type human_eval \
            --eval_dir "${SVEN_ROOT}/experiments/human_eval/${OUT}"
    done
done

echo
echo "=== [$(date)] DONE ==="
