#!/bin/bash
# Shared helpers for SVEN eval sbatch scripts.
# Source this from each job script after setting SVEN_ROOT.

set -e

run_sec_eval() {
    local name="$1"   # e.g. mistral-7b-prefix
    # --gen_only: emit samples to experiments/sec_eval/.../<control>_output/.
    # CodeQL pass is run separately later from a Python 3.8 venv (no GPU needed).
    echo "=== [$(date)] sec_eval gen: ${name} (trained CWEs) ==="
    ( cd "${SVEN_ROOT}/scripts" && bash eval.sh "${name}" trained --gen_only )
}

run_human_eval() {
    local name="$1"          # e.g. mistral-7b-prefix
    local ckpt="${SVEN_ROOT}/trained/${name}/checkpoint-last"
    for ctrl in sec vul; do
        local out="human-eval-${name}-${ctrl}"
        echo "=== [$(date)] human_eval gen: ${out} ==="
        ( cd "${SVEN_ROOT}/scripts" && \
          python human_eval_gen.py \
              --model_type prefix \
              --model_dir "${ckpt}" \
              --control "${ctrl}" \
              --output_name "${out}" )
        echo "=== [$(date)] human_eval exec: ${out} ==="
        ( cd "${SVEN_ROOT}/scripts" && \
          python human_eval_exec.py --output_name "${out}" )
    done
}

eval_model() {
    local name="$1"
    run_sec_eval "${name}"
    run_human_eval "${name}"
}
