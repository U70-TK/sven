#!/bin/bash
# End-to-end redo for deepseek-coder-6.7b-prefix after the tokenizer fix + retrain.
# Usage:  bash sbatch/redo_deepseek-coder-6.7b.sh
# Requires: a GPU (the gen phase loads the model); CodeQL phase is CPU-only.
set -e

SVEN_ROOT=/u901/t577wang/sven
MODEL=deepseek-coder-6.7b-prefix

cd "${SVEN_ROOT}"

# ---- gen phase (GPU): use main .venv (Py3.12, transformers) -------------------
echo "=== [$(date)] gen phase for ${MODEL} ==="
source "${SVEN_ROOT}/.venv/bin/activate"

cd "${SVEN_ROOT}/scripts"

# sec_eval (generation only, no CodeQL yet)
bash eval.sh "${MODEL}" trained --gen_only

# HumanEval (both controls)
CKPT="${SVEN_ROOT}/trained/${MODEL}/checkpoint-last"
for ctrl in sec vul; do
    out="human-eval-${MODEL}-${ctrl}"
    echo "--- human_eval gen: ${out} ---"
    python human_eval_gen.py  --model_type prefix --model_dir "${CKPT}" --control "${ctrl}" --output_name "${out}"
    echo "--- human_eval exec: ${out} ---"
    python human_eval_exec.py --output_name "${out}"
done

deactivate

# ---- codeql phase (CPU): switch to codeql_env (Py3.8) -------------------------
echo "=== [$(date)] codeql phase for ${MODEL} ==="
source "${SVEN_ROOT}/codeql_env/bin/activate"

cd "${SVEN_ROOT}/scripts"
bash eval.sh "${MODEL}" trained --codeql_only

deactivate

echo "=== [$(date)] DONE ${MODEL} ==="
