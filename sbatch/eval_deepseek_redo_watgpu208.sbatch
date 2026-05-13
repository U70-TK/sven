#!/bin/bash
#SBATCH --job-name=sven-eval-deepseek-redo
#SBATCH --partition=ALL
#SBATCH --nodelist=watgpu208
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/u901/t577wang/sven/sbatch/logs/%x-%j.out
#SBATCH --error=/u901/t577wang/sven/sbatch/logs/%x-%j.err

# Re-runs both DeepSeek-Coder prefix models after the tokenizer fix in
# sven/model.py (PreTrainedTokenizerFast load path). Old corrupt samples were
# wiped from experiments/sec_eval/ and experiments/human_eval/ before submit.
#
# No --dependency is set: Slurm will just hold this in PENDING until a GPU on
# watgpu208 becomes free (one of sven-eval-{a,b,c,d} finishes).

export SVEN_ROOT=/u901/t577wang/sven
source "${SVEN_ROOT}/.venv/bin/activate"
source "${SVEN_ROOT}/sbatch/_eval_lib.sh"

nvidia-smi || true

echo "=== [$(date)] sanity: confirming patched tokenizer load path ==="
python - <<'PY'
import sys, os
sys.path.insert(0, os.environ["SVEN_ROOT"])
from sven.model import _load_tokenizer
for m in ("deepseek-coder-1.3b", "deepseek-coder-6.7b"):
    ckpt = f"{os.environ['SVEN_ROOT']}/trained/{m}-prefix/checkpoint-last"
    tok = _load_tokenizer(ckpt)
    text = "from flask import Flask\n"
    rt = tok.decode(tok(text)["input_ids"], skip_special_tokens=True) == text
    print(f"  {m}: class={type(tok).__name__} roundtrip_ok={rt}")
    assert rt, f"tokenizer round-trip failed for {m} -- aborting"
PY

eval_model deepseek-coder-1.3b-prefix
eval_model deepseek-coder-6.7b-prefix

echo "=== [$(date)] DONE deepseek redo ==="
