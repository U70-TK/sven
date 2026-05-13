#!/bin/bash
#SBATCH --job-name=sven-retrain-deepseek
#SBATCH --partition=ALL
#SBATCH --nodelist=watgpu508
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=20:00:00
#SBATCH --output=/u901/t577wang/sven/sbatch/logs/%x-%j.out
#SBATCH --error=/u901/t577wang/sven/sbatch/logs/%x-%j.err

# Retrains both DeepSeek-Coder prefixes after the tokenizer fix in
# sven/model.py:_load_tokenizer. The old checkpoints showed sec≈vul (no
# steering) because they were trained on the wrong (LlamaTokenizer) id space.

set -e

export SVEN_ROOT=/u901/t577wang/sven
source "${SVEN_ROOT}/.venv/bin/activate"

nvidia-smi || true

echo "=== [$(date)] sanity: tokenizer fix active? ==="
python - <<'PY'
import sys, os
sys.path.insert(0, os.environ["SVEN_ROOT"])
from sven.model import _load_tokenizer
for hf in ("deepseek-ai/deepseek-coder-1.3b-instruct",
           "deepseek-ai/DeepSeek-Coder-6.7B-instruct"):
    tok = _load_tokenizer(hf)
    text = "from flask import Flask\n"
    rt = tok.decode(tok(text)["input_ids"], skip_special_tokens=True) == text
    print(f"  {hf}: {type(tok).__name__}  roundtrip_ok={rt}")
    assert rt, f"FATAL: tokenizer fix not active for {hf}"
PY

cd "${SVEN_ROOT}/scripts"

# 1.3B (~5h on a 7b-class GPU, well under 20h)
echo "=== [$(date)] train deepseek-coder-1.3b-prefix ==="
python train.py \
    --output_name deepseek-coder-1.3b-prefix \
    --pretrain_dir deepseek-coder-1.3b

# 6.7B (heavier; if it OOMs, drop --max_num_tokens or split into separate jobs)
echo "=== [$(date)] train deepseek-coder-6.7b-prefix ==="
python train.py \
    --output_name deepseek-coder-6.7b-prefix \
    --pretrain_dir deepseek-coder-6.7b

echo "=== [$(date)] DONE retrain ==="
