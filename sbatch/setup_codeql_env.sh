#!/bin/bash
# Setup for the CodeQL-pass environment on a cluster that already has CodeQL.
# Run from the project root: bash sbatch/setup_codeql_env.sh
#
# Creates: ./codeql_env/  — Python 3.8 venv with libcst, sven, torch-cpu, etc.
# Assumes `uv` is on PATH. If not: pipx install uv
#
# IMPORTANT: sven hardcodes `../codeql/codeql` (relative to scripts/) when
# invoking CodeQL. Point that at your cluster's CodeQL install before running
# the scan, e.g.:
#     ln -s /path/to/cluster/codeql codeql
#     ./codeql/codeql --version   # should print 2.11.1 for paper fidelity

set -e

SVEN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${SVEN_ROOT}"

# Sanity-check the CodeQL pointer exists
if [ ! -x "./codeql/codeql" ]; then
    echo "WARN: ./codeql/codeql not found." >&2
    echo "      Symlink it before scanning, e.g. ln -s /path/to/codeql codeql" >&2
fi

# ---- Python 3.8 venv ---------------------------------------------------------
if [ ! -d "./codeql_env" ]; then
    echo "=== creating codeql_env (Python 3.8) ==="
    uv venv --python 3.8 codeql_env
else
    echo "=== codeql_env already exists ==="
fi

source ./codeql_env/bin/activate

echo "=== installing python deps ==="
uv pip install \
    'libcst==1.1.0' \
    'pyyaml==6.0.2' \
    'tabulate==0.9.0' \
    'lizard==1.17.10' \
    'diff-match-patch==20230430' \
    'yamlize==0.7.1' \
    'ruamel.yaml==0.17.21' \
    'numpy==1.24.4' \
    'scipy==1.10.1'

# CPU-only torch (sven.utils imports it even on codeql_only paths)
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# sven itself (editable)
uv pip install -e .

# ---- Smoke test --------------------------------------------------------------
echo
echo "=== smoke test ==="
python -c "
import libcst, yamlize, lizard, tabulate, yaml, ruamel.yaml, torch
from sven.metric import SecEval
from sven.utils import set_seed, set_logging, set_devices
from sven.constant import BINARY_LABELS, CWES_DICT
print('imports OK; torch:', torch.__version__)
"
echo "python3 -> $(which python3) ($(python3 --version))"
if [ -x "./codeql/codeql" ]; then
    echo "codeql  -> $(./codeql/codeql --version | head -1)"
fi
echo
echo "DONE. Activate later with:  source codeql_env/bin/activate"
