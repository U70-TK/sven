#!/bin/bash
# Usage:
#   bash eval.sh <output_name> [eval_type] [--gen_only] [--codeql_only] [--base]
#
# --base         : evaluate the base LM (no prefix)
# --gen_only     : generate samples only, skip CodeQL analysis
# --codeql_only  : run CodeQL on existing samples, skip generation (no GPU needed)
#
# Examples:
#   bash eval.sh mistral-7b-prefix trained --gen_only      # GPU: generate only
#   bash eval.sh mistral-7b-prefix trained --codeql_only   # CPU: CodeQL on existing samples
#   bash eval.sh mistral-7b-base trained --base --gen_only # GPU: base model, generate only
#   bash eval.sh mistral-7b-base trained --base --codeql_only # CPU: base model CodeQL

set -e

OUTPUT_NAME=${1:?Usage: bash eval.sh <output_name> [eval_type] [--gen_only] [--codeql_only] [--base]}
EVAL_TYPE="trained"
GEN_ONLY=""
CODEQL_ONLY=""
BASE_MODE=""

for arg in "${@:2}"; do
    case "$arg" in
        --gen_only)    GEN_ONLY="--gen_only" ;;
        --codeql_only) CODEQL_ONLY="--codeql_only" ;;
        --base)        BASE_MODE="1" ;;
        *)             EVAL_TYPE="$arg" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -n "$BASE_MODE" ]; then
    MODEL_KEY="${OUTPUT_NAME%-prefix}"
    MODEL_KEY="${MODEL_KEY%-lm}"
    MODEL_KEY="${MODEL_KEY%-base}"
    MODEL_TYPE="lm"
    MODEL_DIR_ARG="--model_dir ${MODEL_KEY}"
    echo "=== Base LM eval: ${OUTPUT_NAME} / ${EVAL_TYPE} ${GEN_ONLY}${CODEQL_ONLY} ==="
else
    TRAINED_DIR="${SCRIPT_DIR}/../trained/${OUTPUT_NAME}/checkpoint-last"
    if [ -z "$CODEQL_ONLY" ] && [ ! -d "$TRAINED_DIR" ]; then
        echo "ERROR: checkpoint not found at $TRAINED_DIR"
        exit 1
    fi
    MODEL_TYPE="prefix"
    MODEL_DIR_ARG="--model_dir ${TRAINED_DIR}"
    echo "=== Prefix eval: ${OUTPUT_NAME} / ${EVAL_TYPE} ${GEN_ONLY}${CODEQL_ONLY} ==="
fi

python sec_eval.py \
    --output_name "${OUTPUT_NAME}" \
    --model_type "${MODEL_TYPE}" \
    ${MODEL_DIR_ARG} \
    --eval_type "${EVAL_TYPE}" \
    ${GEN_ONLY} ${CODEQL_ONLY}

if [ -n "$GEN_ONLY" ]; then
    echo ""
    echo "Samples written to ../experiments/sec_eval/${OUTPUT_NAME}/${EVAL_TYPE}/"
    echo "Run CodeQL on a CPU node with:"
    echo "  bash eval.sh ${OUTPUT_NAME} ${EVAL_TYPE} --codeql_only"
else
    echo ""
    echo "=== Results: ${OUTPUT_NAME} / ${EVAL_TYPE} ==="
    python print_results.py \
        --eval_dir "../experiments/sec_eval/${OUTPUT_NAME}" \
        --eval_type "${EVAL_TYPE}"
fi
