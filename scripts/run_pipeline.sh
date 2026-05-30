#!/usr/bin/env bash
# Run the full canonical baseline pipeline locally.
#
# Usage:
#   ./scripts/run_pipeline.sh                  # defaults: seed=42, order=3, outputs/
#   SEED=7 ORDER=4 ./scripts/run_pipeline.sh   # override
#   OUT_DIR=/tmp/run1 ./scripts/run_pipeline.sh
#
# Requires: .venv/ created and the package installed in editable mode
#   python -m venv .venv
#   source .venv/bin/activate
#   pip install -e ".[dev]"
set -euo pipefail

# Resolve project root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Parameters (override via env).
SEED="${SEED:-42}"
ORDER="${ORDER:-3}"
HOLDOUT="${HOLDOUT:-200}"
INVALID_RATIO="${INVALID_RATIO:-0.39}"
ANOMALY_STRATEGY="${ANOMALY_STRATEGY:-hybrid}"
THRESHOLD="${THRESHOLD:--100.0}"
OUT_DIR="${OUT_DIR:-outputs}"

# Activate .venv only if (a) it exists AND (b) we're not already inside any env.
# Skips silently when invoked via `pixi run`, where pixi has already set up PATH
# and there's no .venv to activate.
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -d .venv ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Final guard: whatever path we came through, the CLI must be reachable.
if ! command -v infineon-baseline >/dev/null 2>&1; then
    echo "ERROR: infineon-baseline not on PATH. Activate your environment first:" >&2
    echo "  pip+venv: source .venv/bin/activate" >&2
    echo "  pixi:     run via 'pixi run ./scripts/run_pipeline.sh' or 'pixi shell' first" >&2
    exit 1
fi

EVAL_DIR="$OUT_DIR/eval"
MODEL_PATH="$OUT_DIR/models/ngram_o${ORDER}.pkl"
SUB_DIR="$OUT_DIR/submissions"
REP_DIR="$OUT_DIR/reports"
mkdir -p "$EVAL_DIR" "$(dirname "$MODEL_PATH")" "$SUB_DIR" "$REP_DIR"

echo "─── 1/4  build-eval  ─────────────────────────────────────────────────"
infineon-baseline --seed "$SEED" build-eval \
    --variants-dir training_data/ \
    --out "$EVAL_DIR" \
    --holdout-per-family "$HOLDOUT" \
    --anomaly-invalid-ratio "$INVALID_RATIO"

echo "─── 2/4  fit (order=$ORDER) ──────────────────────────────────────────"
infineon-baseline --seed "$SEED" fit \
    --train "$EVAL_DIR/train_split.csv" \
    --out "$MODEL_PATH" \
    --order "$ORDER"

echo "─── 3/4  predict (3 tasks) ───────────────────────────────────────────"
infineon-baseline --seed "$SEED" predict --model "$MODEL_PATH" \
    --eval-input "$EVAL_DIR/eval_input_valid.csv" --task next-step \
    --out "$SUB_DIR/task1.csv"
infineon-baseline --seed "$SEED" predict --model "$MODEL_PATH" \
    --eval-input "$EVAL_DIR/eval_input_valid.csv" --task complete \
    --out "$SUB_DIR/task2.csv"
infineon-baseline --seed "$SEED" predict --model "$MODEL_PATH" \
    --eval-input "$EVAL_DIR/eval_input_anomaly.csv" --task anomaly \
    --anomaly-strategy "$ANOMALY_STRATEGY" --threshold "$THRESHOLD" \
    --out "$SUB_DIR/task3.csv"

echo "─── 4/4  score (3 tasks) ─────────────────────────────────────────────"
infineon-baseline --seed "$SEED" score \
    --predictions "$SUB_DIR/task1.csv" \
    --ground-truth "$EVAL_DIR/ground_truth_valid.csv" \
    --task next-step --report-json "$REP_DIR/task1.json"
infineon-baseline --seed "$SEED" score \
    --predictions "$SUB_DIR/task2.csv" \
    --ground-truth "$EVAL_DIR/ground_truth_valid.csv" \
    --task complete --report-json "$REP_DIR/task2.json"
infineon-baseline --seed "$SEED" score \
    --predictions "$SUB_DIR/task3.csv" \
    --ground-truth "$EVAL_DIR/ground_truth_anomaly.csv" \
    --task anomaly --report-json "$REP_DIR/task3.json"

echo
echo "✓ pipeline complete. Reports → $REP_DIR/"
