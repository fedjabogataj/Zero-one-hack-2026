#!/usr/bin/env bash
# Evaluate one or more trained models on the canonical eval set, then print a
# side-by-side comparison table of their headline metrics.
#
# Usage:
#   ./scripts/evaluate_all.sh                                 # auto-discovers all models
#   ./scripts/evaluate_all.sh outputs/models/transformer.pt outputs/models/transformer_5000.pt
#   EVAL_DIR=outputs/eval_5000 ./scripts/evaluate_all.sh      # override eval set
#
# Eval set defaults to outputs/eval/ (the original 1k-derived set) so
# comparisons across different data-size models are apples-to-apples — every
# model is scored on the same held-out sequences.
#
# Per model, this script:
#   1. runs `predict` for next-step, complete, anomaly (writes submission CSVs)
#   2. runs `score` for each (writes JSON reports, auto-resumes the matching
#      wandb run via the .meta.json sibling)
#   3. organises outputs under outputs/{submissions,reports}/<tag>/
# Then it parses every report.json into a single comparison table.

set -euo pipefail
shopt -s nullglob   # let globs expand to empty if no match (instead of literal string)

# ─── Resolve project root + runner ─────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if command -v pixi >/dev/null 2>&1 && [[ -d .pixi || -f pixi.toml ]]; then
    INFINEON=(pixi run infineon-baseline)
    PYTHON=(pixi run python)
elif [[ -d .venv ]] && [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    INFINEON=(infineon-baseline)
    PYTHON=(python)
else
    INFINEON=(infineon-baseline)
    PYTHON=(python)
fi

# ─── Eval set ──────────────────────────────────────────────────────────────
# Default: prefer the canonical (original 1000/family) test set if it exists,
# else fall back to outputs/eval/ for backward compatibility with the older flow.
if [[ -z "${EVAL_DIR:-}" ]]; then
    if [[ -f "outputs/eval_canonical/eval_input_valid.csv" ]]; then
        EVAL_DIR="outputs/eval_canonical"
    else
        EVAL_DIR="outputs/eval"
    fi
fi
EVAL_VALID="$EVAL_DIR/eval_input_valid.csv"
EVAL_ANOMALY="$EVAL_DIR/eval_input_anomaly.csv"
GT_VALID="$EVAL_DIR/ground_truth_valid.csv"
GT_ANOMALY="$EVAL_DIR/ground_truth_anomaly.csv"

for f in "$EVAL_VALID" "$EVAL_ANOMALY" "$GT_VALID" "$GT_ANOMALY"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: $f not found. Run build-eval first." >&2
        exit 2
    fi
done

# ─── Collect models ────────────────────────────────────────────────────────
# Note: when auto-discovering, we EXCLUDE *.best.pt files. Those are
# inference-only sibling checkpoints that TransformerPredictor.load() picks
# up automatically when given the main MODEL.pt path — listing both here
# would double-evaluate the same run.
if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=()
    for p in outputs/models/transformer*.pt outputs/models/ngram*.pkl; do
        [[ -e "$p" ]] || continue
        [[ "$p" == *.best.pt ]] && continue
        MODELS+=("$p")
    done
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: no models found in outputs/models/. Train one first or pass paths explicitly." >&2
    exit 2
fi

echo "[$(date '+%H:%M:%S')] evaluating ${#MODELS[@]} model(s) against $EVAL_DIR/"
echo

# ─── Loop: predict + score per model ───────────────────────────────────────
TAGS=()
for MODEL in "${MODELS[@]}"; do
    if [[ ! -f "$MODEL" ]]; then
        echo "⚠ skipping $MODEL (not found)"; continue
    fi
    BASE=$(basename "$MODEL")
    TAG="${BASE%.*}"
    TAGS+=("$TAG")

    SUB_DIR="outputs/submissions/$TAG"
    REP_DIR="outputs/reports/$TAG"
    mkdir -p "$SUB_DIR" "$REP_DIR"

    echo "──────────────────────────────────────────────────────────────────"
    echo " $TAG  ($MODEL)"
    echo "──────────────────────────────────────────────────────────────────"

    if [[ "$MODEL" == *.pt ]]; then
        MODEL_FLAGS=(--transformer "$MODEL")
    else
        MODEL_FLAGS=(--model "$MODEL")
    fi

    echo "[$(date '+%H:%M:%S')] predict: next-step"
    "${INFINEON[@]}" --seed 42 predict "${MODEL_FLAGS[@]}" \
        --eval-input "$EVAL_VALID" --task next-step \
        --out "$SUB_DIR/task1.csv"
    echo "[$(date '+%H:%M:%S')] predict: complete"
    "${INFINEON[@]}" --seed 42 predict "${MODEL_FLAGS[@]}" \
        --eval-input "$EVAL_VALID" --task complete \
        --out "$SUB_DIR/task2.csv"
    echo "[$(date '+%H:%M:%S')] predict: anomaly"
    "${INFINEON[@]}" --seed 42 predict "${MODEL_FLAGS[@]}" \
        --eval-input "$EVAL_ANOMALY" --task anomaly \
        --out "$SUB_DIR/task3.csv" --threshold -100.0

    # Score: descriptive wandb run name so n-gram-style models (no training
    # run to auto-resume) still get readable labels on the wandb dashboard.
    # When the predict step found a wandb_run_id in the checkpoint, score
    # auto-resumes that run instead and ignores --wandb-run-name.
    SCORE_NAME_BASE="eval-${TAG}"

    echo "[$(date '+%H:%M:%S')] score: next-step"
    "${INFINEON[@]}" score \
        --predictions "$SUB_DIR/task1.csv" --ground-truth "$GT_VALID" \
        --task next-step --report-json "$REP_DIR/task1.json" \
        --wandb-run-name "${SCORE_NAME_BASE}-next-step"
    echo "[$(date '+%H:%M:%S')] score: complete"
    "${INFINEON[@]}" score \
        --predictions "$SUB_DIR/task2.csv" --ground-truth "$GT_VALID" \
        --task complete --report-json "$REP_DIR/task2.json" \
        --wandb-run-name "${SCORE_NAME_BASE}-complete"
    echo "[$(date '+%H:%M:%S')] score: anomaly"
    "${INFINEON[@]}" score \
        --predictions "$SUB_DIR/task3.csv" --ground-truth "$GT_ANOMALY" \
        --task anomaly --report-json "$REP_DIR/task3.json" \
        --wandb-run-name "${SCORE_NAME_BASE}-anomaly"
    echo
done

# ─── Comparison table ──────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════════"
echo "MODEL COMPARISON SUMMARY  (eval set: $EVAL_DIR/)"
echo "═══════════════════════════════════════════════════════════════════════"

"${PYTHON[@]}" - "${TAGS[@]}" <<'PY'
import json, sys
from pathlib import Path

tags = sys.argv[1:]
task_to_file = {"next-step": "task1", "complete": "task2", "anomaly": "task3"}

# Headline metrics per task; "↓" means lower is better.
headline = {
    "next-step": [
        ("top_1_accuracy", "↑"),
        ("top_3_accuracy", "↑"),
        ("top_5_accuracy", "↑"),
        ("mrr",            "↑"),
    ],
    "complete":  [
        ("exact_match_rate",          "↑"),
        ("normalized_edit_distance",  "↓"),
        ("token_accuracy",            "↑"),
        ("block_accuracy",            "↑"),
    ],
    "anomaly":   [
        ("binary_accuracy",            "↑"),
        ("f1",                         "↑"),
        ("roc_auc",                    "↑"),
        ("rule_attribution_accuracy",  "↑"),
    ],
}

def best_idx(values, direction):
    """Return the index of the best numeric value; None if all blank."""
    nums = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
    if not nums:
        return None
    return min(nums, key=lambda x: x[1])[0] if direction == "↓" else max(nums, key=lambda x: x[1])[0]

for task, metrics in headline.items():
    print(f"\nTask: {task}")
    header = ["metric"] + list(tags)
    rows = [header]
    for metric_name, direction in metrics:
        raw_values = []
        for tag in tags:
            path = Path("outputs/reports") / tag / f"{task_to_file[task]}.json"
            if not path.exists():
                raw_values.append(None)
                continue
            data = json.loads(path.read_text())
            raw_values.append(data.get("overall", {}).get(metric_name))
        # Find best to bold/highlight
        b_idx = best_idx(raw_values, direction)
        cells = [f"{metric_name} ({direction})"]
        for i, v in enumerate(raw_values):
            if isinstance(v, (int, float)):
                s = f"{v:.4f}"
                if i == b_idx and len(raw_values) > 1:
                    s = f"*{s}*"   # highlight best with asterisks
            else:
                s = "-"
            cells.append(s)
        rows.append(cells)

    widths = [max(len(str(r[i])) for r in rows) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*rows[0]))
    print("  ".join("-" * w for w in widths))
    for r in rows[1:]:
        print(fmt.format(*r))
PY

echo
echo "[$(date '+%H:%M:%S')] ✓ done — *value* marks the best model per metric (ties go to first)"
echo "    Submissions: outputs/submissions/<model>/"
echo "    Reports:     outputs/reports/<model>/"
echo "    Wandb:       check the wandb project for live charts + eval metrics"
