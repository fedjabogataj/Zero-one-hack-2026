#!/usr/bin/env bash
# Produce organizer-ready submission CSVs from one or more trained models.
#
# Unlike evaluate_all.sh — which runs predict + score against our local
# canonical test set (we have ground truth there) — this script ONLY runs
# predict, against the organizer-provided eval files at official_eval/.
# The organizers do not share ground truth for those, so there is nothing
# to score locally; the only deliverable is the three submission CSVs.
#
# Usage:
#   ./scripts/make_submission.sh                            # all transformer*.pt + ngram*.pkl models
#   ./scripts/make_submission.sh outputs/models/foo.pt      # one specific model
#   OFFICIAL_DIR=other_dir ./scripts/make_submission.sh     # override eval source
#
# Output per model:
#   outputs/submissions/<tag>/official/task1.csv     # next-step (600 rows)
#   outputs/submissions/<tag>/official/task2.csv     # complete  (600 rows)
#   outputs/submissions/<tag>/official/task3.csv     # anomaly   (987 rows)
#
# After each model, the script verifies row counts and column headers match
# the spec and fails loudly if they don't (caught before you upload, not after).

set -euo pipefail
shopt -s nullglob

# ─── Resolve project root + runner ─────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if command -v pixi >/dev/null 2>&1 && [[ -d .pixi || -f pixi.toml ]]; then
    INFINEON=(pixi run infineon-baseline)
elif [[ -d .venv ]] && [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    INFINEON=(infineon-baseline)
else
    INFINEON=(infineon-baseline)
fi

# ─── Eval source ───────────────────────────────────────────────────────────
OFFICIAL_DIR="${OFFICIAL_DIR:-official_eval}"
EVAL_VALID="$OFFICIAL_DIR/eval_input_valid.csv"
EVAL_ANOMALY="$OFFICIAL_DIR/eval_input_anomaly.csv"

for f in "$EVAL_VALID" "$EVAL_ANOMALY"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: $f not found." >&2
        echo "  These are the organizer-provided eval files. They should live in" >&2
        echo "  $OFFICIAL_DIR/ (committed to the repo)." >&2
        exit 2
    fi
done

# Expected row counts straight from the organizer spec (also documented in CLAUDE.md):
#   - eval_input_valid.csv:   100 seqs × 3 families × {0.6, 0.8} = 600 rows
#   - eval_input_anomaly.csv: 987 unlabeled valid + invalid sequences
EXPECTED_VALID=$(($(wc -l < "$EVAL_VALID") - 1))
EXPECTED_ANOMALY=$(($(wc -l < "$EVAL_ANOMALY") - 1))

# ─── Collect models ────────────────────────────────────────────────────────
# Exclude *.best.pt — the loader auto-prefers them when the main .pt is given.
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

echo "[$(date '+%H:%M:%S')] producing submissions for ${#MODELS[@]} model(s)"
echo "[$(date '+%H:%M:%S')]   eval source : $OFFICIAL_DIR/"
echo "[$(date '+%H:%M:%S')]   expected    : valid=$EXPECTED_VALID rows, anomaly=$EXPECTED_ANOMALY rows"
echo

# ─── Sanity-check helper (column header + row count) ───────────────────────
_check_submission() {
    local file="$1" expected_rows="$2" expected_cols="$3" label="$4"
    if [[ ! -f "$file" ]]; then
        echo "    ✗ $label: $file was not written" >&2
        return 1
    fi
    local actual_rows actual_cols
    actual_rows=$(($(wc -l < "$file") - 1))
    actual_cols=$(head -1 "$file")
    if [[ "$actual_rows" != "$expected_rows" || "$actual_cols" != "$expected_cols" ]]; then
        echo "    ✗ $label  EXPECTED rows=$expected_rows cols=$expected_cols"
        echo "                ACTUAL   rows=$actual_rows cols=$actual_cols"
        return 1
    fi
    echo "    ✓ $label  rows=$actual_rows cols=$actual_cols"
    return 0
}

# ─── Loop ──────────────────────────────────────────────────────────────────
N_OK=0
N_FAIL=0
for MODEL in "${MODELS[@]}"; do
    if [[ ! -f "$MODEL" ]]; then
        echo "⚠ skipping $MODEL (not found)"
        continue
    fi
    BASE=$(basename "$MODEL")
    TAG="${BASE%.*}"
    OUT_DIR="outputs/submissions/$TAG/official"
    mkdir -p "$OUT_DIR"

    if [[ "$MODEL" == *.pt ]]; then
        MODEL_FLAG=(--transformer "$MODEL")
        KIND="transformer"
    else
        MODEL_FLAG=(--model "$MODEL")
        KIND="ngram"
    fi

    echo "=== $TAG  ($KIND) ==="

    "${INFINEON[@]}" predict "${MODEL_FLAG[@]}" \
        --eval-input "$EVAL_VALID" --task next-step \
        --out "$OUT_DIR/task1.csv"

    "${INFINEON[@]}" predict "${MODEL_FLAG[@]}" \
        --eval-input "$EVAL_VALID" --task complete \
        --out "$OUT_DIR/task2.csv"

    "${INFINEON[@]}" predict "${MODEL_FLAG[@]}" \
        --eval-input "$EVAL_ANOMALY" --task anomaly \
        --anomaly-strategy hybrid \
        --out "$OUT_DIR/task3.csv"

    echo "  Validation:"
    set +e
    _check_submission "$OUT_DIR/task1.csv" "$EXPECTED_VALID" \
        "EXAMPLE_ID,RANK_1,RANK_2,RANK_3,RANK_4,RANK_5"   "task1"
    R1=$?
    _check_submission "$OUT_DIR/task2.csv" "$EXPECTED_VALID" \
        "EXAMPLE_ID,PREDICTED_SEQUENCE"                    "task2"
    R2=$?
    _check_submission "$OUT_DIR/task3.csv" "$EXPECTED_ANOMALY" \
        "EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE"         "task3"
    R3=$?
    set -e

    if (( R1 == 0 && R2 == 0 && R3 == 0 )); then
        N_OK=$((N_OK + 1))
        echo "  ✓ ready to upload: $OUT_DIR/"
    else
        N_FAIL=$((N_FAIL + 1))
        echo "  ✗ submission INVALID for $TAG — fix before uploading" >&2
    fi
    echo
done

echo "[$(date '+%H:%M:%S')] done.  ✓ $N_OK ok    ✗ $N_FAIL invalid"
[[ $N_FAIL -gt 0 ]] && exit 1
exit 0
