#!/usr/bin/env bash
# Generate variant CSVs for all three product families (MOSFET, IGBT, IC) at a
# given size, plus copy the *_longdescription_parameters.csv files into the
# output dir so it's self-contained for downstream `build-eval` and
# `build-st-embeddings` invocations.
#
# Usage:
#   ./scripts/generate_data.sh <SIZE>                                 # SEED=42, OUTPUT=training_data_<SIZE>/
#   ./scripts/generate_data.sh <SIZE> <SEED>
#   ./scripts/generate_data.sh <SIZE> <SEED> <OUTPUT_DIR>
#
# Examples:
#   ./scripts/generate_data.sh 5000
#   ./scripts/generate_data.sh 25000 1042
#   ./scripts/generate_data.sh 100000 7 training_data_xl
#
# After it finishes:
#   infineon-baseline build-eval --variants-dir <OUTPUT_DIR>/ --out outputs/eval_<SIZE>/
#
# Or just submit the parameterised sbatch which handles generation itself:
#   sbatch --export=ALL,DATA_SIZE=5000 scripts/train_transformer.sbatch

set -euo pipefail

# ─── Args ──────────────────────────────────────────────────────────────────
SIZE="${1:-}"
if [[ -z "$SIZE" ]]; then
    cat >&2 <<EOF
ERROR: <SIZE> is required.

Usage: $0 <SIZE> [SEED=42] [OUTPUT_DIR=training_data_<SIZE>]

  SIZE         sequences per family (e.g. 5000)
  SEED         seed for generate_sequences.py (default 42)
  OUTPUT_DIR   target directory (default training_data_<SIZE>)

Examples:
  $0 5000
  $0 25000 1042
  $0 100000 7 training_data_xl
EOF
    exit 2
fi

if ! [[ "$SIZE" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SIZE must be a positive integer, got '$SIZE'" >&2
    exit 2
fi

SEED="${2:-42}"
OUTPUT_DIR="${3:-training_data_${SIZE}}"

# ─── Resolve project root and runner ───────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Prefer pixi (cluster setup); fall back to .venv (local dev); finally bare python.
if command -v pixi >/dev/null 2>&1 && [[ -d .pixi || -f pixi.toml ]]; then
    RUN=("pixi" "run" "python")
elif [[ -d .venv ]] && [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    RUN=("python")
else
    RUN=("python")
fi

# ─── Generate ──────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

# Copy description CSVs so the new dir is self-contained — build-st-embeddings
# reads *_longdescription_parameters.csv, and we want all sizes to share the
# same per-step text (the descriptions don't depend on dataset size).
cp training_data/*_longdescription_parameters.csv "$OUTPUT_DIR/" 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] generating $SIZE sequences/family (seed $SEED) → $OUTPUT_DIR/"

for fam in mosfet igbt ic; do
    fam_upper=$(echo "$fam" | tr '[:lower:]' '[:upper:]')
    out_csv="$OUTPUT_DIR/${fam_upper}_variants.csv"
    echo "[$(date '+%H:%M:%S')]   → $fam"
    "${RUN[@]}" training_data/generate_sequences.py \
        --family "$fam" --count "$SIZE" --seed "$SEED" --output "$out_csv"
done

# ─── Summarize what landed on disk ─────────────────────────────────────────
echo
echo "[$(date '+%H:%M:%S')] generated:"
for fam in MOSFET IGBT IC; do
    f="$OUTPUT_DIR/${fam}_variants.csv"
    if [[ -f "$f" ]]; then
        rows=$(($(wc -l < "$f") - 1))   # minus the header
        seqs=$(tail -n +2 "$f" | awk -F, '{print $1}' | sort -u | wc -l | tr -d ' ')
        size_mb=$(du -m "$f" | awk '{print $1}')
        printf "    %-30s  %s rows, %s sequences, %s MB\n" \
            "${fam}_variants.csv" "$rows" "$seqs" "$size_mb"
    fi
done

echo
echo "[$(date '+%H:%M:%S')] ✓ done"
echo
echo "Next steps:"
echo "  Build eval set:"
echo "    infineon-baseline --seed 42 build-eval \\"
echo "        --variants-dir $OUTPUT_DIR/ --out outputs/eval_${SIZE}/"
echo
echo "  Or skip ahead and let the sbatch handle the full chain:"
echo "    sbatch --export=ALL,DATA_SIZE=$SIZE scripts/train_transformer.sbatch"
