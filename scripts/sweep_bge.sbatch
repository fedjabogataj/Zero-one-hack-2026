#!/bin/bash
#SBATCH --job-name=sweep-bge
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --account=EUHPC_D30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --array=0-8
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#
# ───────────────────────────────────────────────────────────────────────────────
# Hyperparameter sweep over the flat BGE-init transformer.
#
# Design: 9-run L9 orthogonal (Taguchi) array over three factors. Each level
# of each factor appears exactly 3 times, balanced across the others, so we
# can estimate main effects with 1/3 of the runs a full 3×3×3 grid would
# need. Trade-off: cannot resolve interactions — only main effects. That's
# fine here; we want "which LR is best, which depth is best, which reg is
# best" not "does deep+high-LR interact specially."
#
# Factors (3 levels each):
#   LR            ∈ {1e-4, 3e-4, 1e-3}
#   N_LAYERS      ∈ {4, 6, 8}
#   (DROPOUT, WEIGHT_DECAY) treated as a single 3-level regularisation factor:
#                 {(0.1, 0.0), (0.2, 0.0), (0.1, 0.01)}
#
# Everything else is held fixed:
#   ENCODER=bge     →  BAAI/bge-base-en-v1.5 (768-dim)
#   D_MODEL=768, N_HEADS=12, FF_DIM=3072  →  matches BGE-base init exactly,
#                                            no projection layer needed
#   DATA_SIZE=5000, EPOCHS=30, BATCH_SIZE=64
#
# Grid CSV is encoder-agnostic (LR / N_LAYERS / DROPOUT / WEIGHT_DECAY), so we
# reuse the same L9 array as the MiniLM sweep.
#
# Submit:
#   sbatch scripts/sweep_bge.sbatch                           # all 9 runs
#   sbatch --array=0-2 scripts/sweep_bge.sbatch               # just runs 0-2
#   sbatch --array=4,7 scripts/sweep_bge.sbatch               # just runs 4 and 7
#
# Each array task runs as an independent SLURM job that may be queued, run,
# or cancelled independently. Outputs go to:
#   outputs/slurm/sweep-bge-<ARRAY_JOB_ID>_<TASK_ID>.out
#   outputs/models/transformer_5000_bge-r<IDX>.pt          ← one per task
#
# ⚠ WARNING about MODEL_OUT collision:
# All 9 runs would otherwise write to the same MODEL_OUT path. To keep them
# separate, we override MODEL_OUT per-task by exporting a per-task ENCODER tag
# like "bge-r0", "bge-r1", etc. This gives us 9 distinct .pt files AND 9
# distinct st_step_embeddings caches — BUT the embeddings cache is identical
# content across runs (same model, same descs), so we use a single shared
# symlink to avoid 9x download.
# ───────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

GRID=scripts/sweep_grids/minilm_l9.csv
IDX="${SLURM_ARRAY_TASK_ID:-0}"

# Read row (header is line 1, so row N lives on line N+2).
LINE=$(awk -F, -v n=$((IDX + 2)) 'NR==n' "$GRID")
if [[ -z "$LINE" ]]; then
    echo "ERROR: no grid row at index $IDX in $GRID" >&2
    exit 2
fi

# Parse the CSV row → env vars expected by _train_pipeline.sh.
IFS=',' read -r LR N_LAYERS DROPOUT WEIGHT_DECAY <<<"$LINE"

# Fixed for the whole sweep.
export ENCODER="bge-r${IDX}"             # per-task tag → unique MODEL_OUT
export ST_ENCODER_NAME="BAAI/bge-base-en-v1.5"
export D_MODEL=768
export N_HEADS=12
export FF_DIM=3072
export DATA_SIZE=5000
export EPOCHS=30
export BATCH_SIZE=64

# Per-task knobs from the grid CSV.
export LR
export N_LAYERS
export DROPOUT
export WEIGHT_DECAY

# Self-describing wandb name for the sweep — overrides _train_pipeline's
# auto-name with one that puts the run index up front for sortability.
export WANDB_RUN_NAME="sweep-bge-L9r${IDX}-lr${LR}-L${N_LAYERS}-do${DROPOUT}-wd${WEIGHT_DECAY}-job${SLURM_ARRAY_JOB_ID:-?}_${IDX}"
export WANDB_TAGS="sweep,sweep-bge-l9,run-${IDX}"

# Pre-stage ONE shared BGE embeddings cache so all 9 array tasks can reuse it
# instead of each running stage 4 in parallel and stomping the file.
# The per-task st_step_embeddings_bge-r${IDX}.pkl files will all symlink
# to the canonical one.
SHARED_EMB=outputs/models/st_step_embeddings_bge.pkl
PER_TASK_EMB="outputs/models/st_step_embeddings_${ENCODER}.pkl"
if [[ -f "$SHARED_EMB" && ! -e "$PER_TASK_EMB" ]]; then
    ln -s "$(basename "$SHARED_EMB")" "$PER_TASK_EMB"
fi

echo "[$(date '+%H:%M:%S')] sweep run $IDX/8  →  LR=$LR  N_LAYERS=$N_LAYERS  DROPOUT=$DROPOUT  WEIGHT_DECAY=$WEIGHT_DECAY"

exec bash scripts/_train_pipeline.sh
