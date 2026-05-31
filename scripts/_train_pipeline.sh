#!/bin/bash
# ───────────────────────────────────────────────────────────────────────────────
# Shared training pipeline body. Sourced/exec'd by:
#   - scripts/train_transformer.sbatch  (single run)
#   - scripts/sweep_minilm.sbatch       (array sweep)
#
# This script has NO #SBATCH directives — submit one of the wrappers above.
# All knobs are read from env vars; defaults match the previous behavior so
# unchanged callers stay working.
#
# Env vars (all optional):
#   DATA_SIZE        5000     synthetic sequences/family for training
#   EPOCHS           30       training passes
#   BATCH_SIZE       64       per-step batch size
#   LR               3e-4     peak learning rate
#   WEIGHT_DECAY     0.01     AdamW weight decay
#   DROPOUT          (model default 0.1) override dropout rate
#   GEN_SEED         1042     synthetic-data seed (MUST differ from canonical seed)
#   ENCODER          bge      short tag → ST model id (bge|bge-large|minilm|mpnet)
#                             also goes into model filenames and wandb run name
#   ST_ENCODER_NAME  (unset)  escape hatch — pass an arbitrary HF model id verbatim
#   N_LAYERS         (model default) transformer depth
#   D_MODEL          (model default) transformer width
#   N_HEADS          (model default) attention heads
#   FF_DIM           (model default) FFN inner dim
#   WANDB_RUN_NAME   auto     override the auto-generated wandb run name
#   WANDB_MODE       online   set to "offline" if the proxy is flaky
# ───────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p outputs/slurm outputs/models

# ─── HTTP proxy for compute-node internet access ────────────────────────────
export HTTP_PROXY="http://proxyuser:5dd1d2bd00@10.99.0.1:38425"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"

# ─── Weights & Biases experiment tracking ───────────────────────────────────
export WANDB_ENTITY="${WANDB_ENTITY:-fedja-bogataj-org}"
export WANDB_PROJECT="${WANDB_PROJECT:-infineon-track1}"
export WANDB_MODE="${WANDB_MODE:-online}"

# ─── Tunable params + derived paths ─────────────────────────────────────────
DATA_SIZE="${DATA_SIZE:-5000}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
DROPOUT="${DROPOUT:-}"
GEN_SEED="${GEN_SEED:-1042}"
N_LAYERS="${N_LAYERS:-}"
D_MODEL="${D_MODEL:-}"
N_HEADS="${N_HEADS:-}"
FF_DIM="${FF_DIM:-}"
ENCODER="${ENCODER:-bge}"
# Export ENCODER so the python side (embeddings_st._resolve_model_name)
# sees the same value when subprocesses spawn.
export ENCODER

CANONICAL_DATA="training_data"
CANONICAL_TEST="outputs/eval_canonical"
TRAIN_DATA_DIR="training_data_${DATA_SIZE}"
TRAIN_POOL_DIR="outputs/train_pool_${DATA_SIZE}"
# Encoder-tagged artifact paths — different encoders cannot collide on disk.
MODEL_OUT="outputs/models/transformer_${DATA_SIZE}_${ENCODER}.pt"
ST_EMB="outputs/models/st_step_embeddings_${ENCODER}.pkl"

# ─── Self-describing wandb run name ────────────────────────────────────────
if (( DATA_SIZE >= 1000 && DATA_SIZE % 1000 == 0 )); then
    SIZE_LABEL="$((DATA_SIZE / 1000))k"
else
    SIZE_LABEL="$DATA_SIZE"
fi

DEFAULT_RUN_NAME="xfmr-${ENCODER}-train${SIZE_LABEL}-${EPOCHS}ep-bs${BATCH_SIZE}"
[[ -n "$N_LAYERS" ]]              && DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-L${N_LAYERS}"
[[ -n "$D_MODEL"  ]]              && DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-d${D_MODEL}"
[[ "$LR"       != "3e-4" ]]       && DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-lr${LR}"
[[ -n "$DROPOUT" && "$DROPOUT" != "0.1" ]] && DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-do${DROPOUT}"
[[ "$WEIGHT_DECAY" != "0.01" ]]   && DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-wd${WEIGHT_DECAY}"
[[ "$GEN_SEED" != "1042" ]]       && DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-genseed${GEN_SEED}"

# Unique suffix so identical-config resubmissions don't fight over a name.
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-job${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-job${SLURM_JOB_ID}"
else
    DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-$(date +%Y%m%d-%H%M%S)"
fi

RUN_NAME="${WANDB_RUN_NAME:-$DEFAULT_RUN_NAME}"

echo "[$(date '+%H:%M:%S')] job ${SLURM_JOB_ID:-?} on $(hostname)"
echo "[$(date '+%H:%M:%S')] cwd: $(pwd)"
echo "[$(date '+%H:%M:%S')] params: DATA_SIZE=$DATA_SIZE EPOCHS=$EPOCHS BATCH=$BATCH_SIZE LR=$LR WD=$WEIGHT_DECAY DROPOUT=${DROPOUT:-default}"
echo "[$(date '+%H:%M:%S')] arch:   N_LAYERS=${N_LAYERS:-default} D_MODEL=${D_MODEL:-default} N_HEADS=${N_HEADS:-default} FF_DIM=${FF_DIM:-default}"
echo "[$(date '+%H:%M:%S')] encoder: tag=$ENCODER  hf=${ST_ENCODER_NAME:-<from tag map>}"
echo "[$(date '+%H:%M:%S')] paths:"
echo "    TRAIN  data → $TRAIN_DATA_DIR/   (synthetic, seed $GEN_SEED)"
echo "    TRAIN  pool → $TRAIN_POOL_DIR/   (train_split.csv + internal val)"
echo "    TEST       → $CANONICAL_TEST/    (original 1000/family from $CANONICAL_DATA/)"
echo "    MODEL  out → $MODEL_OUT"
echo "[$(date '+%H:%M:%S')] wandb:  run=$RUN_NAME  mode=$WANDB_MODE"

# ─── Stage 1 of 5: generate synthetic TRAINING data ─────────────────────────
if [[ ! -f "$TRAIN_DATA_DIR/MOSFET_variants.csv" ]]; then
    echo "[$(date '+%H:%M:%S')] stage 1/5 — generate $DATA_SIZE synthetic sequences per family (seed $GEN_SEED) → $TRAIN_DATA_DIR/"
    mkdir -p "$TRAIN_DATA_DIR"
    cp training_data/*_longdescription_parameters.csv "$TRAIN_DATA_DIR/" 2>/dev/null || true
    for fam in mosfet igbt ic; do
        FAM_UPPER=$(echo "$fam" | tr '[:lower:]' '[:upper:]')
        pixi run python training_data/generate_sequences.py \
            --family "$fam" --count "$DATA_SIZE" --seed "$GEN_SEED" \
            --output "$TRAIN_DATA_DIR/${FAM_UPPER}_variants.csv"
    done
else
    echo "[$(date '+%H:%M:%S')] stage 1/5 — generate: skipped ($TRAIN_DATA_DIR/MOSFET_variants.csv exists)"
fi

# ─── Stage 2 of 5: build train pool (train_split + small internal val) ──────
if [[ ! -f "$TRAIN_POOL_DIR/train_split.csv" ]]; then
    echo "[$(date '+%H:%M:%S')] stage 2/5 — build train pool from $TRAIN_DATA_DIR → $TRAIN_POOL_DIR"
    pixi run infineon-baseline --seed 42 build-eval \
        --variants-dir "$TRAIN_DATA_DIR/" --out "$TRAIN_POOL_DIR/" \
        --holdout-per-family 200
else
    echo "[$(date '+%H:%M:%S')] stage 2/5 — train pool: skipped ($TRAIN_POOL_DIR/train_split.csv exists)"
fi

# ─── Stage 3 of 5: build CANONICAL test set from the original 1000/family ───
if [[ ! -f "$CANONICAL_TEST/eval_input_valid.csv" ]]; then
    echo "[$(date '+%H:%M:%S')] stage 3/5 — build canonical test from $CANONICAL_DATA → $CANONICAL_TEST"
    pixi run infineon-baseline --seed 42 build-test-only \
        --variants-dir "$CANONICAL_DATA/" --out "$CANONICAL_TEST/"
else
    echo "[$(date '+%H:%M:%S')] stage 3/5 — canonical test: skipped ($CANONICAL_TEST/eval_input_valid.csv exists)"
fi

# ─── Stage 4 of 5: build sentence-transformer step embeddings ───────────────
# Note: filename is encoder-tagged so different encoders coexist on disk
# and are not accidentally reused across runs.
if [[ ! -f "$ST_EMB" ]]; then
    echo "[$(date '+%H:%M:%S')] stage 4/5 — build-st-embeddings (encoder=$ENCODER, downloads ST model via proxy)"
    pixi run infineon-baseline build-st-embeddings \
        --descriptions-dir training_data/ --out "$ST_EMB"
else
    echo "[$(date '+%H:%M:%S')] stage 4/5 — build-st-embeddings: skipped ($ST_EMB exists)"
fi

# ─── Stage 5 of 5: training ─────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] stage 5/5 — train transformer ($EPOCHS epochs, batch=$BATCH_SIZE)"

ARCH_FLAGS=()
[[ -n "$N_LAYERS" ]] && ARCH_FLAGS+=(--n-layers "$N_LAYERS")
[[ -n "$D_MODEL"  ]] && ARCH_FLAGS+=(--d-model  "$D_MODEL")
[[ -n "$N_HEADS"  ]] && ARCH_FLAGS+=(--n-heads  "$N_HEADS")
[[ -n "$FF_DIM"   ]] && ARCH_FLAGS+=(--ff-dim   "$FF_DIM")
[[ -n "$DROPOUT"  ]] && ARCH_FLAGS+=(--dropout  "$DROPOUT")

pixi run infineon-baseline --seed 42 train \
    --train "$TRAIN_POOL_DIR/train_split.csv" \
    --embeddings "$ST_EMB" \
    --out "$MODEL_OUT" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" --device cuda \
    --weight-decay "$WEIGHT_DECAY" \
    --wandb-run-name "$RUN_NAME" \
    "${ARCH_FLAGS[@]}"

echo "[$(date '+%H:%M:%S')] ✓ all stages complete → $MODEL_OUT"
echo "[$(date '+%H:%M:%S')]   Evaluate with:  EVAL_DIR=$CANONICAL_TEST sbatch scripts/evaluate_all.sbatch"
