#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --account=EUHPC_D30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#
# ───────────────────────────────────────────────────────────────────────────────
# Train on freshly-generated SYNTHETIC data; test against the CANONICAL set
# (original 1000/family in training_data/). No sequence ever appears in both.
#
# Submit:
#   sbatch scripts/train_transformer.sbatch                              # defaults
#   sbatch --export=ALL,DATA_SIZE=5000 scripts/train_transformer.sbatch
#   sbatch --export=ALL,DATA_SIZE=25000,EPOCHS=50 scripts/train_transformer.sbatch
#   sbatch --export=ALL,DATA_SIZE=5000,N_LAYERS=6,BATCH_SIZE=128 \
#          scripts/train_transformer.sbatch
#
# Tunable env vars (all optional, defaults shown):
#   DATA_SIZE        5000     synthetic sequences per family for training
#   EPOCHS           30       passes over training data
#   BATCH_SIZE       64       per-step batch size
#   LR               3e-4     peak learning rate (linear warmup + cosine decay)
#   GEN_SEED         1042     seed for generate_sequences.py (must differ from the
#                             seed that produced training_data/'s original 1000 to
#                             guarantee no overlap with the test set)
#   N_LAYERS         (model default) override transformer depth
#   D_MODEL          (model default) override transformer width
#   WANDB_RUN_NAME   auto     override the auto-generated run name
#   WANDB_MODE       online   set to "offline" if the proxy is flaky
#
# Test set is FIXED: built once from training_data/ (1000/family, all of them) into
# outputs/eval_canonical/. Reused across every DATA_SIZE — that's the whole point.
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
export WANDB_ENTITY="fedja-bogataj-org"
export WANDB_PROJECT="infineon-track1"
export WANDB_MODE="${WANDB_MODE:-online}"

# ─── Tunable params + derived paths ─────────────────────────────────────────
DATA_SIZE="${DATA_SIZE:-5000}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"
GEN_SEED="${GEN_SEED:-1042}"   # MUST differ from the seed that built training_data/
N_LAYERS="${N_LAYERS:-}"
D_MODEL="${D_MODEL:-}"

CANONICAL_DATA="training_data"                       # the original 1000/family — TEST source
CANONICAL_TEST="outputs/eval_canonical"              # built once from CANONICAL_DATA
TRAIN_DATA_DIR="training_data_${DATA_SIZE}"          # synthetic — TRAIN source
TRAIN_POOL_DIR="outputs/train_pool_${DATA_SIZE}"     # train_split.csv + internal val
MODEL_OUT="outputs/models/transformer_${DATA_SIZE}.pt"
ST_EMB="outputs/models/st_step_embeddings.pkl"

# ─── Self-describing wandb run name ────────────────────────────────────────
# Encodes everything that distinguishes this run from another so the wandb
# Compare view can be scanned at a glance. Format:
#   xfmr-train<SIZE>-<EPOCHS>ep-bs<BATCH_SIZE>[-L<N_LAYERS>][-d<D_MODEL>][-lr<LR>][-genseed<G>]-job<SLURM_ID>
#
# Defaults are omitted, so a vanilla run reads:
#   xfmr-train5k-30ep-bs64-job43099000
#
# Ablations make themselves visible:
#   xfmr-train25k-50ep-bs128-L6-d768-job43099001
#   xfmr-train5k-30ep-bs64-lr1e-3-job43099002
#
# Override entirely with WANDB_RUN_NAME=<whatever> in --export.

# Pretty size label: 5000 → "5k", 25000 → "25k", non-multiples stay as digits.
if (( DATA_SIZE >= 1000 && DATA_SIZE % 1000 == 0 )); then
    SIZE_LABEL="$((DATA_SIZE / 1000))k"
else
    SIZE_LABEL="$DATA_SIZE"
fi

DEFAULT_RUN_NAME="xfmr-train${SIZE_LABEL}-${EPOCHS}ep-bs${BATCH_SIZE}"
[[ -n "$N_LAYERS" ]]              && DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-L${N_LAYERS}"
[[ -n "$D_MODEL"  ]]              && DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-d${D_MODEL}"
[[ "$LR"       != "3e-4" ]]       && DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-lr${LR}"
[[ "$GEN_SEED" != "1042" ]]       && DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-genseed${GEN_SEED}"

# Unique suffix so identical-config resubmissions don't fight over a name.
# SLURM_JOB_ID is set when run under sbatch; otherwise fall back to timestamp.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-job${SLURM_JOB_ID}"
else
    DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-$(date +%Y%m%d-%H%M%S)"
fi

RUN_NAME="${WANDB_RUN_NAME:-$DEFAULT_RUN_NAME}"

echo "[$(date '+%H:%M:%S')] job ${SLURM_JOB_ID:-?} on $(hostname)"
echo "[$(date '+%H:%M:%S')] cwd: $(pwd)"
echo "[$(date '+%H:%M:%S')] params: DATA_SIZE=$DATA_SIZE EPOCHS=$EPOCHS BATCH=$BATCH_SIZE LR=$LR"
echo "[$(date '+%H:%M:%S')] arch:   N_LAYERS=${N_LAYERS:-default} D_MODEL=${D_MODEL:-default}"
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
if [[ ! -f "$ST_EMB" ]]; then
    echo "[$(date '+%H:%M:%S')] stage 4/5 — build-st-embeddings (downloads ST model via proxy)"
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

pixi run infineon-baseline --seed 42 train \
    --train "$TRAIN_POOL_DIR/train_split.csv" \
    --embeddings "$ST_EMB" \
    --out "$MODEL_OUT" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" --device cuda \
    --wandb-run-name "$RUN_NAME" \
    "${ARCH_FLAGS[@]}"

echo "[$(date '+%H:%M:%S')] ✓ all stages complete → $MODEL_OUT"
echo "[$(date '+%H:%M:%S')]   Evaluate with:  EVAL_DIR=$CANONICAL_TEST sbatch scripts/evaluate_all.sbatch"
