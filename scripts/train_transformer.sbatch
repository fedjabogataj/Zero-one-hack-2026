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
# Submit:
#   sbatch scripts/train_transformer.sbatch                              # defaults (1000/family)
#   sbatch --export=ALL,DATA_SIZE=5000 scripts/train_transformer.sbatch
#   sbatch --export=ALL,DATA_SIZE=25000,EPOCHS=50 scripts/train_transformer.sbatch
#   sbatch --export=ALL,DATA_SIZE=5000,N_LAYERS=6,D_MODEL=384,BATCH_SIZE=128 \
#          scripts/train_transformer.sbatch
#
# Tunable env vars (all optional, defaults shown):
#   DATA_SIZE        1000     sequences per family — picks/generates training_data_<N>/
#   EPOCHS           30       passes over the training data
#   BATCH_SIZE       64       per-step batch size
#   LR               3e-4     peak learning rate (linear warmup + cosine decay)
#   GEN_SEED         42       seed for generate_sequences.py (only used if data not on disk)
#   N_LAYERS         (model default) override transformer depth
#   D_MODEL          (model default) override transformer width
#   WANDB_RUN_NAME   auto     override the auto-generated run name
#   WANDB_MODE       online   set to "offline" if the proxy is flaky
# ───────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p outputs/slurm outputs/models

# ─── HTTP proxy for compute-node internet access ────────────────────────────
# Compute nodes on Leonardo are firewalled — route through the hackathon-
# provided HTTP proxy. Credentials are hackathon-shared and rotate. The proxy
# restarts roughly every 10 minutes; huggingface_hub and pip both retry.
export HTTP_PROXY="http://proxyuser:5dd1d2bd00@10.99.0.1:38425"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"

# ─── Weights & Biases experiment tracking ───────────────────────────────────
# Set WANDB_API_KEY once via `wandb login` on the login node (writes ~/.netrc,
# which compute nodes can read).
export WANDB_ENTITY="fedja-bogataj-org"
export WANDB_PROJECT="infineon-track1"
export WANDB_MODE="${WANDB_MODE:-online}"

# ─── Resolve tunable params + derived paths ─────────────────────────────────
DATA_SIZE="${DATA_SIZE:-1000}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"
GEN_SEED="${GEN_SEED:-42}"
N_LAYERS="${N_LAYERS:-}"
D_MODEL="${D_MODEL:-}"

# DATA_SIZE=1000 reuses the original `training_data/` and `outputs/eval/` so the
# baseline-vs-larger comparison keeps the existing outputs/models/transformer.pt
# stable. Any other size lives under training_data_<N>/ and outputs/eval_<N>/.
if [[ "$DATA_SIZE" == "1000" ]]; then
    DATA_DIR="training_data"
    EVAL_DIR="outputs/eval"
    MODEL_OUT="outputs/models/transformer.pt"
else
    DATA_DIR="training_data_${DATA_SIZE}"
    EVAL_DIR="outputs/eval_${DATA_SIZE}"
    MODEL_OUT="outputs/models/transformer_${DATA_SIZE}.pt"
fi

# Default wandb run name encodes the variant so the dashboard is self-labelled.
DEFAULT_RUN_NAME="xfmr-${DATA_SIZE}seqs-${EPOCHS}ep-b${BATCH_SIZE}"
if [[ -n "$N_LAYERS" ]]; then DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-L${N_LAYERS}"; fi
if [[ -n "$D_MODEL" ]];  then DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-d${D_MODEL}";  fi
RUN_NAME="${WANDB_RUN_NAME:-$DEFAULT_RUN_NAME}"

ST_EMB="outputs/models/st_step_embeddings.pkl"

echo "[$(date '+%H:%M:%S')] job ${SLURM_JOB_ID:-?} on $(hostname)"
echo "[$(date '+%H:%M:%S')] cwd: $(pwd)"
echo "[$(date '+%H:%M:%S')] params: DATA_SIZE=$DATA_SIZE EPOCHS=$EPOCHS BATCH=$BATCH_SIZE LR=$LR"
echo "[$(date '+%H:%M:%S')] arch:   N_LAYERS=${N_LAYERS:-default} D_MODEL=${D_MODEL:-default}"
echo "[$(date '+%H:%M:%S')] paths:  DATA=$DATA_DIR  EVAL=$EVAL_DIR  OUT=$MODEL_OUT"
echo "[$(date '+%H:%M:%S')] wandb:  run=$RUN_NAME  mode=$WANDB_MODE"

# ─── Stage 1 of 4: generate variants (skipped if dir already populated) ─────
if [[ ! -f "$DATA_DIR/MOSFET_variants.csv" ]]; then
    echo "[$(date '+%H:%M:%S')] stage 1/4 — generate $DATA_SIZE sequences per family (seed $GEN_SEED) → $DATA_DIR/"
    mkdir -p "$DATA_DIR"
    # Reuse the description CSVs from the original training_data/ — same per-step text.
    cp training_data/*_longdescription_parameters.csv "$DATA_DIR/" 2>/dev/null || true
    for fam in mosfet igbt ic; do
        FAM_UPPER=$(echo "$fam" | tr '[:lower:]' '[:upper:]')
        pixi run python training_data/generate_sequences.py \
            --family "$fam" --count "$DATA_SIZE" --seed "$GEN_SEED" \
            --output "$DATA_DIR/${FAM_UPPER}_variants.csv"
    done
else
    echo "[$(date '+%H:%M:%S')] stage 1/4 — generate: skipped ($DATA_DIR/MOSFET_variants.csv exists)"
fi

# ─── Stage 2 of 4: build-eval ───────────────────────────────────────────────
if [[ ! -f "$EVAL_DIR/train_split.csv" ]]; then
    echo "[$(date '+%H:%M:%S')] stage 2/4 — build-eval from $DATA_DIR → $EVAL_DIR"
    pixi run infineon-baseline --seed 42 build-eval \
        --variants-dir "$DATA_DIR/" --out "$EVAL_DIR/" \
        --holdout-per-family 200 --anomaly-invalid-ratio 0.39
else
    echo "[$(date '+%H:%M:%S')] stage 2/4 — build-eval: skipped ($EVAL_DIR/train_split.csv exists)"
fi

# ─── Stage 3 of 4: build-st-embeddings ──────────────────────────────────────
# ST embeddings are per-step (vocab), not per-sequence — same .pkl works for any DATA_SIZE.
if [[ ! -f "$ST_EMB" ]]; then
    echo "[$(date '+%H:%M:%S')] stage 3/4 — build-st-embeddings (downloads ST model via proxy)"
    pixi run infineon-baseline build-st-embeddings \
        --descriptions-dir training_data/ --out "$ST_EMB"
else
    echo "[$(date '+%H:%M:%S')] stage 3/4 — build-st-embeddings: skipped ($ST_EMB exists)"
fi

# ─── Stage 4 of 4: training ─────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] stage 4/4 — train transformer ($EPOCHS epochs, batch=$BATCH_SIZE)"

# Build optional arch override flags only when they're set.
ARCH_FLAGS=()
[[ -n "$N_LAYERS" ]] && ARCH_FLAGS+=(--n-layers "$N_LAYERS")
[[ -n "$D_MODEL"  ]] && ARCH_FLAGS+=(--d-model  "$D_MODEL")

pixi run infineon-baseline --seed 42 train \
    --train "$EVAL_DIR/train_split.csv" \
    --embeddings "$ST_EMB" \
    --out "$MODEL_OUT" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" --device cuda \
    --wandb-run-name "$RUN_NAME" \
    "${ARCH_FLAGS[@]}"

echo "[$(date '+%H:%M:%S')] ✓ all stages complete → $MODEL_OUT"
