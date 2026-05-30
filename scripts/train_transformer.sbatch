#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --account=EUHPC_D30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p outputs/slurm outputs/models

# ─── HTTP proxy for compute-node internet access ────────────────────────────
# Compute nodes on Leonardo are firewalled — route through the hackathon-
# provided HTTP proxy. Credentials are hackathon-shared (provisioned by the
# organizers, expected to rotate). The proxy restarts roughly every 10 minutes
# and may drop long-lived connections — huggingface_hub and pip both retry.
export HTTP_PROXY="http://proxyuser:5dd1d2bd00@10.99.0.1:38425"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"

# ─── Weights & Biases experiment tracking ───────────────────────────────────
# Set WANDB_API_KEY once via `wandb login` on the login node (writes ~/.netrc,
# which compute nodes can read). If the proxy is flaky, override to offline:
#   export WANDB_MODE="offline"
# then sync from the login node after the job with `wandb sync .wandb/...`.
export WANDB_ENTITY="fedja-bogataj-org"
export WANDB_PROJECT="infineon-track1"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "[$(date '+%H:%M:%S')] job ${SLURM_JOB_ID:-?} on $(hostname)"
echo "[$(date '+%H:%M:%S')] cwd: $(pwd)"

# ─── Stage 1 of 3: build-eval (skipped if up-to-date) ───────────────────────
if [[ ! -f outputs/eval/train_split.csv ]]; then
    echo "[$(date '+%H:%M:%S')] stage 1/3 — build-eval"
    pixi run infineon-baseline --seed 42 build-eval \
        --variants-dir training_data/ --out outputs/eval/
else
    echo "[$(date '+%H:%M:%S')] stage 1/3 — build-eval: skipped (train_split.csv exists)"
fi

# ─── Stage 2 of 3: build-st-embeddings (skipped if up-to-date) ──────────────
if [[ ! -f outputs/models/st_step_embeddings.pkl ]]; then
    echo "[$(date '+%H:%M:%S')] stage 2/3 — build-st-embeddings (downloads ST model via proxy)"
    pixi run infineon-baseline build-st-embeddings \
        --descriptions-dir training_data/ \
        --out outputs/models/st_step_embeddings.pkl
else
    echo "[$(date '+%H:%M:%S')] stage 2/3 — build-st-embeddings: skipped (pkl exists)"
fi

# ─── Stage 3 of 3: transformer training ─────────────────────────────────────
echo "[$(date '+%H:%M:%S')] stage 3/3 — train transformer (30 epochs, batch=64)"
pixi run infineon-baseline --seed 42 train \
    --train outputs/eval/train_split.csv \
    --embeddings outputs/models/st_step_embeddings.pkl \
    --out outputs/models/transformer.pt \
    --epochs 30 --batch-size 64 --lr 3e-4 --device cuda

echo "[$(date '+%H:%M:%S')] ✓ all stages complete"
