#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --account=EUHPC_D30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0:30:00
#SBATCH --output=outputs/slurm/eval-%j.out
#SBATCH --error=outputs/slurm/eval-%j.err
#
# Run evaluate_all.sh on a compute node so we get:
#   - GPU acceleration for transformer predict (~10× faster than login CPU)
#   - Immunity from login-node CPU time limits
#   - Wandb env vars set so eval metrics land on each model's training run
#
# Submit:
#   sbatch scripts/evaluate_all.sbatch                              # all discovered models
#   sbatch --export=ALL,MODELS="outputs/models/transformer.pt outputs/models/transformer_5000.pt" \
#          scripts/evaluate_all.sbatch                              # explicit list
#   sbatch --export=ALL,EVAL_DIR=outputs/eval_5000 scripts/evaluate_all.sbatch
#
# Expected runtime on an A100: ~1–2 minutes for 4 models.
# Output: outputs/slurm/eval-<JOBID>.out (and matching .err for warnings).

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p outputs/slurm

# ─── HTTP proxy (same as train_transformer.sbatch) ─────────────────────────
export HTTP_PROXY="http://proxyuser:5dd1d2bd00@10.99.0.1:38425"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"

# ─── Wandb env (so score's auto-resume actually logs to the training runs) ─
export WANDB_ENTITY="fedja-bogataj-org"
export WANDB_PROJECT="infineon-track1"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "[$(date '+%H:%M:%S')] job ${SLURM_JOB_ID:-?} on $(hostname)"
echo "[$(date '+%H:%M:%S')] cwd:  $(pwd)"
echo "[$(date '+%H:%M:%S')] eval: ${EVAL_DIR:-outputs/eval} (override with --export=ALL,EVAL_DIR=...)"

# Pass MODELS through if set (space-separated list); otherwise evaluate_all.sh
# auto-discovers everything under outputs/models/.
if [[ -n "${MODELS:-}" ]]; then
    # shellcheck disable=SC2086
    ./scripts/evaluate_all.sh $MODELS
else
    ./scripts/evaluate_all.sh
fi

echo "[$(date '+%H:%M:%S')] ✓ eval complete"
