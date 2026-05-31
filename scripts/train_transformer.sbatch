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
# Single-run training. Defers to scripts/_train_pipeline.sh for the actual
# stages — that script reads tunables from env vars, so all the usual
# --export=ALL,KEY=VAL overrides keep working unchanged.
#
# Submit:
#   sbatch scripts/train_transformer.sbatch                              # defaults (bge, 5k, 30 ep)
#   sbatch --export=ALL,DATA_SIZE=25000,EPOCHS=50 scripts/train_transformer.sbatch
#   sbatch --export=ALL,ENCODER=minilm,D_MODEL=384,N_HEADS=6,FF_DIM=1536 \
#          scripts/train_transformer.sbatch                              # MiniLM flat
#
# For grid sweeps see scripts/sweep_minilm.sbatch (uses the same pipeline).
# ───────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec bash scripts/_train_pipeline.sh
