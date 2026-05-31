#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --account=EUHPC_D30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=0:30:00
#SBATCH --output=outputs/slurm/score-official-%j.out
#SBATCH --error=outputs/slurm/score-official-%j.err
#
# SLURM wrapper for scripts/score_official.py.
#
# What this does:
#   Re-scores every model in outputs/submissions/<tag>/ using the organizer's
#   official_eval/eval_metrics.py (NOT our internal metrics.py). The ranking
#   that comes out of this is the one the organizers will actually use to
#   grade — pick your submission model from THIS table, not the evaluate_all
#   one.
#
# This job is CPU-only (no model inference, no GPU work). The GPU allocation
# is just to stay in the boost_usr_prod partition; if Leonardo lets you run
# CPU-only jobs on another partition, feel free to swap it out. ~10s per
# model, so 27 models is ~5 minutes total.
#
# Submit:
#   sbatch scripts/score_official.sbatch                                       # all models
#   sbatch --export=ALL,TAGS="transformer_5000_bge-r0 transformer_5000_bge-r8" \
#          scripts/score_official.sbatch
#   sbatch --export=ALL,FORCE=1 scripts/score_official.sbatch                  # ignore cache

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p outputs/slurm

FLAGS=()
[[ "${FORCE:-0}" == "1" ]] && FLAGS+=(--force)

if [[ -n "${TAGS:-}" ]]; then
    # shellcheck disable=SC2086
    exec pixi run python scripts/score_official.py "${FLAGS[@]}" $TAGS
else
    exec pixi run python scripts/score_official.py "${FLAGS[@]}"
fi
