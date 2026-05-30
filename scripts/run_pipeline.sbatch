#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of `srun` tasks executed per node
#SBATCH --gpus-per-task=1 # Number of GPUs (up to 4 on Leonardo)
#SBATCH --mem=120GB # Fair share on Leonardo: 120GB * gpus-per-task
#SBATCH --cpus-per-task=8 # Fair share on Leonardo: 8 * gpus-per-task
#SBATCH --time=0:30:00 # Time limit in HH:MM:SS, up to 24:00:00
#SBATCH --account=EUHPC_D30_031
# ─────────────────────────────────────────────────────────────────────────────
#
# Submit with:
#   sbatch scripts/run_pipeline.sbatch
#
# Override defaults via --export:
#   sbatch --export=ALL,SEED=7,ORDER=4 scripts/run_pipeline.sbatch
#
# NOTE: the n-gram baseline is CPU-only and runs in ~10 seconds end-to-end.
# This sbatch is overkill for the baseline alone, but serves as the scaffold
# for the eventual transformer training job (where it WILL matter).

set -euo pipefail

# Resolve project root.
# When SLURM runs the job, it executes a COPY of this script from a temp dir
# (typically /var/spool/slurm/job.<id>/slurm_script), so BASH_SOURCE points
# there instead of the real script location — making a $SCRIPT_DIR/.. lookup
# resolve to /var/spool/slurm and fail with "Permission denied" on mkdir.
# Use SLURM_SUBMIT_DIR (the directory `sbatch` was run from) when present,
# otherwise fall back to the BASH_SOURCE-based resolution for direct execution.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

mkdir -p outputs/slurm

# ─── Environment setup ───────────────────────────────────────────────────────
# On Leonardo, prefer a module-loaded Python over system Python. Adjust to whatever
# module is current on the cluster when you run this.
#
# module load python/3.11
# module load gcc                       # often needed for numpy/scipy wheels
#
# Activate the venv (create with `python -m venv .venv && pip install -e .[dev]`):
if [[ ! -d .venv ]]; then
    echo "ERROR: .venv/ not found — create it first with:"
    echo "  python -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'"
    exit 1
fi
source .venv/bin/activate

echo "JOB:      ${SLURM_JOB_ID:-(no SLURM)}"
echo "HOST:     $(hostname)"
echo "CWD:      $(pwd)"
echo "PYTHON:   $(python --version)"
echo "PACKAGE:  $(pip show infineon-baseline | grep -E '^(Name|Version|Location)' | tr '\n' ' ')"
echo

# ─── Run the canonical pipeline ──────────────────────────────────────────────
exec ./scripts/run_pipeline.sh
