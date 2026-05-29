#!/bin/bash
#SBATCH --job-name=infineon-baseline
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=00:15:00              # 15 min walltime — pipeline runs in ~10s; padding for setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# ─── Leonardo-specific (customize for your account) ──────────────────────────
# Uncomment + fill in once you have your CINECA Leonardo allocation details:
##SBATCH --account=<YOUR_ACCOUNT>     # e.g. cin_train_2026
##SBATCH --partition=dcgp_usr_prod    # CPU-only partition; use boost_usr_prod for GPU
##SBATCH --qos=normal
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

# Resolve project root: this script lives in <project>/scripts/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
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
