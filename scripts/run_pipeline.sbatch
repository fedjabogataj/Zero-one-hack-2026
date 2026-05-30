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
# Make sure user-installed tools (pixi, etc.) are on PATH on the compute node.
# Compute nodes don't always read ~/.bashrc, so we add pixi's install dir explicitly.
export PATH="$HOME/.pixi/bin:$PATH"

# Decide which environment manager to use. Prefer pixi (created with
# `pixi init --import pyproject.toml && pixi install && pixi run pip install -e .`);
# fall back to a classic pip+venv setup if no pixi env exists.
if command -v pixi >/dev/null 2>&1 && [[ -d .pixi || -f pixi.toml ]]; then
    RUNNER=(pixi run ./scripts/run_pipeline.sh)
    PY_INFO="$(pixi run python --version 2>&1)"
    PKG_INFO="$(pixi run pip show infineon-baseline 2>/dev/null | grep -E '^(Name|Version|Location)' | tr '\n' ' ')"
elif [[ -d .venv ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    RUNNER=(./scripts/run_pipeline.sh)
    PY_INFO="$(python --version 2>&1)"
    PKG_INFO="$(pip show infineon-baseline 2>/dev/null | grep -E '^(Name|Version|Location)' | tr '\n' ' ')"
else
    echo "ERROR: no Python environment found. Set up ONE of:"
    echo "  pixi:  pixi init --import pyproject.toml && pixi install && pixi run pip install -e ."
    echo "  venv:  python -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'"
    exit 1
fi

echo "JOB:      ${SLURM_JOB_ID:-(no SLURM)}"
echo "HOST:     $(hostname)"
echo "CWD:      $(pwd)"
echo "RUNNER:   ${RUNNER[*]}"
echo "PYTHON:   $PY_INFO"
echo "PACKAGE:  $PKG_INFO"
echo

# ─── Run the canonical pipeline ──────────────────────────────────────────────
exec "${RUNNER[@]}"
