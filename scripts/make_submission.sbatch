#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --account=EUHPC_D30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0:15:00
#SBATCH --output=outputs/slurm/submit-%j.out
#SBATCH --error=outputs/slurm/submit-%j.err
#
# GPU wrapper around scripts/make_submission.sh.
#
# Usage:
#   sbatch scripts/make_submission.sbatch                              # all eligible models
#   sbatch --export=ALL,MODELS="outputs/models/foo.pt" scripts/make_submission.sbatch
#   sbatch --export=ALL,MODELS="path1.pt path2.pt" scripts/make_submission.sbatch
#
# Outputs per model:
#   outputs/submissions/<tag>/official/task{1,2,3}.csv  (ready to upload)
# Plus the validation checks (column header + row count) — fails if any task
# CSV doesn't match the spec, so a broken submission can't slip through.

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p outputs/slurm

# Pass any explicitly listed MODELS as positional args; otherwise let
# make_submission.sh auto-discover.
if [[ -n "${MODELS:-}" ]]; then
    # shellcheck disable=SC2086
    exec bash scripts/make_submission.sh $MODELS
else
    exec bash scripts/make_submission.sh
fi
