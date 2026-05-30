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

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p outputs/slurm outputs/models

# Build eval if not already done (so train_split.csv exists)
if [[ ! -f outputs/eval/train_split.csv ]]; then
    pixi run infineon-baseline --seed 42 build-eval \
        --variants-dir training_data/ --out outputs/eval/
fi

# Build ST embeddings if not already done
if [[ ! -f outputs/models/st_step_embeddings.pkl ]]; then
    pixi run infineon-baseline build-st-embeddings \
        --descriptions-dir training_data/ \
        --out outputs/models/st_step_embeddings.pkl
fi

# Train transformer on the GPU
pixi run infineon-baseline --seed 42 train \
    --train outputs/eval/train_split.csv \
    --embeddings outputs/models/st_step_embeddings.pkl \
    --out outputs/models/transformer.pt \
    --epochs 30 --batch-size 64 --lr 3e-4 --device cuda
