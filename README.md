# Industrial AI — Small Transformers for Semiconductor Process Logic

Track 1 ("Learning and Benchmarking Process Logic") submission. A
family-conditioned GPT-style decoder, initialised from
`BAAI/bge-base-en-v1.5` sentence embeddings, trained on synthetic
manufacturing sequences and benchmarked on three tasks: next-step
prediction, sequence completion, and anomaly detection.

**Final submission model:** `transformer_5000_bge-r2`
(LR=1e-4, 8 layers, dropout 0.1, weight_decay 0.01 — picked from an L9
Taguchi sweep by the organizer's own `eval_metrics.py`).

📄 Full writeup, results table, and design decisions: **[REPORT.md](./REPORT.md)**

For the task briefing, see [`TRACK_README.md`](./TRACK_README.md) and
[`Track_industrial_en.md`](./Track_industrial_en.md).
For the design rationale behind every component, see
[`docs/superpowers/specs/2026-05-30-infineon-baseline-design.md`](./docs/superpowers/specs/2026-05-30-infineon-baseline-design.md).

## Install

Python 3.10+. Either dependency manifest works:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .                       # uses pyproject.toml (preferred)
# OR
pip install -r requirements.txt
```

On the Leonardo cluster we use [pixi](https://pixi.sh):

```bash
pixi install
pixi run infineon-baseline --help
```

## Reproduce the final submission

Local run (CPU is fine for everything except training):

```bash
# 1. Generate 5000 synthetic sequences per family (deterministic, seed 1042
#    distinct from the canonical-1000 seed → train and test never overlap)
./scripts/generate_data.sh 5000 1042 training_data_5000/

# 2. Build the held-out canonical TEST set from the organizer-provided
#    training_data/ (1000/family). Never touched by training.
infineon-baseline build-test-only \
    --variants-dir training_data/ \
    --out outputs/eval_canonical/ \
    --seed 42

# 3. Build BGE-base step embeddings (one-time, cached per encoder)
ENCODER=bge infineon-baseline build-st-embeddings \
    --descriptions-dir training_data/ \
    --out outputs/models/st_step_embeddings_bge.pkl

# 4. Train: one of the L9 sweep configs (use index 2 to reproduce bge-r2).
#    Cluster: sbatch --array=2 scripts/sweep_bge.sbatch
#    Local:   pass the exact hyperparameters directly:
infineon-baseline train \
    --train outputs/train_pool_5000/train_split.csv \
    --embeddings outputs/models/st_step_embeddings_bge.pkl \
    --out outputs/models/transformer_5000_bge-r2.pt \
    --epochs 30 --batch-size 64 \
    --lr 1e-4 --n-layers 8 --dropout 0.1 --weight-decay 0.01 \
    --d-model 384 --n-heads 6 --ff-dim 1536 \
    --device cuda --seed 42
#    (Training also writes outputs/models/transformer_5000_bge-r2.best.pt
#     containing the lowest-val-loss weights — auto-preferred at inference.)

# 5. Score every model with the organizer's official scorer
python scripts/score_official.py

# 6. Generate the organizer-format submission CSVs
./scripts/make_submission.sh outputs/models/transformer_5000_bge-r2.pt
cp outputs/submissions/transformer_5000_bge-r2/official/task1.csv nextstep.csv
cp outputs/submissions/transformer_5000_bge-r2/official/task2.csv completion.csv
cp outputs/submissions/transformer_5000_bge-r2/official/task3.csv anomaly.csv
```

Files `nextstep.csv` / `completion.csv` / `anomaly.csv` at the repo root
are the artifacts uploaded to the organizers via the Tally form.

## Cluster training (Leonardo)

Reservation: `s_tra_ncc` · Account: `EUHPC_D30_031` · Partition: `boost_usr_prod`

```bash
# Single model with custom hyperparameters
sbatch --export=ALL,ENCODER=bge,LR=1e-4,N_LAYERS=8,DROPOUT=0.1,WEIGHT_DECAY=0.01 \
       scripts/train_transformer.sbatch

# Full L9 sweep (9 array tasks, ~5h end-to-end across the reservation)
sbatch scripts/sweep_bge.sbatch                  # BGE-init grid
sbatch scripts/sweep_minilm.sbatch               # MiniLM-init grid for comparison

# Evaluate every trained model on the canonical hold-out, with caching
EVAL_DIR=outputs/eval_canonical sbatch scripts/evaluate_all.sbatch

# Re-score every model with the ORGANIZER's eval_metrics.py (CPU, ~10s/model)
sbatch scripts/score_official.sbatch

# Generate organizer-format CSVs from one or more chosen models
sbatch --export=ALL,MODELS="outputs/models/transformer_5000_bge-r2.pt" \
       scripts/make_submission.sbatch
```

## Project layout

```
src/infineon_baseline/   library code (tokenizer, transformer model, train loop,
                         predict/score CLIs, n-gram baseline, anomaly scoring)
training_data/           organizer-provided canonical sequences (1000/family)
                         + grammar definition + validator
training_data_<N>/       generated synthetic sequences (gitignored)
official_eval/           organizer-distributed eval inputs + their eval_metrics.py
                         (used by score_official.py to re-rank)
outputs/                 generated artifacts (gitignored)
  ├ models/              trained checkpoints
  ├ submissions/         per-model predictions
  ├ reports/             per-model scores (internal + official)
  └ slurm/               SLURM job logs
scripts/                 training, evaluation, sweep, submission orchestration
tests/                   unit tests + CLI smoke test
docs/superpowers/        design specs and implementation plans
nextstep.csv             ← final submission file (Task 1)
completion.csv           ← final submission file (Task 2)
anomaly.csv              ← final submission file (Task 3)
REPORT.md                writeup
```

## Reproducibility

Every stochastic operation takes `--seed`. Eval-set construction, anomaly
violation injection, n-gram fitting, transformer initialisation, optimiser
state, and synthetic-data generation are all deterministic given the same
seed. Two runs with the same seed produce byte-identical eval files, models,
submissions, and reports.

For maximum reproducibility of the final submission, we publish:

- **Hyperparameters** (in `REPORT.md` and as a CLI invocation above)
- **Random seeds** (`--seed 42` for training, `--gen-seed 1042` for the
  synthetic data, organizer-provided seeds for the eval splits)
- **Wandb run logs**: https://wandb.ai/fedja-bogataj-org/infineon-track1
- **Per-task official scorer output**: `outputs/reports/transformer_5000_bge-r2/official_*.txt`

## Testing

```bash
pytest -v
```

~60 unit tests + one CLI smoke test. Total runtime < 10 seconds. No GPU
required.

## License

MIT. See [`LICENSE`](./LICENSE).
