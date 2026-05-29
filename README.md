# Industrial AI Baseline — Process Sequence Harness

A Python library + CLI for the Industrial AI hackathon track (Track 1: "Learning
and Benchmarking Process Logic"). Builds an n-gram statistical baseline for the
three submission tasks (next-step prediction, sequence completion, anomaly
detection) with full self-evaluation against a synthesized hold-out eval set.

For the task briefing, see [`TRACK_README.md`](./TRACK_README.md) and
[`Track_industrial_en.md`](./Track_industrial_en.md).

For the design rationale behind every component, see
[`docs/superpowers/specs/2026-05-30-infineon-baseline-design.md`](./docs/superpowers/specs/2026-05-30-infineon-baseline-design.md).

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## End-to-end usage

The CLI has four subcommands. A canonical run looks like this (all paths
relative to this directory, all commands deterministic at `--seed 42`):

```bash
# 1. Build the held-out eval set (and ground truths) from the training variants.
infineon-baseline build-eval \
    --variants-dir training_data/ \
    --out outputs/eval/ \
    --holdout-per-family 200 \
    --anomaly-invalid-ratio 0.39 \
    --seed 42

# 2. Fit the family-conditioned n-gram on the training split.
infineon-baseline fit \
    --train outputs/eval/train_split.csv \
    --out outputs/models/ngram_o3.pkl \
    --order 3

# 3. Produce one submission file per task.
infineon-baseline predict \
    --model outputs/models/ngram_o3.pkl \
    --eval-input outputs/eval/eval_input_valid.csv \
    --task next-step \
    --out outputs/submissions/task1.csv

infineon-baseline predict \
    --model outputs/models/ngram_o3.pkl \
    --eval-input outputs/eval/eval_input_valid.csv \
    --task complete \
    --out outputs/submissions/task2.csv \
    --constrain                              # optional: rule-aware decoder

infineon-baseline predict \
    --model outputs/models/ngram_o3.pkl \
    --eval-input outputs/eval/eval_input_anomaly.csv \
    --task anomaly \
    --anomaly-strategy hybrid \
    --threshold -100.0 \
    --out outputs/submissions/task3.csv

# 4. Score each submission and dump a JSON report (per-family breakdown).
for t in next-step complete anomaly; do
    case "$t" in
        next-step|complete) gt=outputs/eval/ground_truth_valid.csv ;;
        anomaly)            gt=outputs/eval/ground_truth_anomaly.csv ;;
    esac
    infineon-baseline score \
        --predictions outputs/submissions/${t/next-step/task1}.csv \
        --ground-truth $gt \
        --task $t \
        --report-json outputs/reports/$t.json
done
```

## When the organizers' real eval files arrive

Skip step 1; point `--eval-input` (step 3) at their `eval_input_*.csv`. The
predictor and submission writers are format-identical; the rest of the
pipeline is unchanged.

## Testing

```bash
pytest -v
```

About 60 unit tests + one CLI smoke test. Total runtime < 10 seconds.

## Project layout

```
src/infineon_baseline/   # library code (loaders, tokenizer, ngram, eval_set,
                         # anomaly, predictor, submission, metrics, cli)
tests/                   # mirroring tests + tiny in-memory fixture
training_data/           # original starter data (untouched)
outputs/                 # all generated artifacts (gitignored)
docs/superpowers/        # design specs and implementation plans
```

## Reproducibility

Every stochastic operation takes `--seed`; eval-set construction, anomaly
violation injection, and model fitting are all deterministic given the same
seed. Two runs with the same seed produce byte-identical eval files, models,
submissions, and reports.
