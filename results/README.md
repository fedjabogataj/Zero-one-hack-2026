# Archived experiment results

This directory contains the snapshot archive of the **final submitted model**:
`transformer_5000_bge-r2`. Everything else (baseline runs, the other 8 BGE
sweep configs, the 9 MiniLM sweep configs, the un-swept full-dim BGE we
ruled out as overfit) was generated during development and is intentionally
not archived in this repo — see [`REPORT.md`](../REPORT.md) for the full
comparison and design decisions.

## What's archived

| Run | Config | Role |
|---|---|---|
| **[`transformer_5000_bge-r2/`](./transformer_5000_bge-r2/)** | BGE-base init · LR=1e-4 · 8 layers · dropout=0.1 · weight_decay=0.01 | **FINAL SUBMISSION** — selected from a 9-run L9 Taguchi sweep as the Pareto winner under the organizer's `eval_metrics.py` |

## Headline metrics (official scorer)

Numbers below come from `official_eval/eval_metrics.py` — the same script
the organizers grade with. Source artifacts:
`outputs/reports/transformer_5000_bge-r2/official_*.txt`.

| Task | Metric | Value |
|---|---|---|
| Next-step prediction | Top-1 accuracy | **0.7125** |
| Next-step prediction | Top-3 accuracy | 0.9972 |
| Next-step prediction | Top-5 accuracy | 1.0000 |
| Next-step prediction | MRR | **0.8540** |
| Sequence completion | Normalised edit distance ↓ | **0.2126** |
| Sequence completion | Exact match rate | 0.0058 |
| Sequence completion | Token accuracy | 0.4387 |
| Sequence completion | Block-level accuracy | 0.6988 |
| Anomaly detection | Binary accuracy | 1.0000 |
| Anomaly detection | Precision (invalid class) | 1.0000 |
| Anomaly detection | Recall (invalid class) | 1.0000 |
| Anomaly detection | F1 (invalid class) | 1.0000 |
| Anomaly detection | ROC-AUC | 0.8773 |
| Anomaly detection | Rule attribution accuracy | 0.9752 |

For the cross-model comparison (bge-r2 vs MiniLM baselines vs n-gram) see
the [Results section of REPORT.md](../REPORT.md#results). Those baseline
numbers are real — they came from runs we did during development — but
their per-run archives aren't kept here, by design. The wandb project
has the full history:

https://wandb.ai/fedja-bogataj-org/infineon-track1

## To regenerate this archive

The archive is built by `scripts/archive_run.sh` from `outputs/{models,
submissions, reports}/` (which live on the training cluster, not in this
repo since the model checkpoints are >100 MB):

```bash
# On the cluster, after training has finished:
./scripts/archive_run.sh outputs/models/transformer_5000_bge-r2.pt
# → results/transformer_5000_bge-r2/{README.md, metrics/, submissions/}
#   (the .best.pt sibling is auto-preferred for the model copy if present)
```

Then commit and push (verify no `.pt` slips into the staging area — the
default invocation without `--include-model` skips it, but worth checking).
