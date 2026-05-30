# transformer_5000

_Archived 2026-05-30 14:38 UTC_

## Configuration

| Setting | Value |
|---|---|
| Model type | `transformer` |
| Seed | 42 |
| Eval split hash | `(set externally)` |
| Submission timestamp | 2026-05-30T14:33:23.125126+00:00 |
| Checkpoint | `model.pt` (29.4 MB) |
| W&B run | [4n6vqf7p](https://wandb.ai/fedja-bogataj-org/infineon-track1/runs/4n6vqf7p) |

## Next-step prediction  (n=6000)

| Metric | Value | Better |
|---|---|---|
| `top_1_accuracy` | **0.7090** | ↑ |
| `top_3_accuracy` | **0.9967** | ↑ |
| `top_5_accuracy` | **1.0000** | ↑ |
| `mrr` | **0.8523** | ↑ |

## Sequence completion  (n=6000)

| Metric | Value | Better |
|---|---|---|
| `exact_match_rate` | **0.0055** | ↑ |
| `normalized_edit_distance` | **0.2166** | ↓ |
| `token_accuracy` | **0.4009** | ↑ |
| `block_accuracy` | **0.6479** | ↑ |

## Anomaly detection  (n=3000)

| Metric | Value | Better |
|---|---|---|
| `binary_accuracy` | **1.0000** | ↑ |
| `f1` | **1.0000** | ↑ |
| `roc_auc` | **0.7940** | ↑ |
| `rule_attribution_accuracy` | **0.9752** | ↑ |

Confusion matrix (positive = invalid):

|        | predicted invalid | predicted valid |
|---|---|---|
| **actually invalid** | TP = 1170 | FN = 0 |
| **actually valid**   | FP = 0 | TN = 1830 |

---

To reproduce: see commit history and `scripts/train_transformer.sbatch`.
Submission CSVs and metric JSONs in this folder are deterministic from the model checkpoint + the canonical eval set at `outputs/eval_canonical/`.
