# transformer_25000

_Archived 2026-05-30 14:38 UTC_

## Configuration

| Setting | Value |
|---|---|
| Model type | `transformer` |
| Seed | 42 |
| Eval split hash | `(set externally)` |
| Submission timestamp | 2026-05-30T14:32:21.165610+00:00 |
| Checkpoint | `model.pt` (29.4 MB) |
| W&B run | [m4ok54ug](https://wandb.ai/fedja-bogataj-org/infineon-track1/runs/m4ok54ug) |

## Next-step prediction  (n=6000)

| Metric | Value | Better |
|---|---|---|
| `top_1_accuracy` | **0.7027** | ↑ |
| `top_3_accuracy` | **0.9962** | ↑ |
| `top_5_accuracy` | **1.0000** | ↑ |
| `mrr` | **0.8489** | ↑ |

## Sequence completion  (n=6000)

| Metric | Value | Better |
|---|---|---|
| `exact_match_rate` | **0.0047** | ↑ |
| `normalized_edit_distance` | **0.2188** | ↓ |
| `token_accuracy` | **0.3934** | ↑ |
| `block_accuracy` | **0.6432** | ↑ |

## Anomaly detection  (n=3000)

| Metric | Value | Better |
|---|---|---|
| `binary_accuracy` | **1.0000** | ↑ |
| `f1` | **1.0000** | ↑ |
| `roc_auc` | **0.8500** | ↑ |
| `rule_attribution_accuracy` | **0.9752** | ↑ |

Confusion matrix (positive = invalid):

|        | predicted invalid | predicted valid |
|---|---|---|
| **actually invalid** | TP = 1170 | FN = 0 |
| **actually valid**   | FP = 0 | TN = 1830 |

---

To reproduce: see commit history and `scripts/train_transformer.sbatch`.
Submission CSVs and metric JSONs in this folder are deterministic from the model checkpoint + the canonical eval set at `outputs/eval_canonical/`.
