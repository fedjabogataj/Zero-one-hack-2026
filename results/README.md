# Archived experiment results

_Last updated 2026-05-30 14:38 UTC_

Each subdirectory is a single archived experiment. Follow its `README.md`
for the full metrics table; this top-level table shows headline numbers across runs.

## Cross-experiment comparison — headline metrics

| run | Top-1 (↑) | Top-5 (↑) | MRR (↑) | NED (↓) | Block acc (↑) | F1 (↑) | ROC-AUC (↑) |
|---|---|---|---|---|---|---|---|
| [`ngram_o3`](ngram_o3/) | 0.6843 | 1.0000 | 0.8370 | 0.5621 | 0.4786 | 1.0000 | 0.5150 |
| [`transformer_10000`](transformer_10000/) | 0.6902 | 1.0000 | 0.8421 | 0.2241 | 0.6447 | 1.0000 | 0.8389 |
| [`transformer_25000`](transformer_25000/) | 0.7027 | 1.0000 | 0.8489 | 0.2188 | 0.6432 | 1.0000 | 0.8500 |
| [`transformer_5000`](transformer_5000/) | 0.7090 | 1.0000 | 0.8523 | 0.2166 | 0.6479 | 1.0000 | 0.7940 |

_Best per column is whatever maximises the arrow direction. NED (↓) is lower-is-better; the rest are higher-is-better._