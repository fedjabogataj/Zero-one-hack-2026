# REPORT — Small Transformers for Semiconductor Process Logic

**Track:** Industrial AI (Track 1) — Learning & Benchmarking Process Logic
**Team:** [Team Name] · Fedja Bogataj, Luka Premuš
**Mentor:** Simeon (Infineon)

---

## TL;DR

We trained a compact, family-conditioned GPT-style decoder that learns semiconductor
fab process-sequence logic from synthetic data, and benchmarked it against an n-gram
baseline. The transformer matches the baseline on local next-step prediction but
decisively beats it on the metrics that require *global* process understanding —
sequence completion and anomaly detection (anomaly ROC-AUC 0.51 → 0.85). It is small
(~29 MB), trained reproducibly on the Leonardo cluster, and built on a fully open stack.

## Problem

Semiconductor manufacturing is a strict recipe. Each wafer lot runs ~110–150 ordered
steps drawn from a vocabulary of ~120 step types, under a fixed block structure and 10
hard ordering rules (e.g. no patterned etch without a prior lithography develop; no
electrical test before passivation cure). There are three product families — MOSFET
(~126 steps), IGBT (~151), IC (~107) — that share a backbone but differ in prep blocks
and litho-cycle counts.

The catch: a model can score high on *next-step* accuracy purely from local token
frequencies, without understanding anything about long-range ordering. We specifically
set out to measure whether a model captures the global ordering grammar — and to build
toward generalising to a **held-out 4th family** the organizers test post-submission.

## Approach

- **Language modelling over steps.** One step = one token. Sequences run from
  `RECEIVE WAFER LOT` to `SHIP LOT`, modelled autoregressively and conditioned on family.
- **Small GPT decoder.** 4 layers · 6 heads · d_model 384 · ff_dim 1536 · Pre-LN + GELU ·
  dropout 0.1 · max_seq_len 256 · ~29 MB checkpoint. Family conditioning is a learnable
  family embedding summed into every position; the LM head is weight-tied to the token
  embedding table (halves parameters).
- **Key decision — semantic embedding init.** Token embeddings are initialised from
  `sentence-transformers/all-MiniLM-L6-v2` (384-dim), encoding each step as
  *name + description + parameters*. An unseen step from a new family then lands in the
  same semantic space as known steps — the mechanism we expect to soften the OOD drop.
- **Fair comparison by construction.** The transformer predictor exposes exactly the
  n-gram baseline's interface (`top_k`, `log_prob`), so a single evaluation pipeline
  scores both models on identical inputs.
- **Grammar-aware synthetic data.** On top of the 1,000 canonical variants per family,
  we generated 5k / 10k / 25k extra sequences with the validator-backed generator; all
  10 forbidden patterns are enforced, so training data is guaranteed clean.

## How to run it

Python 3.10+. From the repository root:

```bash
# 0. Install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1. Build the held-out eval set (deterministic at --seed 42)
infineon-baseline build-eval \
    --variants-dir training_data/ \
    --out outputs/eval/ \
    --holdout-per-family 200 \
    --anomaly-invalid-ratio 0.39 \
    --seed 42

# 2. Build MiniLM semantic step embeddings (used to init the transformer)
infineon-baseline build-st-embeddings \
    --descriptions-dir training_data/ \
    --out outputs/models/st_minilm.pkl

# 3. Train the family-conditioned transformer (GPU recommended)
infineon-baseline train \
    --train outputs/eval/train_split.csv \
    --embeddings outputs/models/st_minilm.pkl \
    --out outputs/models/transformer_5000.pt \
    --epochs 30 --batch-size 64 --lr 3e-4 \
    --d-model 384 --n-heads 6 --ff-dim 1536 \
    --seed 42

# 4. Produce the three submission files
infineon-baseline predict --transformer outputs/models/transformer_5000.pt \
    --eval-input outputs/eval/eval_input_valid.csv   --task next-step --out nextstep.csv
infineon-baseline predict --transformer outputs/models/transformer_5000.pt \
    --eval-input outputs/eval/eval_input_valid.csv   --task complete  --out completion.csv
infineon-baseline predict --transformer outputs/models/transformer_5000.pt \
    --eval-input outputs/eval/eval_input_anomaly.csv --task anomaly   --out anomaly.csv

# 5. Score (per-family breakdown in the JSON report)
infineon-baseline score --predictions nextstep.csv \
    --ground-truth outputs/eval/ground_truth_valid.csv   --task next-step --report-json outputs/reports/nextstep.json
infineon-baseline score --predictions completion.csv \
    --ground-truth outputs/eval/ground_truth_valid.csv   --task complete  --report-json outputs/reports/completion.json
infineon-baseline score --predictions anomaly.csv \
    --ground-truth outputs/eval/ground_truth_anomaly.csv --task anomaly   --report-json outputs/reports/anomaly.json
```

> **No GPU?** Training runs on CPU but is slow; the trained checkpoints
> (`outputs/models/transformer_{5000,10000,25000}.pt`) are committed, so you can skip
> step 3 and run prediction/scoring directly. Cluster training uses
> `scripts/train_transformer.sbatch` on Leonardo (SLURM).

When the organizers' real eval files arrive, skip step 1 and point `--eval-input` at
their `eval_input_*.csv`; the rest of the pipeline is unchanged.

## Results

Identical eval set for both models: **n = 6,000** (Tasks 1–2), **n = 3,000** (Task 3).

**Baseline (n-gram order 3) vs Transformer (trained on 5k):**

| Metric | N-gram (o3) | Transformer 5k | Δ |
|---|---|---|---|
| Top-1 next-step accuracy | 0.684 | **0.709** | +3.7% |
| MRR (next-step) | 0.837 | **0.852** | +1.8% |
| Token accuracy (completion) | 0.254 | **0.401** | **+57%** |
| Block accuracy (completion) | 0.479 | **0.648** | **+35%** |
| Normalised edit distance ↓ | 0.562 | **0.217** | **−61%** |
| Anomaly ROC-AUC | 0.515 | **0.794** | **+54%** |

The biggest gains are exactly on the structural metrics. **On anomaly detection both
models report F1 = 1.0 — and that is a trap:** the submission can call the rule-based
validator, so binary accuracy is perfect for anyone. The honest signal is **ROC-AUC on
the model's own confidence**: the n-gram scores 0.515 (a coin flip), the transformer
0.794. Rule-attribution accuracy is **97.5%**. Confusion matrix (n=3,000):
TP 1170 · TN 1830 · FP 0 · FN 0.

**Scaling study (transformer, 5k → 10k → 25k sequences):**

| Metric | 5k | 10k | 25k |
|---|---|---|---|
| Top-1 next-step | 0.709 | 0.690 | 0.703 |
| MRR | 0.852 | 0.842 | 0.849 |
| Token accuracy | 0.401 | 0.393 | 0.393 |
| Block accuracy | 0.648 | 0.645 | 0.643 |
| Anomaly ROC-AUC | 0.794 | 0.839 | **0.850** |

Evidence lives in `results/transformer_{5000,10000,25000}/` and `results/ngram_o3/`
(metric JSONs + submission CSVs per model).

## What worked / What didn't

**Worked**
- Semantic-init + family conditioning + weight tying: a ~29 MB model beats the baseline
  on every structural metric.
- ROC-AUC on model confidence cleanly separates "understands ordering" (transformer)
  from "doesn't" (n-gram) — the metric that actually answers the track's question.
- bf16 batched inference made evaluation ~10–30× faster, which made the scaling study
  feasible inside the hackathon.

**Didn't / surprised us**
- **Prediction & completion plateau at 5k.** The synthetic data is too clean, so the
  small model saturates almost immediately — more data barely moves Top-1 or token-acc.
- **F1 = 1.0 is meaningless here** because of the validator shortcut; we had to switch to
  ROC-AUC to get an honest read. Worth flagging loudly to anyone reusing this benchmark.
- Only anomaly ROC-AUC keeps climbing with data (0.79 → 0.85), suggesting the structural
  signal still has headroom while local prediction does not.

## What we'd do with another 36 hours

- **Evaluate on the hidden 4th family** — the real OOD test and the direct payoff of the
  semantic-init design.
- **Sub-word tokenizer** (started on `feat/subword-tokenizer`): split step strings into
  sub-word units so structure transfers to related, never-seen steps.
- **Harder, dirtier training data**: inject controlled near-miss violations so the
  prediction tasks stop saturating and the scaling curve has room to move.
- **Per-rule / per-family diagnostics**: report which of the 10 rules and which families
  are hardest, turning the benchmark into an actionable tool for process engineers.

## Credits & dependencies

- **Libraries:** PyTorch (model + training), sentence-transformers (semantic init),
  scikit-learn, NumPy, pandas; pytest (tests); Weights & Biases (optional tracking).
- **Pretrained model:** `sentence-transformers/all-MiniLM-L6-v2` (embedding init).
- **Infrastructure:** Leonardo cluster (SLURM, GPU) for training.
- **Data:** organizer-provided canonical sequences + our validator-backed synthetic
  generator (`training_data/generate_sequences.py`).
- **AI coding tools:** Claude Code (Anthropic) used during development.

> **Deliverables map:** final submission files `nextstep.csv` / `completion.csv` /
> `anomaly.csv` at the repo root; trained checkpoints in `outputs/models/`; per-task,
> per-family scores in `outputs/reports/`; full metric JSONs and per-model submissions in
> `results/`.
