# REPORT — Small Transformers for Semiconductor Process Logic

**Track:** Industrial AI (Track 1) — Learning & Benchmarking Process Logic
**Team:** Fedja Bogataj, Luka Premuš
**Mentor:** Simeon (Infineon)
**Final submission model:** `transformer_5000_bge-r2` (BGE-init transformer, L9-tuned)

---

## TL;DR

We trained a family-conditioned GPT-style decoder that learns semiconductor fab
process-sequence logic from synthetic data, initialised its token embeddings from
the **BAAI/bge-base-en-v1.5** sentence encoder, and selected the final submission
via a 9-run **L9 Taguchi orthogonal sweep** over learning rate × depth ×
regularisation. The chosen model (`bge-r2`: LR=1e-4, 8 layers, dropout 0.1,
weight_decay 0.01) is the Pareto winner on Tasks 1 and 2 under the **organizer's
own `eval_metrics.py`**, ties everyone on Task 3 F1, and trains in ~2h per run
on a Leonardo A100.

## Problem

Semiconductor manufacturing is a strict recipe. Each wafer lot runs ~110–150
ordered steps drawn from a vocabulary of ~120 step types, under a fixed block
structure and 10 hard ordering rules (e.g. no patterned etch without a prior
lithography develop; no electrical test before passivation cure). Three product
families — MOSFET (~126 steps), IGBT (~151), IC (~107) — share a backbone but
differ in prep blocks and litho-cycle counts.

A model can score high on *next-step* accuracy purely from local token
frequencies without understanding any of the long-range ordering. The track
asks whether a model genuinely captures the global ordering grammar and
generalises to a **held-out 4th family** the organizers test post-submission.

## Approach

- **Language modelling over steps.** One step = one token. Sequences run from
  `RECEIVE WAFER LOT` to `SHIP LOT`, modelled autoregressively and conditioned
  on family. Family identity is a learnable embedding summed into every
  position; the LM head is weight-tied to the token embedding table (halves
  parameters).
- **Semantic embedding init from BGE-base.** Token embeddings are initialised
  from `BAAI/bge-base-en-v1.5` (109M params, 768-dim native), encoding each
  step as *name + description + parameters*. This was a clear upgrade over our
  initial `all-MiniLM-L6-v2` (22M / 384-dim) on every Task 1 / Task 2 metric.
  An unseen step from the OOD 4th family lands in the same semantic space — the
  mechanism we expect to soften the OOD drop.
- **9-run L9 Taguchi orthogonal sweep.** A balanced 3-level design over LR ∈
  {1e-4, 3e-4, 1e-3} × n_layers ∈ {4, 6, 8} × (dropout, weight_decay) ∈
  {(0.1, 0.0), (0.2, 0.0), (0.1, 0.01)}. Each factor level appears 3 times,
  balanced against the others — main-effect estimates with 1/3 the runs of a
  full 3×3×3 grid. Submitted as a SLURM `--array=0-8` job.
- **Best-val-loss checkpointing.** Every end-of-epoch, if `val_loss` improves,
  we CPU-snapshot the model weights. At training end we persist them as a
  sibling `<MODEL>.best.pt`; `TransformerPredictor.load()` auto-prefers this
  file so all downstream eval uses the lowest-val-loss weights, not the
  final-epoch ones. Critical for sweep fairness — high-LR configs overfit
  earlier than low-LR ones, so final-epoch comparisons systematically misrank.
- **Re-ranked with the organizer's official scorer.** Our internal `metrics.py`
  and `official_eval/eval_metrics.py` are independent implementations of the
  same spec. We ran the organizer's scorer (CPU-only, ~10s/model) over every
  trained model and picked the final submission from THAT ranking — the same
  scorer the organizers will use to grade.

## How to run it

Python 3.10+. Reproducing the final submission end-to-end:

```bash
# 0. Install
python -m venv .venv
source .venv/bin/activate
pip install -e .                       # or: pip install -r requirements.txt

# 1. Generate 5000 synthetic sequences per family (deterministic; seed 1042
#    differs from the canonical-1000 seed so train and test never overlap)
./scripts/generate_data.sh 5000 1042 training_data_5000/

# 2. Build the held-out canonical TEST set (the original 1000/family).
#    Submission models are NEVER trained against this set.
infineon-baseline build-test-only \
    --variants-dir training_data/ \
    --out outputs/eval_canonical/ \
    --seed 42

# 3. Build BGE-base step embeddings (one-time, cached per encoder)
ENCODER=bge infineon-baseline build-st-embeddings \
    --descriptions-dir training_data/ \
    --out outputs/models/st_step_embeddings_bge.pkl

# 4. Run the 9-row L9 sweep over BGE (SLURM array job; ~2h/task on A100,
#    9 tasks run in parallel across the reservation, ~3-5h wall-clock total)
sbatch scripts/sweep_bge.sbatch              # or scripts/sweep_minilm.sbatch for the MiniLM grid

# 5. Evaluate every trained model on the canonical test set.
#    Caches per-model; rerun with FORCE=1 to invalidate.
EVAL_DIR=outputs/eval_canonical sbatch scripts/evaluate_all.sbatch

# 6. Re-score every model with the ORGANIZER's official scorer to pick the winner.
#    Translates our GT schema into theirs once (cached), then runs eval_metrics.py
#    per model and prints a comparison table.
sbatch scripts/score_official.sbatch

# 7. Generate the organizer-format submission CSVs from the chosen model
sbatch --export=ALL,MODELS=outputs/models/transformer_5000_bge-r2.pt \
       scripts/make_submission.sbatch
#    Output:
#      outputs/submissions/transformer_5000_bge-r2/official/task{1,2,3}.csv
#    Renamed at repo root as nextstep.csv / completion.csv / anomaly.csv.
```

The `*.best.pt` sibling auto-prefer means **every eval / predict step uses the
lowest-val-loss weights without caller changes**. CPU and login-node fallbacks
exist for everything except training itself.

## Results

All numbers below come from `official_eval/eval_metrics.py` — the organizer's
own scorer — against the canonical hold-out set (n=6000 for Tasks 1-2, n=3000
for Task 3, with 1170 forbidden + 1830 valid examples on Task 3).

### Final submission vs. baselines

| Metric                          | n-gram (o3) | Transformer 5k (MiniLM) | Transformer 25k (MiniLM) | **bge-r2 (final)** |
|---------------------------------|-------------|-------------------------|--------------------------|--------------------|
| Top-1 next-step ↑               | 0.6843      | 0.7090                  | 0.7027                   | **0.7125**         |
| Top-3 next-step ↑               | 0.9900      | 0.9967                  | 0.9962                   | 0.9972             |
| MRR ↑                           | 0.8370      | 0.8523                  | 0.8489                   | **0.8540**         |
| Normalised edit distance ↓      | 0.5621      | 0.2166                  | 0.2188                   | **0.2126**         |
| Token accuracy (completion) ↑   | 0.3275      | 0.4381                  | 0.4302                   | 0.4387             |
| Block accuracy (completion) ↑   | 0.6235      | 0.6984                  | 0.6927                   | 0.6988             |
| F1 (anomaly, invalid class) ↑   | 1.0000      | 1.0000                  | 1.0000                   | 1.0000             |
| Rule attribution accuracy ↑     | 0.9752      | 0.9752                  | 0.9752                   | 0.9752             |
| ROC-AUC (anomaly) ↑             | 0.5150      | 0.7940                  | 0.8500                   | 0.8773             |

**The bge-r2 model is the Pareto winner across Tasks 1 + 2 metrics under the
official scorer**, ties everyone on F1 + rule attribution (oracle-driven), and
clearly beats the n-gram and the MiniLM-init transformers on ROC-AUC.

### The L9 sweep (BGE-init)

3 levels × 3 factors, balanced. Each factor level appears in exactly 3 runs.
Headline metrics for the 9 BGE runs (excluding 3 runs whose ROC-AUC saturated
to 0.5 / random):

| Run    | LR    | n_layers | dropout | wd   | Top-1 ↑ | NED ↓  | ROC-AUC ↑ |
|--------|-------|----------|---------|------|---------|--------|-----------|
| r0     | 1e-4  | 4        | 0.1     | 0.0  | 0.7078  | 0.2132 | 0.7662    |
| r1     | 1e-4  | 6        | 0.2     | 0.0  | 0.6965  | 0.2192 | 0.8299    |
| **r2** | 1e-4  | 8        | 0.1     | 0.01 | **0.7125** | **0.2126** | 0.8773 |
| r3     | 3e-4  | 4        | 0.2     | 0.0  | 0.7032  | 0.2197 | 0.9803    |
| r6     | 1e-3  | 4        | 0.1     | 0.01 | 0.6928  | 0.2189 | 0.9836    |
| r8     | 1e-3  | 8        | 0.2     | 0.0  | 0.7057  | 0.2142 | 0.9977    |

Main-effect read: **higher LR + more layers → much better ROC-AUC**
(r0 0.77 → r8 0.998), but **lower LR wins Task 1 / Task 2** (r2 takes Top-1 +
NED + MRR). We picked r2 as the Pareto winner — best on Tasks 1 + 2, ROC-AUC
still well above the failure threshold.

### Why not the "obvious" pick

A non-swept BGE run with default hyperparameters scored **Top-1 = 0.7672** and
**NED = 0.1834** — apparently +5-6 percentage points over every other config.
We **ruled it out as overfit**: the gap is too large to attribute to anything
but in-distribution pattern memorisation, and the 4th-family OOD set will
punish that. Picking a Pareto winner among more conservatively-trained configs
(low LR, light reg) is the OOD-safer bet.

### Internal scorer vs. official scorer

A useful side-finding from the re-ranking step: our `metrics.py` agrees with
`eval_metrics.py` exactly on **Top-1 / Top-3 / Top-5 / MRR / NED / Exact Match
/ Binary Acc / F1 / ROC-AUC / Rule Attribution**. Two metrics diverge
systematically: **Task 2 token accuracy and block accuracy** are reported
~4-5 percentage points lower by our scorer than by theirs (uniform shift
across every model — likely a normalisation difference). The drift doesn't
flip rankings, but the numbers we reported internally during development
understated those two metrics relative to what the organizers will see.

## What worked / what didn't

**Worked**
- **BGE-base over MiniLM**: same architecture, better tokens → +0.5% Top-1,
  -2% NED across the board. Initialising tokens from a richer semantic space
  is the cheapest model upgrade we've ever made.
- **L9 orthogonal design.** 9 runs is enough to read the main effect of each
  factor cleanly. Three factors × three levels would have needed 27 runs for
  a full grid.
- **Best-val-loss sibling checkpoints.** Without this, low-LR configs (which
  are still improving at epoch 30) would compare unfairly against high-LR
  configs (which overfit by epoch ~20). Auto-prefer in the loader means no
  caller code changes were needed.
- **Re-ranking with the official scorer.** Caught a small drift in our
  internal scorer (Task 2 token/block acc) and confirmed our model ranking
  wasn't sensitive to it. Submitting with the right yardstick.
- **Family conditioning + weight tying.** Tiny architectural choices but
  measurable: family-conditioned token init at start of training, summing
  family embedding into every position, weight-tying the LM head — all
  visible in the gap between baseline and final.

**Didn't / surprised us**
- **The naive BGE single-config training scored suspiciously high** (Top-1
  0.7672 vs sweep best 0.7125). After analysis we believe this was overfit
  to in-distribution synthetic patterns and would degrade hard on the OOD
  4th family. Ruled out of submission.
- **Three sweep configs (r4, r5, r7) had broken anomaly scoring** (ROC-AUC
  exactly 0.5000 — sigmoid saturated, no ranking signal). Pareto-dominated;
  ignored in selection.
- **`F1 = 1.000` is essentially meaningless** on the anomaly task — the hybrid
  strategy uses a symbolic rule checker for the `IS_VALID` decision, so
  every reasonable model gets perfect F1 by routing through the oracle.
  ROC-AUC on the model's own confidence is the only honest read.
- **The BGE sweep was run at D_MODEL=384** (the sweep script was forked from
  the MiniLM grid and inherited the dimension). BGE's native 768-dim
  embeddings get projected down at init — the sweep models don't capture
  BGE's full advantage. The non-swept BGE (full 768-dim) does, but at the
  cost of in-distribution overfit. Trade-off we accepted.

## What we'd do with another 36 hours

- **Evaluate on the hidden 4th family.** The real OOD test and the direct
  payoff of the semantic-init design. We don't have it; the organizers do.
- **Re-run the BGE sweep at native 768-dim.** Same L9, same 9 runs, but
  D_MODEL=768 + N_HEADS=12 + FF_DIM=3072 (BGE's native shape). Would resolve
  whether the non-swept BGE's gap is overfit or a real signal we left on the
  table.
- **Per-family / per-rule diagnostics.** Surface which of the 10 forbidden
  patterns and which families are hardest, turning the benchmark into an
  actionable tool for process engineers rather than just an aggregate score.
- **Subword tokeniser** (started on `feat/subword-tokenizer`, parked when
  it underperformed the flat tokeniser at our scale). With more time +
  ST-init for subwords, it should transfer better to OOD step names.
- **Harder, dirtier training data.** Inject controlled near-miss violations
  so the prediction tasks stop saturating around 5k sequences and the
  scaling curve has room to move.

## Credits & dependencies

- **Libraries:** PyTorch (model + training), `sentence-transformers` (BGE-base
  init), `scikit-learn`, NumPy, pandas, pytest (tests), Weights & Biases
  (optional experiment tracking).
- **Pretrained model:** `BAAI/bge-base-en-v1.5` for token embedding init.
- **Earlier baseline encoder:** `sentence-transformers/all-MiniLM-L6-v2`
  (kept reachable via `ENCODER=minilm` for the cross-encoder comparison).
- **Infrastructure:** Leonardo cluster (SLURM, A100 GPUs) via the
  `boost_usr_prod` partition, `s_tra_ncc` reservation,
  `EUHPC_D30_031` account.
- **Data:** organizer-provided canonical sequences (1000/family) plus our
  validator-backed synthetic generator (`training_data/generate_sequences.py`).
- **AI coding tools:** Claude Code (Anthropic) used during development.

---

**Deliverables map**

- Final submission files: `nextstep.csv` / `completion.csv` / `anomaly.csv` at
  the repo root (generated by `scripts/make_submission.sh` against the
  organizer-provided `official_eval/eval_input_*.csv`)
- Trained model: `outputs/models/transformer_5000_bge-r2.{pt,best.pt}`
  (not committed — re-train with `scripts/sweep_bge.sbatch` then array index 2;
  ~2h on A100)
- Per-task scores (official scorer): `outputs/reports/transformer_5000_bge-r2/official_*.txt`
- Per-task scores (internal scorer): `outputs/reports/transformer_5000_bge-r2/task{1,2,3}.json`
- Cross-model comparison: stdout of `scripts/score_official.sbatch` / `evaluate_all.sbatch`
- L9 sweep + earlier baselines: 9 BGE runs (`transformer_5000_bge-r0..r8`)
  + 9 MiniLM runs + 4 MiniLM-baselines, all evaluated in the same comparison
  table for cross-encoder analysis
- Wandb experiment tracking:
  https://wandb.ai/fedja-bogataj-org/infineon-track1 (every run logged with
  config + per-epoch metrics + eval metrics resumed onto the training run)
