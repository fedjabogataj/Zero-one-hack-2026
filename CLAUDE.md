# CLAUDE.md — Industrial AI (Track 1): Learning & Benchmarking Process Logic

> **Authoritative project reference.** Consolidated from the official track site
> (https://docs.zero-one.lumos-consulting.at/tracks/track-1/) and the in-repo docs
> (`README.md`, `Track_industrial_en.md`, `Track_industrial.md`,
> `training_data/README.md`, `training_data/generation_rules.md`).
> When in doubt, the source files above win — re-read `training_data/generation_rules.md`
> for the full grammar, vocabulary, and rule definitions.

---

## 1. Mission

Build models that learn **semiconductor process-sequence logic** from synthetic fab
sequences, and benchmark whether they genuinely understand process logic or merely
memorize patterns. Three product families: **MOSFET**, **IGBT**, **IC**. A hidden 4th
family is used by organizers post-submission for OOD generalization.

- **Difficulty:** Advanced–Expert. **Scope:** ~36h, synthetic data, open model choice.
- **Sovereignty angle:** own data generation, reproducible training on the Leonardo
  cluster, transparent open stack — not a black-box API wrapper.
- **Minimum viable result:** reproducible end-to-end workflow = synthetic data generation
  + ≥1 trained model + baseline-vs-trained comparison + documented benchmark.
- **Stretch:** multiple model sizes/architectures, scaling analysis, OOD generalization,
  optional fab parameters, small before/after demo dashboard.
- **On-site mentor:** Simeon.

---

## 2. Submission Tasks & Metrics

| # | Task | Input | Metrics |
|---|------|-------|---------|
| 1 | **Next-step prediction** | partial sequence → rank top-5 next steps | Top-1 / Top-3 / Top-5 Accuracy, MRR |
| 2 | **Sequence completion** | partial seq (60% or 80%) → predict remaining steps | Exact Match Rate, Normalized Edit Distance (lower=better), Token Accuracy, Block-level Accuracy |
| 3 | **Anomaly detection** | full sequence → valid or rule-violating | Binary Acc, Precision, Recall, F1, Confusion Matrix, ROC-AUC, Rule Attribution Acc |
| 4* | **OOD generalization** | hidden 4th family | Performance drop ID → OOD (organizers only; no submission) |

Score locally with `training_data/eval_metrics.py` (no external deps):
`python eval_metrics.py --task anomaly --ground-truth <gt.csv> --predictions <out.csv>`

---

## 3. Data

All under `training_data/`. Sequences always start `RECEIVE WAFER LOT`, end `SHIP LOT`.
Each **step string is one token** (~120 distinct across families).

| File | Contents |
|------|----------|
| `synthetic_mosfet.csv` / `syntheticIGBT.csv` / `syntheticIC.csv` | Canonical reference sequence (126 / 151 / 107 steps) — do not modify |
| `*_Longdescr.csv` | Steps + text descriptions |
| `*_longdescription_parameters.csv` | Steps + descriptions + realistic fab-level parameters |
| `MOSFET_variants.csv` / `IGBT_variants.csv` / `IC_variants.csv` | 1,000 validated sequences each (~125 / ~148 / ~115 steps) |
| `generate_sequences.py` | Generator + validator CLI |
| `eval_metrics.py` | Official scoring script |
| `generation_rules.md` | Full grammar, 10 forbidden patterns, eval protocol |

**Long format:** `SEQUENCE_ID,STEP` — one step per row. Load with no extra deps:
```python
from pathlib import Path
from generate_sequences import read_csv_sequences
seqs = read_csv_sequences(Path("MOSFET_variants.csv"))  # dict[seq_id -> list[step_str]]
```

Combinatorial space is huge (MOSFET ~51B, IGBT ~13T, IC ~6B), so generate freely:
```bash
python generate_sequences.py --family mosfet --count 2000 --output extra.csv --seed 42
python generate_sequences.py --validate extra.csv          # check against all 10 rules
python generate_sequences.py --family igbt --estimate-only
```
Families: `mosfet`, `igbt`, `ic`.

---

## 4. Process Grammar (essentials)

Shared backbone (block order is **fixed**):
```
PREFIX → INITIAL_MEASUREMENTS → PRE_PROCESS_CLEAN → FAMILY_SPECIFIC_PREP
→ FIRST_OXIDATION → PROCESS_CYCLES{3..6} → ILD_BLOCK → VIA_BLOCK → METAL_BLOCK
→ PASSIVATION_BLOCK → BACKSIDE_BLOCK → FINAL_INSPECTION → TEST_SUITE → SUFFIX
```
- **PREFIX:** `RECEIVE WAFER LOT → LOT IDENTIFICATION → (INITIAL WAFER INSPECTION | PRE CLEAN INSPECTION)`
- **SUFFIX:** `(LOT RELEASE | FINAL LOT RELEASE) → [PACKAGE PREPARATION (IC)] → SHIP LOT`
- **Litho cycle:** `SPIN COAT PHOTORESIST → SOFT BAKE → ALIGN MASK LEVEL N → EXPOSE LITHO LEVEL N → [POST EXPOSE BAKE] → DEVELOP PHOTORESIST → <PATTERN INSPECTION> → [HARD BAKE]`
- **Family prep:** MOSFET = epitaxy block; IGBT = dual-implant / epi check; IC = early backside grind.
- Litho cycles per family: MOSFET 4, IGBT 6, IC 4.

Full vocabulary (12 categories) and all block definitions live in
`training_data/generation_rules.md` §1–§2. **Variation axes** (what can change while
staying valid) are in §4.

---

## 5. The 10 Forbidden Patterns (anomaly rule set)

These define process-logic violations used in the held-out anomaly eval. Checker:
`validate_sequence(steps) -> list[Violation]` in `generate_sequences.py`.

| Rule ID | Constraint |
|---------|-----------|
| `RULE_DEP_NO_CLEAN` | Every deposition step must be preceded by a cleaning step within the same block (≤12 steps before). |
| `RULE_METAL_ETCH_NO_LITHO` | A metal etch must be preceded by a full litho (`EXPOSE LITHO` + `DEVELOP PHOTORESIST`) within ≤15 steps. |
| `RULE_ETCH_NO_MASK` | Any patterned etch must be preceded by `DEVELOP PHOTORESIST`/`DEVELOP PAD WINDOW` within ≤12 steps. (`ANISOTROPIC ETCH SPACER` is exempt — blanket etch.) |
| `RULE_LITHO_LEVEL_SKIP` | Litho levels must be sequential: `ALIGN MASK LEVEL N+1` cannot precede a completed level N. |
| `RULE_IMPLANT_NO_MASK` | Any implant must be preceded by an oxide/window etch or `DEVELOP PHOTORESIST` within ≤15 steps. |
| `RULE_CMP_NO_DEP` | A CMP step must be preceded by a deposition (or via fill) within ≤6 steps. |
| `RULE_PAD_OPEN_BEFORE_DEP` | Pad-window opening must come **after** `DEPOSIT PASSIVATION` and `CURE PASSIVATION`. |
| `RULE_TEST_BEFORE_PASSIVATION` | All electrical tests must come **after** `CURE PASSIVATION`. |
| `RULE_SHIP_BEFORE_TEST` | `SHIP LOT` must come **after** `WAFER SORT TEST`. |
| `RULE_BACKSIDE_BEFORE_PASSIVATION` | `DEPOSIT BACKSIDE METAL` must come **after** `CURE PASSIVATION`. |

Full trigger/prerequisite step lists and violation examples: `generation_rules.md` §3.

---

## 6. Eval Protocol & Submission Formats

Organizers distribute two input files at hackathon start:

- **`eval_input_valid.csv`** (Tasks 1 & 2): `EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE` — 600 rows (100 seqs × 3 families × {0.6, 0.8} cut). `PARTIAL_SEQUENCE` is pipe-separated steps.
- **`eval_input_anomaly.csv`** (Task 3): `EXAMPLE_ID, FAMILY, SEQUENCE` — 987 unlabeled, shuffled valid + invalid.

**Submission formats:**
```
Task 1: EXAMPLE_ID, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5
Task 2: EXAMPLE_ID, PREDICTED_SEQUENCE          # ONLY steps AFTER the cut, pipe-separated
Task 3: EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE   # IS_VALID 1/0; SCORE=P(valid)∈[0,1]; PREDICTED_RULE if invalid
```

---

## 7. Modeling Notes & Tips

- **Tokenization:** one token per step string; vocab ~120. Family is a strong conditioning signal — include it as a context token.
- **Order matters:** several rules are ordering constraints; positional/structural learning beats pure frequency models.
- **Lengths:** MOSFET ~125, IGBT ~148, IC ~115 steps.
- **Scaling experiment (stretch):** compare 100 vs 1,000 vs 5,000+ training sequences.
- **Suggested stack:** Python + PyTorch (transformer/LLM/sequence models), experiment
  tracking (W&B / MLflow), training on Leonardo cluster (SLURM; GPU quota TBD).
- **Pitfalls:** synthetic-data distribution quality, fair model comparison, meaningful
  generalization tests, robust checkpointing, benchmarks that measure more than memorization.

---

## 8. Source Files (read these for full detail)

- `Track_industrial_en.md` / `Track_industrial.md` — full track briefing (EN / DE)
- `README.md` — repo overview & quickstart
- `training_data/README.md` — data & eval quickstart
- `training_data/generation_rules.md` — **authoritative** grammar, 10 rules, eval protocol
- `training_data/generate_sequences.py` — generator/validator
- `training_data/eval_metrics.py` — official scorer
