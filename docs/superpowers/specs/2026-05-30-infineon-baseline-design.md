# Infineon Baseline — Design Spec

**Date:** 2026-05-30
**Status:** Approved (awaiting user review)
**Track:** Industrial AI (Track 1) — Learning & Benchmarking Process Logic
**Authoritative reference for the track:** `tracks/industrial-infineon/CLAUDE.md`
**Authoritative grammar / rules:** `tracks/industrial-infineon/training_data/generation_rules.md`

---

## 1. Context

The Industrial AI track asks teams to train sequence models on synthetic
semiconductor fabrication sequences and benchmark them on three submission tasks
(next-step prediction, sequence completion, anomaly detection) plus a hidden OOD
generalization eval. The track explicitly rewards a *baseline-vs-trained* comparison
and a reproducible end-to-end workflow.

Two practical gaps shape this design:

1. **The organizers' eval files (`eval_input_valid.csv`, `eval_input_anomaly.csv`)
   and scorer (`eval_metrics.py`) are not yet in the repo** — they ship at hackathon
   start. We need a measurable pipeline *before* they arrive.
2. **The eventual trained transformer needs a harness around it** — data loading,
   tokenization, submission writers, metrics — and that harness should not depend
   on the model choice.

This spec covers building that harness plus a statistical n-gram baseline that
runs through it end-to-end. The transformer is **out of scope** for this spec; it
will plug into the same interfaces in a follow-up spec.

---

## 2. Goal & Non-Goals

### Goal

Deliver a Python package + CLI under `tracks/industrial-infineon/` that:

- Reads the 3,000 training variants from `training_data/*_variants.csv`.
- Builds a held-out evaluation set in the organizers' exact file format,
  including injected rule violations for the anomaly task.
- Fits a family-conditioned n-gram model with backoff on the training split.
- Produces spec-compliant submission files for all three tasks.
- Scores those submissions with our own metric implementations matching
  `generation_rules.md §5.2`.
- Is fully reproducible from a single `--seed`.

### Non-goals

- Training a transformer or any neural model (separate spec).
- Optimizing for the hidden Task 4 (OOD generalization) — out of our control.
- A frontend / dashboard (the track lists it as optional bonus only).
- Cluster orchestration (Leonardo SLURM scripts) — the baseline is CPU-only.

---

## 3. Architecture

```
training_data/*_variants.csv  (3,000 sequences, ~388k rows)
        │
        ▼
   [loaders]  ──► dict[seq_id → list[step_str]]   per family
        │
        ▼
   [tokenizer]  ──► step_str ↔ int id  + family-id; one shared vocab
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
   [eval_set]                              [ngram]
   • 200/family hold-out                   fit on the OTHER 800/family
   • truncate @ 60% & 80%                  family-conditioned, order-3
     → eval_input_valid.csv (600)          stupid backoff
   • inject rule violations                .top_k(prefix, k)
     → eval_input_anomaly.csv (~987)       .log_prob(seq) → perplexity
   • write ground-truth companions
        │                                      │
        └─────────────┬────────────────────────┘
                      ▼
              [predictor]  (task-level orchestrator)
                ├─ Task 1: top-5 next steps via ngram.top_k
                ├─ Task 2: greedy autoregressive completion (optional --constrain)
                └─ Task 3: hybrid → validate_sequence + ngram perplexity
                      │
                      ▼
              [submission]  ──► submission_task{1,2,3}.csv  (spec format)
                      │
                      ▼
              [metrics]  ──► console table + report.json
                            Top-1/3/5, MRR, ExactMatch, EditDist,
                            Token/Block acc, BinaryAcc, P/R/F1, ROC-AUC,
                            RuleAttrAcc; broken down per family
```

When the organizers' real `eval_input_*.csv` arrives, we point the predictor at
those files directly; the rest of the pipeline is unchanged.

---

## 4. File / Module Layout

All new code lives under `tracks/industrial-infineon/`. The existing
`training_data/` directory is **not modified**.

```
tracks/industrial-infineon/
├── CLAUDE.md                        ← existing (track-wide reference)
├── README.md  Track_industrial*.md  ← existing (untouched)
├── training_data/                   ← existing (untouched)
│
├── src/
│   └── infineon_baseline/
│       ├── __init__.py              ← exports public API; one sys.path nudge
│       │                              so `from generate_sequences import validate_sequence`
│       │                              works without copying code
│       ├── loaders.py               ← variant CSV → dict[seq_id, list[step_str]]
│       ├── tokenizer.py             ← Tokenizer: encode/decode, save/load JSON
│       ├── eval_set.py              ← split, truncate, inject_violation,
│       │                              write_eval_inputs, write_ground_truth
│       ├── ngram.py                 ← NGram(order, backoff="stupid"):
│       │                              fit, top_k, log_prob; save/load
│       ├── anomaly.py               ← detect_oracle(steps),
│       │                              detect_perplexity(steps, ngram),
│       │                              detect_hybrid(steps, ngram)
│       ├── predictor.py             ← run_task1/2/3(model, eval_input) → submission rows
│       ├── submission.py            ← write_task{1,2,3}_csv; validates row shape
│       ├── metrics.py               ← pure functions: top_k_acc, mrr, em_rate,
│       │                              ned, token_acc, block_acc, binary_acc,
│       │                              precision/recall/f1, roc_auc, rule_attr_acc;
│       │                              report(task, preds, gt) → dict + console table
│       └── cli.py                   ← argparse; subcommands: build-eval | fit | predict | score
│
├── tests/
│   ├── fixtures/
│   │   └── mini_sequences.py        ← ~20 toy seqs/family, 12-step vocab; in-memory
│   ├── test_loaders.py
│   ├── test_tokenizer.py
│   ├── test_eval_set.py
│   ├── test_ngram.py
│   ├── test_anomaly.py
│   ├── test_predictor.py
│   ├── test_submission.py
│   ├── test_metrics.py
│   └── test_cli_smoke.py            ← e2e on 10 seqs/family from real corpus
│
├── outputs/                         ← gitignored: splits, models, submissions, reports
│   └── .gitkeep
├── docs/
│   └── superpowers/specs/
│       └── 2026-05-30-infineon-baseline-design.md  ← this file
├── pyproject.toml                   ← package metadata, deps, `infineon-baseline` entry point
├── requirements.txt                 ← pinned: numpy, pandas, scikit-learn, pytest
└── .gitignore                       ← outputs/, __pycache__, .venv, *.egg-info, .pytest_cache
```

**Module dependency rule:** files only depend on files to their left in the
above pipeline diagram. `predictor.py` and `cli.py` are the only orchestrators.
No module imports from any module to its right.

---

## 5. Module Responsibilities (Detailed)

### 5.1 `loaders.py`

Single public function:

```python
def load_variants(csv_path: Path, family: str) -> dict[str, list[str]]:
    """Read a *_variants.csv in long format → {sequence_id: [step_str, ...]}."""
```

Validates columns `SEQUENCE_ID`, `STEP`. Raises `ValueError` with file + row
number on malformed input. Pure I/O — no business logic.

### 5.2 `tokenizer.py`

```python
class Tokenizer:
    step_to_id: dict[str, int]
    id_to_step: list[str]
    family_to_id: dict[str, int]   # {"mosfet": 0, "igbt": 1, "ic": 2}

    @classmethod
    def fit(cls, sequences_by_family: dict[str, dict[str, list[str]]]) -> "Tokenizer": ...
    def encode(self, family: str, steps: list[str]) -> tuple[int, list[int]]: ...
    def decode(self, ids: list[int]) -> list[str]: ...
    def save(self, path: Path) -> None: ...      # JSON
    @classmethod
    def load(cls, path: Path) -> "Tokenizer": ...
```

Vocabulary is the **sorted union** of all step strings across families →
deterministic IDs across runs. Family IDs are sorted alphabetically for
the same reason. Unknown step at `encode()` time raises `KeyError` —
the loader is responsible for not feeding garbage.

### 5.3 `eval_set.py`

```python
def split(seqs_by_family: dict[str, dict[str, list[str]]],
          holdout_per_family: int = 200,
          seed: int = 42) -> tuple[TrainSplit, HoldoutSplit]: ...

def truncate(seq: list[str], fraction: float) -> Cut: ...
    # Cut(partial, ground_truth, cut_index)

def inject_violation(seq: list[str], rule_id: str, rng: Random) -> tuple[list[str], str]: ...
    # Returns (corrupted_seq, actual_rule_id_triggered).
    # Round-trips through validate_sequence; retries on no-trigger.
    # Raises RuntimeError if injection cannot trigger the requested rule
    # after N attempts (defensive — should never happen for the 10 supported rules).

def write_eval_inputs(holdout: HoldoutSplit, out_dir: Path, seed: int) -> None: ...
def write_ground_truth(holdout: HoldoutSplit, out_dir: Path) -> None: ...
```

Eval input formats match `generation_rules.md §5.1`:

- `eval_input_valid.csv`: `EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE`
  (600 rows: 100/family × 3 × 2 cuts; `PARTIAL_SEQUENCE` pipe-separated).
- `eval_input_anomaly.csv`: `EXAMPLE_ID, FAMILY, SEQUENCE`
  (~987 rows: 200/family valid + ~129/family invalid, shuffled, unlabeled;
  ratio matches the organizers' 387/987 ≈ 39%).

**Source allocation from the 200/family hold-out:** all 200 held-out sequences
per family are used as full-length valid inputs in `eval_input_anomaly.csv`;
100 of those same 200 (per family) are *also* used for Tasks 1 & 2 in truncated
form. There is no leakage because each task sees a different transformation of
the sequence (truncated vs. full vs. corrupted), and the training split never
overlaps with the held-out 200/family.

Ground-truth companions (never seen by the model):

- `ground_truth_valid.csv`: `EXAMPLE_ID, FULL_SEQUENCE`
- `ground_truth_anomaly.csv`: `EXAMPLE_ID, IS_VALID, RULE_VIOLATED`

One injector function per rule; the 10 rules are dispatched from a registry dict.
Each injector must produce a sequence where `validate_sequence` returns **at least
one violation matching the requested rule**. Cascading violations are tolerated
(e.g. deleting the DEVELOP before a metal etch may trigger both
`RULE_ETCH_NO_MASK` and `RULE_METAL_ETCH_NO_LITHO`); we record the requested
rule as the ground truth. The verification — round-tripping each injection
through `validate_sequence` — is enforced in code. If `max_attempts` injections
all fail to trigger the target rule, the injector raises `RuntimeError`.

### 5.4 `ngram.py`

```python
class NGram:
    order: int                                    # default 3
    counts: dict[str, dict[tuple, Counter]]       # per-family: prefix → next-step counts
    unigram: dict[str, Counter]                   # per-family fallback

    def fit(self, train_split: TrainSplit) -> "NGram": ...
    def top_k(self, family: str, prefix: list[int], k: int = 5) -> list[int]: ...
    def log_prob(self, family: str, sequence: list[int]) -> float: ...
    def save(self, path: Path) -> None: ...       # pickle or JSON
    @classmethod
    def load(cls, path: Path) -> "NGram": ...
```

**Stupid backoff:** if `(order-1)`-step prefix is unseen, drop one step from the
prefix and retry. Falls through to unigram. No smoothing weights. Probability of
an unseen `(prefix, next)` pair is the unigram probability of `next`, with a
floor of `1e-9` to keep `log_prob` finite.

Optional flag `--mask-family` zeroes counts for steps that never appeared in
that family in training — prevents cross-family leakage at the cost of recall on
rare-but-legitimate steps.

### 5.5 `anomaly.py`

```python
def detect_oracle(steps: list[str]) -> AnomalyResult: ...
    # Calls validate_sequence; returns (is_valid, predicted_rule, score=None).

def detect_perplexity(steps: list[str], ngram: NGram, family: str,
                      threshold: float) -> AnomalyResult: ...
    # Computes log_prob; below threshold → is_valid=0; score = sigmoid-normalized lp.

def detect_hybrid(steps: list[str], ngram: NGram, family: str,
                  threshold: float) -> AnomalyResult: ...
    # is_valid + predicted_rule from oracle; score from perplexity.
    # This is the primary submission strategy.
```

Threshold is calibrated on the held-out training split (not the eval set!) —
choose the value that maximizes F1 against synthetic violations injected into
training-set sequences. Calibration is a function in `eval_set.py` that
`predictor.py` invokes once before generating Task 3 outputs.

### 5.6 `predictor.py`

Three task-level functions; each returns an iterator over submission rows.

```python
def run_task1(ngram: NGram, eval_input_valid_path: Path) -> Iterator[Task1Row]: ...
def run_task2(ngram: NGram, eval_input_valid_path: Path,
              constrain: bool = False) -> Iterator[Task2Row]: ...
def run_task3(ngram: NGram, eval_input_anomaly_path: Path,
              threshold: float) -> Iterator[Task3Row]: ...
```

Task 2 uses greedy autoregressive decoding: pick top-1, append, repeat until
`SHIP LOT` or a hard length cap (`1.5 × family_max_seen_length`). With
`--constrain`, the decoder filters top-k candidates to those that don't
immediately trigger a rule violation; if the constrained set is empty, fall
back to unconstrained top-1.

### 5.7 `submission.py`

Three writers, one per task, each enforcing the spec format. Refuses to write
malformed rows — raises `ValueError` with `EXAMPLE_ID` and the violated invariant.
Co-emits `<task>_meta.json` with: model name, n-gram order, seed, eval split
hash (SHA-256 of the sorted EXAMPLE_IDs), timestamp, git commit (if available).

### 5.8 `metrics.py`

Pure functions; signatures `(predictions_df, ground_truth_df) -> float | dict`.
Sklearn for standard classification metrics (`roc_auc_score`,
`precision_recall_fscore_support`, `confusion_matrix`). Hand-rolled for
sequence/task-specific ones (`top_k_accuracy`, `mrr`, `exact_match_rate`,
`normalized_edit_distance`, `token_accuracy`, `block_accuracy`,
`rule_attribution_accuracy`).

`report(task: str, predictions: Path, ground_truth: Path) -> dict` returns the
full per-family + aggregate metrics dict and prints a formatted table.

**Block-level accuracy** groups steps into the 12 functional categories from
`generation_rules.md §1` (Logistics, Cleaning, Thermal/Deposition, Litho,
Etch, Strip, Implant/Diffusion, CMP, Via Fill, Measurement, Electrical Tests,
Substrate Prep), then compares the *sequence of block categories* in the
prediction vs. ground truth. This catches "got the shape right, picked
synonyms" cases that token-level metrics punish.

### 5.9 `cli.py`

`argparse` subcommands. Default seed is `42` everywhere.

```
infineon-baseline build-eval  --variants-dir <dir> --out <dir> --seed 42
                              [--holdout-per-family 200]
                              [--anomaly-invalid-ratio 0.39]

infineon-baseline fit         --train <dir> --out <path> [--order 3]
                              [--mask-family]

infineon-baseline predict     --model <path> --eval-input <path>
                              --task {next-step,complete,anomaly}
                              --out <path> [--constrain]
                              [--anomaly-strategy {oracle,perplexity,hybrid}]

infineon-baseline score       --predictions <path> --ground-truth <path>
                              --task {next-step,complete,anomaly}
                              [--report-json <path>]
```

A top-level `--seed` flag overrides the default for all stochastic operations
in that invocation.

---

## 6. Output Artifacts

All under `outputs/` (gitignored). One canonical run produces:

```
outputs/
├── eval/
│   ├── train_split.csv               ← seqs used to fit n-gram
│   ├── eval_input_valid.csv          ← 600 rows, spec format
│   ├── eval_input_anomaly.csv        ← ~987 rows, spec format
│   ├── ground_truth_valid.csv        ← held back from model
│   ├── ground_truth_anomaly.csv      ← held back from model
│   └── split_meta.json               ← seed, sizes, hash
├── models/
│   └── ngram_o3.pkl                  ← fitted model
├── submissions/
│   ├── submission_task1.csv
│   ├── submission_task1_meta.json
│   ├── submission_task2.csv
│   ├── submission_task2_meta.json
│   ├── submission_task3.csv
│   └── submission_task3_meta.json
└── reports/
    ├── report_task1.json
    ├── report_task2.json
    └── report_task3.json
```

---

## 7. Testing Strategy

### Pyramid

- **Unit tests** (per module) carry most of the weight; all run on
  `tests/fixtures/mini_sequences.py` — an in-memory toy dataset with a 12-step
  vocabulary and ~20 sequences per family. Whole suite < 2 s.
- **Integration tests** (`test_predictor.py`) exercise task-level orchestration
  end-to-end on the fixture.
- **CLI smoke test** (`test_cli_smoke.py`) runs the four real subcommands
  sequentially on 10 sequences per family sampled from the actual corpus.
  Asserts files exist with correct columns. < 2 s.

### TDD discipline

Per the `writing-plans` skill, every implementation task starts with a failing
test, runs to confirm failure, then adds the minimal implementation. Commits
follow the test-then-code pair.

### Critical invariants enforced by tests

- `inject_violation(seq, rule_id) → corrupted` ⇒ `validate_sequence(corrupted)`
  contains exactly one violation, and its `rule_id` equals the requested one.
- `Tokenizer.encode → decode` round-trips losslessly.
- Two runs of the full pipeline with the same seed produce byte-identical
  `eval_input_*.csv`, model, submissions, and reports.
- Submission writers refuse malformed rows (covered by negative tests).

---

## 8. Error Handling

Three strict-validation boundaries; everything inside trusts itself.

1. **CSV loaders** validate column names, non-empty sequences, valid family
   values; bad inputs raise `ValueError` with file path and row index.
2. **Submission writers** refuse to emit rows that violate the spec (missing
   ranks, illegal `IS_VALID`, echo-prefix in Task 2). Loud failure beats a
   silently invalid submission.
3. **CLI** uses `argparse` for arg validation; defaults reproducibility (`--seed`)
   to opt-out, not opt-in.

Between internal modules: type hints, no defensive `if x is None` checks.

---

## 9. Defaults

| Knob | Default | Source |
|---|---|---|
| N-gram order | 3 | Classical trigram; ablation over {1,2,3,4} included in plan |
| Backoff | stupid backoff | Simplest reasonable choice |
| Held-out size | 200 sequences/family | 800/family remain for training |
| Anomaly invalid ratio | 0.39 | Mirrors organizers' 387/987 |
| Random seed | 42 | Single seed for all stochastic operations |
| `--constrain` | off | Honest unconstrained baseline by default |
| `--mask-family` | off | Same reasoning |
| Anomaly strategy | hybrid | Best of both worlds; flag-overridable |

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Organizers' eval CSV format differs subtly from the spec | Loaders are isolated; only the adapter changes if format differs. |
| Our metric implementations differ from `eval_metrics.py` | Cross-check when their script lands; until then our numbers serve as a relative reference. |
| Some rule injections are hard (e.g. `RULE_LITHO_LEVEL_SKIP`) | One injector per rule; verified by round-trip through `validate_sequence`; retry on failure; loud error if not triggerable. |
| N-gram ceiling on Task 2 (no global structure awareness) | `--constrain` flag adds rule-aware decoding; included as ablation. |
| Cross-family leakage in predictions | `--mask-family` flag zeroes non-family steps. |
| Task 4 (OOD) — hidden family | Explicitly out of scope; baseline reports whatever it scores. |

---

## 11. Success Criteria

A binary checklist for "done":

- [ ] `pytest` runs all tests green in under 5 seconds.
- [ ] `infineon-baseline build-eval` produces eval inputs + ground truths in spec format.
- [ ] `infineon-baseline fit` trains the n-gram in under 30 seconds on a laptop.
- [ ] `infineon-baseline predict` produces all three submission CSVs in spec format.
- [ ] `infineon-baseline score` prints metric tables per family and writes `report.json`.
- [ ] Two runs with the same seed produce byte-identical artifacts.
- [ ] A short `README.md` at the project root documents the four commands.

---

## 12. Out of Scope (Explicitly)

- Transformer or any neural model.
- Cluster (Leonardo / SLURM) integration.
- Visualization / dashboard.
- Optimization for Task 4 (OOD generalization).
- Use of the `*_Longdescr.csv` / `*_longdescription_parameters.csv` enriched
  files — bare step sequences are sufficient for the baseline.

---

## 13. References

- Track-wide reference (always loaded): `tracks/industrial-infineon/CLAUDE.md`
- Authoritative grammar, rules, eval protocol: `tracks/industrial-infineon/training_data/generation_rules.md`
- Validator implementation (reused as oracle): `tracks/industrial-infineon/training_data/generate_sequences.py` — `validate_sequence(steps) → list[Violation]`
- Official track docs: https://docs.zero-one.lumos-consulting.at/tracks/track-1/
