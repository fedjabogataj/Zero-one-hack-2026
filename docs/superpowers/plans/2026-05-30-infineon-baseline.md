# Infineon Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Python library + CLI that reads `training_data/*_variants.csv`, builds its own held-out eval set in the organizers' file format, fits a family-conditioned n-gram baseline with backoff, produces spec-compliant submission files for all three Industrial AI tasks, and self-scores with metrics matching `generation_rules.md §5.2`.

**Architecture:** Library under `src/infineon_baseline/` with focused modules (loaders, tokenizer, ngram, eval_set, anomaly, predictor, submission, metrics) wired by a single `cli.py`. Tests mirror modules; a tiny in-memory fixture exercises everything; one CLI smoke test runs end-to-end on 10 sequences/family. The harness is model-agnostic so the eventual transformer plugs in without rework.

**Tech Stack:** Python ≥3.10, numpy, pandas, scikit-learn, pytest. No deep-learning deps — the baseline is statistical.

**Spec:** `docs/superpowers/specs/2026-05-30-infineon-baseline-design.md`
**Track reference (auto-loaded):** `CLAUDE.md`
**Grammar / rules / oracle:** `training_data/generation_rules.md`, `training_data/generate_sequences.py`

**Notes for the executing engineer:**
- All paths in this plan are relative to `tracks/industrial-infineon/` (the repo root for the new git repo).
- The `Violation` dataclass returned by `validate_sequence` exposes a `.rule` field (not `.rule_id`). The 10 rule IDs are the strings `RULE_DEP_NO_CLEAN`, `RULE_METAL_ETCH_NO_LITHO`, `RULE_ETCH_NO_MASK`, `RULE_LITHO_LEVEL_SKIP`, `RULE_IMPLANT_NO_MASK`, `RULE_CMP_NO_DEP`, `RULE_PAD_OPEN_BEFORE_DEP`, `RULE_TEST_BEFORE_PASSIVATION`, `RULE_SHIP_BEFORE_TEST`, `RULE_BACKSIDE_BEFORE_PASSIVATION`.
- Step strings are case-sensitive (always uppercase in the data).
- Never add `Co-Authored-By` trailers to commit messages.

---

## File Structure (locked at plan time)

```
src/infineon_baseline/
├── __init__.py            ← public API + sys.path nudge for generate_sequences
├── loaders.py             ← load_variants(csv, family) → dict[seq_id, list[str]]
├── tokenizer.py           ← Tokenizer.fit/encode/decode/save/load
├── ngram.py               ← NGram: fit/top_k/log_prob/save/load + stupid backoff
├── eval_set.py            ← split/truncate/inject_violation/write_eval_inputs
├── anomaly.py             ← detect_oracle/detect_perplexity/detect_hybrid
├── predictor.py           ← run_task1/run_task2/run_task3
├── submission.py          ← write_task{1,2,3}_csv + meta json
├── metrics.py             ← all metrics + report()
└── cli.py                 ← argparse: build-eval | fit | predict | score
tests/
├── fixtures/mini_sequences.py
├── test_loaders.py / test_tokenizer.py / test_ngram.py
├── test_eval_set.py / test_anomaly.py / test_predictor.py
├── test_submission.py / test_metrics.py
└── test_cli_smoke.py
pyproject.toml             ← package, deps, entry point
```

---

## Task Index

| # | Task | Module |
|---|------|--------|
| 1 | Bootstrap project (pyproject, deps, package skeleton) | — |
| 2 | Test fixtures (mini_sequences) | tests/fixtures |
| 3 | Loaders — read variant CSVs | loaders.py |
| 4 | Tokenizer — fit/encode/decode/save/load | tokenizer.py |
| 5 | NGram — fit + top_k (with stupid backoff) | ngram.py |
| 6 | NGram — log_prob | ngram.py |
| 7 | NGram — save/load | ngram.py |
| 8 | eval_set — split + truncate | eval_set.py |
| 9 | eval_set — inject_violation (10 rules) | eval_set.py |
| 10 | eval_set — write_eval_inputs + write_ground_truth | eval_set.py |
| 11 | anomaly — detect_oracle | anomaly.py |
| 12 | anomaly — detect_perplexity + calibrate_threshold | anomaly.py |
| 13 | anomaly — detect_hybrid | anomaly.py |
| 14 | predictor — run_task1 (next-step) | predictor.py |
| 15 | predictor — run_task2 (greedy completion) | predictor.py |
| 16 | predictor — run_task3 (anomaly) | predictor.py |
| 17 | submission writers (all three tasks) | submission.py |
| 18 | metrics — sequence (top_k, mrr, em, ned, token_acc, block_acc) | metrics.py |
| 19 | metrics — anomaly (binary, P/R/F1, ROC-AUC, rule_attr) | metrics.py |
| 20 | metrics — report() with per-family breakdown + JSON dump | metrics.py |
| 21 | CLI — all four subcommands | cli.py |
| 22 | CLI smoke test (end-to-end on real corpus sample) | tests/test_cli_smoke.py |
| 23 | Constrained decoder for Task 2 (`--constrain`) | predictor.py |
| 24 | Family-vocab mask for ngram (`--mask-family`) | ngram.py |
| 25 | README.md documenting the CLI | README.md |

---

## Task 1: Bootstrap the project

**Files:**
- Create: `pyproject.toml`
- Create: `src/infineon_baseline/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_bootstrap.py`
- Create: `outputs/.gitkeep`

- [ ] **Step 1: Write the failing test**

`tests/test_bootstrap.py`:
```python
def test_package_imports():
    import infineon_baseline
    assert hasattr(infineon_baseline, "__version__")

def test_validate_sequence_accessible_via_package():
    """sys.path nudge in __init__ must make generate_sequences importable."""
    from infineon_baseline import validate_sequence
    assert validate_sequence(["RECEIVE WAFER LOT", "SHIP LOT"]) is not None
```

- [ ] **Step 2: Run test, expect ImportError**

```bash
pytest tests/test_bootstrap.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'infineon_baseline'`).

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "infineon-baseline"
version = "0.1.0"
description = "Statistical baseline + harness for the Industrial AI hackathon track"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.1",
    "scikit-learn>=1.4",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
infineon-baseline = "infineon_baseline.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 4: Write `src/infineon_baseline/__init__.py`**

```python
"""Statistical baseline + harness for the Industrial AI hackathon track."""
from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.1.0"

# Make generate_sequences.py (and validate_sequence) importable from training_data/
# without copying or vendoring its code.
_TRAINING_DATA = Path(__file__).resolve().parents[2] / "training_data"
if str(_TRAINING_DATA) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DATA))

from generate_sequences import validate_sequence, Violation  # noqa: E402

__all__ = ["__version__", "validate_sequence", "Violation"]
```

- [ ] **Step 5: Write `tests/__init__.py` (empty) and `tests/conftest.py`**

`tests/__init__.py` is empty. `tests/conftest.py`:
```python
"""Shared pytest fixtures for the infineon_baseline test suite."""
```

- [ ] **Step 6: Create `outputs/.gitkeep`**

Empty file (so the directory exists in git but its contents are gitignored).

- [ ] **Step 7: Install in editable mode and run tests**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/test_bootstrap.py -v
```
Expected: PASS (both tests).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/ tests/ outputs/.gitkeep
git commit -m "feat: bootstrap infineon_baseline package + first tests"
```

---

## Task 2: Test fixtures (mini sequences)

**Files:**
- Create: `tests/fixtures/__init__.py` (empty)
- Create: `tests/fixtures/mini_sequences.py`
- Create: `tests/test_fixtures.py`

- [ ] **Step 1: Write the failing test**

`tests/test_fixtures.py`:
```python
from tests.fixtures.mini_sequences import MINI_DATASET, MINI_VOCAB, FAMILIES

def test_mini_dataset_has_three_families():
    assert set(MINI_DATASET.keys()) == set(FAMILIES) == {"mosfet", "igbt", "ic"}

def test_each_family_has_at_least_ten_sequences():
    for family in FAMILIES:
        assert len(MINI_DATASET[family]) >= 10

def test_all_sequences_start_with_receive_and_end_with_ship():
    for family in FAMILIES:
        for seq_id, steps in MINI_DATASET[family].items():
            assert steps[0] == "RECEIVE WAFER LOT", f"{family}/{seq_id}"
            assert steps[-1] == "SHIP LOT", f"{family}/{seq_id}"

def test_vocab_is_union_of_all_steps():
    seen = set()
    for family in FAMILIES:
        for steps in MINI_DATASET[family].values():
            seen.update(steps)
    assert seen == set(MINI_VOCAB)
```

- [ ] **Step 2: Run test, expect ImportError**

```bash
pytest tests/test_fixtures.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'tests.fixtures.mini_sequences'`).

- [ ] **Step 3: Write `tests/fixtures/mini_sequences.py`**

```python
"""Tiny in-memory dataset for fast unit tests.

12-step vocabulary, ~10 sequences per family. Exercises the same grammatical
shapes as the real data (prefix, litho cycle, etch, suffix) so we don't need
to load the 125k-row variant CSVs in unit tests.
"""
from __future__ import annotations

FAMILIES = ("mosfet", "igbt", "ic")

MINI_VOCAB = [
    "RECEIVE WAFER LOT",
    "LOT IDENTIFICATION",
    "PRE CLEAN WAFER",
    "THERMAL OXIDATION",
    "SPIN COAT PHOTORESIST",
    "EXPOSE LITHO LEVEL 1",
    "DEVELOP PHOTORESIST",
    "OXIDE ETCH",
    "STRIP PHOTORESIST",
    "CLEAN AFTER ETCH",
    "WAFER SORT TEST",
    "SHIP LOT",
]

_BASE = [
    "RECEIVE WAFER LOT",
    "LOT IDENTIFICATION",
    "PRE CLEAN WAFER",
    "THERMAL OXIDATION",
    "SPIN COAT PHOTORESIST",
    "EXPOSE LITHO LEVEL 1",
    "DEVELOP PHOTORESIST",
    "OXIDE ETCH",
    "STRIP PHOTORESIST",
    "CLEAN AFTER ETCH",
    "WAFER SORT TEST",
    "SHIP LOT",
]


def _vary(seed: int) -> list[str]:
    """Produce a small grammatical variation: optionally double the litho cycle."""
    seq = list(_BASE)
    if seed % 3 == 0:
        # Double the litho cycle to create length variation.
        insert_at = seq.index("EXPOSE LITHO LEVEL 1") + 4  # after CLEAN AFTER ETCH
        seq[insert_at:insert_at] = [
            "SPIN COAT PHOTORESIST",
            "EXPOSE LITHO LEVEL 1",
            "DEVELOP PHOTORESIST",
            "OXIDE ETCH",
            "STRIP PHOTORESIST",
            "CLEAN AFTER ETCH",
        ]
    return seq


MINI_DATASET: dict[str, dict[str, list[str]]] = {
    family: {f"{family}_{i:04d}": _vary(i) for i in range(1, 11)}
    for family in FAMILIES
}
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_fixtures.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/ tests/test_fixtures.py
git commit -m "test: add mini_sequences fixture for fast unit tests"
```

---

## Task 3: Loaders

**Files:**
- Create: `src/infineon_baseline/loaders.py`
- Create: `tests/test_loaders.py`

- [ ] **Step 1: Write the failing test**

`tests/test_loaders.py`:
```python
import csv
from pathlib import Path

import pytest

from infineon_baseline.loaders import load_variants


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SEQUENCE_ID", "STEP"])
        for r in rows:
            w.writerow(r)


def test_load_variants_groups_by_sequence_id(tmp_path):
    csv_path = tmp_path / "MOSFET_variants.csv"
    _write_csv(csv_path, [
        ("seq_0001", "RECEIVE WAFER LOT"),
        ("seq_0001", "LOT IDENTIFICATION"),
        ("seq_0002", "RECEIVE WAFER LOT"),
        ("seq_0002", "SHIP LOT"),
    ])
    result = load_variants(csv_path)
    assert result == {
        "seq_0001": ["RECEIVE WAFER LOT", "LOT IDENTIFICATION"],
        "seq_0002": ["RECEIVE WAFER LOT", "SHIP LOT"],
    }


def test_load_variants_preserves_row_order(tmp_path):
    csv_path = tmp_path / "v.csv"
    _write_csv(csv_path, [
        ("s", "STEP_A"),
        ("s", "STEP_B"),
        ("s", "STEP_C"),
    ])
    assert load_variants(csv_path)["s"] == ["STEP_A", "STEP_B", "STEP_C"]


def test_load_variants_rejects_missing_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("WRONG_COL,OTHER\nx,y\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_variants(csv_path)
```

- [ ] **Step 2: Run test, expect failure**

```bash
pytest tests/test_loaders.py -v
```
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `src/infineon_baseline/loaders.py`**

```python
"""Variant-CSV loaders."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def load_variants(csv_path: Path) -> dict[str, list[str]]:
    """Read a long-format `*_variants.csv` into {sequence_id: [step, ...]}.

    Required columns: SEQUENCE_ID, STEP. Row order within a sequence is preserved.
    """
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "SEQUENCE_ID" not in reader.fieldnames or "STEP" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path}: missing required column(s); "
                f"need SEQUENCE_ID and STEP, got {reader.fieldnames}"
            )
        out: dict[str, list[str]] = defaultdict(list)
        for row in reader:
            out[row["SEQUENCE_ID"]].append(row["STEP"])
    return dict(out)
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_loaders.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/loaders.py tests/test_loaders.py
git commit -m "feat(loaders): read long-format variant CSVs into seq-id dict"
```

---

## Task 4: Tokenizer

**Files:**
- Create: `src/infineon_baseline/tokenizer.py`
- Create: `tests/test_tokenizer.py`

- [ ] **Step 1: Write the failing test**

`tests/test_tokenizer.py`:
```python
import json
from pathlib import Path

import pytest

from infineon_baseline.tokenizer import Tokenizer
from tests.fixtures.mini_sequences import MINI_DATASET, MINI_VOCAB


def test_fit_builds_sorted_deterministic_vocab():
    tok = Tokenizer.fit(MINI_DATASET)
    assert tok.id_to_step == sorted(MINI_VOCAB)
    # All step_to_id values are unique, 0..N-1
    assert sorted(tok.step_to_id.values()) == list(range(len(MINI_VOCAB)))
    # Families sorted alphabetically
    assert tok.family_to_id == {"ic": 0, "igbt": 1, "mosfet": 2}


def test_encode_decode_round_trip():
    tok = Tokenizer.fit(MINI_DATASET)
    steps = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION", "SHIP LOT"]
    family_id, ids = tok.encode("mosfet", steps)
    assert family_id == tok.family_to_id["mosfet"]
    assert tok.decode(ids) == steps


def test_encode_unknown_step_raises():
    tok = Tokenizer.fit(MINI_DATASET)
    with pytest.raises(KeyError):
        tok.encode("mosfet", ["NOT A REAL STEP"])


def test_encode_unknown_family_raises():
    tok = Tokenizer.fit(MINI_DATASET)
    with pytest.raises(KeyError):
        tok.encode("nope", ["RECEIVE WAFER LOT"])


def test_save_and_load_round_trip(tmp_path):
    tok = Tokenizer.fit(MINI_DATASET)
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.step_to_id == tok.step_to_id
    assert loaded.id_to_step == tok.id_to_step
    assert loaded.family_to_id == tok.family_to_id


def test_two_fits_on_same_data_produce_identical_tokenizers():
    a = Tokenizer.fit(MINI_DATASET)
    b = Tokenizer.fit(MINI_DATASET)
    assert a.step_to_id == b.step_to_id
    assert a.family_to_id == b.family_to_id
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/test_tokenizer.py -v
```
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `src/infineon_baseline/tokenizer.py`**

```python
"""Step-string ↔ integer-id mapping plus family ids."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Tokenizer:
    step_to_id: dict[str, int] = field(default_factory=dict)
    id_to_step: list[str] = field(default_factory=list)
    family_to_id: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, sequences_by_family: dict[str, dict[str, list[str]]]) -> "Tokenizer":
        vocab: set[str] = set()
        for fam_seqs in sequences_by_family.values():
            for steps in fam_seqs.values():
                vocab.update(steps)
        id_to_step = sorted(vocab)
        step_to_id = {step: i for i, step in enumerate(id_to_step)}
        family_to_id = {fam: i for i, fam in enumerate(sorted(sequences_by_family.keys()))}
        return cls(step_to_id=step_to_id, id_to_step=id_to_step, family_to_id=family_to_id)

    def encode(self, family: str, steps: list[str]) -> tuple[int, list[int]]:
        if family not in self.family_to_id:
            raise KeyError(f"unknown family {family!r}; known: {sorted(self.family_to_id)}")
        family_id = self.family_to_id[family]
        ids = [self.step_to_id[s] for s in steps]  # raises KeyError on unknown step
        return family_id, ids

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_step[i] for i in ids]

    def save(self, path: Path) -> None:
        data = {
            "id_to_step": self.id_to_step,
            "family_to_id": self.family_to_id,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text())
        id_to_step = list(data["id_to_step"])
        return cls(
            step_to_id={s: i for i, s in enumerate(id_to_step)},
            id_to_step=id_to_step,
            family_to_id=dict(data["family_to_id"]),
        )
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_tokenizer.py -v
```
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/tokenizer.py tests/test_tokenizer.py
git commit -m "feat(tokenizer): step/family id mapping with save/load"
```

---

## Task 5: NGram model — fit + top_k (with stupid backoff)

**Files:**
- Create: `src/infineon_baseline/ngram.py`
- Create: `tests/test_ngram.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ngram.py`:
```python
import pytest

from infineon_baseline.ngram import NGram


def _toy_corpus() -> dict[str, list[list[int]]]:
    # Two MOSFET sequences in id-space; vocab = {0..4}
    return {
        "mosfet": [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 4],
        ],
        "igbt": [[0, 2, 4]],
    }


def test_top_k_after_seen_prefix():
    model = NGram(order=2).fit(_toy_corpus())
    # After "1, 2" we always see "3" twice; after "0,1,2,4" once. So top_k("mosfet", (1,2)) starts with 3.
    top = model.top_k("mosfet", (1, 2), k=2)
    assert top[0] == 3


def test_top_k_returns_at_most_k():
    model = NGram(order=3).fit(_toy_corpus())
    assert len(model.top_k("mosfet", (0, 1, 2), k=10)) <= 10  # bounded by vocab seen


def test_top_k_backoff_to_shorter_prefix_when_unseen():
    model = NGram(order=3).fit(_toy_corpus())
    # (99, 99, 99) never seen at order 3; should back off to order 2 then unigram.
    top = model.top_k("mosfet", (99, 99, 99), k=1)
    assert top != []  # unigram fallback always returns something


def test_top_k_for_unknown_family_returns_empty():
    model = NGram(order=2).fit(_toy_corpus())
    assert model.top_k("unknown", (0, 1), k=5) == []


def test_fit_returns_self_for_chaining():
    m = NGram(order=2)
    assert m.fit(_toy_corpus()) is m
```

- [ ] **Step 2: Run test, expect failure**

```bash
pytest tests/test_ngram.py -v
```
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `src/infineon_baseline/ngram.py`**

```python
"""Family-conditioned n-gram language model with stupid backoff."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class NGram:
    order: int = 3
    # per family: dict mapping prefix-tuple (length <= order-1) to Counter[next_id]
    counts: dict[str, dict[tuple[int, ...], Counter]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(Counter)))
    # per family: total token counts (unigram fallback)
    unigram: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    def fit(self, sequences_by_family: dict[str, list[list[int]]]) -> "NGram":
        """Train on tokenized sequences. {family: [[id, id, ...], ...]}"""
        for family, seqs in sequences_by_family.items():
            for ids in seqs:
                for i, next_id in enumerate(ids):
                    self.unigram[family][next_id] += 1
                    # Record counts for prefixes of every length 0..order-1.
                    for prefix_len in range(0, self.order):
                        if i - prefix_len < 0:
                            continue
                        prefix = tuple(ids[i - prefix_len:i])
                        self.counts[family][prefix][next_id] += 1
        return self

    def top_k(self, family: str, prefix: tuple[int, ...], k: int = 5) -> list[int]:
        """Return the k most-likely next-step ids after `prefix`, with stupid backoff."""
        if family not in self.unigram:
            return []
        # Try progressively shorter prefixes, capped at order-1.
        max_prefix_len = min(len(prefix), self.order - 1)
        for prefix_len in range(max_prefix_len, -1, -1):
            short = tuple(prefix[-prefix_len:]) if prefix_len > 0 else ()
            counter = self.counts[family].get(short)
            if counter:
                return [step_id for step_id, _ in counter.most_common(k)]
        # Final fallback: unigram for that family.
        return [step_id for step_id, _ in self.unigram[family].most_common(k)]
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_ngram.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/ngram.py tests/test_ngram.py
git commit -m "feat(ngram): family-conditioned n-gram fit + top_k with stupid backoff"
```

---

## Task 6: NGram — log_prob

**Files:**
- Modify: `src/infineon_baseline/ngram.py` (add `log_prob` method)
- Modify: `tests/test_ngram.py` (add tests)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_ngram.py`:
```python
import math


def test_log_prob_is_higher_for_seen_sequence():
    model = NGram(order=2).fit(_toy_corpus())
    seen = [0, 1, 2, 3, 4]
    unseen = [4, 3, 2, 1, 0]  # reverse — never seen as transitions
    assert model.log_prob("mosfet", seen) > model.log_prob("mosfet", unseen)


def test_log_prob_is_finite_for_completely_unseen_transitions():
    model = NGram(order=2).fit(_toy_corpus())
    # Sequence of unknown ids — stupid backoff falls to unigram, which uses a floor.
    val = model.log_prob("mosfet", [42, 43, 44])
    assert math.isfinite(val)


def test_log_prob_for_unknown_family_is_minus_inf():
    model = NGram(order=2).fit(_toy_corpus())
    assert model.log_prob("unknown_family", [0, 1, 2]) == float("-inf")
```

- [ ] **Step 2: Run new tests, expect failure**

```bash
pytest tests/test_ngram.py::test_log_prob_is_higher_for_seen_sequence -v
```
Expected: FAIL (`AttributeError: 'NGram' object has no attribute 'log_prob'`).

- [ ] **Step 3: Add `log_prob` to `NGram`**

Append inside the `NGram` class in `src/infineon_baseline/ngram.py`:
```python
    # Floor probability for completely unseen unigram steps (keeps log finite).
    _PROB_FLOOR: float = 1e-9

    def log_prob(self, family: str, ids: list[int]) -> float:
        """Sum log P(id[i] | prefix) over the sequence, with stupid backoff."""
        import math

        if family not in self.unigram:
            return float("-inf")
        family_total = sum(self.unigram[family].values())
        if family_total == 0:
            return float("-inf")

        total_lp = 0.0
        for i, next_id in enumerate(ids):
            prob = self._cond_prob(family, tuple(ids[max(0, i - (self.order - 1)):i]), next_id, family_total)
            total_lp += math.log(max(prob, self._PROB_FLOOR))
        return total_lp

    def _cond_prob(self, family: str, prefix: tuple[int, ...], next_id: int, family_total: int) -> float:
        # Try shortening the prefix until we find a non-empty counter.
        for prefix_len in range(len(prefix), -1, -1):
            short = tuple(prefix[-prefix_len:]) if prefix_len > 0 else ()
            counter = self.counts[family].get(short)
            if counter:
                denom = sum(counter.values())
                count = counter.get(next_id, 0)
                if count > 0:
                    return count / denom
        # Final fallback: unigram with floor.
        return self.unigram[family].get(next_id, 0) / family_total if family_total else 0.0
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_ngram.py -v
```
Expected: all NGram tests PASS (5 + 3 = 8 total).

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/ngram.py tests/test_ngram.py
git commit -m "feat(ngram): log_prob with stupid backoff and unigram floor"
```

---

## Task 7: NGram — save/load

**Files:**
- Modify: `src/infineon_baseline/ngram.py` (add `save`/`load`)
- Modify: `tests/test_ngram.py` (add round-trip test)

- [ ] **Step 1: Add failing test**

Append to `tests/test_ngram.py`:
```python
def test_save_and_load_round_trip(tmp_path):
    model = NGram(order=3).fit(_toy_corpus())
    path = tmp_path / "ngram.pkl"
    model.save(path)
    loaded = NGram.load(path)
    assert loaded.order == model.order
    assert loaded.top_k("mosfet", (1, 2), k=2) == model.top_k("mosfet", (1, 2), k=2)
    assert loaded.log_prob("mosfet", [0, 1, 2]) == model.log_prob("mosfet", [0, 1, 2])
```

- [ ] **Step 2: Run new test, expect failure**

```bash
pytest tests/test_ngram.py::test_save_and_load_round_trip -v
```
Expected: FAIL (`AttributeError: 'NGram' object has no attribute 'save'`).

- [ ] **Step 3: Add `save`/`load`**

Append inside the `NGram` class:
```python
    def save(self, path) -> None:
        import pickle
        from pathlib import Path
        # Convert defaultdicts to plain dicts so pickling is portable.
        data = {
            "order": self.order,
            "counts": {fam: {pre: dict(ctr) for pre, ctr in fam_counts.items()}
                       for fam, fam_counts in self.counts.items()},
            "unigram": {fam: dict(ctr) for fam, ctr in self.unigram.items()},
        }
        Path(path).write_bytes(pickle.dumps(data))

    @classmethod
    def load(cls, path) -> "NGram":
        import pickle
        from collections import Counter, defaultdict
        from pathlib import Path
        data = pickle.loads(Path(path).read_bytes())
        model = cls(order=data["order"])
        for fam, fam_counts in data["counts"].items():
            for pre, ctr in fam_counts.items():
                model.counts[fam][pre] = Counter(ctr)
        for fam, ctr in data["unigram"].items():
            model.unigram[fam] = Counter(ctr)
        return model
```

- [ ] **Step 4: Run all NGram tests**

```bash
pytest tests/test_ngram.py -v
```
Expected: 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/ngram.py tests/test_ngram.py
git commit -m "feat(ngram): pickle-based save/load"
```

---

## Task 8: eval_set — split + truncate

**Files:**
- Create: `src/infineon_baseline/eval_set.py`
- Create: `tests/test_eval_set.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_eval_set.py`:
```python
from infineon_baseline.eval_set import Cut, split, truncate
from tests.fixtures.mini_sequences import MINI_DATASET


def test_split_partitions_by_family_with_exact_holdout_size():
    train, hold = split(MINI_DATASET, holdout_per_family=3, seed=42)
    for family in MINI_DATASET:
        assert len(hold[family]) == 3
        assert len(train[family]) == len(MINI_DATASET[family]) - 3


def test_split_is_deterministic_for_same_seed():
    a = split(MINI_DATASET, holdout_per_family=3, seed=42)
    b = split(MINI_DATASET, holdout_per_family=3, seed=42)
    assert set(a[1]["mosfet"].keys()) == set(b[1]["mosfet"].keys())


def test_split_differs_for_different_seeds():
    a = split(MINI_DATASET, holdout_per_family=3, seed=1)
    b = split(MINI_DATASET, holdout_per_family=3, seed=2)
    assert set(a[1]["mosfet"].keys()) != set(b[1]["mosfet"].keys())


def test_split_raises_when_holdout_exceeds_corpus():
    import pytest
    with pytest.raises(ValueError, match="holdout_per_family"):
        split(MINI_DATASET, holdout_per_family=100, seed=42)


def test_truncate_at_60_percent_splits_partial_and_ground_truth():
    seq = list(range(100))  # 100 items
    cut = truncate(seq, fraction=0.6)
    assert isinstance(cut, Cut)
    assert cut.partial == list(range(60))
    assert cut.ground_truth == list(range(60, 100))
    assert cut.cut_index == 60


def test_truncate_uses_floor_for_non_integer_cut():
    seq = list(range(101))
    cut = truncate(seq, fraction=0.6)  # 60.6 -> floor 60
    assert cut.cut_index == 60
    assert len(cut.partial) + len(cut.ground_truth) == len(seq)


def test_truncate_rejects_invalid_fraction():
    import pytest
    for bad in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="fraction must be in"):
            truncate([1, 2, 3], fraction=bad)
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_eval_set.py -v
```
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `src/infineon_baseline/eval_set.py`**

```python
"""Eval-set construction: split, truncate, inject_violation, write_eval_inputs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import TypeAlias

SequencesByFamily: TypeAlias = dict[str, dict[str, list[str]]]


@dataclass
class Cut:
    partial: list[str]
    ground_truth: list[str]
    cut_index: int


def split(
    sequences_by_family: SequencesByFamily,
    holdout_per_family: int,
    seed: int,
) -> tuple[SequencesByFamily, SequencesByFamily]:
    """Deterministic split: each family gives `holdout_per_family` sequences to the hold-out set."""
    rng = Random(seed)
    train: SequencesByFamily = {}
    hold: SequencesByFamily = {}
    for family, seqs in sequences_by_family.items():
        if holdout_per_family >= len(seqs):
            raise ValueError(
                f"holdout_per_family={holdout_per_family} >= corpus size {len(seqs)} for {family}"
            )
        ids = sorted(seqs.keys())            # sort for determinism across dict-iteration order changes
        rng.shuffle(ids)
        hold_ids = set(ids[:holdout_per_family])
        train[family] = {sid: seqs[sid] for sid in seqs if sid not in hold_ids}
        hold[family] = {sid: seqs[sid] for sid in seqs if sid in hold_ids}
    return train, hold


def truncate(sequence: list[str], fraction: float) -> Cut:
    """Cut a sequence at floor(len*fraction); return Cut(partial, ground_truth, cut_index)."""
    if not (0.0 < fraction < 1.0):
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    cut = math.floor(len(sequence) * fraction)
    return Cut(partial=list(sequence[:cut]), ground_truth=list(sequence[cut:]), cut_index=cut)
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_eval_set.py -v
```
Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/eval_set.py tests/test_eval_set.py
git commit -m "feat(eval_set): deterministic split + fractional truncate"
```

---

## Task 9: eval_set — inject_violation (10 rules)

**Files:**
- Modify: `src/infineon_baseline/eval_set.py` (add `inject_violation` + 10 injectors)
- Modify: `tests/test_eval_set.py` (add round-trip tests against `validate_sequence`)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_eval_set.py`:
```python
import pytest
from random import Random

from infineon_baseline import validate_sequence
from infineon_baseline.eval_set import inject_violation, SUPPORTED_RULES


# A long, fully-valid MOSFET sequence (from the reference data) used as injection substrate.
def _load_reference_mosfet() -> list[str]:
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "training_data" / "synthetic_mosfet.csv"
    rows = path.read_text().splitlines()[1:]  # skip header
    return [r.strip().strip('"') for r in rows if r.strip()]


@pytest.mark.parametrize("rule_id", SUPPORTED_RULES)
def test_injector_triggers_target_rule(rule_id):
    base = _load_reference_mosfet()
    assert validate_sequence(base) == [], "reference must start valid"
    rng = Random(42)
    corrupted, applied = inject_violation(base, rule_id, rng=rng)
    violations = validate_sequence(corrupted)
    assert any(v.rule == rule_id for v in violations), (
        f"expected {rule_id} in violations; got {[v.rule for v in violations]}"
    )
    assert applied == rule_id


def test_inject_violation_rejects_unknown_rule():
    with pytest.raises(ValueError, match="unknown rule"):
        inject_violation(_load_reference_mosfet(), "RULE_DOES_NOT_EXIST", rng=Random(0))


def test_inject_violation_returns_modified_copy_not_in_place():
    base = _load_reference_mosfet()
    snapshot = list(base)
    corrupted, _ = inject_violation(base, "RULE_DEP_NO_CLEAN", rng=Random(0))
    assert base == snapshot, "input must not be mutated"
    assert corrupted is not base
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_eval_set.py -k injector -v
```
Expected: FAIL (`ImportError: cannot import name 'inject_violation'`).

- [ ] **Step 3: Add `inject_violation` and 10 injectors to `src/infineon_baseline/eval_set.py`**

Append at the bottom of `src/infineon_baseline/eval_set.py`:
```python
from random import Random

from generate_sequences import (
    CLEAN_STEPS,
    DEPOSITION_STEPS,
    ETCH_STEPS,
    METAL_ETCH_STEPS,
    validate_sequence,
)


SUPPORTED_RULES: tuple[str, ...] = (
    "RULE_DEP_NO_CLEAN",
    "RULE_METAL_ETCH_NO_LITHO",
    "RULE_ETCH_NO_MASK",
    "RULE_LITHO_LEVEL_SKIP",
    "RULE_IMPLANT_NO_MASK",
    "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP",
    "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST",
    "RULE_BACKSIDE_BEFORE_PASSIVATION",
)

# Trigger sets we look for when injecting. Keep these aligned with generate_sequences.py.
_IMPLANT_STEPS = frozenset({
    "IMPLANT WELL", "IMPLANT SOURCE DRAIN", "IMPLANT SOURCE REGION", "IMPLANT LDD",
    "IMPLANT P BODY", "IMPLANT N BUFFER", "IMPLANT CHANNEL STOP",
    "IMPLANT DRAIN / CATHODE REGION", "IMPLANT N-TYPE",
})
_CMP_STEPS = frozenset({
    "CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC", "CMP METAL", "CMP VIA FILL",
})
_PAD_OPEN_STEPS = frozenset({"OPEN PAD WINDOW", "OPEN BOND PAD WINDOW", "PAD WINDOW LITHO"})
_PASSIVATION_DEP_STEPS = frozenset({"DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"})
_ELECTRICAL_TESTS = frozenset({
    "PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST",
    "THRESHOLD VOLTAGE TEST", "BREAKDOWN VOLTAGE TEST",
    "LEAKAGE TEST", "SWITCHING TEST",
})


def inject_violation(
    sequence: list[str],
    rule_id: str,
    rng: Random,
    max_attempts: int = 8,
) -> tuple[list[str], str]:
    """Return (corrupted_sequence, applied_rule_id).

    Tries `max_attempts` times; raises RuntimeError if no attempt triggers the target rule.
    Operates on a copy — does not mutate `sequence`.
    """
    if rule_id not in _INJECTORS:
        raise ValueError(f"unknown rule {rule_id!r}; supported: {SUPPORTED_RULES}")
    injector = _INJECTORS[rule_id]
    for _ in range(max_attempts):
        attempt = injector(list(sequence), rng)
        if attempt is None:
            continue
        if any(v.rule == rule_id for v in validate_sequence(attempt)):
            return attempt, rule_id
    raise RuntimeError(
        f"inject_violation: could not trigger {rule_id} after {max_attempts} attempts"
    )


# --------------------------------------------------------------------------- #
# Individual injectors — each returns a corrupted list[str] or None on failure #
# --------------------------------------------------------------------------- #

def _delete_clean_before_first_deposition(seq: list[str], rng: Random) -> list[str] | None:
    """RULE_DEP_NO_CLEAN: remove every clean step in the 12 steps before a deposition."""
    for i, s in enumerate(seq):
        if s in DEPOSITION_STEPS:
            window_start = max(0, i - 12)
            to_remove = [j for j in range(window_start, i) if seq[j] in CLEAN_STEPS]
            if to_remove:
                # Remove from highest index down to keep earlier indices valid.
                for j in reversed(to_remove):
                    del seq[j]
                return seq
    return None


def _delete_develop_before_metal_etch(seq: list[str], rng: Random) -> list[str] | None:
    """RULE_METAL_ETCH_NO_LITHO: remove DEVELOP PHOTORESIST in the 15 steps before a metal etch."""
    for i, s in enumerate(seq):
        if s in METAL_ETCH_STEPS:
            window_start = max(0, i - 15)
            for j in range(i - 1, window_start - 1, -1):
                if seq[j] in {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}:
                    del seq[j]
                    return seq
    return None


def _delete_develop_before_etch(seq: list[str], rng: Random) -> list[str] | None:
    """RULE_ETCH_NO_MASK: remove DEVELOP in the 12 steps before a (non-spacer) etch."""
    for i, s in enumerate(seq):
        if s in ETCH_STEPS:
            window_start = max(0, i - 12)
            for j in range(i - 1, window_start - 1, -1):
                if seq[j] in {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}:
                    del seq[j]
                    return seq
    return None


def _skip_litho_level(seq: list[str], rng: Random) -> list[str] | None:
    """RULE_LITHO_LEVEL_SKIP: rename ALIGN MASK LEVEL 2 → ALIGN MASK LEVEL 4 (skipping 3)."""
    targets = [i for i, s in enumerate(seq) if s == "ALIGN MASK LEVEL 2"]
    if not targets:
        return None
    seq[targets[0]] = "ALIGN MASK LEVEL 4"
    return seq


def _delete_mask_before_implant(seq: list[str], rng: Random) -> list[str] | None:
    """RULE_IMPLANT_NO_MASK: remove the oxide/window etch or DEVELOP within 15 steps before an implant."""
    target_prereqs = {"OXIDE ETCH", "OXIDE ETCH DRY", "ETCH SILICON OR OXIDE WINDOW", "DEVELOP PHOTORESIST"}
    for i, s in enumerate(seq):
        if s in _IMPLANT_STEPS:
            window_start = max(0, i - 15)
            to_delete = [j for j in range(window_start, i) if seq[j] in target_prereqs]
            if to_delete:
                for j in reversed(to_delete):
                    del seq[j]
                return seq
    return None


def _delete_deposition_before_cmp(seq: list[str], rng: Random) -> list[str] | None:
    """RULE_CMP_NO_DEP: remove deposition / fill in the 6 steps before a CMP."""
    via_fill = {"FILL VIA METAL", "FILL VIA TUNGSTEN"}
    for i, s in enumerate(seq):
        if s in _CMP_STEPS:
            window_start = max(0, i - 6)
            for j in range(i - 1, window_start - 1, -1):
                if seq[j] in DEPOSITION_STEPS or seq[j] in via_fill:
                    del seq[j]
                    return seq
    return None


def _move_pad_open_before_passivation(seq: list[str], rng: Random) -> list[str] | None:
    """RULE_PAD_OPEN_BEFORE_DEP: move a pad-open step to before DEPOSIT PASSIVATION."""
    pad_idx = next((i for i, s in enumerate(seq) if s in _PAD_OPEN_STEPS), None)
    dep_idx = next((i for i, s in enumerate(seq) if s in _PASSIVATION_DEP_STEPS), None)
    if pad_idx is None or dep_idx is None or pad_idx < dep_idx:
        return None
    step = seq.pop(pad_idx)
    seq.insert(dep_idx, step)  # now appears just before DEPOSIT PASSIVATION
    return seq


def _move_test_before_cure(seq: list[str], rng: Random) -> list[str] | None:
    """RULE_TEST_BEFORE_PASSIVATION: move an electrical test to before CURE PASSIVATION."""
    cure_idx = next((i for i, s in enumerate(seq) if s == "CURE PASSIVATION"), None)
    test_idx = next((i for i, s in enumerate(seq) if s in _ELECTRICAL_TESTS), None)
    if cure_idx is None or test_idx is None or test_idx < cure_idx:
        return None
    step = seq.pop(test_idx)
    seq.insert(cure_idx, step)
    return seq


def _swap_ship_before_sort(seq: list[str], rng: Random) -> list[str] | None:
    """RULE_SHIP_BEFORE_TEST: swap SHIP LOT before WAFER SORT TEST."""
    sort_idx = next((i for i, s in enumerate(seq) if s == "WAFER SORT TEST"), None)
    ship_idx = next((i for i, s in enumerate(seq) if s == "SHIP LOT"), None)
    if sort_idx is None or ship_idx is None or ship_idx < sort_idx:
        return None
    seq[sort_idx], seq[ship_idx] = seq[ship_idx], seq[sort_idx]
    return seq


def _move_backside_before_cure(seq: list[str], rng: Random) -> list[str] | None:
    """RULE_BACKSIDE_BEFORE_PASSIVATION: move DEPOSIT BACKSIDE METAL before CURE PASSIVATION."""
    cure_idx = next((i for i, s in enumerate(seq) if s == "CURE PASSIVATION"), None)
    back_idx = next((i for i, s in enumerate(seq) if s == "DEPOSIT BACKSIDE METAL"), None)
    if cure_idx is None or back_idx is None or back_idx < cure_idx:
        return None
    step = seq.pop(back_idx)
    seq.insert(cure_idx, step)
    return seq


_INJECTORS = {
    "RULE_DEP_NO_CLEAN": _delete_clean_before_first_deposition,
    "RULE_METAL_ETCH_NO_LITHO": _delete_develop_before_metal_etch,
    "RULE_ETCH_NO_MASK": _delete_develop_before_etch,
    "RULE_LITHO_LEVEL_SKIP": _skip_litho_level,
    "RULE_IMPLANT_NO_MASK": _delete_mask_before_implant,
    "RULE_CMP_NO_DEP": _delete_deposition_before_cmp,
    "RULE_PAD_OPEN_BEFORE_DEP": _move_pad_open_before_passivation,
    "RULE_TEST_BEFORE_PASSIVATION": _move_test_before_cure,
    "RULE_SHIP_BEFORE_TEST": _swap_ship_before_sort,
    "RULE_BACKSIDE_BEFORE_PASSIVATION": _move_backside_before_cure,
}
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_eval_set.py -v
```
Expected: 7 (split/truncate) + 10 (parametrized injectors) + 2 (unknown/in-place) = 19 PASS.

If any injector fails the round-trip (e.g., the reference sequence doesn't have an `ALIGN MASK LEVEL 2` to rename), adjust that specific injector to a fallback strategy or pick a different injection point. Re-run.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/eval_set.py tests/test_eval_set.py
git commit -m "feat(eval_set): inject_violation with one injector per rule + round-trip tests"
```

---

## Task 10: eval_set — write_eval_inputs + write_ground_truth

**Files:**
- Modify: `src/infineon_baseline/eval_set.py` (add four writer functions)
- Modify: `tests/test_eval_set.py` (add tests)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_eval_set.py`:
```python
import csv

from infineon_baseline.eval_set import (
    build_valid_eval_inputs,
    build_anomaly_eval_inputs,
    write_eval_inputs_valid,
    write_eval_inputs_anomaly,
    write_ground_truth_valid,
    write_ground_truth_anomaly,
)


def test_build_valid_eval_inputs_has_correct_row_count():
    _, hold = split(MINI_DATASET, holdout_per_family=4, seed=42)
    rows = build_valid_eval_inputs(hold, seqs_per_family=4, fractions=(0.6, 0.8))
    # 4 families × seqs × 2 cuts -- here 3 families × 4 × 2 = 24
    assert len(rows) == 3 * 4 * 2


def test_build_valid_row_format():
    _, hold = split(MINI_DATASET, holdout_per_family=2, seed=42)
    rows = build_valid_eval_inputs(hold, seqs_per_family=2, fractions=(0.6,))
    r = rows[0]
    assert {"EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"} <= r.keys()
    assert "|" in r["PARTIAL_SEQUENCE"]


def test_build_anomaly_eval_inputs_invalid_ratio_is_close_to_target():
    _, hold = split(MINI_DATASET, holdout_per_family=8, seed=42)
    rows = build_anomaly_eval_inputs(hold, seqs_per_family=8, invalid_ratio=0.5, seed=42)
    n_invalid = sum(1 for r in rows if r["_is_invalid"])
    # 8 × 3 = 24 total, invalid_ratio 0.5 → ~12.
    assert abs(n_invalid - 12) <= 3


def test_write_eval_inputs_valid_csv_round_trip(tmp_path):
    _, hold = split(MINI_DATASET, holdout_per_family=2, seed=42)
    rows = build_valid_eval_inputs(hold, seqs_per_family=2, fractions=(0.6, 0.8))
    out = tmp_path / "eval_input_valid.csv"
    write_eval_inputs_valid(rows, out)
    with out.open(newline="") as f:
        readback = list(csv.DictReader(f))
    assert readback[0]["FAMILY"] in {"mosfet", "igbt", "ic"}
    assert "|" in readback[0]["PARTIAL_SEQUENCE"]


def test_write_ground_truth_anomaly_columns(tmp_path):
    _, hold = split(MINI_DATASET, holdout_per_family=2, seed=42)
    rows = build_anomaly_eval_inputs(hold, seqs_per_family=2, invalid_ratio=0.5, seed=42)
    out = tmp_path / "gt_anomaly.csv"
    write_ground_truth_anomaly(rows, out)
    with out.open(newline="") as f:
        readback = list(csv.DictReader(f))
    assert set(readback[0].keys()) == {"EXAMPLE_ID", "IS_VALID", "RULE_VIOLATED"}
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_eval_set.py -k "write or build" -v
```
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Append builders + writers to `src/infineon_baseline/eval_set.py`**

```python
import csv
from pathlib import Path


def build_valid_eval_inputs(
    holdout: SequencesByFamily,
    seqs_per_family: int,
    fractions: tuple[float, ...] = (0.6, 0.8),
) -> list[dict]:
    """Build rows for `eval_input_valid.csv` plus a stashed full-sequence ground truth."""
    rows: list[dict] = []
    for family in sorted(holdout):
        seq_ids = sorted(holdout[family].keys())[:seqs_per_family]
        for sid in seq_ids:
            full = holdout[family][sid]
            for frac in fractions:
                cut = truncate(full, fraction=frac)
                rows.append({
                    "EXAMPLE_ID": f"valid_{family}_{sid}_{int(frac*100):02d}",
                    "FAMILY": family,
                    "COMPLETION_FRACTION": f"{frac:.2f}",
                    "PARTIAL_SEQUENCE": "|".join(cut.partial),
                    "_full_sequence": full,         # stashed for ground-truth writer
                    "_cut_index": cut.cut_index,
                })
    return rows


def build_anomaly_eval_inputs(
    holdout: SequencesByFamily,
    seqs_per_family: int,
    invalid_ratio: float,
    seed: int,
) -> list[dict]:
    """Build rows for `eval_input_anomaly.csv` with `invalid_ratio` of them corrupted."""
    rng = Random(seed)
    rows: list[dict] = []
    for family in sorted(holdout):
        seq_ids = sorted(holdout[family].keys())[:seqs_per_family]
        n_invalid = round(len(seq_ids) * invalid_ratio)
        rng.shuffle(seq_ids)
        invalid_ids = set(seq_ids[:n_invalid])
        for sid in sorted(holdout[family]):           # iterate in sorted order for determinism
            if sid not in seq_ids:
                continue
            full = list(holdout[family][sid])
            if sid in invalid_ids:
                rule = SUPPORTED_RULES[rng.randint(0, len(SUPPORTED_RULES) - 1)]
                try:
                    corrupted, applied = inject_violation(full, rule, rng=rng)
                except RuntimeError:
                    # This particular sequence can't host this rule; pick another.
                    for fallback in SUPPORTED_RULES:
                        try:
                            corrupted, applied = inject_violation(full, fallback, rng=rng)
                            break
                        except RuntimeError:
                            continue
                    else:
                        # As a last resort, keep it valid.
                        corrupted, applied = full, ""
                seq, is_invalid, applied_rule = corrupted, True, applied
            else:
                seq, is_invalid, applied_rule = full, False, ""
            rows.append({
                "EXAMPLE_ID": f"anom_{family}_{sid}",
                "FAMILY": family,
                "SEQUENCE": "|".join(seq),
                "_is_invalid": is_invalid,
                "_applied_rule": applied_rule,
            })
    rng.shuffle(rows)
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_eval_inputs_valid(rows: list[dict], path: Path) -> None:
    _write_csv(path, ["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"], rows)


def write_eval_inputs_anomaly(rows: list[dict], path: Path) -> None:
    _write_csv(path, ["EXAMPLE_ID", "FAMILY", "SEQUENCE"], rows)


def write_ground_truth_valid(rows: list[dict], path: Path) -> None:
    gt_rows = [
        {"EXAMPLE_ID": r["EXAMPLE_ID"], "FULL_SEQUENCE": "|".join(r["_full_sequence"]), "CUT_INDEX": r["_cut_index"]}
        for r in rows
    ]
    _write_csv(path, ["EXAMPLE_ID", "FULL_SEQUENCE", "CUT_INDEX"], gt_rows)


def write_ground_truth_anomaly(rows: list[dict], path: Path) -> None:
    gt_rows = [
        {"EXAMPLE_ID": r["EXAMPLE_ID"], "IS_VALID": 0 if r["_is_invalid"] else 1, "RULE_VIOLATED": r["_applied_rule"]}
        for r in rows
    ]
    _write_csv(path, ["EXAMPLE_ID", "IS_VALID", "RULE_VIOLATED"], gt_rows)
```

- [ ] **Step 4: Run all eval_set tests, expect PASS**

```bash
pytest tests/test_eval_set.py -v
```
Expected: 19 + 5 = 24 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/eval_set.py tests/test_eval_set.py
git commit -m "feat(eval_set): build + write eval inputs and ground-truth CSVs"
```

---

## Task 11: anomaly — detect_oracle

**Files:**
- Create: `src/infineon_baseline/anomaly.py`
- Create: `tests/test_anomaly.py`

- [ ] **Step 1: Write the failing test**

`tests/test_anomaly.py`:
```python
from random import Random

from infineon_baseline import validate_sequence
from infineon_baseline.anomaly import AnomalyResult, detect_oracle
from infineon_baseline.eval_set import inject_violation


def _load_reference_mosfet() -> list[str]:
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    return [r.strip().strip('"') for r in
            (repo_root / "training_data" / "synthetic_mosfet.csv").read_text().splitlines()[1:]
            if r.strip()]


def test_oracle_passes_valid_sequence():
    valid = _load_reference_mosfet()
    assert validate_sequence(valid) == []
    res = detect_oracle(valid)
    assert isinstance(res, AnomalyResult)
    assert res.is_valid == 1 and res.predicted_rule == ""


def test_oracle_detects_injected_violation():
    valid = _load_reference_mosfet()
    corrupted, applied = inject_violation(valid, "RULE_DEP_NO_CLEAN", rng=Random(42))
    res = detect_oracle(corrupted)
    assert res.is_valid == 0
    assert res.predicted_rule == applied
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_anomaly.py -v
```
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `src/infineon_baseline/anomaly.py`**

```python
"""Anomaly detection strategies for Task 3."""
from __future__ import annotations

from dataclasses import dataclass

from generate_sequences import validate_sequence


@dataclass
class AnomalyResult:
    is_valid: int       # 1 or 0
    score: float        # in [0.0, 1.0]; P(valid)
    predicted_rule: str # rule id when invalid; "" when valid


def detect_oracle(steps: list[str]) -> AnomalyResult:
    """Use the rule oracle: any reported violation → invalid + first rule reported."""
    violations = validate_sequence(steps)
    if violations:
        return AnomalyResult(is_valid=0, score=0.0, predicted_rule=violations[0].rule)
    return AnomalyResult(is_valid=1, score=1.0, predicted_rule="")
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_anomaly.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/anomaly.py tests/test_anomaly.py
git commit -m "feat(anomaly): rule-oracle detector via validate_sequence"
```

---

## Task 12: anomaly — detect_perplexity + calibrate_threshold

**Files:**
- Modify: `src/infineon_baseline/anomaly.py`
- Modify: `tests/test_anomaly.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_anomaly.py`:
```python
from infineon_baseline.anomaly import calibrate_threshold, detect_perplexity
from infineon_baseline.ngram import NGram
from infineon_baseline.tokenizer import Tokenizer
from tests.fixtures.mini_sequences import MINI_DATASET


def _fit_mini_ngram() -> tuple[Tokenizer, NGram]:
    tok = Tokenizer.fit(MINI_DATASET)
    train_ids = {
        family: [tok.encode(family, steps)[1] for steps in seqs.values()]
        for family, seqs in MINI_DATASET.items()
    }
    model = NGram(order=2).fit(train_ids)
    return tok, model


def test_perplexity_score_is_lower_for_shuffled_sequence():
    tok, ng = _fit_mini_ngram()
    valid = MINI_DATASET["mosfet"]["mosfet_0001"]
    shuffled = list(reversed(valid))   # reversal preserves first/last but scrambles transitions
    res_v = detect_perplexity(valid, "mosfet", tok, ng, threshold=-50.0)
    res_s = detect_perplexity(shuffled, "mosfet", tok, ng, threshold=-50.0)
    # Score is the log-prob mapped to [0, 1]; valid should be higher.
    assert res_v.score > res_s.score


def test_calibrate_threshold_returns_finite_float():
    tok, ng = _fit_mini_ngram()
    pos = [MINI_DATASET["mosfet"][f"mosfet_{i:04d}"] for i in range(1, 6)]
    neg = [list(reversed(s)) for s in pos]
    families = ["mosfet"] * len(pos)
    th = calibrate_threshold(positives=pos, negatives=neg,
                             families_pos=families, families_neg=families,
                             tokenizer=tok, ngram=ng)
    import math
    assert math.isfinite(th)
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_anomaly.py::test_perplexity_score_is_lower_for_shuffled_sequence -v
```
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Add `detect_perplexity` and `calibrate_threshold` to `anomaly.py`**

Append to `src/infineon_baseline/anomaly.py`:
```python
import math
from typing import Iterable


def detect_perplexity(
    steps: list[str],
    family: str,
    tokenizer,         # infineon_baseline.tokenizer.Tokenizer
    ngram,             # infineon_baseline.ngram.NGram
    threshold: float,  # log-prob threshold; below → invalid
) -> AnomalyResult:
    _, ids = tokenizer.encode(family, steps)
    lp = ngram.log_prob(family, ids)
    is_valid = 1 if lp >= threshold else 0
    score = _sigmoid(lp - threshold)
    return AnomalyResult(is_valid=is_valid, score=score, predicted_rule="")


def _sigmoid(x: float) -> float:
    # Saturates gracefully for very negative/positive x without overflow.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def calibrate_threshold(
    positives: Iterable[list[str]],
    negatives: Iterable[list[str]],
    families_pos: Iterable[str],
    families_neg: Iterable[str],
    tokenizer,
    ngram,
) -> float:
    """Pick the log-prob threshold that maximizes F1 on (pos, neg).

    Computes log-prob for every input, then evaluates every midpoint between
    sorted unique log-probs as a candidate threshold.
    """
    pos_lps = [ngram.log_prob(f, tokenizer.encode(f, s)[1]) for s, f in zip(positives, families_pos)]
    neg_lps = [ngram.log_prob(f, tokenizer.encode(f, s)[1]) for s, f in zip(negatives, families_neg)]
    all_lps = sorted(set(pos_lps + neg_lps))
    if len(all_lps) < 2:
        return all_lps[0] if all_lps else 0.0
    candidates = [(a + b) / 2 for a, b in zip(all_lps, all_lps[1:])]
    best_f1, best_th = -1.0, candidates[0]
    for th in candidates:
        tp = sum(1 for lp in pos_lps if lp >= th)
        fp = sum(1 for lp in neg_lps if lp >= th)
        fn = sum(1 for lp in pos_lps if lp < th)
        if tp + fp == 0 or tp + fn == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_anomaly.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/anomaly.py tests/test_anomaly.py
git commit -m "feat(anomaly): perplexity detector + F1-optimal threshold calibration"
```

---

## Task 13: anomaly — detect_hybrid

**Files:**
- Modify: `src/infineon_baseline/anomaly.py`
- Modify: `tests/test_anomaly.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_anomaly.py`:
```python
from infineon_baseline.anomaly import detect_hybrid


def test_hybrid_uses_oracle_for_decision_and_perplexity_for_score():
    tok, ng = _fit_mini_ngram()
    valid = MINI_DATASET["mosfet"]["mosfet_0001"]
    res = detect_hybrid(valid, "mosfet", tok, ng, threshold=-50.0)
    # Oracle says valid → is_valid=1 and predicted_rule="" (regardless of perplexity score range).
    assert res.is_valid == 1
    assert res.predicted_rule == ""
    # Score should be a real probability in [0,1] from perplexity, not the constant 1.0 from the oracle alone.
    assert 0.0 <= res.score <= 1.0
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_anomaly.py::test_hybrid_uses_oracle_for_decision_and_perplexity_for_score -v
```
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Add `detect_hybrid` to `anomaly.py`**

Append to `src/infineon_baseline/anomaly.py`:
```python
def detect_hybrid(
    steps: list[str],
    family: str,
    tokenizer,
    ngram,
    threshold: float,
) -> AnomalyResult:
    """Oracle determines is_valid + predicted_rule; perplexity provides the continuous score."""
    oracle = detect_oracle(steps)
    perp = detect_perplexity(steps, family, tokenizer, ngram, threshold)
    return AnomalyResult(
        is_valid=oracle.is_valid,
        score=perp.score,
        predicted_rule=oracle.predicted_rule,
    )
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_anomaly.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/anomaly.py tests/test_anomaly.py
git commit -m "feat(anomaly): hybrid oracle-decision + perplexity-score detector"
```

---

## Task 14: predictor — run_task1 (next-step)

**Files:**
- Create: `src/infineon_baseline/predictor.py`
- Create: `tests/test_predictor.py`

- [ ] **Step 1: Write the failing test**

`tests/test_predictor.py`:
```python
from infineon_baseline.predictor import run_task1, Task1Row
from infineon_baseline.ngram import NGram
from infineon_baseline.tokenizer import Tokenizer
from tests.fixtures.mini_sequences import MINI_DATASET


def _fitted():
    tok = Tokenizer.fit(MINI_DATASET)
    train_ids = {
        family: [tok.encode(family, steps)[1] for steps in seqs.values()]
        for family, seqs in MINI_DATASET.items()
    }
    return tok, NGram(order=2).fit(train_ids)


def test_run_task1_returns_one_row_per_example_with_five_ranks():
    tok, ng = _fitted()
    examples = [{
        "EXAMPLE_ID": "x1",
        "FAMILY": "mosfet",
        "COMPLETION_FRACTION": "0.6",
        "PARTIAL_SEQUENCE": "RECEIVE WAFER LOT|LOT IDENTIFICATION|PRE CLEAN WAFER",
    }]
    rows = list(run_task1(examples, tokenizer=tok, ngram=ng))
    assert len(rows) == 1
    r = rows[0]
    assert r.example_id == "x1"
    assert len(r.ranks) == 5
    assert all(isinstance(s, str) for s in r.ranks)


def test_run_task1_handles_empty_topk_by_padding_from_global_unigram():
    tok, ng = _fitted()
    examples = [{
        "EXAMPLE_ID": "x2",
        "FAMILY": "ic",
        "COMPLETION_FRACTION": "0.6",
        # Step IDs all known, but a family with a single training sequence may not have a
        # rich enough distribution to produce 5 unique top_k results. The predictor pads.
        "PARTIAL_SEQUENCE": "RECEIVE WAFER LOT",
    }]
    rows = list(run_task1(examples, tokenizer=tok, ngram=ng))
    assert len(rows[0].ranks) == 5
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_predictor.py -v
```
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `src/infineon_baseline/predictor.py`**

```python
"""Task-level orchestration: turn eval inputs + a fitted model into submission rows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from infineon_baseline.anomaly import AnomalyResult, detect_hybrid, detect_oracle, detect_perplexity
from infineon_baseline.ngram import NGram
from infineon_baseline.tokenizer import Tokenizer


@dataclass
class Task1Row:
    example_id: str
    ranks: list[str]   # length 5


def run_task1(
    examples: Iterable[dict],
    tokenizer: Tokenizer,
    ngram: NGram,
) -> Iterator[Task1Row]:
    """One row per example with the top-5 next steps (always exactly 5)."""
    for ex in examples:
        family = ex["FAMILY"]
        partial_steps = ex["PARTIAL_SEQUENCE"].split("|")
        _, partial_ids = tokenizer.encode(family, partial_steps)
        prefix = tuple(partial_ids[-(ngram.order - 1):]) if ngram.order > 1 else ()
        top_ids = ngram.top_k(family, prefix, k=5)
        # Pad with most-common-overall steps to guarantee exactly 5 ranks.
        if len(top_ids) < 5:
            # Use unigram frequency across all families for tie-break stability.
            from collections import Counter
            global_counter: Counter = Counter()
            for fam_ctr in ngram.unigram.values():
                global_counter.update(fam_ctr)
            for step_id, _ in global_counter.most_common():
                if step_id not in top_ids:
                    top_ids.append(step_id)
                if len(top_ids) >= 5:
                    break
        ranks = tokenizer.decode(top_ids[:5])
        yield Task1Row(example_id=ex["EXAMPLE_ID"], ranks=ranks)
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_predictor.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/predictor.py tests/test_predictor.py
git commit -m "feat(predictor): run_task1 next-step prediction with 5-rank padding"
```

---

## Task 15: predictor — run_task2 (greedy completion)

**Files:**
- Modify: `src/infineon_baseline/predictor.py`
- Modify: `tests/test_predictor.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_predictor.py`:
```python
from infineon_baseline.predictor import run_task2, Task2Row


def test_run_task2_completes_until_ship_lot_or_cap():
    tok, ng = _fitted()
    examples = [{
        "EXAMPLE_ID": "y1",
        "FAMILY": "mosfet",
        "COMPLETION_FRACTION": "0.6",
        "PARTIAL_SEQUENCE": "RECEIVE WAFER LOT|LOT IDENTIFICATION|PRE CLEAN WAFER",
    }]
    rows = list(run_task2(examples, tokenizer=tok, ngram=ng, max_length_multiplier=2.0))
    assert len(rows) == 1
    r = rows[0]
    assert r.example_id == "y1"
    completion = r.predicted_sequence.split("|")
    assert len(completion) > 0
    # First predicted step must not echo the last seen step (no immediate repetition trap).
    # Completion does NOT include the partial sequence.
    assert "RECEIVE WAFER LOT" not in completion or completion.index("RECEIVE WAFER LOT") > 0
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_predictor.py::test_run_task2_completes_until_ship_lot_or_cap -v
```
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Add `run_task2` and `Task2Row` to `predictor.py`**

```python
@dataclass
class Task2Row:
    example_id: str
    predicted_sequence: str   # pipe-separated, steps AFTER the cut only


_SHIP = "SHIP LOT"


def run_task2(
    examples: Iterable[dict],
    tokenizer: Tokenizer,
    ngram: NGram,
    max_length_multiplier: float = 1.5,
    constrain: bool = False,
) -> Iterator[Task2Row]:
    """Greedy autoregressive completion until SHIP LOT or length cap."""
    from generate_sequences import validate_sequence
    ship_id = tokenizer.step_to_id.get(_SHIP)
    for ex in examples:
        family = ex["FAMILY"]
        partial_steps = ex["PARTIAL_SEQUENCE"].split("|")
        _, partial_ids = tokenizer.encode(family, partial_steps)
        max_len = int(len(partial_steps) * max_length_multiplier)
        out_ids: list[int] = []
        prefix_ids = list(partial_ids)
        while len(prefix_ids) - len(partial_ids) < max_len:
            ctx = tuple(prefix_ids[-(ngram.order - 1):]) if ngram.order > 1 else ()
            cands = ngram.top_k(family, ctx, k=5)
            if not cands:
                break
            if constrain:
                # Filter out candidates that immediately trigger a rule violation.
                ok = []
                for cid in cands:
                    trial = tokenizer.decode(prefix_ids + [cid])
                    if not validate_sequence(trial):
                        ok.append(cid)
                cands = ok or cands  # fall back if all options violate
            next_id = cands[0]
            out_ids.append(next_id)
            prefix_ids.append(next_id)
            if ship_id is not None and next_id == ship_id:
                break
        yield Task2Row(
            example_id=ex["EXAMPLE_ID"],
            predicted_sequence="|".join(tokenizer.decode(out_ids)),
        )
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_predictor.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/predictor.py tests/test_predictor.py
git commit -m "feat(predictor): run_task2 greedy sequence completion"
```

---

## Task 16: predictor — run_task3 (anomaly)

**Files:**
- Modify: `src/infineon_baseline/predictor.py`
- Modify: `tests/test_predictor.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_predictor.py`:
```python
from infineon_baseline.predictor import run_task3, Task3Row


def test_run_task3_emits_anomaly_rows_with_required_fields():
    tok, ng = _fitted()
    examples = [{
        "EXAMPLE_ID": "z1",
        "FAMILY": "mosfet",
        "SEQUENCE": "|".join(MINI_DATASET["mosfet"]["mosfet_0001"]),
    }]
    rows = list(run_task3(examples, tokenizer=tok, ngram=ng, threshold=-100.0, strategy="hybrid"))
    assert rows[0].example_id == "z1"
    assert rows[0].is_valid in (0, 1)
    assert 0.0 <= rows[0].score <= 1.0
    assert isinstance(rows[0].predicted_rule, str)
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_predictor.py::test_run_task3_emits_anomaly_rows_with_required_fields -v
```
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Add `run_task3` to `predictor.py`**

```python
@dataclass
class Task3Row:
    example_id: str
    is_valid: int
    score: float
    predicted_rule: str


def run_task3(
    examples: Iterable[dict],
    tokenizer: Tokenizer,
    ngram: NGram,
    threshold: float,
    strategy: str = "hybrid",   # "oracle" | "perplexity" | "hybrid"
) -> Iterator[Task3Row]:
    for ex in examples:
        family = ex["FAMILY"]
        steps = ex["SEQUENCE"].split("|")
        if strategy == "oracle":
            res = detect_oracle(steps)
        elif strategy == "perplexity":
            res = detect_perplexity(steps, family, tokenizer, ngram, threshold)
        elif strategy == "hybrid":
            res = detect_hybrid(steps, family, tokenizer, ngram, threshold)
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
        yield Task3Row(
            example_id=ex["EXAMPLE_ID"],
            is_valid=res.is_valid,
            score=res.score,
            predicted_rule=res.predicted_rule,
        )
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_predictor.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/predictor.py tests/test_predictor.py
git commit -m "feat(predictor): run_task3 anomaly detection with strategy switch"
```

---

## Task 17: submission writers (all three tasks)

**Files:**
- Create: `src/infineon_baseline/submission.py`
- Create: `tests/test_submission.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_submission.py`:
```python
import csv
from pathlib import Path

import pytest

from infineon_baseline.predictor import Task1Row, Task2Row, Task3Row
from infineon_baseline.submission import (
    write_task1_csv, write_task2_csv, write_task3_csv,
)


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_task1_writer_emits_exact_columns(tmp_path):
    out = tmp_path / "t1.csv"
    rows = [Task1Row(example_id="x", ranks=["A", "B", "C", "D", "E"])]
    write_task1_csv(rows, out)
    data = _read(out)
    assert list(data[0].keys()) == ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"]
    assert data[0]["RANK_3"] == "C"


def test_task1_writer_refuses_wrong_rank_count(tmp_path):
    out = tmp_path / "t1.csv"
    rows = [Task1Row(example_id="x", ranks=["A", "B"])]
    with pytest.raises(ValueError, match="exactly 5 ranks"):
        write_task1_csv(rows, out)


def test_task2_writer_pipe_separated(tmp_path):
    out = tmp_path / "t2.csv"
    rows = [Task2Row(example_id="y", predicted_sequence="STEP_A|STEP_B|STEP_C")]
    write_task2_csv(rows, out)
    data = _read(out)
    assert data[0]["PREDICTED_SEQUENCE"] == "STEP_A|STEP_B|STEP_C"


def test_task3_writer_columns_and_value_ranges(tmp_path):
    out = tmp_path / "t3.csv"
    rows = [
        Task3Row(example_id="a", is_valid=1, score=0.95, predicted_rule=""),
        Task3Row(example_id="b", is_valid=0, score=0.08, predicted_rule="RULE_DEP_NO_CLEAN"),
    ]
    write_task3_csv(rows, out)
    data = _read(out)
    assert set(data[0].keys()) == {"EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"}
    assert data[0]["IS_VALID"] == "1" and data[0]["PREDICTED_RULE"] == ""
    assert data[1]["PREDICTED_RULE"] == "RULE_DEP_NO_CLEAN"


def test_task3_writer_rejects_invalid_is_valid(tmp_path):
    out = tmp_path / "t3.csv"
    rows = [Task3Row(example_id="a", is_valid=2, score=0.5, predicted_rule="")]
    with pytest.raises(ValueError, match="IS_VALID must be 0 or 1"):
        write_task3_csv(rows, out)


def test_task3_writer_rejects_score_out_of_range(tmp_path):
    out = tmp_path / "t3.csv"
    rows = [Task3Row(example_id="a", is_valid=1, score=1.5, predicted_rule="")]
    with pytest.raises(ValueError, match="SCORE must be in"):
        write_task3_csv(rows, out)
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_submission.py -v
```
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `src/infineon_baseline/submission.py`**

```python
"""Spec-compliant submission CSV writers with row validation."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from infineon_baseline.predictor import Task1Row, Task2Row, Task3Row


def _ensure_parent(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_task1_csv(rows: Iterable[Task1Row], path: Path) -> None:
    path = _ensure_parent(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        for r in rows:
            if len(r.ranks) != 5:
                raise ValueError(
                    f"Task1Row {r.example_id}: must emit exactly 5 ranks, got {len(r.ranks)}"
                )
            w.writerow([r.example_id, *r.ranks])


def write_task2_csv(rows: Iterable[Task2Row], path: Path) -> None:
    path = _ensure_parent(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        for r in rows:
            if "||" in r.predicted_sequence or r.predicted_sequence.startswith("|") or r.predicted_sequence.endswith("|"):
                raise ValueError(
                    f"Task2Row {r.example_id}: malformed pipe-separated sequence"
                )
            w.writerow([r.example_id, r.predicted_sequence])


def write_task3_csv(rows: Iterable[Task3Row], path: Path) -> None:
    path = _ensure_parent(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
        for r in rows:
            if r.is_valid not in (0, 1):
                raise ValueError(f"Task3Row {r.example_id}: IS_VALID must be 0 or 1, got {r.is_valid}")
            if not (0.0 <= r.score <= 1.0):
                raise ValueError(f"Task3Row {r.example_id}: SCORE must be in [0,1], got {r.score}")
            if r.is_valid == 1 and r.predicted_rule:
                raise ValueError(f"Task3Row {r.example_id}: PREDICTED_RULE must be empty when IS_VALID=1")
            w.writerow([r.example_id, r.is_valid, f"{r.score:.6f}", r.predicted_rule])


def write_meta(
    path: Path,
    task: str,
    model_info: dict,
    seed: int,
    eval_split_hash: str,
) -> None:
    path = _ensure_parent(path)
    path.write_text(json.dumps({
        "task": task,
        "model": model_info,
        "seed": seed,
        "eval_split_hash": eval_split_hash,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_submission.py -v
```
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/submission.py tests/test_submission.py
git commit -m "feat(submission): spec-compliant CSV writers for all three tasks"
```

---

## Task 18: metrics — sequence tasks (Tasks 1 & 2)

**Files:**
- Create: `src/infineon_baseline/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
import pandas as pd

from infineon_baseline.metrics import (
    top_k_accuracy, mrr, exact_match_rate, normalized_edit_distance,
    token_accuracy, block_accuracy, STEP_TO_BLOCK,
)


def test_top_k_accuracy_known_case():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "RANK_1": "X", "RANK_2": "Y", "RANK_3": "Z", "RANK_4": "Q", "RANK_5": "R"},
    ])
    gt = pd.DataFrame([{"EXAMPLE_ID": "a", "NEXT_STEP": "Y"}])
    assert top_k_accuracy(preds, gt, k=2) == 1.0
    assert top_k_accuracy(preds, gt, k=1) == 0.0


def test_mrr_uses_inverse_rank_of_first_hit():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "RANK_1": "X", "RANK_2": "Y", "RANK_3": "Z", "RANK_4": "Q", "RANK_5": "R"},
    ])
    gt = pd.DataFrame([{"EXAMPLE_ID": "a", "NEXT_STEP": "Y"}])
    assert mrr(preds, gt) == 0.5


def test_exact_match_rate():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "PREDICTED_SEQUENCE": "A|B|C"},
        {"EXAMPLE_ID": "b", "PREDICTED_SEQUENCE": "X|Y"},
    ])
    gt = pd.DataFrame([
        {"EXAMPLE_ID": "a", "GROUND_TRUTH_SEQUENCE": "A|B|C"},
        {"EXAMPLE_ID": "b", "GROUND_TRUTH_SEQUENCE": "X|Z"},
    ])
    assert exact_match_rate(preds, gt) == 0.5


def test_normalized_edit_distance():
    # Two identical: 0.0 ; one off-by-one of length 3: 1/3.
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "PREDICTED_SEQUENCE": "A|B|C"},
        {"EXAMPLE_ID": "b", "PREDICTED_SEQUENCE": "A|X|C"},
    ])
    gt = pd.DataFrame([
        {"EXAMPLE_ID": "a", "GROUND_TRUTH_SEQUENCE": "A|B|C"},
        {"EXAMPLE_ID": "b", "GROUND_TRUTH_SEQUENCE": "A|B|C"},
    ])
    val = normalized_edit_distance(preds, gt)
    assert 0.15 < val < 0.2  # (0 + 1/3) / 2 ≈ 0.1666


def test_token_accuracy_overlap_region():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "PREDICTED_SEQUENCE": "A|B|C|D"},
    ])
    gt = pd.DataFrame([
        {"EXAMPLE_ID": "a", "GROUND_TRUTH_SEQUENCE": "A|B|X|D"},  # 3/4 match
    ])
    assert token_accuracy(preds, gt) == 0.75


def test_block_accuracy_uses_step_to_block_mapping():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "PREDICTED_SEQUENCE": "PRE CLEAN WAFER|OXIDE ETCH"},
    ])
    gt = pd.DataFrame([
        # Different step strings but same blocks (Cleaning, Etch)
        {"EXAMPLE_ID": "a", "GROUND_TRUTH_SEQUENCE": "RCA CLEAN 1|OXIDE ETCH DRY"},
    ])
    # block_accuracy compares the sequence of block names → both [CLEAN, ETCH]
    assert block_accuracy(preds, gt) == 1.0


def test_step_to_block_covers_basic_categories():
    # Smoke check that the mapping covers a handful of representative steps.
    assert STEP_TO_BLOCK["RECEIVE WAFER LOT"] == "LOGISTICS"
    assert STEP_TO_BLOCK["PRE CLEAN WAFER"] == "CLEAN"
    assert STEP_TO_BLOCK["OXIDE ETCH"] == "ETCH"
    assert STEP_TO_BLOCK["DEPOSIT POLYSILICON"] == "DEPOSIT"
    assert STEP_TO_BLOCK["IMPLANT WELL"] == "IMPLANT"
    assert STEP_TO_BLOCK["WAFER SORT TEST"] == "TEST"
    assert STEP_TO_BLOCK["SHIP LOT"] == "LOGISTICS"
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_metrics.py -v
```
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `src/infineon_baseline/metrics.py`**

```python
"""Metrics for all three tasks plus a report() helper.

All metric functions are pure: take pandas DataFrames keyed by EXAMPLE_ID,
return a float (or dict for matrix-style outputs).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Block (12 functional categories from generation_rules.md §1)                #
# --------------------------------------------------------------------------- #
# Coarse heuristic mapping. Extend as needed.
_BLOCK_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("LOGISTICS", ("RECEIVE WAFER LOT", "LOT IDENTIFICATION", "LOT RELEASE",
                   "FINAL LOT RELEASE", "SHIP LOT", "PACKAGE PREPARATION")),
    ("INSPECT",   ("INSPECTION", "INSPECT")),
    ("MEASURE",   ("MEASURE",)),
    ("CLEAN",     ("CLEAN", "RCA", "HF DIP", "DRY WAFER", "RINSE")),
    ("DEPOSIT",   ("DEPOSIT", "GROW", "THERMAL OXIDATION", "EPITAXIAL DEPOSITION",
                   "ANNEAL OXIDE", "GATE OXIDE")),
    ("LITHO",     ("SPIN COAT PHOTORESIST", "SOFT BAKE", "ALIGN MASK", "EXPOSE LITHO",
                   "POST EXPOSE BAKE", "DEVELOP PHOTORESIST", "HARD BAKE", "DEVELOP PAD WINDOW",
                   "PAD WINDOW LITHO", "OPEN PAD WINDOW", "OPEN BOND PAD WINDOW")),
    ("ETCH",      ("ETCH", "PASSIVATION ETCH")),
    ("STRIP",     ("STRIP",)),
    ("IMPLANT",   ("IMPLANT", "DRIVE IN DIFFUSION", "RAPID THERMAL ANNEAL", "LIGHT ANNEAL")),
    ("CMP",       ("CMP",)),
    ("VIA_FILL",  ("FILL VIA",)),
    ("TEST",      ("PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST", "THRESHOLD VOLTAGE TEST",
                   "BREAKDOWN VOLTAGE TEST", "LEAKAGE TEST", "SWITCHING TEST",
                   "WAFER SORT TEST", "YIELD ANALYSIS", "FINAL ELECTRICAL TEST PREP")),
]


def _step_to_block(step: str) -> str:
    for block, patterns in _BLOCK_PATTERNS:
        for p in patterns:
            if step == p or step.startswith(p) or p in step:
                return block
    return "OTHER"


# Pre-computed lazy dict-style mapping for tests that introspect it.
class _StepToBlock(dict):
    def __missing__(self, key):
        return _step_to_block(key)


STEP_TO_BLOCK = _StepToBlock()


# --------------------------------------------------------------------------- #
# Task 1 metrics                                                              #
# --------------------------------------------------------------------------- #
def top_k_accuracy(preds: pd.DataFrame, gt: pd.DataFrame, k: int) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID")
    hits = 0
    for _, row in merged.iterrows():
        cols = [f"RANK_{i}" for i in range(1, k + 1)]
        if row["NEXT_STEP"] in [row[c] for c in cols]:
            hits += 1
    return hits / len(merged) if len(merged) else 0.0


def mrr(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID")
    total = 0.0
    for _, row in merged.iterrows():
        for rank_i in range(1, 6):
            if row[f"RANK_{rank_i}"] == row["NEXT_STEP"]:
                total += 1.0 / rank_i
                break
    return total / len(merged) if len(merged) else 0.0


# --------------------------------------------------------------------------- #
# Task 2 metrics                                                              #
# --------------------------------------------------------------------------- #
def exact_match_rate(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID")
    matches = (merged["PREDICTED_SEQUENCE"] == merged["GROUND_TRUTH_SEQUENCE"]).sum()
    return float(matches) / len(merged) if len(merged) else 0.0


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cost = 0 if ai == bj else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def normalized_edit_distance(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID")
    total = 0.0
    for _, row in merged.iterrows():
        p = row["PREDICTED_SEQUENCE"].split("|") if row["PREDICTED_SEQUENCE"] else []
        g = row["GROUND_TRUTH_SEQUENCE"].split("|") if row["GROUND_TRUTH_SEQUENCE"] else []
        if not p and not g:
            continue
        total += _levenshtein(p, g) / max(len(p), len(g))
    return total / len(merged) if len(merged) else 0.0


def token_accuracy(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID")
    correct = total = 0
    for _, row in merged.iterrows():
        p = row["PREDICTED_SEQUENCE"].split("|")
        g = row["GROUND_TRUTH_SEQUENCE"].split("|")
        for a, b in zip(p, g):
            total += 1
            if a == b:
                correct += 1
    return correct / total if total else 0.0


def block_accuracy(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID")
    correct = total = 0
    for _, row in merged.iterrows():
        pb = [STEP_TO_BLOCK[s] for s in row["PREDICTED_SEQUENCE"].split("|")]
        gb = [STEP_TO_BLOCK[s] for s in row["GROUND_TRUTH_SEQUENCE"].split("|")]
        for a, b in zip(pb, gb):
            total += 1
            if a == b:
                correct += 1
    return correct / total if total else 0.0
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_metrics.py -v
```
Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): sequence-task metrics + step→block mapping"
```

---

## Task 19: metrics — anomaly (Task 3)

**Files:**
- Modify: `src/infineon_baseline/metrics.py`
- Modify: `tests/test_metrics.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_metrics.py`:
```python
from infineon_baseline.metrics import (
    binary_accuracy, precision, recall, f1, confusion_matrix_dict,
    roc_auc, rule_attribution_accuracy,
)


def test_binary_accuracy_simple():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 1, "SCORE": 0.9, "PREDICTED_RULE": ""},
        {"EXAMPLE_ID": "b", "IS_VALID": 0, "SCORE": 0.1, "PREDICTED_RULE": "RULE_X"},
    ])
    gt = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 1, "RULE_VIOLATED": ""},
        {"EXAMPLE_ID": "b", "IS_VALID": 1, "RULE_VIOLATED": ""},  # mispredicted
    ])
    assert binary_accuracy(preds, gt) == 0.5


def test_precision_recall_f1_consistency():
    # 1 TP, 1 FP, 1 FN  → P=0.5, R=0.5, F1=0.5
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 0, "SCORE": 0.1, "PREDICTED_RULE": "X"},  # TP (correctly invalid)
        {"EXAMPLE_ID": "b", "IS_VALID": 0, "SCORE": 0.2, "PREDICTED_RULE": "X"},  # FP
        {"EXAMPLE_ID": "c", "IS_VALID": 1, "SCORE": 0.9, "PREDICTED_RULE": ""},   # FN (truly invalid)
    ])
    gt = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 0, "RULE_VIOLATED": "X"},
        {"EXAMPLE_ID": "b", "IS_VALID": 1, "RULE_VIOLATED": ""},
        {"EXAMPLE_ID": "c", "IS_VALID": 0, "RULE_VIOLATED": "Y"},
    ])
    # Positive class = invalid (IS_VALID==0).
    assert precision(preds, gt) == 0.5
    assert recall(preds, gt) == 0.5
    assert abs(f1(preds, gt) - 0.5) < 1e-9


def test_roc_auc_uses_score_column():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 1, "SCORE": 0.9, "PREDICTED_RULE": ""},
        {"EXAMPLE_ID": "b", "IS_VALID": 0, "SCORE": 0.1, "PREDICTED_RULE": "X"},
        {"EXAMPLE_ID": "c", "IS_VALID": 1, "SCORE": 0.8, "PREDICTED_RULE": ""},
        {"EXAMPLE_ID": "d", "IS_VALID": 0, "SCORE": 0.2, "PREDICTED_RULE": "X"},
    ])
    gt = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 1, "RULE_VIOLATED": ""},
        {"EXAMPLE_ID": "b", "IS_VALID": 0, "RULE_VIOLATED": "X"},
        {"EXAMPLE_ID": "c", "IS_VALID": 1, "RULE_VIOLATED": ""},
        {"EXAMPLE_ID": "d", "IS_VALID": 0, "RULE_VIOLATED": "X"},
    ])
    # Perfect separation by SCORE → AUC = 1.0
    assert roc_auc(preds, gt) == 1.0


def test_rule_attribution_accuracy():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 0, "SCORE": 0.1, "PREDICTED_RULE": "RULE_X"},
        {"EXAMPLE_ID": "b", "IS_VALID": 0, "SCORE": 0.1, "PREDICTED_RULE": "RULE_X"},
    ])
    gt = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 0, "RULE_VIOLATED": "RULE_X"},  # correct
        {"EXAMPLE_ID": "b", "IS_VALID": 0, "RULE_VIOLATED": "RULE_Y"},  # incorrect
    ])
    assert rule_attribution_accuracy(preds, gt) == 0.5


def test_confusion_matrix_dict_has_tp_fp_fn_tn():
    preds = pd.DataFrame([{"EXAMPLE_ID": "a", "IS_VALID": 1, "SCORE": 0.9, "PREDICTED_RULE": ""}])
    gt = pd.DataFrame([{"EXAMPLE_ID": "a", "IS_VALID": 1, "RULE_VIOLATED": ""}])
    cm = confusion_matrix_dict(preds, gt)
    assert set(cm.keys()) == {"tp", "fp", "fn", "tn"}
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_metrics.py -k "binary or precision or roc_auc or rule_attr or confusion" -v
```
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Append anomaly metrics to `metrics.py`**

```python
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as _sk_confusion,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _binarize_invalid(df: pd.DataFrame) -> pd.Series:
    """Return Series of 1 = invalid (positive class), 0 = valid."""
    return (df["IS_VALID"] == 0).astype(int)


def binary_accuracy(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID", suffixes=("_pred", "_gt"))
    y_pred = (merged["IS_VALID_pred"] == 0).astype(int)
    y_true = (merged["IS_VALID_gt"] == 0).astype(int)
    return float(accuracy_score(y_true, y_pred))


def precision(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID", suffixes=("_pred", "_gt"))
    y_pred = (merged["IS_VALID_pred"] == 0).astype(int)
    y_true = (merged["IS_VALID_gt"] == 0).astype(int)
    return float(precision_score(y_true, y_pred, zero_division=0))


def recall(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID", suffixes=("_pred", "_gt"))
    y_pred = (merged["IS_VALID_pred"] == 0).astype(int)
    y_true = (merged["IS_VALID_gt"] == 0).astype(int)
    return float(recall_score(y_true, y_pred, zero_division=0))


def f1(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID", suffixes=("_pred", "_gt"))
    y_pred = (merged["IS_VALID_pred"] == 0).astype(int)
    y_true = (merged["IS_VALID_gt"] == 0).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def confusion_matrix_dict(preds: pd.DataFrame, gt: pd.DataFrame) -> dict[str, int]:
    merged = preds.merge(gt, on="EXAMPLE_ID", suffixes=("_pred", "_gt"))
    y_pred = (merged["IS_VALID_pred"] == 0).astype(int)
    y_true = (merged["IS_VALID_gt"] == 0).astype(int)
    cm = _sk_confusion(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def roc_auc(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID", suffixes=("_pred", "_gt"))
    y_true = (merged["IS_VALID_gt"] == 0).astype(int)
    # Higher SCORE = more confident this row is VALID. To predict the invalid class,
    # use 1 - SCORE as the positive-class likelihood.
    y_score = 1.0 - merged["SCORE"].astype(float)
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def rule_attribution_accuracy(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID", suffixes=("_pred", "_gt"))
    both_invalid = merged[(merged["IS_VALID_pred"] == 0) & (merged["IS_VALID_gt"] == 0)]
    if len(both_invalid) == 0:
        return 0.0
    matches = (both_invalid["PREDICTED_RULE"] == both_invalid["RULE_VIOLATED"]).sum()
    return float(matches) / len(both_invalid)
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_metrics.py -v
```
Expected: 12 tests PASS (7 sequence + 5 anomaly).

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): anomaly metrics via sklearn + rule-attribution accuracy"
```

---

## Task 20: metrics — `report()` with per-family breakdown + JSON dump

**Files:**
- Modify: `src/infineon_baseline/metrics.py`
- Modify: `tests/test_metrics.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_metrics.py`:
```python
from infineon_baseline.metrics import report


def test_report_task1_returns_dict_with_aggregate_and_family_breakdowns(tmp_path):
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "valid_mosfet_a_60", "FAMILY": "mosfet",
         "RANK_1": "X", "RANK_2": "Y", "RANK_3": "Z", "RANK_4": "Q", "RANK_5": "R"},
        {"EXAMPLE_ID": "valid_igbt_b_60", "FAMILY": "igbt",
         "RANK_1": "Y", "RANK_2": "Z", "RANK_3": "X", "RANK_4": "Q", "RANK_5": "R"},
    ])
    gt = pd.DataFrame([
        {"EXAMPLE_ID": "valid_mosfet_a_60", "FAMILY": "mosfet", "NEXT_STEP": "Y"},
        {"EXAMPLE_ID": "valid_igbt_b_60",   "FAMILY": "igbt",   "NEXT_STEP": "Y"},
    ])
    rep = report(task="next-step", predictions=preds, ground_truth=gt)
    assert "overall" in rep and "per_family" in rep
    assert "top_1_accuracy" in rep["overall"]
    assert "mosfet" in rep["per_family"]
    out = tmp_path / "report.json"
    out.write_text(json.dumps(rep))
    # Sanity: re-read produces same dict
    import json as _json
    assert _json.loads(out.read_text()) == rep
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_metrics.py::test_report_task1_returns_dict_with_aggregate_and_family_breakdowns -v
```
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Add `report` to `metrics.py`**

```python
def _compute_task1(p: pd.DataFrame, g: pd.DataFrame) -> dict:
    return {
        "top_1_accuracy": top_k_accuracy(p, g, 1),
        "top_3_accuracy": top_k_accuracy(p, g, 3),
        "top_5_accuracy": top_k_accuracy(p, g, 5),
        "mrr": mrr(p, g),
        "n": len(p),
    }


def _compute_task2(p: pd.DataFrame, g: pd.DataFrame) -> dict:
    return {
        "exact_match_rate": exact_match_rate(p, g),
        "normalized_edit_distance": normalized_edit_distance(p, g),
        "token_accuracy": token_accuracy(p, g),
        "block_accuracy": block_accuracy(p, g),
        "n": len(p),
    }


def _compute_task3(p: pd.DataFrame, g: pd.DataFrame) -> dict:
    out = {
        "binary_accuracy": binary_accuracy(p, g),
        "precision": precision(p, g),
        "recall": recall(p, g),
        "f1": f1(p, g),
        "confusion": confusion_matrix_dict(p, g),
        "rule_attribution_accuracy": rule_attribution_accuracy(p, g),
        "n": len(p),
    }
    try:
        out["roc_auc"] = roc_auc(p, g)
    except Exception:
        out["roc_auc"] = float("nan")
    return out


_TASK_DISPATCH = {
    "next-step": _compute_task1,
    "complete":  _compute_task2,
    "anomaly":   _compute_task3,
}


def report(task: str, predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    """Compute the metric dict with overall + per-family breakdowns."""
    if task not in _TASK_DISPATCH:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(_TASK_DISPATCH)}")
    compute = _TASK_DISPATCH[task]
    overall = compute(predictions, ground_truth)
    per_family: dict[str, dict] = {}
    if "FAMILY" in predictions.columns and "FAMILY" in ground_truth.columns:
        for family, p_sub in predictions.groupby("FAMILY"):
            g_sub = ground_truth[ground_truth["FAMILY"] == family]
            if len(p_sub) == 0:
                continue
            per_family[str(family)] = compute(p_sub, g_sub)
    rep = {"task": task, "overall": overall, "per_family": per_family}
    _print_table(rep)
    return rep


def _print_table(rep: dict) -> None:
    """Pretty-print a metrics dict to stdout."""
    print(f"\nTASK: {rep['task']}")
    print("-" * 78)
    overall = rep["overall"]
    keys = [k for k in overall if k not in ("n", "confusion")]
    header = f"{'metric':32s}" + "".join(f"{fam:>10s}" for fam in sorted(rep["per_family"])) + f"{'ALL':>10s}"
    print(header)
    for key in keys:
        row = f"{key:32s}"
        for fam in sorted(rep["per_family"]):
            val = rep["per_family"][fam].get(key, float("nan"))
            row += f"{val:10.3f}" if isinstance(val, (int, float)) else f"{'-':>10s}"
        val = overall.get(key, float("nan"))
        row += f"{val:10.3f}" if isinstance(val, (int, float)) else f"{'-':>10s}"
        print(row)
    print(f"n: {overall['n']}")
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_metrics.py -v
```
Expected: 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): report() with per-family breakdown + console table"
```

---

## Task 21: CLI — all four subcommands

**Files:**
- Create: `src/infineon_baseline/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
import subprocess
import sys


def test_cli_help_shows_all_four_subcommands():
    result = subprocess.run(
        [sys.executable, "-m", "infineon_baseline.cli", "--help"],
        capture_output=True, text=True, check=True,
    )
    out = result.stdout
    for sub in ("build-eval", "fit", "predict", "score"):
        assert sub in out
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_cli.py -v
```
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `src/infineon_baseline/cli.py`**

```python
"""Single CLI entry point with four subcommands.

Usage:
    infineon-baseline build-eval --variants-dir DIR --out DIR --seed 42
    infineon-baseline fit        --train DIR --out PATH [--order 3]
    infineon-baseline predict    --model PATH --eval-input PATH --task TASK --out PATH
    infineon-baseline score      --predictions PATH --ground-truth PATH --task TASK
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from infineon_baseline.anomaly import calibrate_threshold
from infineon_baseline.eval_set import (
    build_anomaly_eval_inputs, build_valid_eval_inputs,
    split, write_eval_inputs_anomaly, write_eval_inputs_valid,
    write_ground_truth_anomaly, write_ground_truth_valid,
)
from infineon_baseline.loaders import load_variants
from infineon_baseline.metrics import report
from infineon_baseline.ngram import NGram
from infineon_baseline.predictor import run_task1, run_task2, run_task3
from infineon_baseline.submission import (
    write_meta, write_task1_csv, write_task2_csv, write_task3_csv,
)
from infineon_baseline.tokenizer import Tokenizer


def _load_corpus(variants_dir: Path) -> dict[str, dict[str, list[str]]]:
    """Load *_variants.csv for all three families from a directory."""
    by_family: dict[str, dict[str, list[str]]] = {}
    for family in ("mosfet", "igbt", "ic"):
        candidates = list(Path(variants_dir).glob(f"*{family.upper()}*variants*.csv"))
        if not candidates:
            raise FileNotFoundError(f"no {family} variants CSV in {variants_dir}")
        by_family[family] = load_variants(candidates[0])
    return by_family


def _hash_split(holdout_ids: list[str]) -> str:
    h = hashlib.sha256()
    for sid in sorted(holdout_ids):
        h.update(sid.encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Subcommand: build-eval
# --------------------------------------------------------------------------- #
def cmd_build_eval(args: argparse.Namespace) -> int:
    corpus = _load_corpus(Path(args.variants_dir))
    train, hold = split(corpus, holdout_per_family=args.holdout_per_family, seed=args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Valid eval inputs (Tasks 1 & 2): use first half of hold-out per family for truncation.
    valid_rows = build_valid_eval_inputs(
        hold, seqs_per_family=args.holdout_per_family // 2, fractions=(0.6, 0.8),
    )
    write_eval_inputs_valid(valid_rows, out_dir / "eval_input_valid.csv")
    write_ground_truth_valid(valid_rows, out_dir / "ground_truth_valid.csv")

    # Anomaly eval inputs (Task 3): full hold-out, with target invalid ratio.
    anom_rows = build_anomaly_eval_inputs(
        hold,
        seqs_per_family=args.holdout_per_family,
        invalid_ratio=args.anomaly_invalid_ratio,
        seed=args.seed,
    )
    write_eval_inputs_anomaly(anom_rows, out_dir / "eval_input_anomaly.csv")
    write_ground_truth_anomaly(anom_rows, out_dir / "ground_truth_anomaly.csv")

    # Persist the training split for `fit`.
    train_path = out_dir / "train_split.csv"
    with train_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SEQUENCE_ID", "FAMILY", "STEP"])
        for family in sorted(train):
            for sid in sorted(train[family]):
                for step in train[family][sid]:
                    w.writerow([sid, family, step])

    holdout_ids = [sid for fam_seqs in hold.values() for sid in fam_seqs]
    meta = {
        "seed": args.seed,
        "holdout_per_family": args.holdout_per_family,
        "anomaly_invalid_ratio": args.anomaly_invalid_ratio,
        "split_hash": _hash_split(holdout_ids),
        "n_train_seqs": sum(len(s) for s in train.values()),
        "n_hold_seqs": sum(len(s) for s in hold.values()),
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"✓ wrote eval files to {out_dir}/  (split_hash={meta['split_hash']})")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: fit
# --------------------------------------------------------------------------- #
def cmd_fit(args: argparse.Namespace) -> int:
    # Read train_split.csv into {family: {sid: [steps]}}
    train_split: dict[str, dict[str, list[str]]] = {}
    with Path(args.train).open(newline="") as f:
        for row in csv.DictReader(f):
            train_split.setdefault(row["FAMILY"], {}).setdefault(row["SEQUENCE_ID"], []).append(row["STEP"])
    tokenizer = Tokenizer.fit(train_split)
    train_ids = {
        family: [tokenizer.encode(family, steps)[1] for steps in seqs.values()]
        for family, seqs in train_split.items()
    }
    model = NGram(order=args.order).fit(train_ids)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    tokenizer.save(out_path.with_suffix(".tokenizer.json"))
    print(f"✓ fitted n-gram (order={args.order}) over {sum(len(s) for s in train_split.values())} sequences")
    print(f"  → {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: predict
# --------------------------------------------------------------------------- #
def _read_eval_input(path: Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def cmd_predict(args: argparse.Namespace) -> int:
    model = NGram.load(args.model)
    tokenizer = Tokenizer.load(Path(args.model).with_suffix(".tokenizer.json"))
    examples = _read_eval_input(args.eval_input)
    out_path = Path(args.out)

    if args.task == "next-step":
        rows = list(run_task1(examples, tokenizer=tokenizer, ngram=model))
        write_task1_csv(rows, out_path)
    elif args.task == "complete":
        rows = list(run_task2(examples, tokenizer=tokenizer, ngram=model, constrain=args.constrain))
        write_task2_csv(rows, out_path)
    elif args.task == "anomaly":
        threshold = args.threshold if args.threshold is not None else 0.0
        rows = list(run_task3(examples, tokenizer=tokenizer, ngram=model,
                              threshold=threshold, strategy=args.anomaly_strategy))
        write_task3_csv(rows, out_path)
    else:
        print(f"unknown task: {args.task}", file=sys.stderr)
        return 2

    write_meta(
        out_path.with_suffix(".meta.json"),
        task=args.task,
        model_info={"type": "ngram", "order": model.order},
        seed=args.seed,
        eval_split_hash="(set externally)",
    )
    print(f"✓ wrote {len(rows)} predictions → {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: score
# --------------------------------------------------------------------------- #
def cmd_score(args: argparse.Namespace) -> int:
    preds = pd.read_csv(args.predictions)
    gt = pd.read_csv(args.ground_truth)
    # Carry FAMILY across from preds if absent in gt, and vice versa (joins on EXAMPLE_ID).
    if "FAMILY" not in gt.columns and "FAMILY" in preds.columns:
        gt = gt.merge(preds[["EXAMPLE_ID", "FAMILY"]], on="EXAMPLE_ID", how="left")
    if "FAMILY" not in preds.columns and "FAMILY" in gt.columns:
        preds = preds.merge(gt[["EXAMPLE_ID", "FAMILY"]], on="EXAMPLE_ID", how="left")
    # For Task 2: gt column rename for our metrics' expected FULL/GROUND_TRUTH names.
    if args.task == "complete" and "FULL_SEQUENCE" in gt.columns and "GROUND_TRUTH_SEQUENCE" not in gt.columns:
        # The ground truth FULL_SEQUENCE is the entire sequence; we want only the post-cut suffix.
        gt["GROUND_TRUTH_SEQUENCE"] = gt.apply(
            lambda r: "|".join(r["FULL_SEQUENCE"].split("|")[int(r["CUT_INDEX"]):]),
            axis=1,
        )
    if args.task == "next-step" and "NEXT_STEP" not in gt.columns and "FULL_SEQUENCE" in gt.columns:
        gt["NEXT_STEP"] = gt.apply(
            lambda r: r["FULL_SEQUENCE"].split("|")[int(r["CUT_INDEX"])],
            axis=1,
        )
    rep = report(args.task, predictions=preds, ground_truth=gt)
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps(rep, indent=2, default=str))
        print(f"✓ wrote report → {args.report_json}")
    return 0


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="infineon-baseline")
    parser.add_argument("--seed", type=int, default=42, help="global random seed")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-eval", help="build held-out eval set + ground truth")
    p.add_argument("--variants-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--holdout-per-family", type=int, default=200)
    p.add_argument("--anomaly-invalid-ratio", type=float, default=0.39)

    p = sub.add_parser("fit", help="fit the n-gram on a train split")
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--order", type=int, default=3)

    p = sub.add_parser("predict", help="produce a submission for one task")
    p.add_argument("--model", required=True)
    p.add_argument("--eval-input", required=True)
    p.add_argument("--task", required=True, choices=["next-step", "complete", "anomaly"])
    p.add_argument("--out", required=True)
    p.add_argument("--constrain", action="store_true", help="rule-aware decoder for complete")
    p.add_argument("--anomaly-strategy", choices=["oracle", "perplexity", "hybrid"], default="hybrid")
    p.add_argument("--threshold", type=float, default=None)

    p = sub.add_parser("score", help="score predictions against ground truth")
    p.add_argument("--predictions", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--task", required=True, choices=["next-step", "complete", "anomaly"])
    p.add_argument("--report-json", default=None)

    args = parser.parse_args(argv)
    dispatch = {
        "build-eval": cmd_build_eval,
        "fit": cmd_fit,
        "predict": cmd_predict,
        "score": cmd_score,
    }
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_cli.py -v
```
Expected: 1 test PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/cli.py tests/test_cli.py
git commit -m "feat(cli): four subcommands wired (build-eval, fit, predict, score)"
```

---

## Task 22: CLI smoke test (end-to-end on real corpus sample)

**Files:**
- Create: `tests/test_cli_smoke.py`
- Create: `tests/fixtures/build_mini_corpus.py` (helper used only by this test)

- [ ] **Step 1: Write the failing test**

`tests/test_cli_smoke.py`:
```python
"""End-to-end smoke test: run the four CLI subcommands on a tiny real-data sample."""
import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA = REPO_ROOT / "training_data"


def _sample_variants(src_csv: Path, dst_csv: Path, n_seqs: int = 10) -> None:
    """Copy the first n_seqs sequences from a variants CSV into dst_csv."""
    seen: set[str] = set()
    with src_csv.open(newline="") as fi, dst_csv.open("w", newline="") as fo:
        reader = csv.DictReader(fi)
        writer = csv.DictWriter(fo, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["SEQUENCE_ID"] not in seen:
                if len(seen) >= n_seqs:
                    break
                seen.add(row["SEQUENCE_ID"])
            writer.writerow(row)


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, "-m", "infineon_baseline.cli", *args], check=True)


def test_full_pipeline_end_to_end(tmp_path):
    # 1. Sample 10 sequences/family into a mini variants dir
    mini = tmp_path / "training_data"
    mini.mkdir()
    for family in ("MOSFET", "IGBT", "IC"):
        _sample_variants(TRAINING_DATA / f"{family}_variants.csv",
                         mini / f"{family}_variants.csv", n_seqs=10)

    eval_dir = tmp_path / "eval"
    submissions = tmp_path / "submissions"
    submissions.mkdir()

    # 2. build-eval
    _run(["build-eval", "--variants-dir", str(mini),
          "--out", str(eval_dir), "--holdout-per-family", "4",
          "--anomaly-invalid-ratio", "0.5", "--seed", "0"])
    assert (eval_dir / "eval_input_valid.csv").exists()
    assert (eval_dir / "eval_input_anomaly.csv").exists()
    assert (eval_dir / "ground_truth_valid.csv").exists()
    assert (eval_dir / "ground_truth_anomaly.csv").exists()
    assert (eval_dir / "train_split.csv").exists()

    # 3. fit
    model_path = tmp_path / "ngram.pkl"
    _run(["fit", "--train", str(eval_dir / "train_split.csv"),
          "--out", str(model_path), "--order", "2"])
    assert model_path.exists()

    # 4. predict (all three tasks)
    _run(["predict", "--model", str(model_path),
          "--eval-input", str(eval_dir / "eval_input_valid.csv"),
          "--task", "next-step",
          "--out", str(submissions / "task1.csv")])
    _run(["predict", "--model", str(model_path),
          "--eval-input", str(eval_dir / "eval_input_valid.csv"),
          "--task", "complete",
          "--out", str(submissions / "task2.csv")])
    _run(["predict", "--model", str(model_path),
          "--eval-input", str(eval_dir / "eval_input_anomaly.csv"),
          "--task", "anomaly",
          "--out", str(submissions / "task3.csv"),
          "--threshold", "-50.0"])
    assert (submissions / "task1.csv").exists()
    assert (submissions / "task2.csv").exists()
    assert (submissions / "task3.csv").exists()

    # 5. score (just verifies the pipeline runs end-to-end; values may be low on tiny data)
    _run(["score", "--predictions", str(submissions / "task3.csv"),
          "--ground-truth", str(eval_dir / "ground_truth_anomaly.csv"),
          "--task", "anomaly",
          "--report-json", str(tmp_path / "report_task3.json")])
    assert (tmp_path / "report_task3.json").exists()
```

- [ ] **Step 2: Run, expect failure** (initially the eval-set CSVs and model don't exist)

```bash
pytest tests/test_cli_smoke.py -v
```
Expected: at minimum, the assertions for the first command may pass but later ones likely fail until subcommands are end-to-end correct. Fix any wiring issues uncovered (e.g., column-name mismatches between `predict` and `score`) by tweaking `cmd_score` argument-handling in `cli.py`. Iterate until green.

- [ ] **Step 3: Make it pass — fix any wiring issues found**

Common wiring fixes that may be needed:
- `cmd_score` for `--task anomaly` expects `IS_VALID` column on both inputs; if `ground_truth_anomaly.csv` has it but typed as int while predictions have it as string, the merge mismatches. Coerce both to int explicitly in `cmd_score` before passing to `report`.
- The `score` flow for `next-step` derives `NEXT_STEP` from `FULL_SEQUENCE[CUT_INDEX]`. Verify `ground_truth_valid.csv` has the `CUT_INDEX` column (`eval_set.write_ground_truth_valid` already writes it).

Run again until green.

- [ ] **Step 4: Re-run all tests to confirm no regressions**

```bash
pytest -v
```
Expected: every test in the suite PASSES (well over 50 tests total).

- [ ] **Step 5: Commit**

```bash
git add tests/test_cli_smoke.py
git commit -m "test: CLI smoke test runs all 4 subcommands end-to-end on real data"
```

---

## Task 23: Constrained decoder for Task 2 (`--constrain`)

**Status:** `--constrain` was already wired in Task 15. This task adds a dedicated test proving it actually changes behavior and improves rule-compliance.

**Files:**
- Modify: `tests/test_predictor.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_predictor.py`:
```python
from infineon_baseline import validate_sequence


def test_constrain_flag_reduces_rule_violations_in_completions():
    tok, ng = _fitted()
    # Build a partial that's known to often complete into a violation on greedy.
    partial = MINI_DATASET["mosfet"]["mosfet_0001"][:6]
    ex = [{
        "EXAMPLE_ID": "c1",
        "FAMILY": "mosfet",
        "COMPLETION_FRACTION": "0.5",
        "PARTIAL_SEQUENCE": "|".join(partial),
    }]
    unconstrained = next(iter(run_task2(ex, tokenizer=tok, ngram=ng, constrain=False)))
    constrained   = next(iter(run_task2(ex, tokenizer=tok, ngram=ng, constrain=True)))

    def violation_count(prefix: list[str], completion: str) -> int:
        full = prefix + completion.split("|")
        return len(validate_sequence(full))

    n_unc = violation_count(partial, unconstrained.predicted_sequence)
    n_con = violation_count(partial, constrained.predicted_sequence)
    # Constrained should be at least as good (fewer or equal violations).
    assert n_con <= n_unc
```

- [ ] **Step 2: Run, expect either PASS (lucky case) or FAIL if mini fixture is too small to differentiate**

```bash
pytest tests/test_predictor.py::test_constrain_flag_reduces_rule_violations_in_completions -v
```
Expected: PASS — the test asserts only `n_con <= n_unc`, which holds by construction in `run_task2` (constrained falls back to unconstrained when no candidate passes the filter). If it surprisingly fails, the regression is in `run_task2`; fix and re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_predictor.py
git commit -m "test(predictor): constrained decoder never increases violations"
```

---

## Task 24: Family-vocab mask (`NGram.fit` option)

**Files:**
- Modify: `src/infineon_baseline/ngram.py`
- Modify: `tests/test_ngram.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_ngram.py`:
```python
def test_family_vocab_mask_zeros_unseen_steps_in_family():
    # Build a corpus where step id 99 appears ONLY in igbt; mosfet never sees it.
    corpus = {
        "mosfet": [[0, 1, 2, 3]],
        "igbt":   [[0, 99, 2, 3]],
    }
    model = NGram(order=2, mask_family_vocab=True).fit(corpus)
    # top_k for mosfet must never return id 99 even under deepest backoff.
    top = model.top_k("mosfet", (), k=10)
    assert 99 not in top
    # igbt should still be able to predict 99.
    top_igbt = model.top_k("igbt", (), k=10)
    assert 99 in top_igbt
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_ngram.py::test_family_vocab_mask_zeros_unseen_steps_in_family -v
```
Expected: FAIL (`TypeError: NGram.__init__() got an unexpected keyword argument 'mask_family_vocab'`).

- [ ] **Step 3: Add `mask_family_vocab` to `NGram`**

Modify `NGram` in `src/infineon_baseline/ngram.py`:

Change the dataclass:
```python
@dataclass
class NGram:
    order: int = 3
    mask_family_vocab: bool = False
    counts: dict[str, dict[tuple[int, ...], Counter]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(Counter)))
    unigram: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    # New: per-family set of "allowed" step ids (everything else is masked to 0).
    _family_vocab: dict[str, frozenset[int]] = field(default_factory=dict)
```

In `fit`, after building counts, finalize the vocab set:
```python
        if self.mask_family_vocab:
            self._family_vocab = {
                fam: frozenset(self.unigram[fam].keys()) for fam in self.unigram
            }
        return self
```

In `top_k`, filter the candidates if masking is active:
```python
    def top_k(self, family: str, prefix: tuple[int, ...], k: int = 5) -> list[int]:
        if family not in self.unigram:
            return []
        allowed = self._family_vocab.get(family) if self.mask_family_vocab else None
        max_prefix_len = min(len(prefix), self.order - 1)
        for prefix_len in range(max_prefix_len, -1, -1):
            short = tuple(prefix[-prefix_len:]) if prefix_len > 0 else ()
            counter = self.counts[family].get(short)
            if counter:
                items = [(sid, c) for sid, c in counter.most_common() if (allowed is None or sid in allowed)]
                if items:
                    return [sid for sid, _ in items[:k]]
        items = [(sid, c) for sid, c in self.unigram[family].most_common()
                 if (allowed is None or sid in allowed)]
        return [sid for sid, _ in items[:k]]
```

In `_cond_prob`, apply the same filter (return 0 for masked-out ids):
```python
    def _cond_prob(self, family: str, prefix: tuple[int, ...], next_id: int, family_total: int) -> float:
        if self.mask_family_vocab:
            allowed = self._family_vocab.get(family, frozenset())
            if next_id not in allowed:
                return 0.0
        # (rest unchanged from Task 6)
        for prefix_len in range(len(prefix), -1, -1):
            short = tuple(prefix[-prefix_len:]) if prefix_len > 0 else ()
            counter = self.counts[family].get(short)
            if counter:
                denom = sum(counter.values())
                count = counter.get(next_id, 0)
                if count > 0:
                    return count / denom
        return self.unigram[family].get(next_id, 0) / family_total if family_total else 0.0
```

Update `save`/`load` to round-trip `mask_family_vocab` and `_family_vocab`:
- In `save()`, include `"mask_family_vocab": self.mask_family_vocab, "_family_vocab": {fam: list(ids) for fam, ids in self._family_vocab.items()}`.
- In `load()`, set `model.mask_family_vocab = data.get("mask_family_vocab", False)` and rebuild `_family_vocab`.

Also expose `--mask-family` on `cli.py fit` to set the flag, and persist it through save/load. Update `cmd_fit`:
```python
    model = NGram(order=args.order, mask_family_vocab=args.mask_family).fit(train_ids)
```
And in the parser:
```python
    p.add_argument("--mask-family", action="store_true",
                   help="zero probabilities for steps unseen in that family")
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_ngram.py -v
```
Expected: all NGram tests PASS (10 total).

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/ngram.py src/infineon_baseline/cli.py tests/test_ngram.py
git commit -m "feat(ngram): optional family-vocab mask + CLI --mask-family flag"
```

---

## Task 25: README documenting the CLI

**Files:**
- Modify: `README.md` (currently the upstream track README; we replace it with our project README and keep the original content in a `TRACK_README.md` sibling for reference)
- Create: `TRACK_README.md` (preserves the original track README content for reference)

Note: The original `README.md` is the *track briefing* from upstream; the project doesn't currently have a project README. We keep the briefing accessible by moving its content to `TRACK_README.md` and writing a project-focused README that documents the four CLI commands.

- [ ] **Step 1: Preserve the upstream track README**

```bash
git mv README.md TRACK_README.md
```

- [ ] **Step 2: Write the new project `README.md`**

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add README.md TRACK_README.md
git commit -m "docs: project README documenting the four CLI commands"
```

---

## Final integration check

After Task 25 is committed:

- [ ] **Step 1: Run the entire test suite**

```bash
pytest -v
```
Expected: every test PASSES; no warnings about deprecations from sklearn/pandas; runtime under 10 seconds.

- [ ] **Step 2: Run the canonical end-to-end pipeline on the real corpus**

```bash
infineon-baseline build-eval --variants-dir training_data/ --out outputs/eval/ --seed 42
infineon-baseline fit --train outputs/eval/train_split.csv --out outputs/models/ngram.pkl --order 3
infineon-baseline predict --model outputs/models/ngram.pkl \
    --eval-input outputs/eval/eval_input_valid.csv --task next-step \
    --out outputs/submissions/task1.csv
infineon-baseline score --predictions outputs/submissions/task1.csv \
    --ground-truth outputs/eval/ground_truth_valid.csv --task next-step \
    --report-json outputs/reports/task1.json
```
Expected: each command exits 0, prints a `✓` line, and produces the expected output file. The final `score` call prints a console table showing per-family Top-1/3/5 accuracy and MRR plus an aggregate row.

- [ ] **Step 3: Verify reproducibility**

Run the same pipeline a second time into a different output directory; diff the resulting `submission_task1.csv` against the first run. They must be byte-identical.

```bash
diff outputs/submissions/task1.csv outputs2/submissions/task1.csv && echo "✓ reproducible"
```

- [ ] **Step 4: Push to origin**

```bash
git push origin main
```

If all four steps above pass, the success-criteria checklist from the spec (§11) is satisfied and the baseline is ready for the eventual transformer comparison.
