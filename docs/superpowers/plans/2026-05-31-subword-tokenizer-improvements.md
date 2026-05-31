# Subword Tokenizer Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four concrete issues hurting subword-tokenizer transformer accuracy: hallucinated beam-search outputs, random embedding init, broken anomaly detection, and undertraining due to identical epoch budget as flat.

**Architecture:** Four independent improvements targeting `transformer_predictor.py`, `transformer_model.py`, `predictor.py`, and `scripts/train_transformer.sbatch` — each in a separate commit. No new dependencies are introduced; all changes are pure-Python additions or edits to existing logic.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pure-Python Levenshtein (no extra deps), bash SLURM sbatch

---

## File Map

| File | Change |
|---|---|
| `src/infineon_baseline/transformer_predictor.py` | Add `_levenshtein`, `_project_to_known_step`; wire into `top_k_steps_batch`; store `_known_steps`; update `load` and `__init__` |
| `src/infineon_baseline/train.py` | Compute `known_steps` list; pass to `_save_checkpoint`; add field to checkpoint dict |
| `src/infineon_baseline/transformer_model.py` | Replace random subword init with average-pooled ST init in `from_tokenizer_and_embedder` |
| `src/infineon_baseline/predictor.py` | Add strategy-aware branch (`hybrid` / `perplexity`) to subword batched and per-example `run_task3` paths |
| `scripts/train_transformer.sbatch` | `EPOCHS` default conditioned on `TOKENIZER`; `--time` bumped to `3:00:00` |

---

## Task 1: Constrained beam search (Improvement 1)

**Files:**
- Modify: `src/infineon_baseline/transformer_predictor.py`
- Modify: `src/infineon_baseline/train.py`

### Step 1.1: Add `_levenshtein` module-level helper to `transformer_predictor.py`

Open `src/infineon_baseline/transformer_predictor.py`. After the `_DTYPE_MAP` dict (line ~45), insert:

```python
def _levenshtein(a: str, b: str) -> int:
    """Standard DP Levenshtein distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]
```

- [ ] Insert the function after the `_DTYPE_MAP` block (around line 46).

### Step 1.2: Add `_project_to_known_step` as a method of `TransformerPredictor`

Add this method directly after `__init__`, before `load`:

```python
def _project_to_known_step(self, candidate: str, known_steps: set[str]) -> str:
    """Snap a beam-search-generated step string to its nearest known step
    by Levenshtein edit distance. Tie-break: shortest known step."""
    if not known_steps or candidate in known_steps:
        return candidate
    best_step: str | None = None
    best_d = float("inf")
    for k in known_steps:
        d = _levenshtein(candidate, k)
        if d < best_d or (d == best_d and best_step is not None and len(k) < len(best_step)):
            best_d, best_step = d, k
    return best_step or candidate
```

- [ ] Add the method to `TransformerPredictor` right after `__init__` ends (before `@classmethod load`).

### Step 1.3: Update `TransformerPredictor.__init__` to accept `known_steps`

At the end of `__init__`, add:

```python
        # Known step strings for constrained beam projection. Empty = no constraint.
        self._known_steps: set[str] = set()
```

- [ ] Add the `_known_steps` attribute initialised to empty set at end of `__init__`.

### Step 1.4: Update `TransformerPredictor.load` to read `known_steps` from checkpoint

After `unigram = {fam: Counter(ctr) for fam, ctr in ckpt["unigram"].items()}` and before `instance = cls(...)`, insert:

```python
        known_steps_raw = ckpt.get("known_steps", [])
```

Then after `instance._is_subword = (tokenizer_kind == "subword")`, add:

```python
        instance._known_steps = set(known_steps_raw)
```

- [ ] Add `known_steps_raw = ckpt.get("known_steps", [])` before `instance = cls(...)`.
- [ ] Add `instance._known_steps = set(known_steps_raw)` after the existing `instance._is_subword` line.

### Step 1.5: Wire constrained projection into `top_k_steps_batch`

In `top_k_steps_batch`, find the final decode loop (around line 470–483):

```python
        results: list[list[str]] = []
        for ex_idx in range(B):
            finished[ex_idx].sort(key=lambda x: -x[1])
            seen: set[str] = set()
            top_steps: list[str] = []
            for gen, _lp in finished[ex_idx]:
                s = tok.decode_step(gen)
                if s and s not in seen:
                    seen.add(s)
                    top_steps.append(s)
                if len(top_steps) >= k:
                    break
            results.append(top_steps)
        return results
```

Replace `s = tok.decode_step(gen)` with:

```python
                s = tok.decode_step(gen)
                if s and self._known_steps:
                    s = self._project_to_known_step(s, self._known_steps)
```

- [ ] Replace the decode+append block so that every decoded step string `s` is projected through `_project_to_known_step` when `_known_steps` is non-empty.

Full replacement — the loop body becomes:

```python
        results: list[list[str]] = []
        for ex_idx in range(B):
            finished[ex_idx].sort(key=lambda x: -x[1])
            seen: set[str] = set()
            top_steps: list[str] = []
            for gen, _lp in finished[ex_idx]:
                s = tok.decode_step(gen)
                if s:
                    if self._known_steps:
                        s = self._project_to_known_step(s, self._known_steps)
                    if s not in seen:
                        seen.add(s)
                        top_steps.append(s)
                if len(top_steps) >= k:
                    break
            results.append(top_steps)
        return results
```

### Step 1.6: Compute `known_steps` in `train.py` and pass to `_save_checkpoint`

In `train.py`'s `train()`, after `corpus = _load_train_split(Path(args.train))` (line ~326), add:

```python
    known_steps = sorted({
        step
        for fam_seqs in corpus.values()
        for steps in fam_seqs.values()
        for step in steps
    })
```

- [ ] Add the `known_steps` computation right after the `corpus = ...` line.

### Step 1.7: Pass `known_steps` through to `_save_checkpoint`

The `_save_checkpoint` function signature needs a new keyword arg. Add `known_steps: list[str] | None = None` to the signature:

```python
def _save_checkpoint(
    out_path: Path,
    *,
    model,
    tokenizer,
    embedder,
    unigram,
    arch_kwargs: dict,
    epoch_completed: int,
    global_step: int,
    optimizer,
    scheduler,
    best_val_loss: float,
    wandb_run_id: str | None,
    training_complete: bool,
    tokenizer_kind: str = "flat",
    known_steps: list[str] | None = None,
) -> None:
```

And inside `_save_checkpoint`, add `"known_steps": known_steps or [],` to the `checkpoint` dict (after e.g. `"unigram": ...`).

- [ ] Add `known_steps: list[str] | None = None` to `_save_checkpoint` signature.
- [ ] Add `"known_steps": known_steps or [],` to the `checkpoint` dict inside `_save_checkpoint`.

### Step 1.8: Pass `known_steps` when calling `_save_checkpoint` in the training loop

In `train.py`, find the two calls to `_save_checkpoint` (one in the training loop, one might be elsewhere). They look like:

```python
            _save_checkpoint(
                out_path,
                model=model, tokenizer=tokenizer, embedder=embedder, unigram=unigram,
                arch_kwargs=arch_kwargs,
                epoch_completed=epoch, global_step=global_step,
                optimizer=optimizer, scheduler=scheduler,
                best_val_loss=best_val_loss,
                wandb_run_id=wandb_run_id,
                training_complete=is_last,
                tokenizer_kind=tokenizer_kind,
            )
```

Add `known_steps=known_steps,` to this call.

- [ ] Add `known_steps=known_steps,` to every `_save_checkpoint(...)` call in `train.py`.

### Step 1.9: Smoke-check backward compatibility

Run:
```bash
cd "/Users/fedjabogataj/Projects/Hackathons/zero one/official_repo/zero_one_hack_01/tracks/industrial-infineon" && source .venv/bin/activate && python -c "
from infineon_baseline.transformer_predictor import TransformerPredictor
from pathlib import Path
p = TransformerPredictor.load(Path('outputs/models/transformer_5000_subword.pt'), device='cpu')
print('known_steps count:', len(p._known_steps))
print('Load OK — no crash expected')
"
```

- [ ] Run the smoke check. Expected output: `known_steps count: 0` (old checkpoint has no `known_steps` field, backward compat defaults to empty set). `Load OK — no crash expected`.

### Step 1.10: Commit

```bash
cd "/Users/fedjabogataj/Projects/Hackathons/zero one/official_repo/zero_one_hack_01/tracks/industrial-infineon" && git add src/infineon_baseline/transformer_predictor.py src/infineon_baseline/train.py && git commit -m "feat(predictor): constrain subword beam search to known step strings"
```

- [ ] Commit with message: `feat(predictor): constrain subword beam search to known step strings`

---

## Task 2: ST-init for subword embedding table (Improvement 2)

**Files:**
- Modify: `src/infineon_baseline/transformer_model.py`

### Step 2.1: Replace random subword init with average-pooled ST init

In `transformer_model.py`, find `from_tokenizer_and_embedder`. The current subword branch (starting at `# Else (subword): leave the embedding table as random init...`) is a comment with no code.

Replace the entire comment block at the end of `from_tokenizer_and_embedder`:

```python
        # Else (subword): leave the embedding table as random init — subword tokens
        # don't have meaningful ST vector counterparts.
```

with:

```python
        else:
            # Subword path: average-pool ST vectors for each subword token.
            # For each subword token, find all step strings whose split contains
            # that token and average their ST vectors. Specials get random init.
            all_steps = list(embedder.step_to_idx.keys())
            emb_dim = embedder.vectors.shape[1]
            vecs = np.zeros((vocab_size, emb_dim), dtype=np.float32)
            for tok_id, tok in enumerate(tokenizer.id_to_token):
                if tok in tokenizer.SPECIAL_TOKENS:
                    vecs[tok_id] = np.random.normal(0, 0.02, size=emb_dim).astype(np.float32)
                    continue
                matching = []
                for step in all_steps:
                    if tok in tokenizer.split_step(step):
                        matching.append(embedder.vectors[embedder.step_to_idx[step]])
                if matching:
                    vecs[tok_id] = np.mean(matching, axis=0)
                else:
                    vecs[tok_id] = np.random.normal(0, 0.02, size=emb_dim).astype(np.float32)
            vecs_t = torch.from_numpy(vecs)
            if vecs_t.shape[1] != d_model:
                proj = nn.Linear(vecs_t.shape[1], d_model, bias=False)
                with torch.no_grad():
                    vecs_t = proj(vecs_t)
            with torch.no_grad():
                model.token_emb.weight[:vocab_size] = vecs_t
```

- [ ] Replace the trailing comment with the full average-pooled ST init block above.

### Step 2.2: Smoke-check non-trivial init

Run:
```bash
cd "/Users/fedjabogataj/Projects/Hackathons/zero one/official_repo/zero_one_hack_01/tracks/industrial-infineon" && source .venv/bin/activate && python -c "
import pickle, numpy as np, torch
from infineon_baseline.subword_tokenizer import SubwordTokenizer
from infineon_baseline.embeddings_st import STStepEmbedder
from infineon_baseline.transformer_model import TransformerLM

# Minimal corpus for the tokenizer
corpus = {'FAM': {'s1': ['DEPOSIT METAL', 'ALIGN MASK']}}
tok = SubwordTokenizer.fit(corpus)
emb = STStepEmbedder.load('outputs/models/st_step_embeddings.pkl')
model = TransformerLM.from_tokenizer_and_embedder(tok, emb)
w = model.token_emb.weight.data.cpu()
specials = list(range(len(tok.SPECIAL_TOKENS)))
non_special_rows = [i for i in range(tok.vocab_size) if i not in specials]
if non_special_rows:
    std = w[non_special_rows].std().item()
    print(f'Non-special token emb std={std:.4f}  (should be > 0.01 if ST-init worked)')
else:
    print('No non-special rows — tiny corpus')
print('Smoke check OK')
"
```

- [ ] Run the smoke check. Expected: `std` should be significantly non-zero (typically >0.1 for ST vectors) — confirms init is not random.

### Step 2.3: Commit

```bash
cd "/Users/fedjabogataj/Projects/Hackathons/zero one/official_repo/zero_one_hack_01/tracks/industrial-infineon" && git add src/infineon_baseline/transformer_model.py && git commit -m "feat(transformer_model): ST-init for subword embedding table"
```

- [ ] Commit with message: `feat(transformer_model): ST-init for subword embedding table`

---

## Task 3: Subword-hybrid anomaly strategy (Improvement 3)

**Files:**
- Modify: `src/infineon_baseline/predictor.py`

### Step 3.1: Add strategy-aware logic to the subword+batched `run_task3` path

In `predictor.py`, find the `if is_subword and is_batched:` block inside `run_task3` (around lines 282–305). Currently it always uses perplexity regardless of `strategy`. Replace the entire block:

```python
    # ── subword + batched ─────────────────────────────────────────────── #
    if is_subword and is_batched:
        for batch in _iter_batches(examples, batch_size):
            families = [ex["FAMILY"] for ex in batch]
            steps_list = [ex["SEQUENCE"].split("|") for ex in batch]
            lps = ngram.log_prob_steps_batch(families, steps_list)
            for ex, lp, steps in zip(batch, lps, steps_list):
                # Per-token average log-prob (subword score-space).
                n_tokens = max(1, sum(
                    len(tokenizer.encode_step(s)) + 1 for s in steps
                ))
                per_tok_lp = lp / n_tokens
                # Decision uses raw per-token lp vs threshold (same units).
                is_valid_flag = int(per_tok_lp >= threshold)
                # SCORE column in the submission must be in [0,1] — sigmoid
                # the gap between observed per-token lp and the threshold.
                # Higher = more likely valid; ROC-AUC reads this column.
                score_norm = _sigmoid(per_tok_lp - threshold)
                yield Task3Row(
                    example_id=ex["EXAMPLE_ID"],
                    is_valid=is_valid_flag,
                    score=float(score_norm),
                    predicted_rule="PERPLEXITY" if not is_valid_flag else "",
                )
        return
```

With:

```python
    # ── subword + batched ─────────────────────────────────────────────── #
    if is_subword and is_batched:
        for batch in _iter_batches(examples, batch_size):
            families = [ex["FAMILY"] for ex in batch]
            steps_list = [ex["SEQUENCE"].split("|") for ex in batch]
            lps = ngram.log_prob_steps_batch(families, steps_list)
            for ex, lp, steps in zip(batch, lps, steps_list):
                # Per-token average log-prob (subword score-space).
                n_tokens = max(1, sum(
                    len(tokenizer.encode_step(s)) + 1 for s in steps
                ))
                per_tok_lp = lp / n_tokens
                # SCORE column in [0,1] — sigmoid of gap vs threshold.
                score_norm = _sigmoid(per_tok_lp - threshold)
                if strategy == "perplexity":
                    is_valid_flag = int(per_tok_lp >= threshold)
                    yield Task3Row(
                        example_id=ex["EXAMPLE_ID"],
                        is_valid=is_valid_flag,
                        score=float(score_norm),
                        predicted_rule="PERPLEXITY" if not is_valid_flag else "",
                    )
                else:  # "hybrid": oracle decides validity, perplexity gives score
                    oracle = detect_oracle(ex["SEQUENCE"].split("|"))
                    yield Task3Row(
                        example_id=ex["EXAMPLE_ID"],
                        is_valid=oracle.is_valid,
                        score=float(score_norm),
                        predicted_rule=oracle.predicted_rule,
                    )
        return
```

- [ ] Replace the `if is_subword and is_batched:` block in `run_task3` with the strategy-aware version above.

### Step 3.2: Add strategy-aware logic to the per-example subword fallback

In the `# ── per-example fallback ─────────────────────────────────────────── #` section of `run_task3`, find the subword branch:

```python
        if is_subword:
            lp = ngram.log_prob_steps(family, steps)
            n_tokens = max(1, sum(
                len(tokenizer.encode_step(s)) + 1 for s in steps
            ))
            per_tok_lp = lp / n_tokens
            is_valid_flag = int(per_tok_lp >= threshold)
            # Same sigmoid normalisation as the batched subword path so
            # SCORE lands in [0,1] for the submission writer + ROC-AUC.
            score_norm = _sigmoid(per_tok_lp - threshold)
            res = AnomalyResult(
                is_valid=is_valid_flag,
                score=float(score_norm),
                predicted_rule="PERPLEXITY" if not is_valid_flag else "",
            )
```

Replace with:

```python
        if is_subword:
            lp = ngram.log_prob_steps(family, steps)
            n_tokens = max(1, sum(
                len(tokenizer.encode_step(s)) + 1 for s in steps
            ))
            per_tok_lp = lp / n_tokens
            score_norm = _sigmoid(per_tok_lp - threshold)
            if strategy == "perplexity":
                is_valid_flag = int(per_tok_lp >= threshold)
                res = AnomalyResult(
                    is_valid=is_valid_flag,
                    score=float(score_norm),
                    predicted_rule="PERPLEXITY" if not is_valid_flag else "",
                )
            else:  # "hybrid"
                oracle = detect_oracle(steps)
                res = AnomalyResult(
                    is_valid=oracle.is_valid,
                    score=float(score_norm),
                    predicted_rule=oracle.predicted_rule,
                )
```

- [ ] Replace the per-example `if is_subword:` block in `run_task3` with the strategy-aware version.

### Step 3.3: Smoke-check — confirm hybrid strategy works on existing subword model

Run a quick inference on the existing checkpoint. This requires an eval set. If canonical test files exist:

```bash
cd "/Users/fedjabogataj/Projects/Hackathons/zero one/official_repo/zero_one_hack_01/tracks/industrial-infineon" && source .venv/bin/activate && python -c "
from infineon_baseline.transformer_predictor import TransformerPredictor
from infineon_baseline.predictor import run_task3
from pathlib import Path
import csv

model_path = Path('outputs/models/transformer_5000_subword.pt')
pred = TransformerPredictor.load(model_path, device='cpu')
tok = pred._tokenizer

eval_file = Path('outputs/eval_canonical/eval_input_anomaly.csv')
if not eval_file.exists():
    print('Eval file not found — skip smoke check')
else:
    examples = []
    with eval_file.open() as f:
        for row in csv.DictReader(f):
            examples.append(dict(row))
            if len(examples) >= 20:
                break
    rows = list(run_task3(examples, tok, pred, threshold=-0.5, strategy='hybrid'))
    valid_count = sum(r.is_valid for r in rows)
    print(f'hybrid strategy: {len(rows)} rows, {valid_count} classified valid')
    print('Smoke check OK — no crash')
"
```

- [ ] Run the smoke check. Expected: no crash, valid/invalid rows returned. If F1 was 0 with perplexity (all same class), hybrid should produce mixed valid/invalid.

### Step 3.4: Commit

```bash
cd "/Users/fedjabogataj/Projects/Hackathons/zero one/official_repo/zero_one_hack_01/tracks/industrial-infineon" && git add src/infineon_baseline/predictor.py && git commit -m "feat(predictor): subword-hybrid anomaly strategy (oracle + perplexity score)"
```

- [ ] Commit with message: `feat(predictor): subword-hybrid anomaly strategy (oracle + perplexity score)`

---

## Task 4: Longer training default for subword (Improvement 4)

**Files:**
- Modify: `scripts/train_transformer.sbatch`

### Step 4.1: Change `--time` from `2:00:00` to `3:00:00`

In `scripts/train_transformer.sbatch`, find:

```bash
#SBATCH --time=2:00:00
```

Replace with:

```bash
#SBATCH --time=3:00:00
```

- [ ] Change `--time=2:00:00` to `--time=3:00:00` in the sbatch header.

### Step 4.2: Replace single `EPOCHS` default with tokenizer-conditioned default

In `scripts/train_transformer.sbatch`, find:

```bash
DATA_SIZE="${DATA_SIZE:-5000}"
EPOCHS="${EPOCHS:-30}"
```

Replace with:

```bash
DATA_SIZE="${DATA_SIZE:-5000}"
TOKENIZER="${TOKENIZER:-flat}"   # "flat" = one id per step; "subword" = word-level split + <sep>
# Subword sequences are ~3x longer per step -> use more epochs by default.
# User-supplied EPOCHS overrides this default.
if [[ "$TOKENIZER" == "subword" ]]; then
    EPOCHS="${EPOCHS:-60}"
else
    EPOCHS="${EPOCHS:-30}"
fi
```

Note: The existing `TOKENIZER="${TOKENIZER:-flat}"` line appears later in the file (around line 66). After this change, remove the duplicate `TOKENIZER` line.

- [ ] Replace `EPOCHS="${EPOCHS:-30}"` with the tokenizer-conditioned block.
- [ ] Remove the duplicate `TOKENIZER="${TOKENIZER:-flat}"` line that previously appeared ~line 66 (now it's set earlier in the block above).

### Step 4.3: Verify the sbatch change is syntactically valid

```bash
bash -n "/Users/fedjabogataj/Projects/Hackathons/zero one/official_repo/zero_one_hack_01/tracks/industrial-infineon/scripts/train_transformer.sbatch" && echo "Syntax OK"
```

- [ ] Run bash syntax check. Expected: `Syntax OK` (exit 0).

### Step 4.4: Commit

```bash
cd "/Users/fedjabogataj/Projects/Hackathons/zero one/official_repo/zero_one_hack_01/tracks/industrial-infineon" && git add scripts/train_transformer.sbatch && git commit -m "feat(sbatch): default EPOCHS=60 for subword (was 30 — too short)"
```

- [ ] Commit with message: `feat(sbatch): default EPOCHS=60 for subword (was 30 — too short)`

---

## Task 5: Push branch

### Step 5.1: Push the branch

```bash
cd "/Users/fedjabogataj/Projects/Hackathons/zero one/official_repo/zero_one_hack_01/tracks/industrial-infineon" && git push 2>&1 | tail -3
```

- [ ] Push and confirm remote accepted the 4 new commits.

---

## Self-Review Checklist

- [x] Improvement 1 (constrained beam): `_levenshtein`, `_project_to_known_step`, `_known_steps` field, `load` reads checkpoint key with default `[]`, wired into `top_k_steps_batch` decode loop. `train.py` computes `known_steps` and passes through `_save_checkpoint`.
- [x] Improvement 2 (ST-init): `from_tokenizer_and_embedder` subword branch replaces comment with actual averaging code. Uses `embedder.step_to_idx.keys()` for step list. Projects to `d_model` if dim differs.
- [x] Improvement 3 (hybrid anomaly): Both `is_subword and is_batched` and per-example fallback paths gain `strategy` branching. `hybrid` uses `detect_oracle` for validity + perplexity for score.
- [x] Improvement 4 (epochs): `--time` bumped, `EPOCHS` default conditioned on `TOKENIZER`. Duplicate `TOKENIZER` line removal noted.
- [x] Backward compat: old checkpoints missing `known_steps` default to `[]` → `_known_steps = set()` → projection no-ops.
- [x] No new dependencies added anywhere.
