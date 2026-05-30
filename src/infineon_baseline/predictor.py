"""Task-level orchestration: turn eval inputs + a fitted model into submission rows.

Three dispatch paths, picked by tokenizer kind and model capabilities:

1. **Flat tokenizer + batched transformer** — pack many prefixes per GPU
   forward via top_k_batch / log_prob_batch / generate_batch.
2. **Subword tokenizer + batched transformer** — same idea, but the model's
   step-level helpers (top_k_steps_batch / complete_steps_batch /
   log_prob_steps_batch) hide the subword decoding from us.
3. **N-gram fallback** — pure Python dict lookups, per-example loop.

The dispatch is duck-typed: tokenizer.sep_id => subword,
model.top_k_batch => batched.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator

from infineon_baseline.anomaly import (
    AnomalyResult, _sigmoid, detect_hybrid, detect_oracle, detect_perplexity,
)
from infineon_baseline.ngram import NGram

_FALLBACK_STEP = "SHIP LOT"


def _is_subword(tokenizer) -> bool:
    return hasattr(tokenizer, "sep_id")


def _is_batched(model) -> bool:
    return hasattr(model, "top_k_batch")


def _iter_batches(items: list, batch_size: int) -> Iterator[list]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


@dataclass
class Task1Row:
    example_id: str
    ranks: list[str]   # length 5


def run_task1(
    examples: Iterable[dict],
    tokenizer,   # Tokenizer (flat) or SubwordTokenizer
    ngram,       # NGram / SoftNGram / TransformerPredictor
    batch_size: int = 128,
) -> Iterator[Task1Row]:
    """One row per example with the top-5 next steps (always exactly 5)."""
    examples = list(examples)
    is_subword = _is_subword(tokenizer)
    is_batched = _is_batched(ngram)

    def _pad_with_fallback(top_steps: list[str]) -> list[str]:
        while len(top_steps) < 5:
            top_steps.append(_FALLBACK_STEP)
        return top_steps[:5]

    # ── subword + batched ─────────────────────────────────────────────── #
    if is_subword and is_batched:
        for batch in _iter_batches(examples, batch_size):
            families = [ex["FAMILY"] for ex in batch]
            partials = [ex["PARTIAL_SEQUENCE"].split("|") for ex in batch]
            top_steps_batch = ngram.top_k_steps_batch(families, partials, k=5)
            for ex, top_steps in zip(batch, top_steps_batch):
                yield Task1Row(
                    example_id=ex["EXAMPLE_ID"],
                    ranks=_pad_with_fallback(list(top_steps)),
                )
        return

    # ── flat + batched ────────────────────────────────────────────────── #
    if is_batched:
        # Global fallback ranking, used to pad rows that come back with < 5.
        global_counter: Counter = Counter()
        for fam_ctr in ngram.unigram.values():
            global_counter.update(fam_ctr)
        fallback_ids = [sid for sid, _ in global_counter.most_common()]

        def _pad_ids(top_ids: list[int]) -> list[int]:
            if len(top_ids) >= 5:
                return top_ids[:5]
            seen = set(top_ids)
            for sid in fallback_ids:
                if sid not in seen:
                    top_ids.append(sid)
                    seen.add(sid)
                if len(top_ids) >= 5:
                    break
            return top_ids[:5]

        for batch in _iter_batches(examples, batch_size):
            families = [ex["FAMILY"] for ex in batch]
            prefixes = []
            for ex in batch:
                partial_steps = ex["PARTIAL_SEQUENCE"].split("|")
                _, partial_ids = tokenizer.encode(ex["FAMILY"], partial_steps)
                prefixes.append(partial_ids)
            top_ids_batch = ngram.top_k_batch(families, prefixes, k=5)
            for ex, top_ids in zip(batch, top_ids_batch):
                top_steps = tokenizer.decode(_pad_ids(list(top_ids)))
                yield Task1Row(
                    example_id=ex["EXAMPLE_ID"],
                    ranks=_pad_with_fallback(list(top_steps)),
                )
        return

    # ── per-example fallback (n-gram, or subword without batched API) ── #
    for ex in examples:
        family = ex["FAMILY"]
        partial_steps = ex["PARTIAL_SEQUENCE"].split("|")
        if is_subword:
            top_steps = ngram.top_k_steps(family, partial_steps, k=5)
        else:
            _, partial_ids = tokenizer.encode(family, partial_steps)
            prefix = tuple(partial_ids[-(ngram.order - 1):]) if ngram.order > 1 else ()
            top_ids = ngram.top_k(family, prefix, k=5)
            if len(top_ids) < 5:
                global_counter = Counter()
                for fam_ctr in ngram.unigram.values():
                    global_counter.update(fam_ctr)
                for step_id, _ in global_counter.most_common():
                    if step_id not in top_ids:
                        top_ids.append(step_id)
                    if len(top_ids) >= 5:
                        break
            top_steps = tokenizer.decode(top_ids[:5])
        yield Task1Row(
            example_id=ex["EXAMPLE_ID"],
            ranks=_pad_with_fallback(list(top_steps)),
        )


@dataclass
class Task2Row:
    example_id: str
    predicted_sequence: str   # pipe-separated, steps AFTER the cut only


_SHIP = "SHIP LOT"


def run_task2(
    examples: Iterable[dict],
    tokenizer,   # Tokenizer (flat) or SubwordTokenizer
    ngram,       # NGram / SoftNGram / TransformerPredictor
    max_length_multiplier: float = 1.5,
    constrain: bool = False,
    batch_size: int = 128,
) -> Iterator[Task2Row]:
    """Greedy autoregressive completion until SHIP LOT (flat) / EOS (subword)
    or length cap."""
    examples = list(examples)
    is_subword = _is_subword(tokenizer)
    is_batched = _is_batched(ngram)

    # ── subword + batched ─────────────────────────────────────────────── #
    # Subword path has no notion of `constrain` (rule validator works on full
    # step strings, not subword tokens), so we always batch when possible.
    if is_subword and is_batched:
        for batch in _iter_batches(examples, batch_size):
            families = [ex["FAMILY"] for ex in batch]
            partials = [ex["PARTIAL_SEQUENCE"].split("|") for ex in batch]
            max_steps = [
                max(10, int(len(p) * max_length_multiplier)) for p in partials
            ]
            completed_batch = ngram.complete_steps_batch(families, partials, max_steps)
            for ex, completed in zip(batch, completed_batch):
                yield Task2Row(
                    example_id=ex["EXAMPLE_ID"],
                    predicted_sequence="|".join(completed),
                )
        return

    # ── flat + batched (constrain=False) ──────────────────────────────── #
    # Constrained decoding validates each candidate per step via the symbolic
    # rule checker — per-example sequential, not worth batching.
    if is_batched and not constrain and not is_subword:
        ship_id = tokenizer.step_to_id.get(_SHIP)
        for batch in _iter_batches(examples, batch_size):
            families = [ex["FAMILY"] for ex in batch]
            prefixes = []
            max_steps_list = []
            for ex in batch:
                partial_steps = ex["PARTIAL_SEQUENCE"].split("|")
                _, partial_ids = tokenizer.encode(ex["FAMILY"], partial_steps)
                prefixes.append(partial_ids)
                max_steps_list.append(int(len(partial_steps) * max_length_multiplier))
            out_ids_batch = ngram.generate_batch(
                families, prefixes, max_steps_list, stop_id=ship_id,
            )
            for ex, out_ids in zip(batch, out_ids_batch):
                yield Task2Row(
                    example_id=ex["EXAMPLE_ID"],
                    predicted_sequence="|".join(tokenizer.decode(out_ids)),
                )
        return

    # ── per-example fallback ─────────────────────────────────────────── #
    if not is_subword:
        ship_id = tokenizer.step_to_id.get(_SHIP)

    for ex in examples:
        family = ex["FAMILY"]
        partial_steps = ex["PARTIAL_SEQUENCE"].split("|")

        if is_subword:
            max_steps = max(10, int(len(partial_steps) * max_length_multiplier))
            completed = ngram.complete_steps(family, partial_steps, max_steps=max_steps)
            yield Task2Row(
                example_id=ex["EXAMPLE_ID"],
                predicted_sequence="|".join(completed),
            )
            continue

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
                from generate_sequences import validate_sequence
                ok = []
                for cid in cands:
                    trial = tokenizer.decode(prefix_ids + [cid])
                    if not validate_sequence(trial):
                        ok.append(cid)
                cands = ok or cands
            next_id = cands[0]
            out_ids.append(next_id)
            prefix_ids.append(next_id)
            if ship_id is not None and next_id == ship_id:
                break
        yield Task2Row(
            example_id=ex["EXAMPLE_ID"],
            predicted_sequence="|".join(tokenizer.decode(out_ids)),
        )


@dataclass
class Task3Row:
    example_id: str
    is_valid: int
    score: float
    predicted_rule: str


def run_task3(
    examples: Iterable[dict],
    tokenizer,   # Tokenizer (flat) or SubwordTokenizer
    ngram,       # NGram / SoftNGram / TransformerPredictor
    threshold: float,
    strategy: str = "hybrid",   # "oracle" | "perplexity" | "hybrid"
    batch_size: int = 128,
) -> Iterator[Task3Row]:
    examples = list(examples)
    is_subword = _is_subword(tokenizer)
    is_batched = _is_batched(ngram)

    # ── oracle: no model call, no batching needed ─────────────────────── #
    if strategy == "oracle":
        for ex in examples:
            res = detect_oracle(ex["SEQUENCE"].split("|"))
            yield Task3Row(
                example_id=ex["EXAMPLE_ID"],
                is_valid=res.is_valid,
                score=res.score,
                predicted_rule=res.predicted_rule,
            )
        return

    # ── subword + batched ─────────────────────────────────────────────── #
    if is_subword and is_batched:
        for batch in _iter_batches(examples, batch_size):
            families = [ex["FAMILY"] for ex in batch]
            steps_list = [ex["SEQUENCE"].split("|") for ex in batch]
            lps = ngram.log_prob_steps_batch(families, steps_list)
            for ex, lp, steps in zip(batch, lps, steps_list):
                # Normalise log-prob by token count (matches per-example subword path).
                n_tokens = max(1, sum(
                    len(tokenizer.encode_step(s)) + 1 for s in steps
                ))
                score = lp / n_tokens
                is_valid_flag = int(score >= threshold)
                yield Task3Row(
                    example_id=ex["EXAMPLE_ID"],
                    is_valid=is_valid_flag,
                    score=float(score),
                    predicted_rule="PERPLEXITY" if not is_valid_flag else "NONE",
                )
        return

    # ── flat + batched ───────────────────────────────────────────────── #
    if is_batched and strategy in ("perplexity", "hybrid"):
        for batch in _iter_batches(examples, batch_size):
            families = [ex["FAMILY"] for ex in batch]
            ids_list = []
            for ex in batch:
                steps = ex["SEQUENCE"].split("|")
                _, ids = tokenizer.encode(ex["FAMILY"], steps)
                ids_list.append(ids)
            lps = ngram.log_prob_batch(families, ids_list)
            for ex, lp in zip(batch, lps):
                score = _sigmoid(lp - threshold)
                if strategy == "perplexity":
                    is_valid = 1 if lp >= threshold else 0
                    yield Task3Row(
                        example_id=ex["EXAMPLE_ID"],
                        is_valid=is_valid,
                        score=float(score),
                        predicted_rule="",
                    )
                else:  # hybrid: oracle decides validity, model gives the score
                    oracle = detect_oracle(ex["SEQUENCE"].split("|"))
                    yield Task3Row(
                        example_id=ex["EXAMPLE_ID"],
                        is_valid=oracle.is_valid,
                        score=float(score),
                        predicted_rule=oracle.predicted_rule,
                    )
        return

    # ── per-example fallback ─────────────────────────────────────────── #
    for ex in examples:
        family = ex["FAMILY"]
        steps = ex["SEQUENCE"].split("|")

        if is_subword:
            lp = ngram.log_prob_steps(family, steps)
            n_tokens = max(1, sum(
                len(tokenizer.encode_step(s)) + 1 for s in steps
            ))
            score = lp / n_tokens
            is_valid_flag = int(score >= threshold)
            res = AnomalyResult(
                is_valid=is_valid_flag,
                score=float(score),
                predicted_rule="PERPLEXITY" if not is_valid_flag else "NONE",
            )
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
