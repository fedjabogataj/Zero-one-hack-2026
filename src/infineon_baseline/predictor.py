"""Task-level orchestration: turn eval inputs + a fitted model into submission rows.

For NGram/SoftNGram models, each example is processed one at a time (cheap —
pure dict lookups). For TransformerPredictor, we dispatch through batched
fast paths (top_k_batch, log_prob_batch, generate_batch) that pack many
examples into one GPU forward pass — typically 10–50× speedup on A100.
The dispatch is duck-typed: any model exposing `top_k_batch` opts into the
batched path.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator

from infineon_baseline.anomaly import (
    AnomalyResult, _sigmoid, detect_hybrid, detect_oracle, detect_perplexity,
)
from infineon_baseline.ngram import NGram
from infineon_baseline.tokenizer import Tokenizer


def _is_batched(model) -> bool:
    """True if the model exposes a batched fast path."""
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
    tokenizer: Tokenizer,
    ngram: NGram,
    batch_size: int = 128,
) -> Iterator[Task1Row]:
    """One row per example with the top-5 next steps (always exactly 5)."""
    examples = list(examples)

    # Build a global fallback ranking once — used to pad rows that come back
    # with fewer than 5 candidates (e.g. unseen prefixes on n-gram).
    global_counter: Counter = Counter()
    for fam_ctr in ngram.unigram.values():
        global_counter.update(fam_ctr)
    fallback_ids = [sid for sid, _ in global_counter.most_common()]

    def _pad_to_5(top_ids: list[int]) -> list[int]:
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

    if _is_batched(ngram):
        for batch in _iter_batches(examples, batch_size):
            families = [ex["FAMILY"] for ex in batch]
            prefixes = []
            for ex in batch:
                partial_steps = ex["PARTIAL_SEQUENCE"].split("|")
                _, partial_ids = tokenizer.encode(ex["FAMILY"], partial_steps)
                # TransformerPredictor.order == max_seq_len, so the per-example
                # path would slice [-(order-1):] — i.e. the whole prefix for any
                # realistic sequence. Pass the full prefix; top_k_batch handles
                # the model-cap truncation internally.
                prefixes.append(partial_ids)
            top_ids_batch = ngram.top_k_batch(families, prefixes, k=5)
            for ex, top_ids in zip(batch, top_ids_batch):
                ranks = tokenizer.decode(_pad_to_5(list(top_ids)))
                yield Task1Row(example_id=ex["EXAMPLE_ID"], ranks=ranks)
        return

    # Per-example n-gram path.
    for ex in examples:
        family = ex["FAMILY"]
        partial_steps = ex["PARTIAL_SEQUENCE"].split("|")
        _, partial_ids = tokenizer.encode(family, partial_steps)
        prefix = tuple(partial_ids[-(ngram.order - 1):]) if ngram.order > 1 else ()
        top_ids = list(ngram.top_k(family, prefix, k=5))
        ranks = tokenizer.decode(_pad_to_5(top_ids))
        yield Task1Row(example_id=ex["EXAMPLE_ID"], ranks=ranks)


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
    batch_size: int = 128,
) -> Iterator[Task2Row]:
    """Greedy autoregressive completion until SHIP LOT or length cap."""
    ship_id = tokenizer.step_to_id.get(_SHIP)
    examples = list(examples)

    # Batched path: only when the model supports it AND constrain is off.
    # Constrained decoding validates each candidate per step via the symbolic
    # rule checker — that's per-example sequential and not worth batching.
    if _is_batched(ngram) and not constrain:
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

    # Per-example path (n-gram, or transformer with --constrain).
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
                from generate_sequences import validate_sequence
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
    batch_size: int = 128,
) -> Iterator[Task3Row]:
    examples = list(examples)

    # Oracle strategy never touches the model — no batching to do.
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

    if _is_batched(ngram) and strategy in ("perplexity", "hybrid"):
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

    # Per-example fallback (n-gram or anything without batched API).
    for ex in examples:
        family = ex["FAMILY"]
        steps = ex["SEQUENCE"].split("|")
        if strategy == "perplexity":
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
