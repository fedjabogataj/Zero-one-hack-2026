"""Task-level orchestration: turn eval inputs + a fitted model into submission rows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from infineon_baseline.anomaly import AnomalyResult, detect_hybrid, detect_oracle, detect_perplexity
from infineon_baseline.ngram import NGram
from infineon_baseline.tokenizer import Tokenizer

_FALLBACK_STEP = "SHIP LOT"


@dataclass
class Task1Row:
    example_id: str
    ranks: list[str]   # length 5


def run_task1(
    examples: Iterable[dict],
    tokenizer,   # Tokenizer (flat) or SubwordTokenizer
    ngram,       # NGram / SoftNGram / TransformerPredictor
) -> Iterator[Task1Row]:
    """One row per example with the top-5 next steps (always exactly 5)."""
    is_subword = hasattr(tokenizer, "sep_id")

    for ex in examples:
        family = ex["FAMILY"]
        partial_steps = ex["PARTIAL_SEQUENCE"].split("|")

        if is_subword:
            # Beam search via TransformerPredictor.top_k_steps
            top_steps = ngram.top_k_steps(family, partial_steps, k=5)
        else:
            # Flat tokenizer path (NGram / SoftNGram / TransformerPredictor flat)
            _, partial_ids = tokenizer.encode(family, partial_steps)
            prefix = tuple(partial_ids[-(ngram.order - 1):]) if ngram.order > 1 else ()
            top_ids = ngram.top_k(family, prefix, k=5)
            # Pad with most-common-overall steps to guarantee exactly 5 ranks.
            if len(top_ids) < 5:
                from collections import Counter
                global_counter: Counter = Counter()
                for fam_ctr in ngram.unigram.values():
                    global_counter.update(fam_ctr)
                for step_id, _ in global_counter.most_common():
                    if step_id not in top_ids:
                        top_ids.append(step_id)
                    if len(top_ids) >= 5:
                        break
            top_steps = tokenizer.decode(top_ids[:5])

        # Always pad to exactly 5 using a safe default step.
        while len(top_steps) < 5:
            top_steps.append(_FALLBACK_STEP)
        yield Task1Row(example_id=ex["EXAMPLE_ID"], ranks=top_steps[:5])


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
) -> Iterator[Task2Row]:
    """Greedy autoregressive completion until SHIP LOT or length cap."""
    is_subword = hasattr(tokenizer, "sep_id")

    if not is_subword:
        ship_id = tokenizer.step_to_id.get(_SHIP)

    for ex in examples:
        family = ex["FAMILY"]
        partial_steps = ex["PARTIAL_SEQUENCE"].split("|")

        if is_subword:
            # Greedy subword completion via TransformerPredictor.complete_steps
            max_steps = max(10, int(len(partial_steps) * max_length_multiplier))
            completed = ngram.complete_steps(family, partial_steps, max_steps=max_steps)
            yield Task2Row(
                example_id=ex["EXAMPLE_ID"],
                predicted_sequence="|".join(completed),
            )
            continue

        # Flat tokenizer path
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
    tokenizer,   # Tokenizer (flat) or SubwordTokenizer
    ngram,       # NGram / SoftNGram / TransformerPredictor
    threshold: float,
    strategy: str = "hybrid",   # "oracle" | "perplexity" | "hybrid"
) -> Iterator[Task3Row]:
    is_subword = hasattr(tokenizer, "sep_id")

    for ex in examples:
        family = ex["FAMILY"]
        steps = ex["SEQUENCE"].split("|")

        if is_subword:
            # For subword mode: use oracle rule detection (no perplexity strategy
            # available without threshold calibration for subword sequences).
            # Perplexity path uses log_prob_steps if strategy requests it.
            if strategy == "oracle":
                res = detect_oracle(steps)
            else:
                # Perplexity / hybrid: compute log-prob via teacher-forcing.
                lp = ngram.log_prob_steps(family, steps)
                # Normalise by sequence length (in subword tokens) to get per-token lp.
                n_tokens = max(1, sum(
                    len(tokenizer.encode_step(s)) + 1 for s in steps
                ))
                score = lp / n_tokens
                # Lower (more negative) score = more anomalous.
                is_valid_flag = int(score >= threshold)
                res = AnomalyResult(
                    is_valid=is_valid_flag,
                    score=float(score),
                    predicted_rule="PERPLEXITY" if not is_valid_flag else "NONE",
                )
        else:
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
