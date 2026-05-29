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
