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
