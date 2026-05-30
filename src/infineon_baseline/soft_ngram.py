"""Embedding-aware wrapper around `NGram` for OOD-friendly prediction.

Composes a fitted `NGram` with a fitted `StepEmbedder` + `Tokenizer` to support
two new behaviors on top of the plain n-gram:

1. **OOD step aliasing.** When the input sequence contains a step name the
   tokenizer doesn't know (e.g. from the hidden 4th product family), we look
   up its embedding and replace it with the nearest known step before feeding
   it to the n-gram. This means the model can produce predictions for
   completely new step strings instead of falling back to unigram.

2. **Soft prefix lookup.** When the exact prefix has no count in the n-gram
   (typical OOD case even after aliasing, or unseen combinations within the
   3 known families), we aggregate the next-step distributions of the K most
   embedding-similar SEEN prefixes, weighted by cosine similarity.

For *in-distribution* inputs whose exact prefix is in the n-gram, behavior is
identical to the plain `NGram.top_k` — no smearing, no slowdown.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from infineon_baseline.embeddings import StepEmbedder
from infineon_baseline.ngram import NGram
from infineon_baseline.tokenizer import Tokenizer


@dataclass
class SoftNGram:
    ngram: NGram
    embedder: StepEmbedder
    tokenizer: Tokenizer
    n_neighbors: int = 8         # how many similar prefixes to aggregate on soft fallback
    sim_floor: float = 0.05      # ignore neighbors below this cosine similarity

    # Cached per family: aligned arrays of (prefix_tuple, prefix_vector, counter)
    _family_prefix_list: dict[str, list[tuple[int, ...]]] = field(default_factory=dict)
    _family_prefix_matrix: dict[str, np.ndarray] = field(default_factory=dict)
    _family_prefix_norms: dict[str, np.ndarray] = field(default_factory=dict)
    _step_vectors: np.ndarray | None = None  # shape (vocab, D) — id-indexed

    def __post_init__(self) -> None:
        self._build_prefix_index()
        self._build_step_vectors()

    # Attribute forwarders so SoftNGram is a drop-in replacement for NGram in
    # `predictor.run_task*` (those functions read .order and .unigram directly).
    @property
    def order(self) -> int:
        return self.ngram.order

    @property
    def unigram(self):
        return self.ngram.unigram

    # --------------------------------------------------------------- #
    # One-time precomputation                                         #
    # --------------------------------------------------------------- #
    def _build_step_vectors(self) -> None:
        """Cache one embedding per token id (rows aligned with tokenizer.id_to_step)."""
        vecs = np.stack([self.embedder.encode(s) for s in self.tokenizer.id_to_step])
        self._step_vectors = vecs

    def _build_prefix_index(self) -> None:
        """For each family, build a list of seen (order-1)-length prefixes + their
        mean-embedding matrix. Used by the soft fallback to find similar prefixes."""
        ord_minus_1 = self.ngram.order - 1
        for family, fam_counts in self.ngram.counts.items():
            prefixes = [p for p in fam_counts if len(p) == ord_minus_1]
            if not prefixes:
                self._family_prefix_list[family] = []
                self._family_prefix_matrix[family] = np.zeros((0,))
                self._family_prefix_norms[family] = np.zeros((0,))
                continue
            # Mean step-embedding per prefix.
            vecs = np.stack([
                np.mean([self.embedder.encode(self.tokenizer.id_to_step[i]) for i in p], axis=0)
                for p in prefixes
            ])
            norms = np.linalg.norm(vecs, axis=1)
            norms = np.where(norms == 0, 1.0, norms)
            self._family_prefix_list[family] = prefixes
            self._family_prefix_matrix[family] = vecs
            self._family_prefix_norms[family] = norms

    # --------------------------------------------------------------- #
    # OOD step aliasing                                                #
    # --------------------------------------------------------------- #
    def alias_step_name(self, step: str) -> int:
        """Map a (possibly unknown) step name to the nearest known token id.

        - Known step: returns its tokenizer id directly.
        - Unknown step: encodes the name string, finds the nearest known step
          in embedding space, returns that step's tokenizer id.
        """
        if step in self.tokenizer.step_to_id:
            return self.tokenizer.step_to_id[step]
        nearest = self.embedder.nearest(step, k=1, exclude_self=False)
        if not nearest:
            return -1
        return self.tokenizer.step_to_id.get(nearest[0][0], -1)

    def encode_with_aliasing(self, family: str, steps: list[str]) -> tuple[int, list[int]]:
        """Tokenize a sequence, replacing unknown step names with nearest known."""
        if family not in self.tokenizer.family_to_id:
            raise KeyError(f"unknown family {family!r}")
        family_id = self.tokenizer.family_to_id[family]
        ids: list[int] = []
        for s in steps:
            aid = self.alias_step_name(s)
            if aid < 0:
                # Truly unencodable — skip; the n-gram will see a shorter prefix.
                continue
            ids.append(aid)
        return family_id, ids

    # --------------------------------------------------------------- #
    # Prediction                                                       #
    # --------------------------------------------------------------- #
    def top_k(self, family: str, prefix: tuple[int, ...], k: int = 5) -> list[int]:
        """Return the top-k next-step ids.

        Algorithm:
          1. If the exact prefix has a counter, use it (identical to NGram).
          2. Otherwise, try the soft fallback: aggregate counters of the N most
             embedding-similar SEEN prefixes, weighted by cosine similarity.
          3. If no similar prefix passes `sim_floor`, fall back to NGram's
             standard stupid backoff path.
        """
        if family not in self.ngram.counts:
            return []
        order_minus_1 = self.ngram.order - 1
        short = tuple(prefix[-order_minus_1:]) if order_minus_1 > 0 else ()

        # 1. Exact match path — preserves in-distribution accuracy and speed.
        exact = self.ngram.counts[family].get(short)
        if exact:
            return [sid for sid, _ in exact.most_common(k)]

        # 2. Soft fallback — aggregate similar prefixes.
        soft = self._soft_aggregate(family, short, top_n=self.n_neighbors)
        if soft:
            return [sid for sid, _ in soft.most_common(k)]

        # 3. Stupid-backoff fallback (unigram in the limit).
        return self.ngram.top_k(family, prefix, k)

    def log_prob(self, family: str, ids: list[int]) -> float:
        """Sum of log P(id_i | prefix_i), using soft prob for unseen prefixes."""
        import math

        if family not in self.ngram.unigram:
            return float("-inf")
        family_total = sum(self.ngram.unigram[family].values())
        if family_total == 0:
            return float("-inf")

        order_minus_1 = self.ngram.order - 1
        total_lp = 0.0
        for i, next_id in enumerate(ids):
            prefix = tuple(ids[max(0, i - order_minus_1):i])
            prob = self._cond_prob(family, prefix, next_id, family_total)
            total_lp += math.log(max(prob, self.ngram._PROB_FLOOR))
        return total_lp

    # --------------------------------------------------------------- #
    # Internals                                                        #
    # --------------------------------------------------------------- #
    def _soft_aggregate(self, family: str, prefix: tuple[int, ...], top_n: int) -> Counter | None:
        prefixes = self._family_prefix_list.get(family, [])
        if not prefixes or len(prefix) == 0:
            return None
        # Mean embedding of the query prefix.
        if self._step_vectors is None:
            return None
        # Some ids in `prefix` may be out of range if the prefix was obtained
        # via aliasing on a corrupt input — defend against that.
        valid_ids = [i for i in prefix if 0 <= i < len(self._step_vectors)]
        if not valid_ids:
            return None
        q_vec = self._step_vectors[valid_ids].mean(axis=0)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm == 0:
            return None
        matrix = self._family_prefix_matrix[family]
        norms = self._family_prefix_norms[family]
        sims = (matrix @ q_vec) / (norms * q_norm)
        # Pick top-N above floor.
        top_idx = np.argsort(-sims)[:top_n]
        agg: Counter = Counter()
        for idx in top_idx:
            sim = float(sims[idx])
            if sim < self.sim_floor:
                break
            counter = self.ngram.counts[family][prefixes[idx]]
            for next_id, count in counter.items():
                agg[next_id] += sim * count
        return agg if agg else None

    def _cond_prob(self, family: str, prefix: tuple[int, ...], next_id: int, family_total: int) -> float:
        # 1. Exact prefix?
        counter = self.ngram.counts[family].get(prefix)
        if counter:
            denom = sum(counter.values())
            count = counter.get(next_id, 0)
            if count > 0:
                return count / denom

        # 2. Soft aggregate.
        soft = self._soft_aggregate(family, prefix, top_n=self.n_neighbors)
        if soft:
            denom = sum(soft.values())
            count = soft.get(next_id, 0)
            if count > 0:
                return count / denom

        # 3. NGram's own backoff path.
        return self.ngram._cond_prob(family, prefix, next_id, family_total)


# ----------------------------------------------------------------------- #
# Convenience constructor                                                 #
# ----------------------------------------------------------------------- #
def build_step_embedder_from_training_data(
    descriptions_dir: Path,
) -> StepEmbedder:
    """Convenience: locate all `*_longdescription_parameters.csv` in a dir and
    build a `StepEmbedder` from them."""
    paths = sorted(Path(descriptions_dir).glob("*longdescription_parameters.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no *longdescription_parameters.csv files in {descriptions_dir}"
        )
    return StepEmbedder.from_description_csvs(paths)
