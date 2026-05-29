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
