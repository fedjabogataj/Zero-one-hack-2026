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
