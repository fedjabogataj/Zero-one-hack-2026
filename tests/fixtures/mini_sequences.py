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
