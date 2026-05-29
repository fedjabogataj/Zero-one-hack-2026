"""Step-string <-> integer-id mapping plus family ids."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Tokenizer:
    step_to_id: dict[str, int] = field(default_factory=dict)
    id_to_step: list[str] = field(default_factory=list)
    family_to_id: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, sequences_by_family: dict[str, dict[str, list[str]]]) -> "Tokenizer":
        vocab: set[str] = set()
        for fam_seqs in sequences_by_family.values():
            for steps in fam_seqs.values():
                vocab.update(steps)
        id_to_step = sorted(vocab)
        step_to_id = {step: i for i, step in enumerate(id_to_step)}
        family_to_id = {fam: i for i, fam in enumerate(sorted(sequences_by_family.keys()))}
        return cls(step_to_id=step_to_id, id_to_step=id_to_step, family_to_id=family_to_id)

    def encode(self, family: str, steps: list[str]) -> tuple[int, list[int]]:
        if family not in self.family_to_id:
            raise KeyError(f"unknown family {family!r}; known: {sorted(self.family_to_id)}")
        family_id = self.family_to_id[family]
        ids = [self.step_to_id[s] for s in steps]  # raises KeyError on unknown step
        return family_id, ids

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_step[i] for i in ids]

    def save(self, path: Path) -> None:
        data = {
            "id_to_step": self.id_to_step,
            "family_to_id": self.family_to_id,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text())
        id_to_step = list(data["id_to_step"])
        return cls(
            step_to_id={s: i for i, s in enumerate(id_to_step)},
            id_to_step=id_to_step,
            family_to_id=dict(data["family_to_id"]),
        )
