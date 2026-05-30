"""Word-level subword tokenizer with step-boundary <sep> tokens.

Splits step strings on whitespace and isolates trailing digit runs so that
related steps like 'DEPOSIT METAL 1', 'DEPOSIT METAL SEED', and
'DEPOSIT BACKSIDE METAL' share common subword tokens ('DEPOSIT', 'METAL').

Special tokens: <pad> (id 0), <bos> (id 1), <eos> (id 2), <sep> (id 3), <unk> (id 4).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SubwordTokenizer:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    SEP = "<sep>"
    UNK = "<unk>"
    SPECIAL_TOKENS = (PAD, BOS, EOS, SEP, UNK)

    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: list[str] = field(default_factory=list)
    family_to_id: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def split_step(step: str) -> list[str]:
        """Split one step string into subwords. Whitespace + digit-run splits.

        Example: "ALIGN MASK LEVEL 2" -> ["ALIGN", "MASK", "LEVEL", "2"]
                 "DEPOSIT METAL 1"    -> ["DEPOSIT", "METAL", "1"]
                 "RECEIVE WAFER LOT"  -> ["RECEIVE", "WAFER", "LOT"]
        Use regex: [A-Za-z]+ and \\d+ tokens capture words and numbers.
        """
        return re.findall(r"[A-Za-z][A-Za-z0-9_]*|\d+", step)

    @classmethod
    def fit(cls, sequences_by_family: dict[str, dict[str, list[str]]]) -> "SubwordTokenizer":
        """Build vocab from union of subwords across all training steps."""
        vocab: set[str] = set()
        for fam_seqs in sequences_by_family.values():
            for steps in fam_seqs.values():
                for step in steps:
                    vocab.update(cls.split_step(step))
        # Specials first (so PAD has id 0, etc.), then alphabetical subwords.
        id_to_token = list(cls.SPECIAL_TOKENS) + sorted(vocab)
        token_to_id = {tok: i for i, tok in enumerate(id_to_token)}
        family_to_id = {fam: i for i, fam in enumerate(sorted(sequences_by_family))}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token, family_to_id=family_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    @property
    def sep_id(self) -> int:
        return self.token_to_id[self.SEP]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK]

    def encode_step(self, step: str) -> list[int]:
        """One step string -> list of subword ids (no separators)."""
        return [self.token_to_id.get(t, self.unk_id) for t in self.split_step(step)]

    def encode_sequence(self, family: str, steps: list[str]) -> tuple[int, list[int]]:
        """Full sequence -> (family_id, [<bos>, step1, <sep>, step2, <sep>, ..., stepN, <eos>])."""
        if family not in self.family_to_id:
            raise KeyError(f"unknown family {family!r}; known: {sorted(self.family_to_id)}")
        ids = [self.bos_id]
        for i, step in enumerate(steps):
            ids.extend(self.encode_step(step))
            ids.append(self.sep_id if i < len(steps) - 1 else self.eos_id)
        return self.family_to_id[family], ids

    def encode_prefix(self, family: str, partial_steps: list[str]) -> tuple[int, list[int]]:
        """Like encode_sequence but the last step is followed by <sep>, not <eos>.

        This is the right form for next-step prediction: the model will generate
        from where the partial sequence ended, expecting to emit step content
        followed by a separator.
        """
        if family not in self.family_to_id:
            raise KeyError(f"unknown family {family!r}")
        ids = [self.bos_id]
        for step in partial_steps:
            ids.extend(self.encode_step(step))
            ids.append(self.sep_id)
        return self.family_to_id[family], ids

    def decode_step(self, ids: list[int]) -> str:
        """Decode a stream of (presumed single-step) subword ids into a step string.

        Strips specials, joins with spaces.
        """
        toks = [self.id_to_token[i] for i in ids if i < len(self.id_to_token)]
        toks = [t for t in toks if t not in self.SPECIAL_TOKENS]
        return " ".join(toks)

    def decode_sequence(self, ids: list[int]) -> list[str]:
        """Decode a full sequence back into a list of step strings.

        Splits on <sep>/<eos>, decodes each segment with decode_step,
        drops empty segments.
        """
        steps: list[str] = []
        current: list[int] = []
        for i in ids:
            if i in (self.sep_id, self.eos_id):
                if current:
                    s = self.decode_step(current)
                    if s:
                        steps.append(s)
                    current = []
                if i == self.eos_id:
                    break
            elif i == self.bos_id or i == self.pad_id:
                continue
            else:
                current.append(i)
        if current:
            s = self.decode_step(current)
            if s:
                steps.append(s)
        return steps

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "id_to_token": self.id_to_token,
            "family_to_id": self.family_to_id,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "SubwordTokenizer":
        data = json.loads(Path(path).read_text())
        id_to_token = list(data["id_to_token"])
        return cls(
            token_to_id={t: i for i, t in enumerate(id_to_token)},
            id_to_token=id_to_token,
            family_to_id=dict(data["family_to_id"]),
        )
