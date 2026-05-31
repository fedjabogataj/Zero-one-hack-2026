"""Sentence-transformer-based step embedder.

Encodes each step as: step_name + " " + description + " " + parameters
using `sentence-transformers/all-MiniLM-L6-v2` (384-dim).

Public API is identical to `StepEmbedder` so both embedders can be used
interchangeably. Class name `STStepEmbedder` distinguishes it.

Unknown step names at inference time are encoded from the step name string
alone (same model, same space).

Model object is cached in a class attribute so it is loaded only once per
process.
"""
from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


_MODEL_NAME = "BAAI/bge-base-en-v1.5"
# 109M params, 768-dim sentence-embedding model. Meaningfully stronger than
# the previous all-MiniLM-L6-v2 (22M params, 384-dim) on technical /
# domain-specific text, while still small enough to encode the 136 step
# strings in well under a minute on the login node.


@dataclass
class STStepEmbedder:
    vectors: np.ndarray           # shape (N_steps, 768)
    step_to_idx: dict[str, int]
    idx_to_step: list[str]
    _row_norms: np.ndarray        # shape (N_steps,) — precomputed L2 norms

    # Class-level model cache — loaded once per process.
    _model: "object | None" = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def _get_model(cls):
        """Return the cached SentenceTransformer, loading it on first call."""
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer(_MODEL_NAME)
        return cls._model

    @classmethod
    def fit(cls, descriptions_by_step: dict[str, str]) -> "STStepEmbedder":
        """Fit on a {step_name: description_text} mapping.

        `description_text` should be a single string per step — concatenate
        multi-family descriptions before calling.
        """
        model = cls._get_model()
        steps = sorted(descriptions_by_step.keys())
        texts = [f"{s} {descriptions_by_step[s]}" for s in steps]
        vecs = model.encode(texts, batch_size=64, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=False)
        vecs = vecs.astype(np.float32)
        row_norms = np.linalg.norm(vecs, axis=1)
        row_norms = np.where(row_norms == 0, 1.0, row_norms)
        return cls(
            vectors=vecs,
            step_to_idx={s: i for i, s in enumerate(steps)},
            idx_to_step=steps,
            _row_norms=row_norms,
        )

    @classmethod
    def from_description_csvs(cls, csv_paths: Iterable[Path]) -> "STStepEmbedder":
        """Build by reading one or more `*_longdescription_parameters.csv` files.

        Merges descriptions/parameters across all input files per step name.
        """
        merged: dict[str, list[str]] = {}
        for path in csv_paths:
            path = Path(path)
            with path.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                params_col = next(
                    (c for c in (reader.fieldnames or []) if "PARAMETERS" in c.upper()),
                    None,
                )
                desc_col = next(
                    (c for c in (reader.fieldnames or []) if "DESCRIPTION" in c.upper()),
                    None,
                )
                for row in reader:
                    step = (row.get("STEP") or "").strip()
                    if not step:
                        continue
                    pieces = [step]
                    if desc_col:
                        pieces.append(row.get(desc_col, "") or "")
                    if params_col:
                        pieces.append(row.get(params_col, "") or "")
                    merged.setdefault(step, []).append(". ".join(pieces))
        descriptions_by_step = {
            step: ". ".join(dict.fromkeys(texts))
            for step, texts in merged.items()
        }
        return cls.fit(descriptions_by_step)

    def encode(self, step: str) -> np.ndarray:
        """Return the 384-dim embedding for a step string.

        Known steps return the cached vector. Unknown steps are encoded
        on the fly from the step name alone.
        """
        idx = self.step_to_idx.get(step)
        if idx is not None:
            return self.vectors[idx]
        model = self._get_model()
        v = model.encode([step], convert_to_numpy=True, normalize_embeddings=False)[0]
        return v.astype(np.float32)

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.encode(a), self.encode(b)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return float(va @ vb / denom) if denom > 0 else 0.0

    def nearest(self, step: str, k: int = 5, exclude_self: bool = True) -> list[tuple[str, float]]:
        """Return the k most-similar known steps by cosine similarity."""
        v = self.encode(step)
        v_norm = float(np.linalg.norm(v))
        if v_norm == 0:
            return []
        sims = (self.vectors @ v) / (self._row_norms * v_norm)
        order = np.argsort(-sims)
        out: list[tuple[str, float]] = []
        for idx in order:
            name = self.idx_to_step[idx]
            if exclude_self and name == step:
                continue
            out.append((name, float(sims[idx])))
            if len(out) >= k:
                break
        return out

    def similarity_matrix(self, query_vec: np.ndarray) -> np.ndarray:
        """Cosine similarity of a single query vector vs every known step."""
        q_norm = float(np.linalg.norm(query_vec))
        if q_norm == 0:
            return np.zeros(len(self.idx_to_step))
        return (self.vectors @ query_vec) / (self._row_norms * q_norm)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({
            "vectors": self.vectors,
            "step_to_idx": self.step_to_idx,
            "idx_to_step": self.idx_to_step,
            "row_norms": self._row_norms,
        }))

    @classmethod
    def load(cls, path: Path) -> "STStepEmbedder":
        data = pickle.loads(Path(path).read_bytes())
        return cls(
            vectors=data["vectors"],
            step_to_idx=data["step_to_idx"],
            idx_to_step=data["idx_to_step"],
            _row_norms=data["row_norms"],
        )
