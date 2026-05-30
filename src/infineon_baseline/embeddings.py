"""Step-string embeddings derived from the description + parameter CSVs.

Each unique step string in the training data gets a single vector built from
the concatenation of its natural-language descriptions and fab parameters
(merged across all families it appears in). Embeddings are computed via TF-IDF
over word + bigram features, kept dense (small vocab, ~150 steps), and cached
to disk for fast load.

At inference time, unknown step names (e.g. from the hidden OOD family) are
encoded by feeding the step name itself through the same vectorizer — yielding
a vector in the same TF-IDF space that can be matched against known steps via
cosine similarity.
"""
from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class StepEmbedder:
    vectorizer: TfidfVectorizer
    vectors: np.ndarray            # shape (N_steps, D)
    step_to_idx: dict[str, int]
    idx_to_step: list[str]
    _row_norms: np.ndarray         # shape (N_steps,) — precomputed L2 norms

    @classmethod
    def fit(cls, descriptions_by_step: dict[str, str]) -> "StepEmbedder":
        """Fit on a {step_name: description_text} mapping.

        `description_text` should be a single string per step — concatenate
        multi-family descriptions before calling.
        """
        steps = sorted(descriptions_by_step.keys())
        texts = [descriptions_by_step[s] for s in steps]
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            lowercase=True,
            sublinear_tf=True,
        )
        X = vectorizer.fit_transform(texts).toarray()
        row_norms = np.linalg.norm(X, axis=1)
        row_norms = np.where(row_norms == 0, 1.0, row_norms)  # avoid div-by-0
        return cls(
            vectorizer=vectorizer,
            vectors=X,
            step_to_idx={s: i for i, s in enumerate(steps)},
            idx_to_step=steps,
            _row_norms=row_norms,
        )

    @classmethod
    def from_description_csvs(cls, csv_paths: Iterable[Path]) -> "StepEmbedder":
        """Build by reading one or more `*_longdescription_parameters.csv` files.

        Merges descriptions/parameters across all input files per step name —
        same step appearing in multiple families gets a concatenated text.
        """
        merged: dict[str, list[str]] = {}
        for path in csv_paths:
            path = Path(path)
            with path.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                # The PARAMETERS column header uses a non-breaking hyphen in some files;
                # we tolerate both Unicode variants by matching on a substring.
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
        # Concatenate per-step text (deduplicate identical phrasings).
        descriptions_by_step = {
            step: ". ".join(dict.fromkeys(texts))  # dedup but preserve order
            for step, texts in merged.items()
        }
        return cls.fit(descriptions_by_step)

    def encode(self, step: str) -> np.ndarray:
        """Return the embedding vector for a step string.

        Known steps return the cached vector; unknown ones (e.g. OOD family
        step names) are vectorized on the fly from the step name itself.
        """
        idx = self.step_to_idx.get(step)
        if idx is not None:
            return self.vectors[idx]
        v = self.vectorizer.transform([step]).toarray()[0]
        return v

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

    # ------------------------------------------------------------------ #
    # Batch helpers for use inside SoftNGram                              #
    # ------------------------------------------------------------------ #
    def similarity_matrix(self, query_vec: np.ndarray) -> np.ndarray:
        """Cosine similarity of a single query vector vs every known step.

        Returns shape (N_steps,). Used by SoftNGram to score prefixes in batch.
        """
        q_norm = float(np.linalg.norm(query_vec))
        if q_norm == 0:
            return np.zeros(len(self.idx_to_step))
        return (self.vectors @ query_vec) / (self._row_norms * q_norm)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({
            "vectorizer": self.vectorizer,
            "vectors": self.vectors,
            "step_to_idx": self.step_to_idx,
            "idx_to_step": self.idx_to_step,
            "row_norms": self._row_norms,
        }))

    @classmethod
    def load(cls, path: Path) -> "StepEmbedder":
        data = pickle.loads(Path(path).read_bytes())
        return cls(
            vectorizer=data["vectorizer"],
            vectors=data["vectors"],
            step_to_idx=data["step_to_idx"],
            idx_to_step=data["idx_to_step"],
            _row_norms=data["row_norms"],
        )
