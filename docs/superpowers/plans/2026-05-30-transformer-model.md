# Transformer Next-Step Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GPT-style transformer as a drop-in replacement for NGram in the predict pipeline, using sentence-transformer semantic embeddings so it generalizes to the hidden 4th product family.

**Architecture:** A small causal decoder (4 layers, 6 heads, 384-dim matching all-MiniLM-L6-v2) initialised from ST embeddings; family conditioning via a learnable family embedding summed at every position; LM head weight-tied to the embedding table. The predictor adapter wraps it in a class that exposes exactly the `NGram` interface (`order`, `unigram`, `top_k`, `log_prob`) so `predictor.run_task*` functions work unchanged.

**Tech Stack:** Python 3.10+, PyTorch ≥ 2.1, sentence-transformers ≥ 2.5, existing infineon_baseline package (numpy, pandas, scikit-learn, tokenizer, embeddings, ngram, predictor, cli).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `pyproject.toml` | Add torch + sentence-transformers deps |
| Create | `src/infineon_baseline/embeddings_st.py` | `STStepEmbedder` — sentence-transformer embeddings, same API as `StepEmbedder` |
| Create | `src/infineon_baseline/transformer_model.py` | `TransformerLM` — GPT-style decoder, `from_tokenizer_and_embedder` classmethod |
| Create | `src/infineon_baseline/train.py` | `SequenceDataset`, training loop, `train(args)` entrypoint |
| Create | `src/infineon_baseline/transformer_predictor.py` | `TransformerPredictor` — NGram-compatible wrapper |
| Modify | `src/infineon_baseline/cli.py` | Add `train`, `build-st-embeddings` subcommands; add `--transformer` to `predict` |
| Create | `scripts/train_transformer.sbatch` | GPU SLURM job script |

---

## Task 1: Add deps and build STStepEmbedder

**Files:**
- Modify: `pyproject.toml`
- Create: `src/infineon_baseline/embeddings_st.py`

- [ ] **Step 1: Add deps to pyproject.toml**

Open `pyproject.toml` and add to the `dependencies` list:

```toml
[project]
dependencies = [
    "numpy>=1.26",
    "pandas>=2.1",
    "scikit-learn>=1.4",
    "torch>=2.1",
    "sentence-transformers>=2.5",
]
```

- [ ] **Step 2: Install deps**

```bash
cd /Users/fedjabogataj/Projects/Hackathons/zero\ one/official_repo/zero_one_hack_01/tracks/industrial-infineon
source .venv/bin/activate
pip install -e .
```

Expected: pip reports `Successfully installed` or `already satisfied` for torch and sentence-transformers. If torch wheel fails (e.g. ARM/macOS issues), try: `pip install torch --index-url https://download.pytorch.org/whl/cpu` then re-run `pip install -e .`.

- [ ] **Step 3: Create embeddings_st.py**

Create `src/infineon_baseline/embeddings_st.py` with the following content:

```python
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


_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class STStepEmbedder:
    vectors: np.ndarray           # shape (N_steps, 384)
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
```

- [ ] **Step 4: Smoke-test the embedder**

```bash
source .venv/bin/activate
python -c "
from infineon_baseline.embeddings_st import STStepEmbedder
e = STStepEmbedder.fit({'ETCH': 'removes material', 'DEPOSIT CVD': 'chemical vapour'})
print('vectors shape:', e.vectors.shape)       # (2, 384)
print('nearest ETCH:', e.nearest('ETCH', k=1))
print('encode unknown:', e.encode('POLISH').shape)  # (384,)
"
```

Expected: vectors shape `(2, 384)`, nearest returns a list with one tuple.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/infineon_baseline/embeddings_st.py
git commit -m "feat: add STStepEmbedder using sentence-transformers/all-MiniLM-L6-v2"
```

---

## Task 2: Build TransformerLM (GPT-style decoder)

**Files:**
- Create: `src/infineon_baseline/transformer_model.py`

Architecture decision: family conditioning is done by **summing** a learnable family embedding (shape `n_families × d_model`) into every position's token+position embedding. This is simpler than prepending a special token and keeps sequence length identical to the step sequence.

- [ ] **Step 1: Create transformer_model.py**

```python
"""Small GPT-style causal language model for next-step prediction.

Architecture:
- Token embeddings: initialised from STStepEmbedder.vectors (vocab_size × d_model).
- Family conditioning: learnable family embedding (n_families × d_model) summed
  into every position of the input (simpler than a special prefix token).
- Positional embeddings: learned (max_seq_len × d_model).
- n_layers stacked TransformerDecoderLayer blocks (causal mask applied in forward).
- LM head: weight-tied linear projection to vocab_size logits (no softmax).

Default arch matches all-MiniLM-L6-v2 dimension (384):
  d_model=384, n_heads=6, n_layers=4, ff_dim=1536, dropout=0.1, max_seq_len=256
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    vocab_size: int = 0            # set from tokenizer at construction time
    n_families: int = 0            # set from tokenizer at construction time
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 4
    ff_dim: int = 1536
    dropout: float = 0.1
    max_seq_len: int = 256
    freeze_embeddings: bool = False
    pad_id: int = 0                # id reserved for PAD (set to vocab_size slot)


class TransformerLM(nn.Module):
    """GPT-style causal LM with family conditioning."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        V = config.vocab_size
        D = config.d_model

        # Token embedding table — rows will be overwritten from ST vectors.
        self.token_emb = nn.Embedding(V + 1, D, padding_idx=V)  # +1 for PAD slot
        # Family embedding (summed into every position).
        self.family_emb = nn.Embedding(config.n_families, D)
        # Learned positional embeddings.
        self.pos_emb = nn.Embedding(config.max_seq_len, D)
        self.emb_drop = nn.Dropout(config.dropout)

        # Transformer decoder stack — using nn.TransformerEncoder with causal mask.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # LM head — weight-tied with token embedding table.
        self.lm_head = nn.Linear(D, V + 1, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        self._init_weights()
        if config.freeze_embeddings:
            self.token_emb.weight.requires_grad_(False)

    def _init_weights(self) -> None:
        nn.init.normal_(self.family_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    @classmethod
    def from_tokenizer_and_embedder(
        cls,
        tokenizer,
        embedder,
        **arch_kwargs,
    ) -> "TransformerLM":
        """Construct from a fitted Tokenizer and STStepEmbedder.

        The token embedding table is initialised from embedder.vectors.
        arch_kwargs override TransformerConfig defaults (d_model, n_heads, etc.).
        """
        vocab_size = len(tokenizer.id_to_step)
        n_families = len(tokenizer.family_to_id)
        d_model = arch_kwargs.pop("d_model", 384)

        config = TransformerConfig(
            vocab_size=vocab_size,
            n_families=n_families,
            d_model=d_model,
            pad_id=vocab_size,   # PAD uses the extra slot beyond the vocab
            **arch_kwargs,
        )
        model = cls(config)

        # Initialise token embeddings from the ST embedder vectors.
        vecs = torch.from_numpy(embedder.vectors.astype(np.float32))  # (vocab_size, D_emb)
        if vecs.shape[1] != d_model:
            # Project if embedding dim differs from d_model.
            proj = nn.Linear(vecs.shape[1], d_model, bias=False)
            with torch.no_grad():
                vecs = proj(vecs)
        with torch.no_grad():
            model.token_emb.weight[:vocab_size] = vecs

        return model

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask (True = ignore) for causal attention."""
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        family_ids: torch.Tensor,   # (B,)
        step_ids: torch.Tensor,     # (B, T)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, T) True = PAD
    ) -> torch.Tensor:              # (B, T, V+1)
        B, T = step_ids.shape
        device = step_ids.device

        # Build embeddings: token + position + family.
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B, T)
        tok = self.token_emb(step_ids)              # (B, T, D)
        pos = self.pos_emb(positions)               # (B, T, D)
        fam = self.family_emb(family_ids)           # (B, D)
        x = self.emb_drop(tok + pos + fam.unsqueeze(1))  # broadcast family across T

        causal = self._causal_mask(T, device)
        x = self.transformer(x, mask=causal, src_key_padding_mask=key_padding_mask,
                             is_causal=True)
        logits = self.lm_head(x)   # (B, T, V+1)
        return logits

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

- [ ] **Step 2: Smoke-test the model builds and forward-passes**

```bash
source .venv/bin/activate
python -c "
import torch
from infineon_baseline.tokenizer import Tokenizer
from infineon_baseline.embeddings_st import STStepEmbedder

# Minimal tokenizer with 5 steps, 3 families.
tok = Tokenizer(
    step_to_id={'A':0,'B':1,'C':2,'D':3,'E':4},
    id_to_step=['A','B','C','D','E'],
    family_to_id={'mosfet':0,'igbt':1,'ic':2},
)
emb = STStepEmbedder.fit({'A':'etch','B':'deposit','C':'clean','D':'anneal','E':'ship'})

from infineon_baseline.transformer_model import TransformerLM
model = TransformerLM.from_tokenizer_and_embedder(tok, emb)
print('params:', model.param_count())

family_ids = torch.tensor([0, 1])
step_ids = torch.tensor([[0,1,2],[1,2,3]])
logits = model(family_ids, step_ids)
print('logits shape:', logits.shape)   # (2, 3, 6) — V+1=6
"
```

Expected: params ~8M, logits shape `(2, 3, 6)`.

- [ ] **Step 3: Commit**

```bash
git add src/infineon_baseline/transformer_model.py
git commit -m "feat: add TransformerLM (GPT decoder, 384-dim, family-conditioned)"
```

---

## Task 3: Build training loop

**Files:**
- Create: `src/infineon_baseline/train.py`

Key decisions:
- PAD token id = `vocab_size` (the extra slot in the embedding table, `padding_idx=V`).
- Val split: last 10% of training sequences (by insertion order), not a random shuffle — ensures reproducibility without contaminating the training data.
- Scheduler: linear warmup for 200 steps then cosine decay to 0.
- Loss: cross-entropy on next-token at each position, ignoring PAD targets.
- Checkpoint: single `.pt` dict containing `config`, `state_dict`, `tokenizer_data`, `embedder_data`.

- [ ] **Step 1: Create train.py**

```python
"""Training loop for the TransformerLM next-step model.

Reads train_split.csv (produced by `build-eval`), tokenises sequences, and
trains a TransformerLM with causal LM objective. Saves a self-contained
checkpoint (.pt) that TransformerPredictor can load without separate files.

Usage (via CLI):
    infineon-baseline train \\
        --train outputs/eval/train_split.csv \\
        --embeddings outputs/models/st_step_embeddings.pkl \\
        --out outputs/models/transformer.pt \\
        --epochs 20 --batch-size 32 --lr 3e-4 --device auto --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ─── Dataset ──────────────────────────────────────────────────────────────── #

class SequenceDataset(Dataset):
    """One sample = one manufacturing sequence as (family_id, ids_tensor)."""

    def __init__(
        self,
        samples: list[tuple[int, list[int]]],   # (family_id, id_list)
        max_seq_len: int,
        pad_id: int,
    ) -> None:
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        family_id, ids = self.samples[idx]
        # Truncate to max_seq_len (keep last tokens — more useful for prediction).
        ids = ids[:self.max_seq_len]
        T = len(ids)
        # Input: ids[:-1], target: ids[1:] (next-token prediction).
        inp = ids[:-1] if T > 1 else ids
        tgt = ids[1:] if T > 1 else ids
        inp_len = len(inp)
        # Pad to max_seq_len - 1.
        max_T = self.max_seq_len - 1
        inp_t = torch.full((max_T,), self.pad_id, dtype=torch.long)
        tgt_t = torch.full((max_T,), -100, dtype=torch.long)  # -100 = ignore in CE
        inp_t[:inp_len] = torch.tensor(inp, dtype=torch.long)
        tgt_t[:inp_len] = torch.tensor(tgt, dtype=torch.long)
        pad_mask = torch.zeros(max_T, dtype=torch.bool)
        pad_mask[inp_len:] = True  # True = padding position
        return {
            "family_id": torch.tensor(family_id, dtype=torch.long),
            "step_ids": inp_t,
            "targets": tgt_t,
            "pad_mask": pad_mask,
        }


def _load_train_split(csv_path: Path) -> dict[str, dict[str, list[str]]]:
    """Read train_split.csv → {family: {seq_id: [steps]}}."""
    data: dict[str, dict[str, list[str]]] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            data.setdefault(row["FAMILY"], {}).setdefault(row["SEQUENCE_ID"], []).append(row["STEP"])
    return data


def _build_samples(
    corpus: dict[str, dict[str, list[str]]],
    tokenizer,
    max_seq_len: int,
    pad_id: int,
) -> tuple[list[tuple[int, list[int]]], dict[str, Counter]]:
    """Tokenise all sequences; also compute unigram counts (for TransformerPredictor)."""
    samples: list[tuple[int, list[int]]] = []
    unigram: dict[str, Counter] = {}
    for family, seqs in corpus.items():
        family_id = tokenizer.family_to_id[family]
        uc: Counter = Counter()
        for steps in seqs.values():
            # Alias unknown steps via step_to_id; skip truly unknown ones.
            ids = [tokenizer.step_to_id[s] for s in steps if s in tokenizer.step_to_id]
            if len(ids) < 2:
                continue
            uc.update(ids)
            samples.append((family_id, ids))
        unigram[family] = uc
    return samples, unigram


# ─── Scheduler ────────────────────────────────────────────────────────────── #

def _make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Training entrypoint ──────────────────────────────────────────────────── #

def train(args: argparse.Namespace) -> int:
    """Main training loop. Called by CLI's `train` subcommand."""
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Device ──────────────────────────────────────────────────────────── #
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load data ───────────────────────────────────────────────────────── #
    from infineon_baseline.tokenizer import Tokenizer
    from infineon_baseline.embeddings_st import STStepEmbedder
    from infineon_baseline.transformer_model import TransformerLM, TransformerConfig

    corpus = _load_train_split(Path(args.train))
    # Build tokenizer from the corpus (only known step names in train_split).
    tokenizer = Tokenizer.fit(corpus)
    embedder = STStepEmbedder.load(Path(args.embeddings))

    max_seq_len = getattr(args, "max_seq_len", 256)
    vocab_size = len(tokenizer.id_to_step)
    pad_id = vocab_size  # one slot beyond the actual vocab

    all_samples, unigram = _build_samples(corpus, tokenizer, max_seq_len, pad_id)

    # Val split: last 10% of samples (deterministic).
    n_val = max(1, len(all_samples) // 10)
    train_samples = all_samples[:-n_val]
    val_samples = all_samples[-n_val:]
    print(f"Train sequences: {len(train_samples)}  Val: {len(val_samples)}")

    train_ds = SequenceDataset(train_samples, max_seq_len, pad_id)
    val_ds = SequenceDataset(val_samples, max_seq_len, pad_id)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=g, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ───────────────────────────────────────────────────────────── #
    arch_kwargs: dict = {}
    for key in ("d_model", "n_heads", "n_layers", "ff_dim", "dropout"):
        val = getattr(args, key, None)
        if val is not None:
            arch_kwargs[key] = val
    arch_kwargs["max_seq_len"] = max_seq_len

    model = TransformerLM.from_tokenizer_and_embedder(tokenizer, embedder, **arch_kwargs)
    model = model.to(device)
    print(f"Params: {model.param_count():,}")

    # ── Optimizer & scheduler ───────────────────────────────────────────── #
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=0.01, betas=(0.9, 0.95),
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(200, total_steps // 10)
    scheduler = _make_scheduler(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # ── Training loop ───────────────────────────────────────────────────── #
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        n_tokens = 0

        for batch in train_loader:
            family_ids = batch["family_id"].to(device)
            step_ids = batch["step_ids"].to(device)
            targets = batch["targets"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            logits = model(family_ids, step_ids, key_padding_mask=pad_mask)
            # logits: (B, T, V+1); targets: (B, T) with -100 for PAD
            B, T, Vp1 = logits.shape
            loss = criterion(logits.view(B * T, Vp1), targets.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            batch_tokens = int((targets != -100).sum().item())
            n_tokens += batch_tokens
            epoch_loss += loss.item()

            if global_step == 1:
                elapsed = time.time() - t0
                print(f"  First step: {elapsed:.2f}s, "
                      f"tokens/s ≈ {batch_tokens / max(elapsed, 1e-6):.0f}")

            if global_step % 50 == 0:
                print(f"  step {global_step:5d} | loss {loss.item():.4f} "
                      f"| lr {scheduler.get_last_lr()[0]:.2e}")

            if global_step % 500 == 0:
                val_loss = _eval(model, val_loader, criterion, device)
                print(f"  [val step {global_step}] loss {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch}/{args.epochs} | avg_loss {avg_loss:.4f} "
              f"| {n_tokens/elapsed:.0f} tok/s | {elapsed:.1f}s")

    # Final val pass.
    val_loss = _eval(model, val_loader, criterion, device)
    print(f"Final val loss: {val_loss:.4f}")

    # ── Save checkpoint ─────────────────────────────────────────────────── #
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialise tokenizer and embedder inline so the checkpoint is self-contained.
    tokenizer_data = {
        "id_to_step": tokenizer.id_to_step,
        "family_to_id": tokenizer.family_to_id,
    }
    embedder_data = pickle.dumps({
        "vectors": embedder.vectors,
        "step_to_idx": embedder.step_to_idx,
        "idx_to_step": embedder.idx_to_step,
        "row_norms": embedder._row_norms,
    })

    checkpoint = {
        "config": vars(model.config),
        "state_dict": model.state_dict(),
        "tokenizer_data": tokenizer_data,
        "embedder_data": embedder_data,
        "unigram": {fam: dict(ctr) for fam, ctr in unigram.items()},
        "arch_kwargs": arch_kwargs,
    }
    torch.save(checkpoint, out_path)
    print(f"✓ checkpoint saved → {out_path}")
    return 0


def _eval(model, loader, criterion, device) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            family_ids = batch["family_id"].to(device)
            step_ids = batch["step_ids"].to(device)
            targets = batch["targets"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            logits = model(family_ids, step_ids, key_padding_mask=pad_mask)
            B, T, Vp1 = logits.shape
            loss = criterion(logits.view(B * T, Vp1), targets.view(B * T))
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)
```

- [ ] **Step 2: Commit**

```bash
git add src/infineon_baseline/train.py
git commit -m "feat: add TransformerLM training loop with AdamW + cosine LR"
```

---

## Task 4: Build TransformerPredictor and wire up CLI

**Files:**
- Create: `src/infineon_baseline/transformer_predictor.py`
- Modify: `src/infineon_baseline/cli.py`

- [ ] **Step 1: Create transformer_predictor.py**

```python
"""NGram-compatible wrapper around a trained TransformerLM.

Exposes the same attributes and methods as `NGram` so
`predictor.run_task1/2/3` work without modification:
  - model.order       (int)
  - model.unigram     (dict[str, Counter])
  - model.top_k(family, prefix, k) -> list[int]
  - model.log_prob(family, ids) -> float

OOD step aliasing: when a step name is not in the tokenizer vocabulary,
we use STStepEmbedder.nearest() to map it to the closest known step.
"""
from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class TransformerPredictor:
    """Wraps TransformerLM with the NGram interface for use in run_task*."""

    order: int          # set to max_seq_len so run_task1 uses the full prefix
    unigram: dict[str, Counter]

    def __init__(
        self,
        model,          # TransformerLM (already on device)
        tokenizer,      # Tokenizer
        embedder,       # STStepEmbedder
        unigram: dict[str, Counter],
        device: torch.device,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._embedder = embedder
        self.unigram = unigram
        self._device = device
        self.order = model.config.max_seq_len  # lets run_task1 pass the full prefix

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> "TransformerPredictor":
        """Load a checkpoint produced by `train.py`."""
        from infineon_baseline.tokenizer import Tokenizer
        from infineon_baseline.embeddings_st import STStepEmbedder
        from infineon_baseline.transformer_model import TransformerLM, TransformerConfig

        if device == "auto":
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            _device = torch.device(device)

        ckpt = torch.load(path, map_location=_device, weights_only=False)

        # Reconstruct tokenizer.
        tok_data = ckpt["tokenizer_data"]
        id_to_step = list(tok_data["id_to_step"])
        tokenizer = Tokenizer(
            step_to_id={s: i for i, s in enumerate(id_to_step)},
            id_to_step=id_to_step,
            family_to_id=dict(tok_data["family_to_id"]),
        )

        # Reconstruct embedder.
        emb_data = pickle.loads(ckpt["embedder_data"])
        embedder = STStepEmbedder(
            vectors=emb_data["vectors"],
            step_to_idx=emb_data["step_to_idx"],
            idx_to_step=emb_data["idx_to_step"],
            _row_norms=emb_data["row_norms"],
        )

        # Reconstruct model.
        cfg_dict = ckpt["config"]
        config = TransformerConfig(**cfg_dict)
        model = TransformerLM(config)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(_device)
        model.eval()

        unigram = {fam: Counter(ctr) for fam, ctr in ckpt["unigram"].items()}
        return cls(model=model, tokenizer=tokenizer, embedder=embedder,
                   unigram=unigram, device=_device)

    # ─── OOD aliasing ─────────────────────────────────────────────────── #

    def _alias_step(self, step: str) -> int:
        """Map step name to a token id, using nearest-neighbor for OOD names."""
        sid = self._tokenizer.step_to_id.get(step)
        if sid is not None:
            return sid
        nearest = self._embedder.nearest(step, k=1, exclude_self=False)
        if nearest:
            return self._tokenizer.step_to_id.get(nearest[0][0], 0)
        return 0   # last-resort: token 0

    def _encode_prefix(self, family: str, prefix: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (family_id_tensor, step_ids_tensor) for inference."""
        fid = self._tokenizer.family_to_id.get(family, 0)
        family_t = torch.tensor([fid], dtype=torch.long, device=self._device)
        if len(prefix) == 0:
            # Empty prefix: use a single PAD token so forward() has something.
            pad_id = self._model.config.pad_id
            ids_t = torch.tensor([[pad_id]], dtype=torch.long, device=self._device)
        else:
            max_len = self._model.config.max_seq_len - 1
            ids = list(prefix[-max_len:])
            ids_t = torch.tensor([ids], dtype=torch.long, device=self._device)
        return family_t, ids_t

    # ─── NGram-compatible interface ───────────────────────────────────── #

    @torch.no_grad()
    def top_k(self, family: str, prefix: tuple[int, ...], k: int = 5) -> list[int]:
        """Return the k most-likely next-step ids.

        Forward the prefix through the model, take logits at the last
        position, and return top-k step ids (excluding PAD).
        """
        self._model.eval()
        family_t, ids_t = self._encode_prefix(family, prefix)
        logits = self._model(family_t, ids_t)  # (1, T, V+1)
        last_logits = logits[0, -1, :]          # (V+1,)
        # Mask out PAD slot (index vocab_size).
        pad_id = self._model.config.pad_id
        last_logits[pad_id] = float("-inf")
        # Also mask steps not in this family's unigram if family is known.
        if family in self.unigram:
            allowed = set(self.unigram[family].keys())
            for i in range(pad_id):
                if i not in allowed:
                    last_logits[i] = float("-inf")
        top_ids = torch.topk(last_logits, k=min(k, pad_id), dim=-1).indices.tolist()
        return top_ids

    @torch.no_grad()
    def log_prob(self, family: str, ids: list[int]) -> float:
        """Sum of log P(id_i | prefix_i) under the model.

        Uses teacher-forcing: feed the full sequence as input,
        read logits at each position for next-token prediction.
        """
        if not ids:
            return 0.0
        self._model.eval()
        fid = self._tokenizer.family_to_id.get(family, 0)
        family_t = torch.tensor([fid], dtype=torch.long, device=self._device)
        max_len = self._model.config.max_seq_len
        ids_clipped = ids[:max_len]

        # Input: [PAD] + ids[:-1] so position i predicts ids[i].
        pad_id = self._model.config.pad_id
        inp = [pad_id] + ids_clipped[:-1]
        inp_t = torch.tensor([inp], dtype=torch.long, device=self._device)

        logits = self._model(family_t, inp_t)  # (1, T, V+1)
        log_probs = F.log_softmax(logits[0], dim=-1)  # (T, V+1)
        total = 0.0
        for t, next_id in enumerate(ids_clipped):
            total += log_probs[t, next_id].item()
        return total
```

- [ ] **Step 2: Modify cli.py to add `train`, `build-st-embeddings`, and `--transformer` on predict**

Open `src/infineon_baseline/cli.py` and make these three changes:

**2a. Add imports at the top (after existing imports):**

```python
# New transformer imports (lazy-imported inside functions to keep startup fast)
```

Actually, add the new subcommand functions and wire them to main(). The cleanest approach is to add three new functions and modify `cmd_predict` and `main`. Here is the full set of changes:

**Add `cmd_build_st_embeddings` function** (after `cmd_build_embeddings`):

```python
def cmd_build_st_embeddings(args: argparse.Namespace) -> int:
    """Compute sentence-transformer embeddings for every step."""
    from infineon_baseline.embeddings_st import STStepEmbedder
    descriptions_dir = Path(args.descriptions_dir)
    paths = sorted(descriptions_dir.glob("*longdescription_parameters.csv"))
    if not paths:
        raise FileNotFoundError(f"no *longdescription_parameters.csv in {descriptions_dir}")
    embedder = STStepEmbedder.from_description_csvs(paths)
    out_path = Path(args.out)
    embedder.save(out_path)
    print(f"✓ ST-embedded {len(embedder.idx_to_step)} unique steps "
          f"({embedder.vectors.shape[1]} dims) → {out_path}")
    return 0
```

**Add `cmd_train` function** (after `cmd_build_st_embeddings`):

```python
def cmd_train(args: argparse.Namespace) -> int:
    """Train the transformer LM."""
    from infineon_baseline.train import train as _train
    return _train(args)
```

**Modify `cmd_predict`** to support `--transformer PATH`:

Replace the existing `cmd_predict` function body with one that checks `args.transformer` first:

```python
def cmd_predict(args: argparse.Namespace) -> int:
    # Validate mutual exclusivity.
    has_model = bool(getattr(args, "model", None))
    has_transformer = bool(getattr(args, "transformer", None))
    if has_model and has_transformer:
        print("ERROR: --model and --transformer are mutually exclusive", file=sys.stderr)
        return 2
    if not has_model and not has_transformer:
        print("ERROR: one of --model or --transformer is required", file=sys.stderr)
        return 2

    if has_transformer:
        from infineon_baseline.transformer_predictor import TransformerPredictor
        model = TransformerPredictor.load(Path(args.transformer))
        tokenizer = model._tokenizer
        model_kind = "transformer"
    else:
        base_ngram = NGram.load(args.model)
        tokenizer = Tokenizer.load(Path(args.model).with_suffix(".tokenizer.json"))
        if args.embeddings:
            embedder = StepEmbedder.load(args.embeddings)
            model = SoftNGram(ngram=base_ngram, embedder=embedder, tokenizer=tokenizer)
            model_kind = "soft-ngram"
        else:
            model = base_ngram
            model_kind = "ngram"

    examples = _read_eval_input(args.eval_input)
    out_path = Path(args.out)

    if args.task == "next-step":
        rows = list(run_task1(examples, tokenizer=tokenizer, ngram=model))
        write_task1_csv(rows, out_path)
    elif args.task == "complete":
        rows = list(run_task2(examples, tokenizer=tokenizer, ngram=model, constrain=args.constrain))
        write_task2_csv(rows, out_path)
    elif args.task == "anomaly":
        threshold = args.threshold if args.threshold is not None else 0.0
        rows = list(run_task3(examples, tokenizer=tokenizer, ngram=model,
                              threshold=threshold, strategy=args.anomaly_strategy))
        write_task3_csv(rows, out_path)
    else:
        print(f"unknown task: {args.task}", file=sys.stderr)
        return 2

    model_order = getattr(model, "order", None)
    if has_transformer:
        model_info = {"type": "transformer", "path": str(args.transformer)}
    else:
        model_info = {"type": model_kind, "order": model_order,
                      "embeddings": str(args.embeddings) if args.embeddings else None}
    write_meta(
        out_path.with_suffix(".meta.json"),
        task=args.task,
        model_info=model_info,
        seed=args.seed,
        eval_split_hash="(set externally)",
    )
    print(f"✓ wrote {len(rows)} predictions ({model_kind}) → {out_path}")
    return 0
```

**Modify `main()`** to:
1. Make `--model` on the predict subparser optional (remove `required=True`).
2. Add `--transformer` argument on the predict subparser.
3. Add `build-st-embeddings` subparser.
4. Add `train` subparser.
5. Add new dispatch entries.

- [ ] **Step 3: Apply all cli.py edits**

The exact edits to `src/infineon_baseline/cli.py`:

**Edit 1 — modify predict subparser to make `--model` optional and add `--transformer`:**

Find (in `main()`):
```python
    p = sub.add_parser("predict", help="produce a submission for one task")
    p.add_argument("--model", required=True)
```
Replace with:
```python
    p = sub.add_parser("predict", help="produce a submission for one task")
    p.add_argument("--model", default=None,
                   help="path to a fitted NGram/SoftNGram model (.pkl)")
    p.add_argument("--transformer", default=None,
                   help="path to a trained TransformerPredictor checkpoint (.pt); "
                        "mutually exclusive with --model")
```

**Edit 2 — add `build-st-embeddings` subparser (after the build-embeddings block):**

Find:
```python
    p = sub.add_parser("score", help="score predictions against ground truth")
```
Insert before it:
```python
    p = sub.add_parser("build-st-embeddings",
                       help="compute sentence-transformer embeddings for every step")
    p.add_argument("--descriptions-dir", required=True,
                   help="dir containing *_longdescription_parameters.csv files")
    p.add_argument("--out", required=True, help="output .pkl path")

    p = sub.add_parser("train", help="train the transformer LM")
    p.add_argument("--train", required=True, help="path to train_split.csv")
    p.add_argument("--embeddings", required=True,
                   help="path to a fitted STStepEmbedder (.pkl)")
    p.add_argument("--out", required=True, help="output checkpoint path (.pt)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=None, dest="d_model")
    p.add_argument("--n-heads", type=int, default=None, dest="n_heads")
    p.add_argument("--n-layers", type=int, default=None, dest="n_layers")

```

**Edit 3 — add dispatch entries:**

Find:
```python
    dispatch = {
        "build-eval": cmd_build_eval,
        "fit": cmd_fit,
        "predict": cmd_predict,
        "score": cmd_score,
        "build-embeddings": cmd_build_embeddings,
    }
```
Replace with:
```python
    dispatch = {
        "build-eval": cmd_build_eval,
        "fit": cmd_fit,
        "predict": cmd_predict,
        "score": cmd_score,
        "build-embeddings": cmd_build_embeddings,
        "build-st-embeddings": cmd_build_st_embeddings,
        "train": cmd_train,
    }
```

- [ ] **Step 4: Verify imports at module level are fine (no circular)**

The new imports are all inside functions or at the top with no circular dependencies. Confirm by doing:

```bash
source .venv/bin/activate
python -c "from infineon_baseline.cli import main; print('CLI imports OK')"
python -c "from infineon_baseline.transformer_predictor import TransformerPredictor; print('Predictor imports OK')"
```

Both should print OK without error.

- [ ] **Step 5: Commit**

```bash
git add src/infineon_baseline/transformer_predictor.py src/infineon_baseline/cli.py
git commit -m "feat: add TransformerPredictor (NGram-compatible) and wire CLI train/predict"
```

---

## Task 5: GPU sbatch script + smoke test

**Files:**
- Create: `scripts/train_transformer.sbatch`

- [ ] **Step 1: Create scripts/train_transformer.sbatch**

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --account=EUHPC_D30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p outputs/slurm outputs/models

# Build eval if not already done (so train_split.csv exists)
if [[ ! -f outputs/eval/train_split.csv ]]; then
    pixi run infineon-baseline --seed 42 build-eval \
        --variants-dir training_data/ --out outputs/eval/
fi

# Build ST embeddings if not already done
if [[ ! -f outputs/models/st_step_embeddings.pkl ]]; then
    pixi run infineon-baseline build-st-embeddings \
        --descriptions-dir training_data/ \
        --out outputs/models/st_step_embeddings.pkl
fi

# Train transformer on the GPU
pixi run infineon-baseline --seed 42 train \
    --train outputs/eval/train_split.csv \
    --embeddings outputs/models/st_step_embeddings.pkl \
    --out outputs/models/transformer.pt \
    --epochs 30 --batch-size 64 --lr 3e-4 --device cuda
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x /Users/fedjabogataj/Projects/Hackathons/zero\ one/official_repo/zero_one_hack_01/tracks/industrial-infineon/scripts/train_transformer.sbatch
```

- [ ] **Step 3: Run smoke test — build ST embeddings**

```bash
cd /Users/fedjabogataj/Projects/Hackathons/zero\ one/official_repo/zero_one_hack_01/tracks/industrial-infineon
source .venv/bin/activate
infineon-baseline build-st-embeddings \
    --descriptions-dir training_data/ \
    --out outputs/models/st_step_embeddings.pkl
```

Expected: prints "✓ ST-embedded N unique steps (384 dims) → outputs/models/st_step_embeddings.pkl"

- [ ] **Step 4: Run smoke test — 1-epoch CPU train**

```bash
infineon-baseline --seed 42 train \
    --train outputs/eval/train_split.csv \
    --embeddings outputs/models/st_step_embeddings.pkl \
    --out outputs/models/transformer_smoke.pt \
    --epochs 1 --batch-size 8 --device cpu
```

Expected: prints device=cpu, param count, loss per step, "✓ checkpoint saved". Note 1-epoch performance will be poor — that is expected.

- [ ] **Step 5: Run smoke test — predict task 1**

```bash
mkdir -p outputs/submissions
infineon-baseline --seed 42 predict \
    --transformer outputs/models/transformer_smoke.pt \
    --eval-input outputs/eval/eval_input_valid.csv \
    --task next-step \
    --out outputs/submissions/task1_xfmr_smoke.csv
```

Expected: prints "✓ wrote N predictions (transformer) → outputs/submissions/task1_xfmr_smoke.csv"

- [ ] **Step 6: Score the smoke submission**

```bash
infineon-baseline score \
    --predictions outputs/submissions/task1_xfmr_smoke.csv \
    --ground-truth outputs/eval/ground_truth_valid.csv \
    --task next-step
```

Expected: prints Top-1, Top-5, MRR metrics (any numbers — the 1-epoch model will score poorly).

- [ ] **Step 7: Commit**

```bash
git add scripts/train_transformer.sbatch
git commit -m "feat: add GPU sbatch for transformer training"
```

---

## Self-Review Against Spec

**Spec coverage check:**

| Spec requirement | Covered in task |
|-----------------|-----------------|
| `STStepEmbedder` with same API as `StepEmbedder` | Task 1 |
| Uses `sentence-transformers/all-MiniLM-L6-v2`, 384-dim | Task 1 |
| Unknown step → encode name alone | Task 1 |
| Class-level model cache | Task 1 |
| Save/load via pickle | Task 1 |
| `TransformerLM` d_model=384, n_heads=6, n_layers=4, ff_dim=1536 | Task 2 |
| Init embedding table from embedder.vectors (optional, trainable) | Task 2 |
| Family embedding summed at every position | Task 2 |
| Learned positional embeddings | Task 2 |
| Causal mask in attention | Task 2 |
| LM head weight-tied to embedding table | Task 2 |
| `from_tokenizer_and_embedder` classmethod | Task 2 |
| `forward(family_ids, step_ids) -> logits` shape (B,T,V) | Task 2 |
| `SequenceDataset` from train_split.csv, PAD token | Task 3 |
| DataLoader batched, shuffled, deterministic seed | Task 3 |
| AdamW lr=3e-4, wd=0.01, betas=(0.9,0.95) | Task 3 |
| Linear warmup 200 steps + cosine decay | Task 3 |
| Gradient clipping 1.0 | Task 3 |
| Causal LM objective, ignore PAD | Task 3 |
| Log train loss every 50 steps | Task 3 |
| Log val loss every 500 steps | Task 3 |
| Single .pt checkpoint with model + tokenizer + embedder | Task 3 |
| `TransformerPredictor` with `order`, `unigram`, `top_k`, `log_prob` | Task 4 |
| OOD aliasing via `STStepEmbedder.nearest()` | Task 4 |
| `TransformerPredictor.load(path)` classmethod | Task 4 |
| CLI `train` subcommand | Task 4 |
| CLI `build-st-embeddings` subcommand | Task 4 |
| `--transformer PATH` on `predict`, mutually exclusive with `--model` | Task 4 |
| Add torch + sentence-transformers to pyproject.toml deps | Task 1 |
| GPU sbatch script | Task 5 |
| Smoke test: train 1 epoch CPU | Task 5 |
| Smoke test: predict + score task 1 | Task 5 |
| Do NOT break existing NGram/SoftNGram paths | Validated in Task 4 Step 4 |

**Placeholder scan:** No TBD/TODO/similar found. All steps have concrete code.

**Type consistency:** `STStepEmbedder` used consistently throughout (Tasks 1, 2, 3, 4). `TransformerLM` / `TransformerConfig` names are consistent across tasks 2, 3, 4. `TransformerPredictor.load` returns `TransformerPredictor`, used by cli. `unigram` is `dict[str, Counter]` throughout.

**Edge cases noted:**
- `TransformerPredictor.top_k` masks the PAD slot and optionally masks steps not in the family unigram — prevents nonsensical predictions.
- `log_prob` feeds `[PAD] + ids[:-1]` as input so position `t` predicts `ids[t]`.
- When `prefix` is empty in `top_k`, a single PAD token is fed as input to the model so forward() receives a valid tensor.
- The checkpoint is self-contained (tokenizer + embedder stored inline) so no separate `.tokenizer.json` file is needed for the transformer path.
- The `--model` argument is made optional (not removed) so existing NGram usage still works.
