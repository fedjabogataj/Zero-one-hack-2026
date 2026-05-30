"""NGram-compatible wrapper around a trained TransformerLM.

Exposes the same attributes and methods as `NGram` so
`predictor.run_task1/2/3` work without modification:
  - model.order       (int)
  - model.unigram     (dict[str, Counter])
  - model.top_k(family, prefix, k) -> list[int]
  - model.log_prob(family, ids) -> float

OOD step aliasing: when a step name is not in the tokenizer vocabulary,
we use STStepEmbedder.nearest() to map it to the closest known step,
mirroring the SoftNGram.alias_step_name() behaviour.
"""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F


class TransformerPredictor:
    """Wraps TransformerLM with the NGram interface for use in run_task*."""

    order: int          # set to max_seq_len so run_task1 passes the full prefix
    unigram: dict[str, Counter]

    def __init__(
        self,
        model,          # TransformerLM (already on device, eval mode)
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
        # order = max_seq_len so that run_task1 slices prefix[-(order-1):]
        # which will just be the full prefix for any real sequence.
        self.order = model.config.max_seq_len

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> "TransformerPredictor":
        """Load a self-contained checkpoint produced by `train.py`."""
        from infineon_baseline.tokenizer import Tokenizer
        from infineon_baseline.embeddings_st import STStepEmbedder
        from infineon_baseline.transformer_model import TransformerLM, TransformerConfig

        if device == "auto":
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            _device = torch.device(device)

        ckpt = torch.load(path, map_location=_device, weights_only=False)

        # Reconstruct tokenizer from embedded data.
        tok_data = ckpt["tokenizer_data"]
        id_to_step = list(tok_data["id_to_step"])
        tokenizer = Tokenizer(
            step_to_id={s: i for i, s in enumerate(id_to_step)},
            id_to_step=id_to_step,
            family_to_id=dict(tok_data["family_to_id"]),
        )

        # Reconstruct embedder from embedded data.
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
        """Map a step name to a token id, using nearest-neighbor for OOD names."""
        sid = self._tokenizer.step_to_id.get(step)
        if sid is not None:
            return sid
        nearest = self._embedder.nearest(step, k=1, exclude_self=False)
        if nearest:
            return self._tokenizer.step_to_id.get(nearest[0][0], 0)
        return 0   # last-resort fallback: token 0

    def _encode_prefix(self, family: str, prefix: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (family_id_tensor, step_ids_tensor) for a single inference call."""
        fid = self._tokenizer.family_to_id.get(family, 0)
        family_t = torch.tensor([fid], dtype=torch.long, device=self._device)
        if len(prefix) == 0:
            # Empty prefix: feed a single PAD token so forward() gets valid input.
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

        Forwards the prefix through the model, takes logits at the last
        position, masks out the PAD slot, and returns top-k step ids.
        """
        self._model.eval()
        family_t, ids_t = self._encode_prefix(family, prefix)
        logits = self._model(family_t, ids_t)  # (1, T, V+1)
        last_logits = logits[0, -1, :].clone()  # (V+1,) — clone before in-place ops

        # Mask out PAD slot.
        pad_id = self._model.config.pad_id
        last_logits[pad_id] = float("-inf")

        # Optionally restrict to steps seen in this family (same as NGram mask_family_vocab).
        if family in self.unigram:
            allowed = set(self.unigram[family].keys())
            mask = torch.ones(pad_id, dtype=torch.bool, device=self._device)
            for sid in allowed:
                mask[sid] = False
            last_logits[:pad_id][mask] = float("-inf")

        top_ids = torch.topk(last_logits, k=min(k, pad_id), dim=-1).indices.tolist()
        return top_ids

    @torch.no_grad()
    def log_prob(self, family: str, ids: list[int]) -> float:
        """Sum of log P(id_i | prefix_i) under the model (teacher-forcing).

        Feed [PAD] + ids[:-1] as input so that position t predicts ids[t].
        """
        if not ids:
            return 0.0
        self._model.eval()
        fid = self._tokenizer.family_to_id.get(family, 0)
        family_t = torch.tensor([fid], dtype=torch.long, device=self._device)
        max_len = self._model.config.max_seq_len
        ids_clipped = ids[:max_len]

        # Build input: prepend PAD so index 0 predicts ids[0].
        pad_id = self._model.config.pad_id
        inp = [pad_id] + ids_clipped[:-1]
        inp_t = torch.tensor([inp], dtype=torch.long, device=self._device)

        logits = self._model(family_t, inp_t)        # (1, T, V+1)
        log_probs = F.log_softmax(logits[0], dim=-1)  # (T, V+1)
        total = 0.0
        for t, next_id in enumerate(ids_clipped):
            total += log_probs[t, next_id].item()
        return total
