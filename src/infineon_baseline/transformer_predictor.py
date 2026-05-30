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
        tokenizer,      # Tokenizer or SubwordTokenizer
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
        # Detect subword mode from tokenizer duck-type.
        self._is_subword: bool = hasattr(tokenizer, "sep_id")

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> "TransformerPredictor":
        """Load a self-contained checkpoint produced by `train.py`.

        Reads `tokenizer_kind` from the checkpoint (defaults to "flat" for
        backward compatibility with old checkpoints) and reconstructs the
        appropriate tokenizer class.
        """
        from infineon_baseline.tokenizer import Tokenizer
        from infineon_baseline.subword_tokenizer import SubwordTokenizer
        from infineon_baseline.embeddings_st import STStepEmbedder
        from infineon_baseline.transformer_model import TransformerLM, TransformerConfig

        if device == "auto":
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            _device = torch.device(device)

        ckpt = torch.load(path, map_location=_device, weights_only=False)

        tokenizer_kind = ckpt.get("tokenizer_kind", "flat")
        tok_data = ckpt["tokenizer_data"]

        if tokenizer_kind == "subword":
            id_to_token = list(tok_data["id_to_token"])
            tokenizer = SubwordTokenizer(
                token_to_id={t: i for i, t in enumerate(id_to_token)},
                id_to_token=id_to_token,
                family_to_id=dict(tok_data["family_to_id"]),
            )
        else:
            # Flat tokenizer (default / backward compat).
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
        instance = cls(model=model, tokenizer=tokenizer, embedder=embedder,
                       unigram=unigram, device=_device)
        instance._is_subword = (tokenizer_kind == "subword")
        return instance

    # ─── OOD aliasing (flat mode only) ────────────────────────────────── #

    def _alias_step(self, step: str) -> int:
        """Map a step name to a token id, using nearest-neighbor for OOD names.

        Only used in flat tokenizer mode. In subword mode, OOD handling is
        implicit: unknown subwords map to <unk>.
        """
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

    # ─── Subword-mode methods ─────────────────────────────────────────── #

    @torch.no_grad()
    def top_k_steps(self, family: str, partial_steps: list[str], k: int = 5) -> list[str]:
        """Beam search to find the k most-likely next step strings (subword mode).

        Generates subword tokens from the prefix until <sep> or <eos> is emitted.
        Returns up to k decoded step strings sorted by cumulative log-probability.
        """
        self._model.eval()
        tok = self._tokenizer
        fid, prefix_ids = tok.encode_prefix(family, partial_steps)

        beam_size = k
        max_step_len = 16   # safety cap: max subwords per step

        # Each beam: (full_input_ids, generated_step_ids, cumulative_log_prob)
        beams: list[tuple[list[int], list[int], float]] = [(list(prefix_ids), [], 0.0)]
        finished: list[tuple[list[int], float]] = []

        for _ in range(max_step_len):
            if not beams:
                break
            candidates: list[tuple[list[int], list[int], float]] = []
            for input_ids, gen, lp in beams:
                # If the last generated token is already a step boundary, record as done.
                if gen and gen[-1] in (tok.sep_id, tok.eos_id):
                    finished.append((gen[:-1], lp))
                    continue

                max_len = self._model.config.max_seq_len - 1
                ids_tensor = torch.tensor(
                    [input_ids[-max_len:]], dtype=torch.long, device=self._device,
                )
                family_t = torch.tensor([fid], dtype=torch.long, device=self._device)
                logits = self._model(family_t, ids_tensor)   # (1, T, V+1)
                last_logits = logits[0, -1, :]               # (V+1,)
                log_prob_vec = F.log_softmax(last_logits, dim=-1)

                topv, topi = log_prob_vec.topk(beam_size + 1)
                for v, i in zip(topv.tolist(), topi.tolist()):
                    new_gen = gen + [i]
                    # Terminate immediately if eos or sep emitted.
                    if i in (tok.sep_id, tok.eos_id):
                        finished.append((gen, lp + v))
                    else:
                        candidates.append((input_ids + [i], new_gen, lp + v))

            # Keep top beam_size candidates by log-prob.
            candidates.sort(key=lambda x: -x[2])
            beams = candidates[:beam_size]

            if len(finished) >= beam_size:
                break

        # Stragglers: if we never hit a boundary within max_step_len, take what we have.
        for _input_ids, gen, lp in beams:
            if gen:
                finished.append((gen, lp))

        # Sort by log-prob (descending), decode, deduplicate while preserving order.
        finished.sort(key=lambda x: -x[1])
        seen: set[str] = set()
        top_steps: list[str] = []
        for gen, _ in finished:
            s = tok.decode_step(gen)
            if s and s not in seen:
                seen.add(s)
                top_steps.append(s)
            if len(top_steps) >= k:
                break
        return top_steps

    @torch.no_grad()
    def complete_steps(
        self, family: str, partial_steps: list[str], max_steps: int = 200,
    ) -> list[str]:
        """Greedy autoregressive completion from the subword prefix (subword mode).

        Generates subword tokens one at a time, accumulates full steps from
        <sep> boundaries, and stops at <eos> or when max_steps steps are emitted.
        """
        self._model.eval()
        tok = self._tokenizer
        fid, prefix_ids = tok.encode_prefix(family, partial_steps)
        input_ids = list(prefix_ids)
        completed_steps: list[str] = []
        current_step_ids: list[int] = []
        max_total_tokens = self._model.config.max_seq_len * 2   # hard cap

        for _ in range(max_total_tokens):
            max_len = self._model.config.max_seq_len - 1
            ids_t = torch.tensor(
                [input_ids[-max_len:]], dtype=torch.long, device=self._device,
            )
            fam_t = torch.tensor([fid], dtype=torch.long, device=self._device)
            logits = self._model(fam_t, ids_t)
            next_id = int(logits[0, -1, :].argmax().item())
            input_ids.append(next_id)

            if next_id == tok.eos_id:
                if current_step_ids:
                    s = tok.decode_step(current_step_ids)
                    if s:
                        completed_steps.append(s)
                break
            elif next_id == tok.sep_id:
                s = tok.decode_step(current_step_ids)
                if s:
                    completed_steps.append(s)
                current_step_ids = []
                if len(completed_steps) >= max_steps:
                    break
            else:
                current_step_ids.append(next_id)

        return completed_steps

    @torch.no_grad()
    def log_prob_steps(self, family: str, steps: list[str]) -> float:
        """Teacher-forcing log-probability of a full step sequence (subword mode).

        Tokenises as encode_sequence, then sums log P(token | context) over all
        non-pad positions, mirroring the flat log_prob() interface.
        """
        if not steps:
            return 0.0
        self._model.eval()
        tok = self._tokenizer
        fid, ids = tok.encode_sequence(family, steps)
        return self.log_prob(family, ids)
