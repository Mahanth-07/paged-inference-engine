"""The correctness oracle: a single-sequence generator with NO cache at all.

Recomputes the prefix for each generated token. It shares model weights with the
paged path but does not use its block tables or cache, so it helps isolate cache
bookkeeping errors from model errors. This path is intended for tests.

    ref = ReferenceGenerator(runner.model, device, dtype)
    ids = ref.greedy(prompt_ids, max_tokens=64)
"""

from __future__ import annotations

import torch

from ..model_runner import ModelInput


class ReferenceGenerator:
    def __init__(self, model, device, dtype=None):
        self.model = model
        self.device = torch.device(device)
        self.dtype = dtype

    @torch.inference_mode()
    def forward(self, token_ids: list[int]) -> torch.Tensor:
        """Logits for every position of `token_ids`. (len, vocab)."""
        n = len(token_ids)
        ids = torch.tensor(token_ids, dtype=torch.int64, device=self.device)
        pos = torch.arange(n, dtype=torch.int64, device=self.device)
        mi = ModelInput(
            input_ids=ids, positions=pos,
            slot_mapping=torch.empty(0, dtype=torch.int64, device=self.device),
        )
        # kv_caches=None routes PagedAttention to its cacheless path: plain causal SDPA
        # over the K/V computed in this very call. No block table is involved.
        hidden = self.model(ids, pos, None, mi)
        return self.model.compute_logits(hidden, None)

    @torch.inference_mode()
    def greedy(self, prompt_ids: list[int], max_tokens: int,
               eos_token_id: int | None = None) -> list[int]:
        ids = list(prompt_ids)
        out: list[int] = []
        for _ in range(max_tokens):
            nxt = int(self.forward(ids)[-1].argmax())
            ids.append(nxt)
            out.append(nxt)
            if eos_token_id is not None and nxt == eos_token_id:
                break
        return out
