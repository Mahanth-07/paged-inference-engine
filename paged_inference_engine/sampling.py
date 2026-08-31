"""Sampling parameters and the sampler itself.

The sampler is fully batched: temperature/top-p/top-k are built as tensors once per
step and applied to the whole (num_seqs, vocab) logits matrix. A per-sequence Python
loop here silently becomes the bottleneck at batch 256 -- it is an embarrassing thing
to find in your own profile, and it is easy to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence as _Seq


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 128
    min_tokens: int = 0
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False
    """Benchmarks MUST set this True so every system generates the same token count."""
    seed: Optional[int] = None

    @property
    def greedy(self) -> bool:
        return self.temperature < 1e-5


class Sampler:
    """Stateful only in its RNG, so a step is reproducible given a seed.

    Kept as a class (rather than a bare function) so the per-step parameter tensors
    can be cached: at steady state the same sequences are decoding with the same
    params step after step, and rebuilding three float tensors per step for 256 rows
    is pure overhead.
    """

    def __init__(self, device: str = "cpu", seed: int = 0):
        import torch

        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self._cache_key: Optional[tuple] = None
        self._cache_tensors: Optional[tuple] = None

    def _params_tensors(self, batch: _Seq[SamplingParams]):
        import torch

        key = tuple((p.temperature, p.top_p, p.top_k) for p in batch)
        if key == self._cache_key:
            return self._cache_tensors
        temps = torch.tensor([p.temperature for p in batch], dtype=torch.float32,
                             device=self.device)
        top_ps = torch.tensor([p.top_p for p in batch], dtype=torch.float32,
                              device=self.device)
        top_ks = torch.tensor([p.top_k for p in batch], dtype=torch.int64,
                              device=self.device)
        greedy = temps < 1e-5
        tensors = (temps, top_ps, top_ks, greedy)
        self._cache_key, self._cache_tensors = key, tensors
        return tensors

    def __call__(self, logits, batch: _Seq[SamplingParams]):
        import torch

        assert logits.shape[0] == len(batch), "logits rows must align with sampling params"
        logits = logits.float()
        greedy_ids = logits.argmax(dim=-1)

        # These predicates come from the Python params, NOT from a tensor reduction.
        # `bool(some_tensor.all())` is a device sync, and there would be three of them
        # in this function, every step, for a decision the host already knows.
        if all(p.greedy for p in batch):
            return greedy_ids     # fast path: benchmarks are greedy; skip all masking
        any_top_k = any(p.top_k > 0 for p in batch)
        any_top_p = any(p.top_p < 1.0 for p in batch)

        temps, top_ps, top_ks, greedy = self._params_tensors(batch)

        # temperature (guard the greedy rows against a divide-by-zero)
        safe_t = torch.where(greedy, torch.ones_like(temps), temps)
        logits = logits / safe_t.unsqueeze(1)

        # top-k: one batched topk at the largest k in the batch, then threshold per row
        if any_top_k:
            vocab = logits.shape[-1]
            k = min(vocab, max(p.top_k for p in batch))
            kth = logits.topk(k, dim=-1).values                       # (n, k) descending
            row_k = torch.where(top_ks > 0, torch.clamp(top_ks, max=k), k) - 1
            thresh = kth.gather(1, row_k.unsqueeze(1))
            logits = logits.masked_fill(logits < thresh, float("-inf"))

        # top-p: sort desc, cumulative probability, drop the tail beyond p
        if any_top_p:
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            probs = sorted_logits.softmax(dim=-1)
            cum = probs.cumsum(dim=-1)
            # keep the token that crosses p; `cum - probs` is the mass strictly before it
            drop = (cum - probs) > top_ps.unsqueeze(1)
            sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
            logits = torch.empty_like(sorted_logits).scatter_(1, sorted_idx, sorted_logits)

        probs = logits.softmax(dim=-1)
        sampled = torch.multinomial(probs, num_samples=1, generator=self.generator).squeeze(1)
        return torch.where(greedy, greedy_ids, sampled)


def sample(logits, sampling_params_batch, generator=None):
    """Functional entry point, mostly for tests. The engine holds a Sampler."""
    import torch

    sampler = Sampler(device=str(logits.device))
    if generator is not None:
        sampler.generator = generator
    with torch.no_grad():
        return sampler(logits, sampling_params_batch)
