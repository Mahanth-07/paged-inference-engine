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
