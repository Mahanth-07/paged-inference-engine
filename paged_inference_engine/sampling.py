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
