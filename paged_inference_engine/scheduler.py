"""Iteration-level scheduler: the thing that makes this "continuous batching".

Static batching (what HF `generate` does): pick N prompts, run until the LONGEST one
finishes, then start the next batch. Every sequence that finished early wastes its slot
for the rest of the batch. At realistic output-length variance this wastes most of the GPU.

Continuous batching: reschedule every single forward pass. A sequence that emits EOS is
evicted immediately and a waiting request takes its slot on the next step.

This file is pure Python and torch-free on purpose -- drive it with `bench/scheduler_sim.py`
on your laptop to tune policy for free.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .block_manager import AllocStatus, BlockManager
from .config import SchedulerConfig
from .sequence import Sequence, SequenceStatus


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_tokens: int
    """How many tokens of this sequence run in THIS step.
    Decode: always 1. Prefill: the whole prompt, or a chunk of it under chunked prefill."""


@dataclass
class SchedulerOutput:
    prefills: list[ScheduledSeq] = field(default_factory=list)
    decodes: list[ScheduledSeq] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    blocks_to_copy: list[tuple[int, int]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes

    @property
    def num_batched_tokens(self) -> int:
        return sum(s.num_tokens for s in self.prefills) + sum(s.num_tokens for s in self.decodes)

    @property
    def all_seqs(self) -> list[Sequence]:
        return [s.seq for s in self.prefills] + [s.seq for s in self.decodes]
