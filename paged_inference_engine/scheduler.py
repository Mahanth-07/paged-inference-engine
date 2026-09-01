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
