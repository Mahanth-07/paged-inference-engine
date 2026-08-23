"""Sequence state. Pure Python, fully testable on your laptop."""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import Optional

from .sampling import SamplingParams


class SequenceStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_STOPPED = enum.auto()      # hit EOS or a stop token
    FINISHED_LENGTH = enum.auto()       # hit max_tokens
    FINISHED_ABORTED = enum.auto()      # cancelled by the caller
    FINISHED_IGNORED = enum.auto()      # rejected at admission; can never be scheduled

    @property
    def is_finished(self) -> bool:
        return self.name.startswith("FINISHED")


_counter = itertools.count()


# eq=False so Sequence keeps identity hashing/equality. Two sequences with the same
# tokens are still different requests, and the engine keys dicts by sequence object.
@dataclass(eq=False)
class Sequence:
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    seq_id: int = field(default_factory=lambda: next(_counter))

    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING

    block_table: list[int] = field(default_factory=list)
    """logical block index -> physical block id. Owned by BlockManager."""

    num_computed_tokens: int = 0
    """How many tokens already have KV in the cache. With chunked prefill this
    advances in chunks during prefill; afterwards it tracks generation."""

    # Timestamps for metrics (set by the engine).
    arrival_time: float = 0.0
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    token_times: list[float] = field(default_factory=list)

    num_preemptions: int = 0
    finish_reason_detail: Optional[str] = None

    # ---- token accounting -------------------------------------------------

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_uncomputed_tokens(self) -> int:
        return self.total_len - self.num_computed_tokens

    @property
