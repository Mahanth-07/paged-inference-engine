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
