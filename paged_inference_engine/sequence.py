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
    def is_prefill(self) -> bool:
        """True while this sequence still has tokens whose KV is not in the cache.

        The bound is total_len, NOT prompt_len. They are the same for a fresh request,
        but a sequence preempted by recompute has thrown away the KV for its already
        GENERATED tokens too, and those have to be recomputed before it can decode
        again. Using prompt_len here means a preempted sequence resumes decoding with
        a hole in its cache: the output stays fluent and is quietly wrong from the
        preemption point onward.
        """
        return self.num_computed_tokens < self.total_len

    def num_blocks(self, block_size: int) -> int:
        return (self.total_len + block_size - 1) // block_size

    def last_block_occupancy(self, block_size: int) -> int:
        """Tokens used in the final block. In [1, block_size]. FlashInfer calls
        this kv_last_page_len and requires it to be > 0."""
        r = self.total_len % block_size
        return r if r else block_size

    # ---- slot mapping -----------------------------------------------------

    def slot_for_token(self, token_index: int, block_size: int) -> int:
        """Flat index into the paged cache for logical position `token_index`.

        This one line is the whole idea of paging: a logical position maps to
        (block, offset) and the physical location is block_table[block] * block_size + offset.
        """
        logical_block = token_index // block_size
        offset = token_index % block_size
        return self.block_table[logical_block] * block_size + offset

    # ---- mutation ---------------------------------------------------------

    def append_token(self, token_id: int, now: float = 0.0) -> None:
        self.output_token_ids.append(token_id)
        self.num_computed_tokens += 1
        if self.first_token_time is None:
            self.first_token_time = now
        self.token_times.append(now)
