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


class Scheduler:
    def __init__(self, config: SchedulerConfig, block_manager: BlockManager):
        self.config = config
        self.block_manager = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.ignored: list[Sequence] = []
        """Requests rejected at admission. The engine drains this every step and emits
        a finished output for each, so a caller gets an answer instead of hanging on a
        request the scheduler quietly dropped."""

        # Cumulative scheduling counters.
        self.num_preemptions = 0
        self.num_rejected = 0
        self.num_steps = 0

    # ---- queue management -------------------------------------------------

    def add_seq(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def abort(self, seq_id: int) -> None:
        for seq in list(self.waiting):
            if seq.seq_id == seq_id:
                self.waiting.remove(seq)
                seq.status = SequenceStatus.FINISHED_ABORTED
                return
        for seq in list(self.running):
            if seq.seq_id == seq_id:
                self.running.remove(seq)
                self.block_manager.free(seq)
                seq.status = SequenceStatus.FINISHED_ABORTED
                return

    @property
    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running or self.ignored)

    def pop_ignored(self) -> list[Sequence]:
        out, self.ignored = self.ignored, []
        return out

    def _reject(self, seq: Sequence, reason: str) -> None:
        seq.status = SequenceStatus.FINISHED_IGNORED
        seq.finish_reason_detail = reason
        self.ignored.append(seq)
        self.num_rejected += 1

    # ---- the main event ---------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        """Decide what runs this step.

        Order matters and is a real policy choice:
          1. Decodes first. Already-running sequences have already paid TTFT; starving
             them spikes ITL, which users feel more than a slightly worse TTFT.
          2. Preempt from the TAIL of running (newest first) when the cache is full.
             Preempting the oldest would waste the most work.
          3. Fill the leftover token budget with prefills.

        Decode priority trades some prefill latency for steadier inter-token latency.
        """
        out = SchedulerOutput()
        self.num_steps += 1
        token_budget = self.config.max_num_batched_tokens

        # --- 1. decodes, with preemption on memory pressure ---
        running_queue = deque(self.running)
        surviving: list[Sequence] = []

        while running_queue:
            seq = running_queue.popleft()
            if seq.is_prefill:
                # Partially-prefilled sequence from a previous chunked-prefill step.
                surviving.append(seq)
                continue

            while not self.block_manager.can_append_slot(seq):
                # Not enough memory. Evict someone -- the newest running sequence,
                # or this one if it's the only candidate left.
                if running_queue:
                    victim = running_queue.pop()
                else:
                    victim = seq
                self._preempt(victim, out)
                if victim is seq:
                    break
            else:
                cow = self.block_manager.append_slot(seq)
                if cow is not None:
                    out.blocks_to_copy.append(cow)
                out.decodes.append(ScheduledSeq(seq, 1))
                surviving.append(seq)
                token_budget -= 1

        self.running = surviving

        # --- 2. resume partially-prefilled sequences before admitting new ones ---
        for seq in self.running:
            if seq.is_prefill and token_budget > 0:
                chunk = self._prefill_chunk_size(seq, token_budget)
                if chunk > 0 and self._reserve_output_slot(seq, chunk, out):
                    out.prefills.append(ScheduledSeq(seq, chunk))
                    token_budget -= chunk

        # --- 3. admit new requests from the waiting queue ---
        while self.waiting and token_budget > 0:
            if len(self.running) >= self.config.max_num_seqs:
                break
            seq = self.waiting[0]

            # Without chunked prefill, a prompt wider than the WHOLE token budget can
            # never be scheduled, and under FCFS it blocks the queue behind it forever.
            # (Reproduce the stall by removing this: `python -m bench.scheduler_sim
            # --max-num-batched-tokens 1024 --no-chunked-prefill`.) Real engines reject
            # at admission; a clear error beats a silent hang.
            if (not self.config.enable_chunked_prefill
                    and seq.num_uncomputed_tokens > self.config.max_num_batched_tokens):
                self.waiting.popleft()
                self._reject(seq, "prompt exceeds max_num_batched_tokens and chunked "
                                  "prefill is disabled")
                continue

            status = self.block_manager.can_allocate(seq)
            if status == AllocStatus.NEVER:
                self.waiting.popleft()
                self._reject(seq, "sequence can never fit in the KV cache")
                continue
            if status == AllocStatus.LATER:
                break  # FCFS: do not skip ahead, that's how you get starvation

            self.waiting.popleft()
            # Allocate BEFORE sizing the chunk: with prefix caching, allocate() may
            # hand back blocks that are already computed and advance
            # seq.num_computed_tokens, which shrinks the prefill this step needs.
            self.block_manager.allocate(seq)
            chunk = self._prefill_chunk_size(seq, token_budget)
            if chunk <= 0:
                self.block_manager.free(seq)
                seq.num_computed_tokens = 0
                self.waiting.appendleft(seq)
                break
            if not self._reserve_output_slot(seq, chunk, out):
                # Shouldn't happen -- can_allocate budgets for the +1 -- but if the
                # watermark is misconfigured, roll back rather than corrupt the cache.
                self.block_manager.free(seq)
                seq.num_computed_tokens = 0
                self.waiting.appendleft(seq)
                break
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            out.prefills.append(ScheduledSeq(seq, chunk))
            token_budget -= chunk

        return out

    def _reserve_output_slot(self, seq: Sequence, chunk: int, out: SchedulerOutput) -> bool:
        """A prefill chunk that COMPLETES the sequence's uncomputed tokens also emits a
        token this step, so it needs cache room for that token. A partial chunk doesn't.

        Forgetting this is a fun bug: it only fires when a prompt length lands exactly on
        a block boundary and the cache is nearly full.
        """
