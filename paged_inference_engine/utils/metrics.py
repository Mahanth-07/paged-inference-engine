"""Metric collection. These definitions are the field standard -- match them exactly
so your numbers are comparable to vLLM's published benchmarks.

    TTFT  time to first token          = first_token_time - arrival_time
    TPOT  time per output token        = (finish_time - first_token_time) / (output_len - 1)
    ITL   inter-token latency          = the full distribution of gaps between tokens
    E2EL  end-to-end latency           = finish_time - arrival_time

Report percentiles, not just means. A mean ITL hides exactly the stalls that chunked
prefill is designed to fix, and p99 is what a user actually experiences.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional

from ..sequence import Sequence


@dataclass
class RequestMetrics:
    seq_id: int
    prompt_len: int
    output_len: int
    arrival_time: float
    first_token_time: Optional[float]
    finish_time: Optional[float]
    num_preemptions: int
    itls: list[float] = field(default_factory=list)

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def e2el(self) -> Optional[float]:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    @property
    def tpot(self) -> Optional[float]:
        if self.finish_time is None or self.first_token_time is None or self.output_len < 2:
            return None
        return (self.finish_time - self.first_token_time) / (self.output_len - 1)


@dataclass
class StepMetrics:
    num_batched_tokens: int
    num_prefill_seqs: int
    num_decode_seqs: int
    step_latency: float
    kv_utilization: float
    num_preempted: int


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class MetricsCollector:
    def __init__(self) -> None:
        self.requests: list[RequestMetrics] = []
        self.steps: list[StepMetrics] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def record_step(self, **kwargs) -> None:
        self.steps.append(StepMetrics(**kwargs))

    def record_request(self, seq: Sequence) -> None:
        itls = [b - a for a, b in zip(seq.token_times, seq.token_times[1:])]
        self.requests.append(
            RequestMetrics(
                seq_id=seq.seq_id,
                prompt_len=seq.prompt_len,
                output_len=seq.output_len,
                arrival_time=seq.arrival_time,
                first_token_time=seq.first_token_time,
                finish_time=seq.finish_time,
                num_preemptions=seq.num_preemptions,
                itls=itls,
            )
        )

    # ---- summary ----------------------------------------------------------

    def summary(self, duration: Optional[float] = None) -> dict:
        reqs = self.requests
        if not reqs:
            return {}
        if duration is None:
            duration = max(r.finish_time or 0 for r in reqs) - min(r.arrival_time for r in reqs)
        duration = max(duration, 1e-9)

        ttfts = [r.ttft for r in reqs if r.ttft is not None]
        tpots = [r.tpot for r in reqs if r.tpot is not None]
        e2els = [r.e2el for r in reqs if r.e2el is not None]
        itls = [x for r in reqs for x in r.itls]
