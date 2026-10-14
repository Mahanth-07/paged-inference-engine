"""CPU scheduler simulator with a configurable model-latency estimate.

It drives the real Scheduler and the real BlockManager with a fake model runner whose
latency comes from a cost model. That means you can tune scheduling policy, block size,
token budgets, and chunked-prefill settings against realistic traces in seconds instead
of burning GPU-hours on a rented box.

    python -m bench.scheduler_sim --num-requests 500 --request-rate 8

Sanity-check the cost model against real measurements once you have a GPU engine
working, then trust it for policy exploration.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

from paged_inference_engine.block_manager import BlockManager
from paged_inference_engine.config import SchedulerConfig
from paged_inference_engine.sampling import SamplingParams
from paged_inference_engine.scheduler import Scheduler
from paged_inference_engine.sequence import Sequence
from paged_inference_engine.utils.metrics import MetricsCollector


@dataclass
class CostModel:
    """Crude but directionally correct. Calibrate against a real box in week 4.

    prefill: compute-bound, ~linear in tokens
    decode:  memory-bound, ~constant per step (weight sweep) + small per-seq term
             (KV sweep) -- which is exactly why batching helps decode so much
    """

    prefill_us_per_token: float = 22.0
    decode_base_us: float = 3800.0
    decode_us_per_seq: float = 21.0
    decode_us_per_kv_token: float = 0.004

    def step_time(self, prefill_tokens: int, decode_seqs: int, decode_kv_tokens: int) -> float:
        t = self.prefill_us_per_token * prefill_tokens
        if decode_seqs:
            t += (
                self.decode_base_us
                + self.decode_us_per_seq * decode_seqs
                + self.decode_us_per_kv_token * decode_kv_tokens
