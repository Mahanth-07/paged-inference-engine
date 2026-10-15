"""Async load generator. Points at ANY OpenAI-compatible server, so the exact same
client benchmarks paged-inference-engine, vLLM and TGI. That is what makes the comparison fair.

    python -m bench.client --backend paged-inference-engine --port 8000 \
        --dataset sharegpt --num-prompts 500 --request-rate 8 --out results/ns_r8.json

Methodology rules that make the numbers defensible. Skipping any of them turns the
result into a number nobody should believe, including you:

  * Poisson arrivals, not a fixed batch. A fixed batch measures a different system --
    one with no queueing -- and hides exactly the tail behaviour that matters.
  * ignore_eos + a fixed output length, so every system generates the same token count.
    Otherwise you are comparing generation lengths, not engines.
  * Discard warmup requests (first-request compile/alloc costs are real but not the
    steady state you are reporting).
  * Same seed, same prompts, same max-model-len, same dtype across systems.
  * >= 3 runs; report median and spread.
  * Check the CLIENT is not the bottleneck -- see --check-client-capacity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from paged_inference_engine.utils.metrics import percentile


@dataclass
class RequestResult:
    success: bool
    prompt_len: int
    output_len: int
    ttft: float = 0.0
    latency: float = 0.0
    itls: list[float] = field(default_factory=list)
    error: str = ""


# ---- datasets -------------------------------------------------------------


def load_sharegpt(path: str, num_prompts: int, tokenizer, seed: int = 0,
                  fixed_output_len: Optional[int] = None):
    """ShareGPT is what vLLM's own benchmarks use, so the numbers are comparable.

