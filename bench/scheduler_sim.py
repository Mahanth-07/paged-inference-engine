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


