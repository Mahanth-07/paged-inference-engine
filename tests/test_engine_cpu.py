"""End-to-end engine correctness on the CPU fixture model.

These tests need no GPU and check block tables, slot
mapping, chunked prefill, preemption, cross-sequence contamination, and prefix cache
sharing. The oracle is `ReferenceGenerator`, which has no cache at all and therefore
cannot share a bug with the paged path.

Everything here is greedy, so "correct" means token-for-token identical, not close.
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from paged_inference_engine.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig  # noqa: E402
from paged_inference_engine.engine import LLMEngine  # noqa: E402
from paged_inference_engine.models.reference import ReferenceGenerator  # noqa: E402
from paged_inference_engine.sampling import SamplingParams  # noqa: E402

GREEDY = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)
