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


def make_engine(**overrides) -> LLMEngine:
    cfg = EngineConfig(
        model=ModelConfig(model="__random__", dtype="float32", max_model_len=512,
                          hidden_size=128, num_layers=2, num_heads=4, num_kv_heads=2,
                          head_dim=32, intermediate_size=256, vocab_size=1000, seed=0),
        cache=CacheConfig(block_size=overrides.pop("block_size", 16),
                          num_gpu_blocks=overrides.pop("num_gpu_blocks", 256),
                          enable_prefix_caching=overrides.pop("prefix_caching", False)),
        scheduler=SchedulerConfig(
            max_num_batched_tokens=overrides.pop("max_num_batched_tokens", 128),
            max_num_seqs=overrides.pop("max_num_seqs", 8),
            enable_chunked_prefill=overrides.pop("chunked_prefill", True)),
        attention_backend="naive", device="cpu",
    )
    return LLMEngine(cfg)


PROMPTS = [
    [5, 6, 7, 8, 9, 10, 11, 12, 13],
