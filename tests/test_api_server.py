"""OpenAI-compatible server tests, driven over real HTTP against the CPU fixture model.

The server is the thing the benchmark client actually measures, so a bug here is a bug
in every number the project reports. The two that matter most: streaming must emit one
chunk per token (or TTFT and ITL are fiction), and requests must be served
concurrently by the background engine loop rather than serialised by the handlers.
"""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from paged_inference_engine.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig  # noqa: E402
from paged_inference_engine.engine import LLMEngine  # noqa: E402
from paged_inference_engine.entrypoints.api_server import build_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    cfg = EngineConfig(
        model=ModelConfig(model="__random__", dtype="float32", max_model_len=512,
                          hidden_size=128, num_layers=2, num_heads=4, num_kv_heads=2,
                          head_dim=32, intermediate_size=256, vocab_size=1000),
        cache=CacheConfig(block_size=16, num_gpu_blocks=256),
