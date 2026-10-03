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
        scheduler=SchedulerConfig(max_num_batched_tokens=128, max_num_seqs=8),
        attention_backend="naive", device="cpu",
    )
    with TestClient(build_app(LLMEngine(cfg))) as c:
        yield c


def sse_chunks(text):
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block.startswith("data:"):
            continue
        payload = block[5:].strip()
        if payload == "[DONE]":
            out.append(None)
        else:
            out.append(json.loads(payload))
    return out


def test_health_and_models(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v1/models").json()["data"][0]["object"] == "model"


def test_completion_generates_exactly_max_tokens(client):
    r = client.post("/v1/completions", json={
        "prompt_token_ids": [5, 6, 7, 8, 9], "max_tokens": 7,
        "temperature": 0, "ignore_eos": True})
    assert r.status_code == 200
    body = r.json()
    assert body["usage"]["completion_tokens"] == 7
    assert body["choices"][0]["finish_reason"] == "length"


def test_streaming_emits_one_chunk_per_token(client):
    """If the server buffers, TTFT and ITL are measuring the wrong thing entirely."""
    r = client.post("/v1/completions", json={
        "prompt_token_ids": [5, 6, 7, 8, 9], "max_tokens": 6,
