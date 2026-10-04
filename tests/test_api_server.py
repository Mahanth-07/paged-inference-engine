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
        "temperature": 0, "ignore_eos": True, "stream": True})
    chunks = sse_chunks(r.text)
    assert chunks[-1] is None, "stream did not terminate with [DONE]"
    data = [c for c in chunks if c]
    assert len(data) == 6, f"expected 6 token chunks, got {len(data)}"
    assert data[-1]["choices"][0]["finish_reason"] == "length"
    assert all(c["choices"][0]["finish_reason"] is None for c in data[:-1])


def test_chat_completions_shape(client):
    r = client.post("/v1/chat/completions", json={
        "prompt_token_ids": [1, 2, 3], "max_tokens": 4,
        "temperature": 0, "ignore_eos": True})
    msg = r.json()["choices"][0]["message"]
    assert msg["role"] == "assistant"


def test_greedy_is_deterministic_across_requests(client):
    body = {"prompt_token_ids": [11, 12, 13, 14], "max_tokens": 8,
            "temperature": 0, "ignore_eos": True}
    a = client.post("/v1/completions", json=body).json()["choices"][0]["text"]
    b = client.post("/v1/completions", json=body).json()["choices"][0]["text"]
    assert a == b


def test_concurrent_requests_are_batched_not_serialised(client):
    """Fire several requests at once and check the engine actually ran them together.

    mean_batch_size > 1 is the observable difference between continuous batching and
    a server that has accidentally serialised itself in the request handler.
    """
    import concurrent.futures as cf

    def body(i):
        return {"prompt_token_ids": [i, i + 1, i + 2, i + 3], "max_tokens": 24,
                "temperature": 0, "ignore_eos": True}

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(
            lambda i: client.post("/v1/completions", json=body(i)).status_code, range(6)))
    assert results == [200] * 6
    assert client.get("/metrics").json()["mean_batch_size"] > 1.0


def test_metrics_endpoint_reports_the_standard_fields(client):
    m = client.get("/metrics").json()
    for key in ("p99_ttft_ms", "median_tpot_ms", "p99_itl_ms",
                "output_throughput_tok_s", "kv_cache_utilization", "num_gpu_blocks"):
        assert key in m, f"missing {key}"


def test_bad_request_without_tokenizer_is_a_400(client):
    r = client.post("/v1/completions", json={"prompt": "text", "max_tokens": 4})
    assert r.status_code == 400
