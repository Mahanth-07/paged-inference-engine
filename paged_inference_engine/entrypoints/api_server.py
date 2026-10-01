"""OpenAI-compatible server.

Exists so the SAME benchmark client can hit paged-inference-engine and vLLM. That is what makes
the comparison credible: identical client, identical arrival process, identical
timing code, only the server changes.

    python -m paged_inference_engine.entrypoints.api_server --model meta-llama/Llama-3.2-1B-Instruct \
        --port 8000 --backend flashinfer

Endpoints: /health, /v1/models, /v1/completions, /v1/chat/completions, /metrics

Architecture that matters: the engine loop runs in ONE background task calling
engine.step() continuously; HTTP handlers only enqueue a request and await their own
asyncio.Queue of output chunks. If a handler ever called engine.step() itself the whole
server would serialise and the benchmark would measure the wrong system entirely.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from typing import Any, Optional

from ..config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from ..sampling import SamplingParams


class AsyncEngine:
    """Wraps the synchronous LLMEngine in a background stepping loop."""

    def __init__(self, engine):
        self.engine = engine
        self.streams: dict[int, asyncio.Queue] = {}
        self._task: Optional[asyncio.Task] = None
        self._loop_started = asyncio.Event()

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        self._loop_started.set()
        while True:
            if not self.engine.has_unfinished_requests():
                # Yield rather than spin. asyncio.sleep(0) alone starves the accept
                # loop under load; a tiny sleep costs nothing at idle and nothing at
                # load (we only get here when there is genuinely no work).
                await asyncio.sleep(0.0005)
                continue
            outputs = self.engine.step()
            for out in outputs:
                q = self.streams.get(out.seq_id)
                if q is not None:
                    q.put_nowait(out)
            await asyncio.sleep(0)   # let the event loop drain sockets between steps

    async def generate(self, prompt_token_ids: list[int], params: SamplingParams):
        """Async generator of RequestOutput deltas for one request."""
        q: asyncio.Queue = asyncio.Queue()
        seq_id = self.engine.add_request(prompt_token_ids=prompt_token_ids,
                                         sampling_params=params)
        self.streams[seq_id] = q
        try:
            while True:
                out = await q.get()
                yield out
                if out.finished:
                    return
        finally:
            self.streams.pop(seq_id, None)


def build_app(engine):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse

    from contextlib import asynccontextmanager

    async_engine = AsyncEngine(engine)
    tokenizer = engine.tokenizer

    @asynccontextmanager
    async def lifespan(_app):
        # The engine loop is a background task for the life of the process. Handlers
        # only enqueue; if one ever called engine.step() the server would serialise.
        async_engine.start()
        yield
        await async_engine.stop()

    app = FastAPI(title="paged-inference-engine", lifespan=lifespan)
    app.state.engine = async_engine

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        name = engine.config.model.model
        return {"object": "list",
                "data": [{"id": name, "object": "model", "owned_by": "paged-inference-engine"}]}

    @app.get("/metrics")
    async def metrics():
        return JSONResponse({**engine.stats(), **engine.metrics.summary()})

    def _params(body: dict) -> SamplingParams:
        return SamplingParams(
            temperature=float(body.get("temperature", 1.0)),
            top_p=float(body.get("top_p", 1.0)),
            top_k=int(body.get("top_k", -1)),
            max_tokens=int(body.get("max_tokens", 128)),
            min_tokens=int(body.get("min_tokens", 0)),
            # The benchmark needs this: every system must generate the same token count
            # or the throughput comparison is meaningless.
            ignore_eos=bool(body.get("ignore_eos", False)),
            stop_token_ids=tuple(body.get("stop_token_ids", ()) or ()),
            seed=body.get("seed"),
        )

    def _encode(body: dict, chat: bool) -> list[int]:
        if body.get("prompt_token_ids"):
            return list(body["prompt_token_ids"])
        if tokenizer is None:
            raise HTTPException(400, "server has no tokenizer; send prompt_token_ids")
        if chat:
            return tokenizer.apply_chat_template(
                body["messages"], add_generation_prompt=True, tokenize=True)
        return tokenizer.encode(body["prompt"])

    async def _stream(req_id: str, model: str, prompt_ids: list[int],
                      params: SamplingParams, chat: bool):
        created = int(time.time())
        first = True
        async for out in async_engine.generate(prompt_ids, params):
            # Emit the moment a TOKEN exists, not the moment text exists. A token that
            # only completes half a UTF-8 codepoint still counts for ITL, and buffering
            # it would flatter both TTFT and ITL.
            if not out.new_token_ids and not out.finished:
                continue
            if chat:
                delta: dict[str, Any] = {"content": out.delta_text}
                if first:
                    delta["role"] = "assistant"
                choice = {"index": 0, "delta": delta,
                          "finish_reason": out.finish_reason if out.finished else None}
                obj = "chat.completion.chunk"
            else:
                choice = {"index": 0, "text": out.delta_text, "logprobs": None,
                          "finish_reason": out.finish_reason if out.finished else None}
                obj = "text_completion"
            first = False
            payload = {"id": req_id, "object": obj, "created": created,
                       "model": model, "choices": [choice]}
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    async def _complete(body: dict, chat: bool):
        model = body.get("model", engine.config.model.model)
        prompt_ids = _encode(body, chat)
        params = _params(body)
        req_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        if body.get("stream"):
            return StreamingResponse(
                _stream(req_id, model, prompt_ids, params, chat),
                media_type="text/event-stream",
            )
        final = None
        async for out in async_engine.generate(prompt_ids, params):
            final = out
        text = final.text if final else ""
        choice = ({"index": 0, "message": {"role": "assistant", "content": text},
                   "finish_reason": final.finish_reason if final else "stop"}
                  if chat else
                  {"index": 0, "text": text, "logprobs": None,
                   "finish_reason": final.finish_reason if final else "stop"})
        return {
            "id": req_id,
            "object": "chat.completion" if chat else "text_completion",
            "created": int(time.time()), "model": model, "choices": [choice],
            "usage": {"prompt_tokens": len(prompt_ids),
                      "completion_tokens": len(final.token_ids) if final else 0,
                      "total_tokens": len(prompt_ids) + (len(final.token_ids) if final else 0)},
        }
