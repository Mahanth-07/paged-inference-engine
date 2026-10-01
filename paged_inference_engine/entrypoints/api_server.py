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

