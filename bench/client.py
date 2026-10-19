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

    Download:
      https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
      ShareGPT_V3_unfiltered_cleaned_split.json

    Its length distribution is heavy-tailed, which is the whole point: uniform lengths
    make a scheduler look better than it is, because the long-prompt stalls and the
    early-finisher waste both disappear.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download ShareGPT_V3_unfiltered_cleaned_split.json from "
            "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered "
            "or use --dataset synthetic."
        )
    with open(path) as f:
        raw = json.load(f)
    raw = [d for d in raw if len(d.get("conversations", [])) >= 2]
    rng = random.Random(seed)
    rng.shuffle(raw)

    out = []
    for d in raw:
        prompt = d["conversations"][0]["value"]
        completion = d["conversations"][1]["value"]
        p_ids = tokenizer.encode(prompt)
        c_len = len(tokenizer.encode(completion))
        if len(p_ids) < 4 or len(p_ids) > 1024:
            continue
        o_len = fixed_output_len if fixed_output_len else c_len
        if o_len < 4:
            continue
        out.append((prompt, len(p_ids), min(o_len, 2048)))
        if len(out) >= num_prompts:
            break
    if len(out) < num_prompts:
        raise ValueError(f"only {len(out)} usable prompts in {path}, wanted {num_prompts}")
    return out


def synthetic(num_prompts: int, input_len: int, output_len: int, tokenizer,
              seed: int = 0):
    """Fixed-length synthetic prompts, for clean ablations where you want to isolate
    one variable. Use ShareGPT for headline numbers -- fixed lengths make continuous
    batching look worse than it is, because there are no early finishers to reclaim."""
    rng = random.Random(seed)
    vocab = getattr(tokenizer, "vocab_size", 32000)
    out = []
    for _ in range(num_prompts):
        ids = [rng.randrange(1000, min(vocab, 20000)) for _ in range(input_len)]
        out.append((tokenizer.decode(ids), input_len, output_len))
    return out


# ---- request driver -------------------------------------------------------


async def send_openai_completion(session, url: str, model: str, prompt: str,
                                 output_len: int, prompt_len: int,
                                 sem=None) -> RequestResult:
    """POST /v1/completions with stream=True, ignore_eos=True, temperature=0.

    The subtle part is TTFT. Servers often send a first SSE chunk carrying only role or
    metadata and no token; counting it as the first token gives a flattering, wrong
    number. So we only start the clock on a chunk that actually contains generated text
    (or, for a tokens-only server, a chunk at all after the metadata one).
    """
    payload = {
        "model": model, "prompt": prompt, "max_tokens": output_len,
        "min_tokens": output_len,       # pin the count on servers that honour it
        "temperature": 0.0, "stream": True, "ignore_eos": True,
    }
    result = RequestResult(success=False, prompt_len=prompt_len, output_len=0)
    ctx = sem if sem is not None else _NullCtx()
    async with ctx:
        st = time.perf_counter()
        last = st
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    result.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                    return result
                async for raw in resp.content:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choice = chunk["choices"][0]
                    text = choice.get("text") or choice.get("delta", {}).get("content") or ""
                    if not text and choice.get("finish_reason") is None and result.output_len == 0:
                        continue          # metadata-only opening chunk
                    now = time.perf_counter()
                    if result.output_len == 0:
                        result.ttft = now - st
                    else:
                        result.itls.append(now - last)
                    last = now
                    result.output_len += 1
                result.latency = time.perf_counter() - st
                result.success = result.output_len > 0
        except Exception as e:  # noqa: BLE001 -- a failed request is data, not a crash
            result.error = f"{type(e).__name__}: {e}"
        return result


class _NullCtx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *a):
        return False


async def _arrivals(requests, rate: float):
    """Poisson arrivals. rate == inf fires everything at once (max-throughput mode)."""
    for r in requests:
        yield r
        if rate == float("inf"):
            continue
        await asyncio.sleep(random.expovariate(rate))


async def run_benchmark(args) -> dict:
    import aiohttp

    tokenizer = _get_tokenizer(args.model)
    if args.dataset == "sharegpt":
        data = load_sharegpt(args.dataset_path, args.num_prompts + args.warmup,
                             tokenizer, args.seed,
                             fixed_output_len=args.output_len if args.fixed_output else None)
    else:
        data = synthetic(args.num_prompts + args.warmup, args.input_len,
                         args.output_len, tokenizer, args.seed)

    warmup, requests = data[:args.warmup], data[args.warmup:]
    url = f"http://{args.host}:{args.port}/v1/completions"
    sem = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None
    conn = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=args.timeout)

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        if warmup:
            await asyncio.gather(*[
                send_openai_completion(session, url, args.model, p, min(o, 16), pl)
                for p, pl, o in warmup
            ])

        tasks = []
        bench_start = time.perf_counter()
        async for prompt, plen, olen in _arrivals(requests, args.request_rate):
            tasks.append(asyncio.create_task(send_openai_completion(
                session, url, args.model, prompt, olen, plen, sem)))
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - bench_start

    return summarize(results, duration, args)


def summarize(results: list[RequestResult], duration: float, args) -> dict:
    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    if not ok:
        return {"error": "all requests failed",
                "sample_errors": [r.error for r in failed[:5]]}

    ttfts = [r.ttft * 1000 for r in ok]
    itls = [x * 1000 for r in ok for x in r.itls]
    e2els = [r.latency * 1000 for r in ok]
    # TPOT: steady-state per-token cost, so it excludes the prefill (hence output_len-1)
    tpots = [(r.latency - r.ttft) * 1000 / (r.output_len - 1) for r in ok if r.output_len > 1]
    out_tokens = sum(r.output_len for r in ok)
    in_tokens = sum(r.prompt_len for r in ok)

    def dist(name, xs):
        return {f"mean_{name}_ms": sum(xs) / len(xs) if xs else float("nan"),
                f"median_{name}_ms": percentile(xs, 50),
                f"p90_{name}_ms": percentile(xs, 90),
                f"p99_{name}_ms": percentile(xs, 99)}

    good = sum(1 for r in ok
               if r.ttft * 1000 <= args.ttft_slo_ms
               and r.output_len > 1
               and (r.latency - r.ttft) * 1000 / (r.output_len - 1) <= args.tpot_slo_ms)

