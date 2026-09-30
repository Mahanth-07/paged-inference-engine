"""Offline CLI. Handy for eyeballing output without standing up the server.

    paged-inference-engine generate --prompt "The capital of France is" --max-tokens 32
    paged-inference-engine info
    paged-inference-engine throughput --num-prompts 200 --input-len 512 --output-len 128
"""

from __future__ import annotations

import argparse
import time

from ..config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from ..sampling import SamplingParams


def _config(args) -> EngineConfig:
    return EngineConfig(
        model=ModelConfig(model=args.model, dtype=args.dtype,
                          max_model_len=args.max_model_len),
        cache=CacheConfig(block_size=args.block_size,
                          gpu_memory_utilization=args.gpu_memory_utilization,
                          enable_prefix_caching=args.enable_prefix_caching),
        scheduler=SchedulerConfig(max_num_batched_tokens=args.max_num_batched_tokens,
                                  max_num_seqs=args.max_num_seqs),
        attention_backend=args.backend,
        enforce_eager=not args.cuda_graphs,
        device=args.device,
    )


def cmd_generate(args) -> None:
    from ..engine import LLMEngine

    engine = LLMEngine(_config(args))
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_tokens)
    ids = [engine.tokenizer.encode(p) for p in args.prompt]
    t0 = time.perf_counter()
    outs = engine.generate(prompt_token_ids=ids, sampling_params=sp)
    dt = time.perf_counter() - t0
    for prompt, out in zip(args.prompt, outs):
        print(f"\n=== {prompt!r}\n{out.text}")
    total = sum(len(o.token_ids) for o in outs)
    print(f"\n[{total} tokens in {dt:.2f}s = {total/dt:.1f} tok/s]")


def cmd_info(args) -> None:
    from ..engine import LLMEngine

    engine = LLMEngine(_config(args))
