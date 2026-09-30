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
    mc = engine.config.model
    bm = engine.block_manager
    kv_per_token = 2 * mc.num_layers * mc.num_kv_heads * mc.head_dim * 2
    print(f"model              {mc.model}")
    print(f"layers/heads/kv    {mc.num_layers} / {mc.num_heads} / {mc.num_kv_heads} "
          f"(GQA ratio {mc.gqa_ratio})")
    print(f"head_dim / hidden  {mc.head_dim} / {mc.hidden_size}")
    print(f"params             {sum(p.numel() for p in engine.model_runner.model.parameters())/1e9:.3f} B")
    print(f"block size         {bm.block_size} tokens")
    print(f"blocks             {bm.allocator.num_blocks:,} "
          f"= {bm.allocator.num_blocks * bm.block_size:,} cached tokens")
    print(f"KV bytes/token     {kv_per_token:,} (bf16)")
    print(f"max concurrent     ~{bm.allocator.num_blocks * bm.block_size // max(1, mc.max_model_len)} "
          f"seqs at max_model_len={mc.max_model_len}")


def cmd_throughput(args) -> None:
    """Offline max-throughput number: submit everything at once, drain, divide.

    This is the number to quote for OFFLINE batch throughput. It is NOT the number to
    quote for serving -- for that you need bench/client.py's Poisson arrivals and a
    latency axis, because "tokens/sec" with no latency bound is not an operating point.
    """
    import random

    from ..engine import LLMEngine

    engine = LLMEngine(_config(args))
    rng = random.Random(args.seed)
    vocab = engine.config.model.vocab_size
    prompts = [[rng.randrange(1000, min(vocab, 20000)) for _ in range(args.input_len)]
               for _ in range(args.num_prompts)]
    sp = SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True)
    t0 = time.perf_counter()
    outs = engine.generate(prompt_token_ids=prompts, sampling_params=sp)
    dt = time.perf_counter() - t0
    out_tok = sum(len(o.token_ids) for o in outs)
    in_tok = sum(len(o.prompt_token_ids) for o in outs)
    print(f"requests           {len(outs)}")
    print(f"elapsed            {dt:.2f}s")
    print(f"output throughput  {out_tok/dt:,.1f} tok/s")
    print(f"total throughput   {(in_tok+out_tok)/dt:,.1f} tok/s")
    print(f"requests/s         {len(outs)/dt:,.2f}")
    for k, v in engine.stats().items():
        print(f"  {k:<22} {v}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="paged-inference-engine")

    def common(p):
        p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
        p.add_argument("--backend", default="flashinfer",
                       choices=["naive", "flashinfer", "flash_attn", "triton"])
        p.add_argument("--dtype", default="bfloat16")
        p.add_argument("--device", default="cuda")
        p.add_argument("--block-size", type=int, default=16)
        p.add_argument("--max-model-len", type=int, default=4096)
        p.add_argument("--max-num-batched-tokens", type=int, default=8192)
        p.add_argument("--max-num-seqs", type=int, default=256)
        p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
        p.add_argument("--enable-prefix-caching", action="store_true")
        p.add_argument("--cuda-graphs", action="store_true")
        return p

    sub = ap.add_subparsers(dest="cmd", required=True)

    g = common(sub.add_parser("generate"))
    g.add_argument("--prompt", action="append", required=True)
    g.add_argument("--max-tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=-1)
    g.set_defaults(func=cmd_generate)

    common(sub.add_parser("info")).set_defaults(func=cmd_info)

    t = common(sub.add_parser("throughput"))
    t.add_argument("--num-prompts", type=int, default=200)
    t.add_argument("--input-len", type=int, default=512)
    t.add_argument("--output-len", type=int, default=128)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_throughput)
    return ap


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
