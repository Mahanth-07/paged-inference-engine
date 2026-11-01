"""M5 profiling helpers. GPU box only.

    # timeline: where does a decode step actually spend its time?
    nsys profile -o results/decode --force-overwrite true \
        python scripts/profile.py timeline --batch 1 --steps 50

    # kernel counters: is the decode kernel near the memory roofline?
    ncu --set full --kernel-name-base demangled -k _paged_attn_decode \
        -o results/decode_ncu python scripts/profile.py kernel --batch 1

    # no external profiler needed: the launch-overhead story in one table
    python scripts/profile.py breakdown

What to look for in the nsys timeline, and what each thing means:

  * GAPS between kernels at batch size 1. If the GPU is idle 40-60% of the timeline,
    you have a LAUNCH OVERHEAD problem, not a kernel problem. The fix is CUDA graphs
    (--cuda-graphs), and the before/after timeline is the single most demoable image
    in the whole project.
  * A fat block of CPU time before each step. That is prepare_inputs building Python
    lists; it should be a few hundred microseconds, not milliseconds.
  * cudaMemcpyAsync followed immediately by a sync. That is a .item() or .tolist()
    somewhere in the hot path serialising the pipeline.
  * The lm_head GEMM dominating a prefill step. That means compute_logits is running
    over every batched token instead of just the ones that need a token.
"""

from __future__ import annotations

import argparse
import time

import torch

from paged_inference_engine.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from paged_inference_engine.engine import LLMEngine
from paged_inference_engine.sampling import SamplingParams


def build(args, eager=True) -> LLMEngine:
    return LLMEngine(EngineConfig(
        model=ModelConfig(model=args.model, dtype=args.dtype,
                          max_model_len=args.max_model_len),
        cache=CacheConfig(block_size=args.block_size, num_gpu_blocks=args.num_gpu_blocks),
        scheduler=SchedulerConfig(max_num_batched_tokens=args.max_num_batched_tokens,
                                  max_num_seqs=max(args.batch, 8)),
        attention_backend=args.backend, device="cuda", enforce_eager=eager,
    ))


def _fill(engine, batch: int, prompt_len: int, max_tokens: int = 10_000) -> None:
    """Drive the engine into steady-state decode with `batch` live sequences."""
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    for i in range(batch):
        engine.add_request(prompt_token_ids=[100 + (i % 500)] * prompt_len,
                           sampling_params=sp)
    while any(s.is_prefill for s in engine.scheduler.running) or engine.scheduler.waiting:
        engine.step()


def cmd_timeline(args) -> None:
    engine = build(args, eager=not args.cuda_graphs)
    _fill(engine, args.batch, args.prompt_len)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.steps):
        engine.step()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"profiled {args.steps} decode steps at batch {args.batch}")


def cmd_kernel(args) -> None:
    """Isolate the decode kernel so ncu profiles it and nothing else."""
    from paged_inference_engine.kernels.paged_attention_triton import paged_attention_decode

    b, ctx, h, kvh, d, bs = (args.batch, args.context, args.num_heads,
                             args.num_kv_heads, args.head_dim, args.block_size)
    pages = (ctx + bs - 1) // bs
    nb = pages * b + 8
    bt = torch.randperm(nb)[:pages * b].view(b, pages).to(torch.int32).cuda()
    kc = torch.randn(nb, bs, kvh, d, dtype=torch.float16, device="cuda")
    vc = torch.randn_like(kc)
    q = torch.randn(b, h, d, dtype=torch.float16, device="cuda")
    cl = torch.full((b,), ctx, dtype=torch.int32, device="cuda")
    for _ in range(5):
        paged_attention_decode(q, kc, vc, bt, cl, d ** -0.5, split_k=args.split_k)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    paged_attention_decode(q, kc, vc, bt, cl, d ** -0.5, split_k=args.split_k)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    kv_bytes = b * ctx * kvh * d * 2 * 2
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        paged_attention_decode(q, kc, vc, bt, cl, d ** -0.5, split_k=args.split_k)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f"decode kernel: {dt*1e6:.1f} us, {kv_bytes/dt/1e9:.0f} GB/s effective")
    print("Decode attention is memory bound -- compare that against the device's peak "
          "HBM bandwidth, not against its TFLOPs.")


def cmd_breakdown(args) -> None:
    """Where the step time goes, without needing nsys installed.

    Splits a decode step into: input preparation (pure Python/host), the model forward,
    and sampling. The interesting column is `gpu busy %` -- at batch size 1 on a 1B
    model it is usually well under half, and that gap IS the launch overhead that CUDA
    graphs remove.
    """
    print(f"{'batch':>6} {'mode':>12} {'step ms':>9} {'prep ms':>9} {'fwd ms':>9} "
          f"{'gpu busy':>9} {'tok/s':>9}")
    print("-" * 68)
    for batch in args.batches:
        for mode, eager in (("eager", True), ("cuda graphs", False)):
            if mode == "cuda graphs" and args.backend == "flashinfer":
                continue
            engine = build(argparse.Namespace(**{**vars(args), "batch": batch}), eager=eager)
            _fill(engine, batch, args.prompt_len)
            runner, bm = engine.model_runner, engine.block_manager

            for _ in range(5):
                engine.step()
            torch.cuda.synchronize()

            t_prep = t_step = t_fwd = 0.0
            n = args.steps
            for _ in range(n):
                sched = engine.scheduler.schedule()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                mi, out_seqs = runner.prepare_inputs(sched, bm)
                t1 = time.perf_counter()
                with torch.inference_mode():
                    graph = runner._maybe_graph(mi)
                    if graph is not None:
                        logits = graph.replay(mi)
                    else:
                        h = runner.model(mi.input_ids, mi.positions, runner.kv_caches, mi)
                        logits = runner.model.compute_logits(h, mi.logits_indices)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                tokens = runner.sampler(logits, [s.sampling_params for s in out_seqs])
                for seq, tid in zip(out_seqs, tokens.tolist()):
                    seq.append_token(tid)
                engine.scheduler.free_finished(engine.eos_token_id)
                t3 = time.perf_counter()
                t_prep += t1 - t0
                t_fwd += t2 - t1
                t_step += t3 - t0

            step_ms = t_step / n * 1000
            print(f"{batch:>6} {mode:>12} {step_ms:>9.3f} {t_prep/n*1000:>9.3f} "
                  f"{t_fwd/n*1000:>9.3f} {100*t_fwd/t_step:>8.1f}% "
                  f"{batch/(t_step/n):>9.0f}")
            del engine
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["timeline", "kernel", "breakdown"])
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--backend", default="triton",
                    choices=["naive", "flashinfer", "flash_attn", "triton"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-gpu-blocks", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--num-heads", type=int, default=32)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--split-k", type=int, default=None)
    ap.add_argument("--cuda-graphs", action="store_true")
    args = ap.parse_args()
