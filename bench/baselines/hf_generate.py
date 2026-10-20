"""HuggingFace `generate` baselines -- the lines paged-inference-engine must beat.

Two variants, because "beats HF" is only impressive if the HF config was reasonable:

  1. STATIC BATCHING (the naive line)
     Fixed batches, generate until the LONGEST member finishes. With realistic
     output-length variance this wastes most of the GPU: a sequence that finishes in
     20 tokens still occupies a slot for all 500 steps. This script measures the waste
     directly and reports it:
         batch_utilization = sum(output_lens) / (batch_size * max(output_lens))
     On ShareGPT that is routinely under 40%. That number is the entire argument for
     continuous batching, and it is far more persuasive than a speedup ratio.

  2. STATIC CACHE + torch.compile (the fair line)
     Removes cache reallocation and launch overhead, so what is left is genuinely the
     batching and memory-management difference -- which is the thing paged-inference-engine built.

Also reported: peak memory and bytes-of-KV-cache-per-served-token. HF pads the KV
cache to max_len for every sequence in the batch; paging does not. That is a memory
story, not a speed story, and it is the more fundamental one.

    python -m bench.baselines.hf_generate --model ... --num-prompts 500 \
        --mode static --out results/hf_static_rateinf_rep1.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time


def build_prompts(args, tokenizer):
    if args.dataset == "sharegpt":
        from bench.client import load_sharegpt

        data = load_sharegpt(args.dataset_path, args.num_prompts, tokenizer, args.seed,
                             fixed_output_len=args.output_len if args.fixed_output else None)
    else:
        from bench.client import synthetic

        data = synthetic(args.num_prompts, args.input_len, args.output_len,
                         tokenizer, args.seed)
    return data


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--mode", default="static", choices=["static", "static_cache_compile"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-prompts", type=int, default=500)
    ap.add_argument("--dataset", default="sharegpt", choices=["sharegpt", "synthetic"])
    ap.add_argument("--dataset-path",
                    default="data/ShareGPT_V3_unfiltered_cleaned_split.json")
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--fixed-output", action="store_true", default=True)
    ap.add_argument("--variable-output", dest="fixed_output", action="store_false")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ttft-slo-ms", type=float, default=500.0)
    ap.add_argument("--tpot-slo-ms", type=float, default=50.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"     # right padding corrupts generation

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).cuda().eval()
    gen_kwargs = dict(do_sample=False, temperature=None, top_p=None,
                      pad_token_id=tokenizer.pad_token_id)
    if args.mode == "static_cache_compile":
        model.generation_config.cache_implementation = "static"
        model.forward = torch.compile(model.forward, mode="reduce-overhead",
                                      fullgraph=True)

    data = build_prompts(args, tokenizer)
    torch.cuda.reset_peak_memory_stats()

    # ---- warmup (compile + alloc costs are real but not the steady state) ----
    warm = data[:2]
    enc = tokenizer([p for p, _, _ in warm], return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        model.generate(**enc, max_new_tokens=8, min_new_tokens=8, **gen_kwargs)
    torch.cuda.synchronize()

    # ---- the run ----
    per_request = []
    batch_utils = []
    total_out = total_in = 0
    t_start = time.perf_counter()
    for i in range(0, len(data), args.batch_size):
        batch = data[i:i + args.batch_size]
        prompts = [p for p, _, _ in batch]
        out_lens = [o for _, _, o in batch]
        # Static batching: the whole batch runs for max(out_lens) steps, no matter that
        # some members finished long before. This IS the inefficiency being measured.
        steps = max(out_lens)
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(**enc, max_new_tokens=steps, min_new_tokens=steps, **gen_kwargs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        batch_utils.append(sum(out_lens) / (len(batch) * steps))
        total_out += sum(out_lens)
        total_in += sum(pl for _, pl, _ in batch)
        for (_, plen, olen) in batch:
            # Every member of a static batch waits for the whole batch: its E2E latency
            # is the batch's wall time, and its "TTFT" is the prefill of the widest
            # padded prompt. That is the honest accounting, and it is why static
            # batching loses on tail latency even before it loses on throughput.
            per_request.append({"prompt_len": plen, "output_len": olen,
                                "latency_ms": dt * 1000,
                                "ttft_ms": dt * 1000 / steps,
                                "tpot_ms": dt * 1000 / max(1, steps - 1)})
    duration = time.perf_counter() - t_start

    from paged_inference_engine.utils.metrics import percentile

    def dist(name, xs):
        return {f"mean_{name}_ms": sum(xs) / len(xs),
                f"median_{name}_ms": percentile(xs, 50),
                f"p90_{name}_ms": percentile(xs, 90),
                f"p99_{name}_ms": percentile(xs, 99)}

    ttfts = [r["ttft_ms"] for r in per_request]
    tpots = [r["tpot_ms"] for r in per_request]
    e2els = [r["latency_ms"] for r in per_request]
    good = sum(1 for r in per_request
               if r["ttft_ms"] <= args.ttft_slo_ms and r["tpot_ms"] <= args.tpot_slo_ms)

    peak = torch.cuda.max_memory_allocated()
    summary = {
        "num_requests": len(per_request),
        "num_failed": 0,
        "duration_s": duration,
        "request_throughput_rps": len(per_request) / duration,
