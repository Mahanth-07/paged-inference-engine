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
