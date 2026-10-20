"""Attention backend microbenchmark and roofline check (GPU only).

    python -m bench.microbench_attention
    python -m bench.microbench_attention --roofline --context 8192 --batch 1

Decode attention is MEMORY BOUND: one token of query against a full sweep of the KV
cache. So the number that matters is not TFLOPs, it is achieved HBM bandwidth as a
fraction of peak. `--roofline` computes exactly that:

    bytes_moved = num_seqs * context_len * num_kv_heads * head_dim * 2 (K and V) * dtype
    achieved GB/s = bytes_moved / elapsed

A kernel at 80% of peak has essentially nothing left to give and the remaining gap to
vLLM is somewhere else. A kernel at 25% has a real problem -- usually uncoalesced
loads from a too-small page tile, or too few programs to saturate the SMs (which is
what SPLIT_K exists to fix).
"""

from __future__ import annotations

import argparse
import time

import torch

from paged_inference_engine.kernels.paged_attention_triton import (
    paged_attention_decode,
    paged_attention_prefill,
    reference_paged_attention_decode,
