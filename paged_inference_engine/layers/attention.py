"""Paged attention: the one layer that has to know about the block table.

Four backends, built in this order, each a fully working engine:

  1. "naive"      -- gather each sequence's blocks into a dense tensor and call SDPA.
                     Slow, obviously correct, runs on CPU/MPS, and is the oracle the
                     other three are tested against. Never delete it.
  2. "flashinfer" -- production paged kernels. Fast and correct, which unblocks M3/M4.
  3. "flash_attn" -- FA2/FA3 with a paged block table. A second opinion on (2).
  4. "triton"     -- our own kernels (kernels/paged_attention_triton.py).

Shapes (num_tokens is the FLATTENED prefill+decode batch for the step):
    query            : (num_tokens, num_heads, head_dim)
    k_cache/v_cache  : (num_blocks, block_size, num_kv_heads, head_dim)
    output           : (num_tokens, num_heads, head_dim)

Layout convention, relied on everywhere: prefill tokens come FIRST in the flattened
batch, then decode tokens. That is what lets the two paths write into disjoint slices
of one output tensor without a gather.

GQA: query head h attends to KV head h // (num_heads // num_kv_heads). Getting this
mapping wrong produces fluent-but-wrong output that you will not catch by reading it.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

