"""Minimal Llama 3.x, written from scratch.

Architecture:
    embed_tokens
    N x DecoderLayer:
        RMSNorm -> Attention(qkv proj, RoPE, PagedAttention, o proj) -> residual
        RMSNorm -> MLP(gate/up fused, SwiGLU, down)                  -> residual
    RMSNorm
    lm_head   (tied to embed_tokens on the 1B model)

Things that silently corrupt output, all handled below:
  - RoPE theta is 500000.0 on Llama 3, not 10000.0.
  - Llama 3.1/3.2 use a piecewise frequency rescale ("llama3" rope_scaling) that only
    matters above 8k but is cheap to get right.
  - Position ids during chunked prefill are ABSOLUTE, not chunk-relative.
  - RMSNorm accumulates in fp32 then casts back, even in a bf16 model.
  - q/k/v and gate/up are fused into single GEMMs at load time; the checkpoint stores
    them separately, so the loader has to slice into the fused buffer.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..kernels.fused import fused_add_rms_norm, rms_norm, silu_and_mul
