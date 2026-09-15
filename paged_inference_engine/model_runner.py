"""Turns a SchedulerOutput into tensors, runs the model, returns sampled tokens.

This is where CPU overhead goes to hide. At batch size 1 a naive implementation spends
more time in Python building tensors than the GPU spends computing. The fixes are
boring and all live here: build flat Python lists once, do ONE pinned host-to-device
copy per tensor, never call .item() in the loop, and capture the decode path in a
CUDA graph.

Batch layout convention, relied on by every attention backend:
    [ prefill tokens (sequences concatenated) | decode tokens (one per sequence) ]
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .block_manager import BlockManager, compute_num_gpu_blocks
from .config import EngineConfig
from .layers.attention import copy_blocks
from .sampling import Sampler
from .scheduler import SchedulerOutput
from .sequence import Sequence

_DTYPES = {
    "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
    "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "auto": None,
}


@dataclass
class ModelInput:
    """Flattened batch. Prefill and decode tokens live in ONE tensor -- that is what
    makes mixed prefill+decode batches (chunked prefill) possible at all."""

    input_ids: torch.Tensor            # (num_tokens,) int32
    positions: torch.Tensor            # (num_tokens,) int32, ABSOLUTE, for RoPE
    slot_mapping: torch.Tensor         # (num_tokens,) int64, where each K/V is written

    num_prefills: int = 0
    num_prefill_tokens: int = 0
    num_decodes: int = 0

