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
