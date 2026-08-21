"""Configuration objects. Keep every tunable knob here so benchmarks can sweep them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

AttentionBackend = Literal["naive", "flashinfer", "flash_attn", "triton"]


@dataclass
class ModelConfig:
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    trust_remote_code: bool = False
