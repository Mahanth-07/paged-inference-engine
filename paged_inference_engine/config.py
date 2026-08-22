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
    seed: int = 0

    # Populated from the HF config at load time.
    hidden_size: int = 0
    num_layers: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    rope_scaling: Optional[dict] = None

    @property
    def gqa_ratio(self) -> int:
        assert self.num_kv_heads > 0
        return self.num_heads // self.num_kv_heads


@dataclass
class CacheConfig:
    block_size: int = 16
    """Tokens per physical block. Ablate {8, 16, 32, 64} in M4."""

    gpu_memory_utilization: float = 0.90
    """Fraction of total VRAM the engine may use. Blocks = (budget - weights - activations) / bytes_per_block."""

    num_gpu_blocks: Optional[int] = None
    """Set explicitly to make benchmarks deterministic; otherwise profiled at startup."""

    enable_prefix_caching: bool = False
    """M5. Requires content-hashing blocks and refcounting."""

    kv_cache_dtype: str = "auto"  # "auto" | "fp8"


@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 8192
    """Token budget per forward pass. The main throughput/latency dial."""

    max_num_seqs: int = 256
    """Max sequences in a running batch."""

    enable_chunked_prefill: bool = True
    """Cap prefill tokens per step so long prompts don't stall decodes. Big ITL p99 win."""

    max_num_partial_prefills: int = 1

    preemption_mode: Literal["recompute", "swap"] = "recompute"

    policy: Literal["fcfs", "priority"] = "fcfs"


@dataclass
class EngineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    attention_backend: AttentionBackend = "naive"
    enforce_eager: bool = True
    """False enables CUDA graph capture for the decode path (M5)."""

    cuda_graph_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256)

    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str) -> "EngineConfig":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            cache=CacheConfig(**raw.get("cache", {})),
            scheduler=SchedulerConfig(**raw.get("scheduler", {})),
            **{k: v for k, v in raw.items() if k not in ("model", "cache", "scheduler")},
        )
