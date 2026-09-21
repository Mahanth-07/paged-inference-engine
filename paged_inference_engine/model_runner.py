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

    # prefill (varlen) metadata
    qo_indptr: Optional[torch.Tensor] = None            # (num_prefills+1,) int32
    prefill_query_lens: Optional[torch.Tensor] = None   # (num_prefills,) int32
    prefill_context_lens: Optional[torch.Tensor] = None # (num_prefills,) int32, AFTER this chunk
    prefill_block_tables: Optional[torch.Tensor] = None # (num_prefills, max_pages) int32
    prefill_kv_cu_seqlens: Optional[torch.Tensor] = None
    max_query_len: int = 0
    max_prefill_context_len: int = 0

    # decode metadata
    decode_block_tables: Optional[torch.Tensor] = None  # (num_decodes, max_pages) int32
    decode_context_lens: Optional[torch.Tensor] = None  # (num_decodes,) int32
    max_decode_context_len: int = 0

    # FlashInfer wrappers, planned ONCE per step (never per layer)
    fi_prefill: Any = None
    fi_decode: Any = None

    logits_indices: Optional[torch.Tensor] = None
    use_split_k: Optional[int] = None

    @property
    def is_decode_only(self) -> bool:
        return self.num_prefills == 0 and self.num_decodes > 0


class AttentionState:
    """Owns the FlashInfer workspace and wrappers.

    They live here rather than on each attention layer for one reason: plan() is the
    expensive part of FlashInfer and it depends only on the step's page table, not on
    the layer. Planning once per step instead of once per layer is a 16x reduction in
    that cost on a 1B model.
    """

    def __init__(self, config: EngineConfig, num_kv_heads: int, num_heads: int,
                 head_dim: int, dtype: torch.dtype, device: str):
        self.enabled = config.attention_backend == "flashinfer"
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.dtype, self.device = dtype, device
        self.page_size = config.cache.block_size
        self.prefill_wrapper = None
        self.decode_wrapper = None
        if not self.enabled:
            return
        import flashinfer

        # 128MB is plenty and it is allocated once, at startup, on purpose.
        self.workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, "NHD")
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace, "NHD")

    def plan(self, mi: ModelInput, prefill_csr, decode_csr) -> None:
        if not self.enabled:
            return
        if mi.num_prefills:
            self.prefill_wrapper.plan(
                mi.qo_indptr, *prefill_csr,
                num_qo_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim, page_size=self.page_size,
                causal=True, q_data_type=self.dtype,
            )
            mi.fi_prefill = self.prefill_wrapper
        if mi.num_decodes:
            self.decode_wrapper.plan(
                *decode_csr,
                num_qo_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim, page_size=self.page_size,
                q_data_type=self.dtype,
            )
            mi.fi_decode = self.decode_wrapper


class ModelRunner:
    def __init__(self, config: EngineConfig, load: bool = True):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = _DTYPES.get(config.model.dtype) or torch.float32
        self.model = None
        self.tokenizer = None
        self.kv_caches: list[torch.Tensor] = []
        self.graphs: dict[int, "CUDAGraphRunner"] = {}
        self.num_blocks = 0
        self.attn_state: Optional[AttentionState] = None
        self.sampler = Sampler(device=str(self.device), seed=config.model.seed)
        # Reserved scratch block, one past the pool the BlockManager hands out. Padded
        # rows in a CUDA-graph batch write their garbage K/V here instead of needing a
        # host-side branch. Costs one block; buys a branch-free capture.
        self.scratch_block = 0
        if load:
            self.load_model()

    # ---- setup ------------------------------------------------------------

    def load_model(self) -> None:
        """Build the model and load HF safetensors directly.

        Deliberately does NOT go through transformers' from_pretrained -- the point of
        the exercise is the modeling code. What it does use transformers for is the
        tokenizer and the config json, which are not the interesting part.
        """
        from .models.llama import LlamaForCausalLM

        mc = self.config.model
        if mc.model == "__random__":
            self._init_random_model()
            return

        from transformers import AutoConfig, AutoTokenizer

        hf = AutoConfig.from_pretrained(mc.model, trust_remote_code=mc.trust_remote_code)
        mc.hidden_size = hf.hidden_size
        mc.num_layers = hf.num_hidden_layers
        mc.num_heads = hf.num_attention_heads
        mc.num_kv_heads = getattr(hf, "num_key_value_heads", hf.num_attention_heads)
        mc.head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
        mc.intermediate_size = hf.intermediate_size
        mc.vocab_size = hf.vocab_size
        mc.rope_theta = float(getattr(hf, "rope_theta", 500000.0))
        mc.rms_norm_eps = float(getattr(hf, "rms_norm_eps", 1e-5))
        mc.tie_word_embeddings = bool(getattr(hf, "tie_word_embeddings", False))
        mc.rope_scaling = getattr(hf, "rope_scaling", None)
        mc.max_model_len = min(mc.max_model_len, getattr(hf, "max_position_embeddings", 1 << 30))

        self.tokenizer = AutoTokenizer.from_pretrained(mc.model)

        with torch.device("meta"):
            model = LlamaForCausalLM(
                mc, mc.intermediate_size, backend=self.config.attention_backend,
                tie_word_embeddings=mc.tie_word_embeddings, rope_scaling=mc.rope_scaling,
            )
        model.to_empty(device=self.device)
        model = model.to(self.dtype)
        # RoPE tables were on `meta`; rebuild them for real on the target device.
        model.model.rotary = type(model.model.rotary)(
            head_dim=mc.head_dim, max_position=mc.max_model_len, base=mc.rope_theta,
            rope_scaling=mc.rope_scaling, dtype=torch.float32, device=self.device,
        )

        counts = model.load_weights(self._iter_hf_weights(mc.model))
        expected = model.expected_write_counts()
        missing = {k: (counts.get(k, 0), v) for k, v in expected.items() if counts.get(k, 0) != v}
        if missing:
            raise RuntimeError(
                "checkpoint did not fully cover the model -- these params would be left "
                f"at random init: {missing}"
            )
        self.model = model.eval()
        self._post_init()

    @staticmethod
    def _iter_hf_weights(model_id: str):
        """Stream (name, tensor) pairs out of the safetensors shards, one at a time."""
        import glob

        from safetensors.torch import safe_open

        path = model_id
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download

            path = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no safetensors in {path}")
        for f in files:
            with safe_open(f, framework="pt", device="cpu") as sf:
                for name in sf.keys():
                    yield name, sf.get_tensor(name)

    def _init_random_model(self) -> None:
        """The CPU correctness fixture from configs/tiny.yaml: a real model with random
        weights, small enough that the naive backend is fast. Everything the engine does
        -- paging, chunked prefill, preemption, batching -- is exercised for free."""
        from .models.llama import LlamaForCausalLM

        mc = self.config.model
        mc.intermediate_size = mc.intermediate_size or 4 * mc.hidden_size
        torch.manual_seed(mc.seed)
        model = LlamaForCausalLM(
            mc, mc.intermediate_size, backend=self.config.attention_backend,
            tie_word_embeddings=mc.tie_word_embeddings, rope_scaling=mc.rope_scaling,
        )
        # std=0.1, not the usual 0.02. A small-std random transformer is close to the
        # identity: with tied embeddings its argmax is just the last input token, every
        # step, for every prompt. Every "output is unchanged" test would then pass
        # vacuously -- including with a completely broken block table. At 0.1 the
        # outputs actually depend on the attended context, which is the whole point of
        # having a fixture.
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, std=0.1)
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        self.tokenizer = None
        self._post_init()

    def _post_init(self) -> None:
        mc = self.config.model
        self.attn_state = AttentionState(
            self.config, mc.num_kv_heads, mc.num_heads, mc.head_dim, self.dtype,
            str(self.device),
        )

    def allocate_kv_cache(self, num_blocks: int) -> None:
        """One tensor per layer: (2, num_blocks + 1, block_size, num_kv_heads, head_dim).

        Allocated ALL at once, up front. Growing the cache at runtime reintroduces
        exactly the fragmentation and unpredictable OOMs that paging exists to remove.

        The +1 is the scratch block described in __init__; the BlockManager is only ever
        told about `num_blocks`, so it can never hand it out.
        """
        mc = self.config.model
        bs = self.config.cache.block_size
        self.num_blocks = num_blocks
        self.scratch_block = num_blocks
        self.kv_caches = [
            torch.zeros(2, num_blocks + 1, bs, mc.num_kv_heads, mc.head_dim,
                        dtype=self.dtype, device=self.device)
            for _ in range(mc.num_layers)
        ]

    def profile_num_blocks(self) -> int:
