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
        """Measure the activation peak with a real forward pass, then divide up what's left.

        Hardcoding a guess for the activation peak is the classic source of "works at
        batch 8, OOMs at batch 64" -- the peak scales with max_num_batched_tokens, so
        profile at exactly that width.
        """
        cc = self.config.cache
        if cc.num_gpu_blocks is not None:
            self.allocate_kv_cache(cc.num_gpu_blocks)
            return cc.num_gpu_blocks
        if self.device.type != "cuda":
            n = 1024
            self.allocate_kv_cache(n)
            return n

        mc = self.config.model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        weight_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

        n_tok = self.config.scheduler.max_num_batched_tokens
        self._dummy_forward(n_tok)
        torch.cuda.synchronize()
        activation_peak = torch.cuda.max_memory_allocated() - weight_bytes
        gc.collect()
        torch.cuda.empty_cache()

        num_blocks = compute_num_gpu_blocks(
            total_memory_bytes=total, weight_bytes=weight_bytes,
            activation_peak_bytes=max(activation_peak, 0),
            num_layers=mc.num_layers, num_kv_heads=mc.num_kv_heads, head_dim=mc.head_dim,
            block_size=cc.block_size,
            dtype_bytes=torch.tensor([], dtype=self.dtype).element_size(),
            utilization=cc.gpu_memory_utilization,
        )
        if num_blocks <= 0:
            raise RuntimeError(
                f"no room for a KV cache: weights {weight_bytes/2**30:.1f} GiB + "
                f"activations {activation_peak/2**30:.1f} GiB at "
                f"{cc.gpu_memory_utilization:.0%} of {total/2**30:.1f} GiB"
            )
        print(f"[paged-inference-engine] KV cache: {num_blocks:,} blocks = "
              f"{num_blocks * cc.block_size:,} tokens "
              f"({num_blocks * 2 * cc.block_size * mc.num_kv_heads * mc.head_dim * mc.num_layers * torch.tensor([], dtype=self.dtype).element_size() / 2**30:.2f} GiB)")
        self.allocate_kv_cache(num_blocks)
        return num_blocks

    def _dummy_forward(self, num_tokens: int) -> None:
        """One forward pass at the widest batch the scheduler can produce, with a
        throwaway one-block cache, purely to observe the activation high-water mark."""
        mc = self.config.model
        bs = self.config.cache.block_size
        pages = (num_tokens + bs - 1) // bs + 1
        caches = [
            torch.zeros(2, pages, bs, mc.num_kv_heads, mc.head_dim,
                        dtype=self.dtype, device=self.device)
            for _ in range(mc.num_layers)
        ]
        dev = self.device
        mi = ModelInput(
            input_ids=torch.zeros(num_tokens, dtype=torch.int64, device=dev),
            positions=torch.arange(num_tokens, dtype=torch.int64, device=dev) % mc.max_model_len,
            slot_mapping=torch.arange(num_tokens, dtype=torch.int64, device=dev),
            num_prefills=1, num_prefill_tokens=num_tokens, num_decodes=0,
            qo_indptr=torch.tensor([0, num_tokens], dtype=torch.int32, device=dev),
            prefill_query_lens=torch.tensor([num_tokens], dtype=torch.int32, device=dev),
            prefill_context_lens=torch.tensor([num_tokens], dtype=torch.int32, device=dev),
            prefill_block_tables=torch.arange(pages, dtype=torch.int32, device=dev).unsqueeze(0),
            prefill_kv_cu_seqlens=torch.tensor([0, num_tokens], dtype=torch.int32, device=dev),
            max_query_len=num_tokens, max_prefill_context_len=num_tokens,
            logits_indices=torch.tensor([num_tokens - 1], dtype=torch.int64, device=dev),
        )
        if self.attn_state and self.attn_state.enabled:
            self.attn_state.plan(
                mi,
                (torch.tensor([0, pages], dtype=torch.int32, device=dev),
                 torch.arange(pages, dtype=torch.int32, device=dev),
                 torch.tensor([bs], dtype=torch.int32, device=dev)),
                None,
            )
        with torch.inference_mode():
            h = self.model(mi.input_ids, mi.positions, caches, mi)
            self.model.compute_logits(h, mi.logits_indices)
        del caches

    # ---- the hot path -----------------------------------------------------

    def prepare_inputs(self, sched: SchedulerOutput, bm: BlockManager) -> tuple[ModelInput, list[Sequence]]:
        """Build every tensor for this step, and return the sequences that will get a token.

        Everything is accumulated into flat Python lists and shipped in one pinned,
        non-blocking copy per tensor. A per-sequence `.to(device)` is a sync per
        sequence and dominates the profile at batch 256.
        """
        bsz = bm.block_size
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        logits_indices: list[int] = []
        out_seqs: list[Sequence] = []

        qo_indptr = [0]
        prefill_qlens: list[int] = []
        prefill_ctx: list[int] = []
        prefill_seqs: list[Sequence] = []

        for s in sched.prefills:
            seq = s.seq
            start = seq.num_computed_tokens
            end = start + s.num_tokens
            toks = seq.all_token_ids
            input_ids.extend(toks[start:end])
            positions.extend(range(start, end))
            slot_mapping.extend(bm.slot_mapping(seq, start, end))
            qo_indptr.append(qo_indptr[-1] + s.num_tokens)
            prefill_qlens.append(s.num_tokens)
            prefill_ctx.append(end)
            prefill_seqs.append(seq)
            # A chunk that computes the sequence's last uncomputed token emits a token;
            # a partial chunk does not.
            if end >= seq.total_len:
                logits_indices.append(len(input_ids) - 1)
                out_seqs.append(seq)

        num_prefill_tokens = len(input_ids)

        decode_ctx: list[int] = []
        decode_seqs: list[Sequence] = []
        for s in sched.decodes:
            seq = s.seq
            pos = seq.total_len - 1                  # the token sampled last step
            input_ids.append(seq.all_token_ids[pos])
            positions.append(pos)
            slot_mapping.append(seq.slot_for_token(pos, bsz))
            decode_ctx.append(seq.total_len)
            decode_seqs.append(seq)
            logits_indices.append(len(input_ids) - 1)
            out_seqs.append(seq)

        t = self._to_device

        mi = ModelInput(
            input_ids=t(input_ids, torch.int64),
            positions=t(positions, torch.int64),
            slot_mapping=t(slot_mapping, torch.int64),
            num_prefills=len(prefill_seqs),
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=len(decode_seqs),
            logits_indices=t(logits_indices, torch.int64),
        )

        prefill_csr = decode_csr = None
        if prefill_seqs:
            tables = bm.block_tables_padded(prefill_seqs)
            mi.qo_indptr = t(qo_indptr, torch.int32)
            mi.prefill_query_lens = t(prefill_qlens, torch.int32)
            mi.prefill_context_lens = t(prefill_ctx, torch.int32)
            mi.prefill_block_tables = t(tables, torch.int32)
            cu = [0]
            for c in prefill_ctx:
                cu.append(cu[-1] + c)
            mi.prefill_kv_cu_seqlens = t(cu, torch.int32)
            mi.max_query_len = max(prefill_qlens)
            mi.max_prefill_context_len = max(prefill_ctx)
            if self.attn_state and self.attn_state.enabled:
                meta = bm.paged_kv_metadata(prefill_seqs)
                prefill_csr = (t(meta["kv_indptr"], torch.int32),
                               t(meta["kv_indices"], torch.int32),
                               t(meta["kv_last_page_len"], torch.int32))

        if decode_seqs:
            tables = bm.block_tables_padded(decode_seqs)
            mi.decode_block_tables = t(tables, torch.int32)
            mi.decode_context_lens = t(decode_ctx, torch.int32)
            mi.max_decode_context_len = max(decode_ctx)
            if self.attn_state and self.attn_state.enabled:
                meta = bm.paged_kv_metadata(decode_seqs)
                decode_csr = (t(meta["kv_indptr"], torch.int32),
                              t(meta["kv_indices"], torch.int32),
                              t(meta["kv_last_page_len"], torch.int32))

        if self.attn_state:
            self.attn_state.plan(mi, prefill_csr, decode_csr)
        return mi, out_seqs

    def _to_device(self, data, dtype):
        """One pinned, non-blocking H2D copy. Pinning only helps on CUDA; elsewhere it
        is an error, so it is conditional."""
        if self.device.type == "cuda":
            cpu = torch.tensor(data, dtype=dtype, pin_memory=True)
            return cpu.to(self.device, non_blocking=True)
        return torch.tensor(data, dtype=dtype, device=self.device)

    @torch.inference_mode()
    def execute_model(self, sched: SchedulerOutput, bm: BlockManager) -> dict[Sequence, int]:
        """Run one forward pass, return {sequence: sampled_token_id}.

        A sequence mid chunked-prefill contributes tokens to the batch but gets no
        output this step -- it is not in the returned dict.
        """
        # Copy-on-write first: a shared block must be duplicated BEFORE this step writes
        # new K/V into it, or we corrupt whoever else is still reading it.
        if sched.blocks_to_copy:
            copy_blocks(self.kv_caches, sched.blocks_to_copy)

        mi, out_seqs = self.prepare_inputs(sched, bm)
        if not out_seqs and mi.input_ids.numel() == 0:
            return {}

        graph = self._maybe_graph(mi)
        if graph is not None:
            logits = graph.replay(mi)
        else:
            hidden = self.model(mi.input_ids, mi.positions, self.kv_caches, mi)
            logits = self.model.compute_logits(hidden, mi.logits_indices)

        if not out_seqs:
            return {}
        params = [s.sampling_params for s in out_seqs]
        tokens = self.sampler(logits, params)
        # ONE device->host sync per step. Not zero, but not per-sequence either.
        return dict(zip(out_seqs, tokens.tolist()))

    # ---- CUDA graphs (M5) --------------------------------------------------

    def _maybe_graph(self, mi: ModelInput):
        if not self.graphs or not mi.is_decode_only:
            return None
        for size in sorted(self.graphs):
            if size >= mi.num_decodes:
                return self.graphs[size]
        return None

    def capture_cuda_graphs(self) -> None:
        """Capture the decode-only path at bucketed batch sizes.

        At batch size 1 on a 1B model, kernel launch overhead is typically 40-60% of the
        step: ~16 layers x ~8 kernels each, every one of them a few microseconds of GPU
        work behind a few microseconds of launch. A graph replays the whole step as a
        single submission. This is the single biggest and most demoable win in M5 --
        capture an nsys timeline before and after.

        Requirements that will bite you, all handled by CUDAGraphRunner:
          - every input tensor must be preallocated at max shape and reused, because a
            graph captures POINTERS, not values
          - no CPU-GPU sync inside the captured region (so: no .item(), no .tolist())
          - block_tables must be a fixed-shape buffer you memcpy into
          - padded rows must be harmless: they write K/V to the scratch block and read
            a context length of 1
          - warm up eagerly first, or the capture records lazy-init work
        """
        if self.config.enforce_eager or self.device.type != "cuda":
            return
        if self.config.attention_backend == "flashinfer":
            print("[paged-inference-engine] flashinfer + CUDA graphs needs the wrapper's own capture "
                  "mode; staying eager. Use --backend triton for the graph ablation.")
            return
        max_bs = self.config.scheduler.max_num_seqs
        sizes = [s for s in self.config.cuda_graph_batch_sizes if s <= max_bs]
        max_pages = (self.config.model.max_model_len + self.config.cache.block_size - 1) \
            // self.config.cache.block_size
        pool = None
        for size in sizes:
            runner = CUDAGraphRunner(self, size, max_pages)
            pool = runner.capture(pool)
            self.graphs[size] = runner
        print(f"[paged-inference-engine] captured CUDA graphs for batch sizes {sizes}")


class CUDAGraphRunner:
    """One captured decode step at a fixed batch size.

    All graphs share a memory pool so capturing 13 buckets does not cost 13 copies of
    the activation workspace.
    """

    def __init__(self, runner: ModelRunner, batch_size: int, max_pages: int):
        self.runner = runner
        self.batch_size = batch_size
        dev = runner.device
        self.input_ids = torch.zeros(batch_size, dtype=torch.int64, device=dev)
        self.positions = torch.zeros(batch_size, dtype=torch.int64, device=dev)
        # scratch slot: padded rows write their garbage K/V into the reserved block
        self.slot_mapping = torch.full((batch_size,),
                                       runner.scratch_block * runner.config.cache.block_size,
                                       dtype=torch.int64, device=dev)
        self.block_tables = torch.full((batch_size, max_pages), runner.scratch_block,
                                       dtype=torch.int32, device=dev)
        self.context_lens = torch.ones(batch_size, dtype=torch.int32, device=dev)
        self.logits_indices = torch.arange(batch_size, dtype=torch.int64, device=dev)
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.output: Optional[torch.Tensor] = None

    def _model_input(self) -> ModelInput:
        return ModelInput(
            input_ids=self.input_ids, positions=self.positions,
            slot_mapping=self.slot_mapping,
            num_prefills=0, num_prefill_tokens=0, num_decodes=self.batch_size,
            decode_block_tables=self.block_tables, decode_context_lens=self.context_lens,
            logits_indices=self.logits_indices,
            use_split_k=1,   # split-K picks its grid from a host-side max(); pin it
        )

    def capture(self, pool):
        r = self.runner
        mi = self._model_input()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
