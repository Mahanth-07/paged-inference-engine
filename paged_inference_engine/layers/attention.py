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

# ---------------------------------------------------------------------------
# cache writes
# ---------------------------------------------------------------------------


def reshape_and_cache_torch(key, value, k_cache, v_cache, slot_mapping) -> None:
    """Scatter newly computed K/V into the paged cache (torch reference).

    slot_mapping[i] = block_table[pos // block_size] * block_size + pos % block_size,
    computed on the CPU by BlockManager.slot_mapping. Every token in the batch --
    prefill and decode alike -- gets written, every layer, every step.
    """
    num_kv_heads, head_dim = key.shape[-2], key.shape[-1]
    # Index-assignment rather than index_copy_: MPS has no aten::index_copy, and this
    # is the CPU/MPS path (CUDA goes through the fused Triton kernel). Slots are unique
    # by construction -- one token, one position, one slot -- so the duplicate-index
    # nondeterminism of index_put_ cannot arise.
    k_cache.view(-1, num_kv_heads, head_dim)[slot_mapping] = key
    v_cache.view(-1, num_kv_heads, head_dim)[slot_mapping] = value


def reshape_and_cache(key, value, k_cache, v_cache, slot_mapping) -> None:
    from ..kernels.cache_ops import reshape_and_cache_triton, use_triton_cache_op

    if use_triton_cache_op(key):
        reshape_and_cache_triton(key, value, k_cache, v_cache, slot_mapping)
    else:
        reshape_and_cache_torch(key, value, k_cache, v_cache, slot_mapping)


def copy_blocks(kv_caches: list[torch.Tensor], blocks_to_copy: list[tuple[int, int]]) -> None:
    """Copy-on-write for shared blocks (prefix caching / forked sequences).

    The scheduler tells us (src, dst) pairs; every layer's cache needs the same copy.
    Batch them into one index_copy_ per layer rather than a loop of slice assignments.
    """
    if not blocks_to_copy or not kv_caches:
        return
    device = kv_caches[0].device
    src = torch.tensor([s for s, _ in blocks_to_copy], dtype=torch.long, device=device)
    dst = torch.tensor([d for _, d in blocks_to_copy], dtype=torch.long, device=device)
    for cache in kv_caches:
        # cache: (2, num_blocks, block_size, num_kv_heads, head_dim)
        cache[:, dst] = cache[:, src]


# ---------------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------------


class PagedAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: Optional[float] = None,
        backend: str = "naive",
        layer_idx: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale if scale is not None else head_dim ** -0.5
        self.backend = backend
        self.layer_idx = layer_idx
        assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads"
        self.gqa_ratio = num_heads // num_kv_heads

    def forward(self, query, key, value, kv_cache, model_input):
        """
        1. Write this step's K/V into the paged cache -- always, every token.
        2. Prefill tokens: causal attention over [0, context_len), where the queries
           align to the END of the context (chunked prefill means a chunk's queries
           sit at absolute positions ctx-q_len .. ctx-1, not 0 .. q_len-1).
        3. Decode tokens: paged decode attention, query length 1.
        """
        mi = model_input
        if kv_cache is not None:
            k_cache, v_cache = kv_cache[0], kv_cache[1]
            reshape_and_cache(key, value, k_cache, v_cache, mi.slot_mapping)
        else:  # no cache: single-shot causal attention (used by the M1 oracle path)
            return self._forward_cacheless(query, key, value)

        out = torch.empty_like(query)
        npt = mi.num_prefill_tokens

        if mi.num_prefills:
            out[:npt] = self._prefill(query[:npt], k_cache, v_cache, mi)
        if mi.num_decodes:
            out[npt:] = self._decode(query[npt:], k_cache, v_cache, mi)
        return out

    # ---- dispatch ---------------------------------------------------------

    def _prefill(self, query, k_cache, v_cache, mi):
        if self.backend == "flashinfer" and mi.fi_prefill is not None:
            return self._forward_flashinfer_prefill(query, k_cache, v_cache, mi)
        if self.backend == "flash_attn":
            return self._forward_flash_attn_prefill(query, k_cache, v_cache, mi)
        if self.backend == "triton":
            from ..kernels.paged_attention_triton import paged_attention_prefill

            return paged_attention_prefill(
                query, k_cache, v_cache,
                mi.prefill_block_tables, mi.prefill_context_lens,
                mi.prefill_query_lens, mi.qo_indptr, self.scale,
                max_query_len=mi.max_query_len,
            )
        return self._forward_naive(
            query, k_cache, v_cache,
            mi.prefill_block_tables, mi.prefill_context_lens,
            mi.prefill_query_lens, mi.qo_indptr,
        )

    def _decode(self, query, k_cache, v_cache, mi):
        if self.backend == "flashinfer" and mi.fi_decode is not None:
            return self._forward_flashinfer_decode(query, k_cache, v_cache, mi)
        if self.backend == "flash_attn":
            return self._forward_flash_attn_decode(query, k_cache, v_cache, mi)
        if self.backend == "triton":
            from ..kernels.paged_attention_triton import paged_attention_decode

            return paged_attention_decode(
                query, k_cache, v_cache,
                mi.decode_block_tables, mi.decode_context_lens, self.scale,
                split_k=mi.use_split_k,
                # The runner already has this on the host. Letting the kernel read it
                # off the device would be a sync per layer -- 16 of them per step.
                max_context=mi.max_decode_context_len,
            )
        ones = torch.ones_like(mi.decode_context_lens)
        return self._forward_naive(
            query, k_cache, v_cache,
            mi.decode_block_tables, mi.decode_context_lens, ones, None,
        )

    # ---- backend 1: naive gather (the oracle) -----------------------------

    def _forward_naive(self, query, k_cache, v_cache, block_tables, context_lens,
                       query_lens, qo_indptr=None):
        """Gather each sequence's blocks back into a dense tensor and call SDPA.

        Materialises the whole context per sequence, so it is slow and memory hungry.
        That is fine: it differs from the fast kernels in exactly one way (it does not
        touch the block table during the math), which is what makes it a useful oracle.
        """
        block_size = k_cache.shape[1]
        outputs = []
        offset = 0
        ctx_list = context_lens.tolist()
        qlen_list = query_lens.tolist()
        for i, (ctx, q_len) in enumerate(zip(ctx_list, qlen_list)):
            q = query[offset:offset + q_len]                       # (q_len, H, D)
            offset += q_len
            num_blocks = (ctx + block_size - 1) // block_size
            blocks = block_tables[i, :num_blocks]
            k = k_cache[blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:ctx]
            v = v_cache[blocks].reshape(-1, self.num_kv_heads, self.head_dim)[:ctx]
            if self.gqa_ratio > 1:
                k = k.repeat_interleave(self.gqa_ratio, dim=1)
                v = v.repeat_interleave(self.gqa_ratio, dim=1)

            # Bottom-right aligned causal mask: query j sits at absolute position
            # ctx - q_len + j. This is THE chunked-prefill correctness detail.
            if q_len == 1:
                mask = None
            else:
                pos = torch.arange(ctx - q_len, ctx, device=q.device).unsqueeze(1)
                keys = torch.arange(ctx, device=q.device).unsqueeze(0)
                mask = keys <= pos

            o = F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),      # (1, H, q_len, D)
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                attn_mask=mask,
                scale=self.scale,
            )
            outputs.append(o.squeeze(0).transpose(0, 1))           # (q_len, H, D)
        return torch.cat(outputs, dim=0)

    def _forward_cacheless(self, query, key, value):
        """Plain causal attention with no cache at all. Used by the M1 reference model
        so that path shares zero code with the paging machinery."""
        k = key.repeat_interleave(self.gqa_ratio, dim=1) if self.gqa_ratio > 1 else key
        v = value.repeat_interleave(self.gqa_ratio, dim=1) if self.gqa_ratio > 1 else value
        o = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            is_causal=True,
            scale=self.scale,
        )
        return o.squeeze(0).transpose(0, 1).contiguous()

    # ---- backend 2: FlashInfer -------------------------------------------

    def _forward_flashinfer_prefill(self, query, k_cache, v_cache, mi):
        # The wrapper was planned ONCE for this step by AttentionState.plan(); calling
        # plan() per layer would dominate the step time.
        return mi.fi_prefill.run(query, (k_cache, v_cache))

    def _forward_flashinfer_decode(self, query, k_cache, v_cache, mi):
