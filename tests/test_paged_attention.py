"""Paged attention tests.

The parametrization IS the point. 90% of paged attention bugs live in exactly these
axes, and specifically in the combination "context_len is not a multiple of block_size"
+ "GQA ratio > 1".

Everything that does not need a kernel runs on the MacBook: the pure-torch reference,
the block-table gather, reshape_and_cache, and the GQA head mapping. Only the Triton
comparisons need CUDA, and they skip cleanly without it.
"""

import pytest

torch = pytest.importorskip("torch")

from paged_inference_engine.kernels.paged_attention_triton import (  # noqa: E402
    reference_paged_attention_decode,
    reference_paged_attention_prefill,
)
from paged_inference_engine.layers.attention import (  # noqa: E402
    PagedAttention,
    reshape_and_cache_torch,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def make_paged_fixture(context_lens, num_heads, num_kv_heads, head_dim, block_size,
                       dtype=torch.float32, device="cpu", seed=0, query_lens=None):
    """Build a paged cache whose blocks are SHUFFLED and NON-CONTIGUOUS.

    If the fixture used contiguous block ids, every one of these tests would pass with
    a completely broken block-table lookup. Shuffling is what makes them mean something.
    Returns the paged cache plus a dense per-sequence view for independent checking.
    """
    g = torch.Generator().manual_seed(seed)
    pages = [(c + block_size - 1) // block_size for c in context_lens]
    total_pages = sum(pages) + 5
    perm = torch.randperm(total_pages, generator=g)

    k_cache = torch.randn(total_pages, block_size, num_kv_heads, head_dim,
                          generator=g, dtype=torch.float32).to(dtype).to(device)
    v_cache = torch.randn(total_pages, block_size, num_kv_heads, head_dim,
                          generator=g, dtype=torch.float32).to(dtype).to(device)

    tables, cursor = [], 0
    for p in pages:
        tables.append(perm[cursor:cursor + p].tolist())
        cursor += p
    max_pages = max(pages)
    block_tables = torch.tensor([t + [0] * (max_pages - len(t)) for t in tables],
                                dtype=torch.int32, device=device)
    ctx = torch.tensor(context_lens, dtype=torch.int32, device=device)

    qlens = query_lens or [1] * len(context_lens)
    total_q = sum(qlens)
    query = torch.randn(total_q, num_heads, head_dim,
                        generator=g, dtype=torch.float32).to(dtype).to(device)
    return query, k_cache, v_cache, block_tables, ctx, qlens


def dense_reference(query, k_cache, v_cache, block_tables, context_lens, qlens, scale):
    """Independent oracle: gather with plain Python indexing and do the math by hand.

    Deliberately does NOT reuse reference_paged_attention_*: two implementations that
    share a helper share its bugs.
    """
    block_size = k_cache.shape[1]
    num_heads, head_dim = query.shape[1], query.shape[2]
    num_kv_heads = k_cache.shape[2]
    ratio = num_heads // num_kv_heads
    outs = []
    offset = 0
    for i, q_len in enumerate(qlens):
        ctx = int(context_lens[i])
        q = query[offset:offset + q_len].float()
        offset += q_len
        ks, vs = [], []
        for pos in range(ctx):
            blk = int(block_tables[i, pos // block_size])
            ks.append(k_cache[blk, pos % block_size].float())
            vs.append(v_cache[blk, pos % block_size].float())
        k = torch.stack(ks)                       # (ctx, kv_heads, dim)
        v = torch.stack(vs)
        out = torch.empty(q_len, num_heads, head_dim, dtype=torch.float32)
        for h in range(num_heads):
            kh = k[:, h // ratio]                  # (ctx, dim)
            vh = v[:, h // ratio]
            logits = (q[:, h] @ kh.T) * scale      # (q_len, ctx)
            for j in range(q_len):
                allowed = ctx - q_len + j + 1      # bottom-right causal
                row = logits[j, :allowed].softmax(-1)
                out[j, h] = row @ vh[:allowed]
        outs.append(out)
    return torch.cat(outs).to(query.dtype)


# ---------------------------------------------------------------------------
# reference correctness -- runs on the MacBook
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (8, 2), (4, 1)])
@pytest.mark.parametrize("context_lens", [
    [1],                       # single token
    [16],                      # exactly one full block
    [17],                      # one token into a second block  <-- the classic bug
    [15, 16, 17, 100, 129],    # ragged batch spanning boundaries
    [257],                     # long-ish, exercises the block loop
])
def test_reference_decode_matches_dense(block_size, num_heads, num_kv_heads, context_lens):
    q, kc, vc, bt, ctx, qlens = make_paged_fixture(
        context_lens, num_heads, num_kv_heads, 32, block_size)
    scale = 32 ** -0.5
    got = reference_paged_attention_decode(q, kc, vc, bt, ctx, scale)
    want = dense_reference(q, kc, vc, bt, ctx, qlens, scale)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 2), (4, 4)])
@pytest.mark.parametrize("context_lens,query_lens", [
    ([10], [10]),              # fresh prefill: chunk starts at position 0
    ([37], [5]),               # chunked prefill: queries sit at positions 32..36
    ([64], [64]),              # exact block multiple
    ([33, 65, 17], [33, 8, 1]),  # ragged mix of fresh and continued chunks
])
def test_reference_prefill_matches_dense(block_size, num_heads, num_kv_heads,
                                         context_lens, query_lens):
    """The chunked cases are the ones that matter: query j of a chunk attends to keys
    0..(ctx - q_len + j), not 0..j. Top-left causal masking passes the first case and
    fails every other one."""
    q, kc, vc, bt, ctx, _ = make_paged_fixture(
        context_lens, num_heads, num_kv_heads, 32, block_size, query_lens=query_lens)
    scale = 32 ** -0.5
    ql = torch.tensor(query_lens, dtype=torch.int32)
    got = reference_paged_attention_prefill(q, kc, vc, bt, ctx, ql, scale)
    want = dense_reference(q, kc, vc, bt, ctx, query_lens, scale)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_naive_backend_matches_reference():
    """PagedAttention's `naive` backend is the engine's oracle, so it gets its own test
    rather than being trusted because it looks obviously right."""
    ctx_lens, q_lens = [33, 65, 17], [33, 8, 1]
    q, kc, vc, bt, ctx, _ = make_paged_fixture(ctx_lens, 8, 2, 32, 16, query_lens=q_lens)
    attn = PagedAttention(8, 2, 32, backend="naive")
    qo = torch.tensor([0, 33, 41, 42], dtype=torch.int32)
    got = attn._forward_naive(q, kc, vc, bt, ctx,
                              torch.tensor(q_lens, dtype=torch.int32), qo)
    want = dense_reference(q, kc, vc, bt, ctx, q_lens, 32 ** -0.5)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_gqa_head_mapping():
    """Query head h must attend to kv head h // (num_heads // num_kv_heads).

    Constructed so the answer is unmistakable: kv head g is filled with the constant g,
    so the attention output for query head h IS the index of the kv head it read.
    """
    num_heads, num_kv_heads, head_dim, block_size = 8, 2, 4, 16
    ratio = num_heads // num_kv_heads
    k_cache = torch.zeros(4, block_size, num_kv_heads, head_dim)
    v_cache = torch.zeros(4, block_size, num_kv_heads, head_dim)
    for g in range(num_kv_heads):
        v_cache[:, :, g, :] = float(g)
    query = torch.randn(1, num_heads, head_dim)
    bt = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32)
    ctx = torch.tensor([16], dtype=torch.int32)

    out = reference_paged_attention_decode(query, k_cache, v_cache, bt, ctx, 1.0)
    for h in range(num_heads):
        expected = float(h // ratio)
        assert torch.allclose(out[0, h], torch.full((head_dim,), expected), atol=1e-5), (
            f"query head {h} read kv head {out[0, h, 0].item():.0f}, expected {expected:.0f}"
        )


def test_partial_last_block_is_masked():
    """A context length that is not a multiple of block_size leaves garbage in the tail
    of the last page. If it is not masked, the output changes when that garbage does --
    which is exactly what this test forces."""
    q, kc, vc, bt, ctx, _ = make_paged_fixture([17], 4, 4, 32, 16)
    scale = 32 ** -0.5
    before = reference_paged_attention_decode(q, kc, vc, bt, ctx, scale).clone()
    last_block = int(bt[0, 1])
    kc[last_block, 1:] = 1e3          # poison positions 17.. of the last page
    vc[last_block, 1:] = 1e3
    after = reference_paged_attention_decode(q, kc, vc, bt, ctx, scale)
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)


def test_reshape_and_cache_writes_to_the_right_slots():
    """Non-monotonic slots on purpose: that is what a real ragged batch produces, and a
    scatter that quietly assumes sorted indices will pass a monotonic test."""
    num_kv_heads, head_dim, block_size, num_blocks = 2, 4, 8, 6
    kc = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
    vc = torch.zeros_like(kc)
    slots = torch.tensor([37, 2, 19, 8, 40, 3], dtype=torch.int64)
    key = torch.arange(len(slots) * num_kv_heads * head_dim, dtype=torch.float32) \
        .view(len(slots), num_kv_heads, head_dim)
    value = key * -1

    reshape_and_cache_torch(key, value, kc, vc, slots)
    for i, slot in enumerate(slots.tolist()):
        b, off = slot // block_size, slot % block_size
        torch.testing.assert_close(kc[b, off], key[i])
        torch.testing.assert_close(vc[b, off], value[i])
