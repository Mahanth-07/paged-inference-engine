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
