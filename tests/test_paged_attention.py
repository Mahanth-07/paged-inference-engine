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
