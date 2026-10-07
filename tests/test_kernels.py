"""Fused elementwise kernels: the Triton path must agree with the torch path.

The torch implementations are not dead fallbacks -- they are the oracle, and they are
what the MacBook runs. The Triton comparisons need CUDA and skip cleanly without it.
"""

import pytest

torch = pytest.importorskip("torch")

from paged_inference_engine.kernels import fused  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_rms_norm_torch_matches_hand_computation():
