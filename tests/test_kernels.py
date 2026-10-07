"""Fused elementwise kernels: the Triton path must agree with the torch path.

The torch implementations are not dead fallbacks -- they are the oracle, and they are
what the MacBook runs. The Triton comparisons need CUDA and skip cleanly without it.
"""

import pytest

torch = pytest.importorskip("torch")

from paged_inference_engine.kernels import fused  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_rms_norm_torch_matches_hand_computation():
    x = torch.randn(5, 16)
    w = torch.randn(16)
    got = fused.rms_norm_torch(x, w, 1e-6)
    want = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_rms_norm_accumulates_in_fp32():
    """An fp32 accumulator is not optional even in a bf16 model: bf16 has 8 mantissa
    bits, and mean(x^2) over 2048 elements loses most of them."""
    x = (torch.randn(1, 2048) * 8).to(torch.bfloat16)
    w = torch.ones(2048, dtype=torch.bfloat16)
    got = fused.rms_norm_torch(x, w, 1e-6).float()
    want = fused.rms_norm_torch(x.float(), w.float(), 1e-6)
    torch.testing.assert_close(got, want, atol=5e-2, rtol=5e-2)


def test_fused_add_rms_norm_matches_unfused():
    x, res, w = torch.randn(4, 32), torch.randn(4, 32), torch.randn(32)
    out, new_res = fused.fused_add_rms_norm_torch(x, res.clone(), w, 1e-6)
    torch.testing.assert_close(new_res, res + x)
