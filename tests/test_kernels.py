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
    torch.testing.assert_close(out, fused.rms_norm_torch(res + x, w, 1e-6))


def test_silu_and_mul_matches_unfused():
    gate_up = torch.randn(6, 64)
    gate, up = gate_up.chunk(2, dim=-1)
    torch.testing.assert_close(fused.silu_and_mul_torch(gate_up),
                               torch.nn.functional.silu(gate) * up)


def test_rope_is_a_rotation():
    """RoPE preserves the norm of each half-pair; if it doesn't, the cos/sin pairing
    is wrong (a classic: interleaved vs half-split layout)."""
    q = torch.randn(3, 2, 8)
    k = torch.randn(3, 1, 8)
    ang = torch.randn(3, 4)
    q2, _ = fused.apply_rope_torch(q, k, ang.cos(), ang.sin())
    a, b = q.chunk(2, dim=-1)
    a2, b2 = q2.chunk(2, dim=-1)
    torch.testing.assert_close((a ** 2 + b ** 2), (a2 ** 2 + b2 ** 2), atol=1e-5, rtol=1e-5)


def test_rope_at_position_zero_is_identity():
    q, k = torch.randn(2, 2, 8), torch.randn(2, 1, 8)
    zeros = torch.zeros(2, 4)
    q2, k2 = fused.apply_rope_torch(q, k, torch.ones_like(zeros), zeros)
    torch.testing.assert_close(q2, q)
    torch.testing.assert_close(k2, k)


def test_llama3_rope_scaling_leaves_short_wavelengths_alone():
    """The Llama 3.1/3.2 rescale only touches long wavelengths. If it changed the high
    frequencies too, short-context output would break -- and that would be obvious.
    The dangerous direction is the opposite: silently not applying it at all."""
    from paged_inference_engine.models.llama import _llama3_scaled_inv_freq

    inv = 1.0 / (500000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    scaled = _llama3_scaled_inv_freq(inv, 32.0, 1.0, 4.0, 8192)
    assert torch.allclose(scaled[0], inv[0]), "high frequency was rescaled"
    assert scaled[-1] < inv[-1], "low frequency was NOT rescaled"


@CUDA
@pytest.mark.parametrize("hidden", [128, 2048, 3072])
def test_triton_rms_norm_matches_torch(hidden):
    x = torch.randn(17, hidden, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(fused.rms_norm(x, w, 1e-6),
                               fused.rms_norm_torch(x, w, 1e-6), atol=2e-2, rtol=2e-2)


@CUDA
def test_triton_fused_add_rms_norm_matches_torch():
    x = torch.randn(9, 2048, dtype=torch.bfloat16, device="cuda")
    res = torch.randn_like(x)
    w = torch.randn(2048, dtype=torch.bfloat16, device="cuda")
    o1, r1 = fused.fused_add_rms_norm(x, res.clone(), w, 1e-6)
    o2, r2 = fused.fused_add_rms_norm_torch(x, res.clone(), w, 1e-6)
    torch.testing.assert_close(r1, r2, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(o1, o2, atol=2e-2, rtol=2e-2)

