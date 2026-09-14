"""End-to-end correctness against HuggingFace, on real Llama weights. GPU required.

Compare logits and generated tokens against HuggingFace to check model math
independently of the paging tests.

The engine-level invariants (batched == unbatched, preemption preserves output, chunked
prefill preserves output, prefix caching preserves output) are ALSO tested on the CPU
fixture in tests/test_engine_cpu.py, which runs on a laptop in seconds. These versions
re-check them on real weights, where numerical drift is a genuine second failure mode.

    PAGED_INFERENCE_ENGINE_TEST_MODEL=meta-llama/Llama-3.2-1B-Instruct pytest tests/test_correctness.py
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

MODEL = os.environ.get("PAGED_INFERENCE_ENGINE_TEST_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
PROMPT = "The capital of France is Paris, and the capital of Germany is"


def build_engine(dtype="float32", backend="naive", **kw):
