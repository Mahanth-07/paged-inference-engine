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
    from paged_inference_engine.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
    from paged_inference_engine.engine import LLMEngine

    cfg = EngineConfig(
        model=ModelConfig(model=MODEL, dtype=dtype,
                          max_model_len=kw.pop("max_model_len", 2048)),
        cache=CacheConfig(block_size=kw.pop("block_size", 16),
                          num_gpu_blocks=kw.pop("num_gpu_blocks", 2048),
                          enable_prefix_caching=kw.pop("prefix_caching", False)),
        scheduler=SchedulerConfig(
            max_num_batched_tokens=kw.pop("max_num_batched_tokens", 2048),
            max_num_seqs=kw.pop("max_num_seqs", 64),
            enable_chunked_prefill=kw.pop("chunked_prefill", True)),
        attention_backend=backend, device="cuda", enforce_eager=kw.pop("eager", True),
    )
    return LLMEngine(cfg)


@pytest.fixture(scope="module")
def engine():
    return build_engine()


@pytest.fixture(scope="module")
def hf():
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).cuda().eval()
    yield m
    del m
    torch.cuda.empty_cache()


def greedy(engine, prompt_ids, n):
    from paged_inference_engine.sampling import SamplingParams

    out = engine.generate(prompt_token_ids=[list(prompt_ids)],
                          sampling_params=SamplingParams(temperature=0.0, max_tokens=n,
                                                         ignore_eos=True))
    return out[0].token_ids


