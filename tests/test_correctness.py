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


# ---------------------------------------------------------------------------
# M1: does our modeling code compute the same function?
# ---------------------------------------------------------------------------


def test_logits_match_hf(engine, hf):
    """fp32, single forward pass, first 20 positions.

    If this fails, bisect layer by layer -- compare hidden states after the embedding,
    after layer 0's input norm, after attention, after the MLP. The first mismatch tells
    you exactly which component is wrong. Do not debug the whole model at once.
    """
    from paged_inference_engine.models.reference import ReferenceGenerator

    ids = engine.tokenizer.encode(PROMPT)[:20]
    with torch.inference_mode():
        want = hf(torch.tensor([ids], device="cuda")).logits[0].float()
    got = ReferenceGenerator(engine.model_runner.model, "cuda").forward(ids).float()
    torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)


def test_paged_engine_logits_match_the_cacheless_reference(engine):
    """Separates 'our model is wrong' from 'our paging is wrong'. If the previous test
    passes and this one fails, the bug is in the block table, not the math."""
    from paged_inference_engine.models.reference import ReferenceGenerator

    ids = engine.tokenizer.encode(PROMPT)
    want = ReferenceGenerator(engine.model_runner.model, "cuda").forward(ids)[-1]

    runner = engine.model_runner
    from paged_inference_engine.sampling import SamplingParams

    engine.add_request(prompt_token_ids=ids, sampling_params=SamplingParams(max_tokens=1))
    sched = engine.scheduler.schedule()
    mi, _ = runner.prepare_inputs(sched, engine.block_manager)
    with torch.inference_mode():
        h = runner.model(mi.input_ids, mi.positions, runner.kv_caches, mi)
        got = runner.model.compute_logits(h, mi.logits_indices)[0]
    torch.testing.assert_close(got.float(), want.float(), atol=1e-3, rtol=1e-3)
    engine.scheduler.abort(sched.all_seqs[0].seq_id)


def test_greedy_output_matches_hf_token_for_token(engine, hf):
    """64 greedy tokens, identical ids, in fp32.

    bf16 legitimately diverges after ~30 tokens from numerical drift alone, so this
    test runs in fp32. If it still diverges there, the bug is real.
    """
    ids = engine.tokenizer.encode(PROMPT)
    with torch.inference_mode():
        want = hf.generate(torch.tensor([ids], device="cuda"), max_new_tokens=64,
                           min_new_tokens=64, do_sample=False,
                           pad_token_id=engine.tokenizer.eos_token_id)
    want_ids = want[0, len(ids):].tolist()
    assert greedy(engine, ids, 64) == want_ids


# ---------------------------------------------------------------------------
# M2c: do all the backends agree?
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["naive", "flashinfer", "flash_attn", "triton"])
def test_all_backends_agree(backend):
    """Same prompt, same seed, greedy -- every backend produces identical text.

    This is the test that certifies the kernel. Skips (rather than fails) when the
    backend's library is not installed, so it is useful on any box.
    """
    if backend == "flashinfer":
        pytest.importorskip("flashinfer")
    if backend == "flash_attn":
        pytest.importorskip("flash_attn")
    if backend == "triton":
        pytest.importorskip("triton")

    ref = build_engine(dtype="float16", backend="naive")
    ids = ref.tokenizer.encode(PROMPT)
    want = greedy(ref, ids, 32)
    del ref
    torch.cuda.empty_cache()

    got = greedy(build_engine(dtype="float16", backend=backend), ids, 32)
    assert got == want, f"{backend} diverged from naive"


@pytest.mark.parametrize("block_size", [8, 16, 32, 64])
def test_block_size_does_not_change_output(block_size):
    """Block size is a memory/efficiency knob, not a semantic one. If it changes the
    answer, the slot mapping or the last-page masking is wrong."""
    ids = build_engine(dtype="float16").tokenizer.encode(PROMPT)
    want = greedy(build_engine(dtype="float16", block_size=16), ids, 24)
    torch.cuda.empty_cache()
    assert greedy(build_engine(dtype="float16", block_size=block_size), ids, 24) == want


# ---------------------------------------------------------------------------
# M3: scheduling must not change the answer
# ---------------------------------------------------------------------------


def test_batched_matches_unbatched(engine):
    """A prompt generated alone must produce the same tokens as when it is batched with
    31 others of different lengths.

    This catches the nastiest class of bug in the project: cross-sequence contamination
    from a wrong block table, attention mask, or query_start_loc. The output stays
    fluent, so you will not notice by reading it.
    """
    from paged_inference_engine.sampling import SamplingParams

    tok = engine.tokenizer
    prompts = [tok.encode(PROMPT[:n]) for n in (12, 20, 31, 44, 57, 63, 70, 77)] * 4
    sp = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)

    solo = [greedy(engine, p, 16) for p in prompts[:8]]
    batched = [o.token_ids for o in
               engine.generate(prompt_token_ids=[list(p) for p in prompts],
                               sampling_params=sp)]
    assert batched[:8] == solo
    assert batched[:8] == batched[8:16] == batched[16:24] == batched[24:32]


@pytest.mark.parametrize("budget", [16, 64, 4096])
def test_chunked_prefill_is_output_preserving(budget):
    """Chunked prefill splits a prompt across steps, so the later chunks' queries sit at
    absolute positions that are not 0..n. Bottom-right causal alignment is what makes
    that correct; top-left alignment passes the unchunked case and fails these."""
    e0 = build_engine(dtype="float16", max_num_batched_tokens=4096)
    ids = e0.tokenizer.encode(PROMPT * 4)
    want = greedy(e0, ids, 16)
    del e0
    torch.cuda.empty_cache()
    got = greedy(build_engine(dtype="float16", max_num_batched_tokens=budget), ids, 16)
    assert got == want


def test_preemption_is_output_preserving():
    """A KV cache small enough to force preemption must not change a single token. If
    it does, the recompute path is losing or duplicating tokens -- most likely by
    re-prefilling only the prompt and not the already-generated tokens."""
    from paged_inference_engine.sampling import SamplingParams

    e0 = build_engine(dtype="float16", num_gpu_blocks=4096)
    tok = e0.tokenizer
    prompts = [tok.encode(PROMPT[:n]) for n in (20, 40, 60, 77)]
    sp = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
    want = [o.token_ids for o in e0.generate(prompt_token_ids=[list(p) for p in prompts],
                                             sampling_params=sp)]
    del e0
    torch.cuda.empty_cache()

    tight = build_engine(dtype="float16", num_gpu_blocks=48, max_num_seqs=8,
                         max_num_batched_tokens=128)
    got = [o.token_ids for o in tight.generate(prompt_token_ids=[list(p) for p in prompts],
                                               sampling_params=sp)]
    assert tight.scheduler.num_preemptions > 0, "cache was not small enough to preempt"
    assert got == want


# ---------------------------------------------------------------------------
# M5
# ---------------------------------------------------------------------------

