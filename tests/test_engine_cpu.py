"""End-to-end engine correctness on the CPU fixture model.

These tests need no GPU and check block tables, slot
mapping, chunked prefill, preemption, cross-sequence contamination, and prefix cache
sharing. The oracle is `ReferenceGenerator`, which has no cache at all and therefore
cannot share a bug with the paged path.

Everything here is greedy, so "correct" means token-for-token identical, not close.
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from paged_inference_engine.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig  # noqa: E402
from paged_inference_engine.engine import LLMEngine  # noqa: E402
from paged_inference_engine.models.reference import ReferenceGenerator  # noqa: E402
from paged_inference_engine.sampling import SamplingParams  # noqa: E402

GREEDY = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)


def make_engine(**overrides) -> LLMEngine:
    cfg = EngineConfig(
        model=ModelConfig(model="__random__", dtype="float32", max_model_len=512,
                          hidden_size=128, num_layers=2, num_heads=4, num_kv_heads=2,
                          head_dim=32, intermediate_size=256, vocab_size=1000, seed=0),
        cache=CacheConfig(block_size=overrides.pop("block_size", 16),
                          num_gpu_blocks=overrides.pop("num_gpu_blocks", 256),
                          enable_prefix_caching=overrides.pop("prefix_caching", False)),
        scheduler=SchedulerConfig(
            max_num_batched_tokens=overrides.pop("max_num_batched_tokens", 128),
            max_num_seqs=overrides.pop("max_num_seqs", 8),
            enable_chunked_prefill=overrides.pop("chunked_prefill", True)),
        attention_backend="naive", device="cpu",
    )
    return LLMEngine(cfg)


PROMPTS = [
    [5, 6, 7, 8, 9, 10, 11, 12, 13],
    [1, 2, 3],
    list(range(20, 60)),
    [7] * 35,
    list(range(200, 217)),          # 17 tokens: one past a 16-block boundary
    list(range(300, 332)),          # 32 tokens: exactly two blocks
]


def gen_outputs(engine, prompts, sp=GREEDY):
    return engine.generate(prompt_token_ids=copy.deepcopy(prompts), sampling_params=sp)


def gen(engine, prompts, sp=GREEDY):
    return [o.token_ids for o in gen_outputs(engine, prompts, sp)]


# ---------------------------------------------------------------------------


def test_paged_engine_matches_cacheless_reference():
    """The M1 oracle. If this fails, nothing below it means anything."""
    engine = make_engine()
    ref = ReferenceGenerator(engine.model_runner.model, "cpu")
    got = gen(engine, PROMPTS)
    for prompt, tokens in zip(PROMPTS, got):
        assert tokens == ref.greedy(prompt, GREEDY.max_tokens), f"diverged on {prompt[:5]}..."


def test_logits_match_reference_not_just_argmax():
    """argmax agreement is a weak signal on a random-weight model: a broken attention
    can still pick the same token. Compare the logits themselves."""
    engine = make_engine()
    ref = ReferenceGenerator(engine.model_runner.model, "cpu")
    prompt = list(range(20, 60))
    want = ref.forward(prompt)[-1]

    runner = engine.model_runner
    engine.add_request(prompt_token_ids=prompt, sampling_params=GREEDY)
    sched = engine.scheduler.schedule()
    mi, _ = runner.prepare_inputs(sched, engine.block_manager)
    with torch.inference_mode():
