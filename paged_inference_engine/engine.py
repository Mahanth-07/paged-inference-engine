"""The engine loop. Everything else is policy; this is the crank you turn.

    engine = LLMEngine(EngineConfig())
    engine.add_request(prompt="hello world", sampling_params=SamplingParams(max_tokens=32))
    while engine.has_unfinished_requests():
        for out in engine.step():
            print(out.delta_text, end="")

One `step()` == one forward pass == one scheduling decision. That is what makes this
continuous batching rather than static batching: there is no notion of "a batch"
that has to drain before the next one starts.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

from .block_manager import BlockManager
from .config import EngineConfig
from .sampling import SamplingParams
from .scheduler import Scheduler, SchedulerOutput
from .sequence import RequestOutput, Sequence, SequenceStatus
from .utils.metrics import MetricsCollector


class Detokenizer:
    """Incremental detokenization for streaming.

    A token boundary is not a character boundary: a multi-byte UTF-8 codepoint can
    span two tokens, and emitting the first half produces a replacement character in
    the user's terminal. So we hold back any trailing partial codepoint until the
    next token completes it.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._text: dict[int, str] = {}

    def decode_delta(self, seq: Sequence) -> tuple[str, str]:
        if self.tokenizer is None:
            return "", ""
        full = self.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True)
        prev = self._text.get(seq.seq_id, "")
        if full.endswith("�"):
            return prev, ""            # incomplete codepoint: wait for the next token
        self._text[seq.seq_id] = full
        return full, full[len(prev):]

    def finish(self, seq: Sequence) -> str:
        text = self._text.pop(seq.seq_id, "")
        return text


class LLMEngine:
    def __init__(self, config: EngineConfig, model_runner=None):
        self.config = config
        self.metrics = MetricsCollector()

        # model_runner is injectable so bench/scheduler_sim.py can pass a fake one and
        # scheduling policy can be tuned on a laptop with no GPU.
        if model_runner is None:
            from .model_runner import ModelRunner

            model_runner = ModelRunner(config)
        self.model_runner = model_runner

        # profile_num_blocks() both sizes AND allocates the cache -- the two must agree,
        # so they are deliberately not separate calls the caller can get out of sync.
        num_blocks = self.model_runner.profile_num_blocks()

        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=config.cache.block_size,
            enable_prefix_caching=config.cache.enable_prefix_caching,
        )
        self.scheduler = Scheduler(config.scheduler, self.block_manager)
