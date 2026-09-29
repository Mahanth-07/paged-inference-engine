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
