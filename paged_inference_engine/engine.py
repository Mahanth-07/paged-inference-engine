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
