"""The correctness oracle: a single-sequence generator with NO cache at all.

Recomputes the prefix for each generated token. It shares model weights with the
paged path but does not use its block tables or cache, so it helps isolate cache
bookkeeping errors from model errors. This path is intended for tests.

    ref = ReferenceGenerator(runner.model, device, dtype)
    ids = ref.greedy(prompt_ids, max_tokens=64)
"""

from __future__ import annotations

import torch

from ..model_runner import ModelInput
