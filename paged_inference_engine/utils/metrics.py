"""Metric collection. These definitions are the field standard -- match them exactly
so your numbers are comparable to vLLM's published benchmarks.

    TTFT  time to first token          = first_token_time - arrival_time
    TPOT  time per output token        = (finish_time - first_token_time) / (output_len - 1)
    ITL   inter-token latency          = the full distribution of gaps between tokens
    E2EL  end-to-end latency           = finish_time - arrival_time

Report percentiles, not just means. A mean ITL hides exactly the stalls that chunked
prefill is designed to fix, and p99 is what a user actually experiences.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional

from ..sequence import Sequence


@dataclass
class RequestMetrics:
    seq_id: int
    prompt_len: int
