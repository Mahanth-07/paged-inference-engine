"""Offline CLI. Handy for eyeballing output without standing up the server.

    paged-inference-engine generate --prompt "The capital of France is" --max-tokens 32
    paged-inference-engine info
    paged-inference-engine throughput --num-prompts 200 --input-len 512 --output-len 128
"""

from __future__ import annotations

