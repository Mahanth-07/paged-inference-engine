"""HuggingFace `generate` baselines -- the lines paged-inference-engine must beat.

Two variants, because "beats HF" is only impressive if the HF config was reasonable:

  1. STATIC BATCHING (the naive line)
     Fixed batches, generate until the LONGEST member finishes. With realistic
     output-length variance this wastes most of the GPU: a sequence that finishes in
     20 tokens still occupies a slot for all 500 steps. This script measures the waste
     directly and reports it:
         batch_utilization = sum(output_lens) / (batch_size * max(output_lens))
     On ShareGPT that is routinely under 40%. That number is the entire argument for
     continuous batching, and it is far more persuasive than a speedup ratio.

  2. STATIC CACHE + torch.compile (the fair line)
     Removes cache reallocation and launch overhead, so what is left is genuinely the
     batching and memory-management difference -- which is the thing paged-inference-engine built.

Also reported: peak memory and bytes-of-KV-cache-per-served-token. HF pads the KV
cache to max_len for every sequence in the batch; paging does not. That is a memory
story, not a speed story, and it is the more fundamental one.

    python -m bench.baselines.hf_generate --model ... --num-prompts 500 \
        --mode static --out results/hf_static_rateinf_rep1.json
"""

from __future__ import annotations

import argparse
import json
