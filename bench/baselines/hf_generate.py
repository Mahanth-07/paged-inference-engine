"""HuggingFace `generate` baselines -- the lines paged-inference-engine must beat.

Two variants, because "beats HF" is only impressive if the HF config was reasonable:

  1. STATIC BATCHING (the naive line)
     Fixed batches, generate until the LONGEST member finishes. With realistic
     output-length variance this wastes most of the GPU: a sequence that finishes in
     20 tokens still occupies a slot for all 500 steps. This script measures the waste
     directly and reports it:
