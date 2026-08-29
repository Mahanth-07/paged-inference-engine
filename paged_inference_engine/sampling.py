"""Sampling parameters and the sampler itself.

The sampler is fully batched: temperature/top-p/top-k are built as tensors once per
step and applied to the whole (num_seqs, vocab) logits matrix. A per-sequence Python
loop here silently becomes the bottleneck at batch 256 -- it is an embarrassing thing
to find in your own profile, and it is easy to avoid.
"""

