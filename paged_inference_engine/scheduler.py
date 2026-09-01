"""Iteration-level scheduler: the thing that makes this "continuous batching".

Static batching (what HF `generate` does): pick N prompts, run until the LONGEST one
finishes, then start the next batch. Every sequence that finished early wastes its slot
for the rest of the batch. At realistic output-length variance this wastes most of the GPU.

Continuous batching: reschedule every single forward pass. A sequence that emits EOS is
