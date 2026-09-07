"""Paged attention: the one layer that has to know about the block table.

Four backends, built in this order, each a fully working engine:

  1. "naive"      -- gather each sequence's blocks into a dense tensor and call SDPA.
                     Slow, obviously correct, runs on CPU/MPS, and is the oracle the
                     other three are tested against. Never delete it.
  2. "flashinfer" -- production paged kernels. Fast and correct, which unblocks M3/M4.
  3. "flash_attn" -- FA2/FA3 with a paged block table. A second opinion on (2).
  4. "triton"     -- our own kernels (kernels/paged_attention_triton.py).

