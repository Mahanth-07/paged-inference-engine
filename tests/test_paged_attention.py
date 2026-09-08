"""Paged attention tests.

The parametrization IS the point. 90% of paged attention bugs live in exactly these
axes, and specifically in the combination "context_len is not a multiple of block_size"
+ "GQA ratio > 1".

Everything that does not need a kernel runs on the MacBook: the pure-torch reference,
the block-table gather, reshape_and_cache, and the GQA head mapping. Only the Triton
