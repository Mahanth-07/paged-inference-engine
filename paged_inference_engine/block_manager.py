"""Paged KV cache memory management.

Pure Python with no torch dependency, so allocation can be tested without a GPU.

Mental model
------------
Physical memory is a flat pool of `num_blocks` blocks, each holding `block_size`
tokens' worth of K and V for every layer. A sequence owns a *block table*: a list
mapping its logical block i to some physical block id. Blocks need not be contiguous,
which is the entire point -- it removes internal fragmentation and lets sequences
grow without reserving max_model_len up front.

Reference counting exists so two sequences can share blocks: identical prefixes
(prefix caching) or forked sequences (beam search / parallel sampling). A block
returns to the free list only when its refcount hits zero.
"""
