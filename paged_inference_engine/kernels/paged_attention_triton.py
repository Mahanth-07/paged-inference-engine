"""Our own paged-attention kernels. This is what makes paged-inference-engine a kernels project.

Why decode attention specifically:
  - Decode has query length 1, so there is no big GEMM -- it is a pure memory sweep
    over the KV cache. Arithmetic intensity is roughly the batch size.
  - That makes it MEMORY BOUND. The optimisation target is percent of peak HBM
    bandwidth, not TFLOPs (4090 ~1008 GB/s, A100 80GB SXM ~2039 GB/s, H100 SXM
    ~3350 GB/s). `bench/microbench_attention.py --roofline` reports the achieved
    fraction; that number is the honest measure of this kernel.
  - Paging is what makes it non-trivial: one sequence's K/V is scattered across
    arbitrary physical blocks, so you cannot stride through memory -- you gather
    block ids from the block table and loop.

Three kernels here:
  1. _paged_attn_decode_kernel        one program per (seq, kv_head). The GQA_RATIO
                                      query heads that share a KV head are processed
                                      together, so their shared K/V is loaded ONCE
                                      instead of GQA_RATIO times. On Llama-3.2-1B
                                      (ratio 4) that is a 4x cut in KV traffic, and
                                      KV traffic is the whole cost.
  2. _paged_attn_decode_split_kernel  flash-decoding: split the sequence's KV across
     + _combine_kernel                SPLIT_K programs, each producing a partial
