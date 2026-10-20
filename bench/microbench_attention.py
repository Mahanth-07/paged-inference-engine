"""Attention backend microbenchmark and roofline check (GPU only).

    python -m bench.microbench_attention
    python -m bench.microbench_attention --roofline --context 8192 --batch 1

Decode attention is MEMORY BOUND: one token of query against a full sweep of the KV
cache. So the number that matters is not TFLOPs, it is achieved HBM bandwidth as a
fraction of peak. `--roofline` computes exactly that:

