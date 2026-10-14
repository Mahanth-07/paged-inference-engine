"""Metric collection. These definitions are the field standard -- match them exactly
so your numbers are comparable to vLLM's published benchmarks.

    TTFT  time to first token          = first_token_time - arrival_time
    TPOT  time per output token        = (finish_time - first_token_time) / (output_len - 1)
