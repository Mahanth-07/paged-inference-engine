"""Async load generator. Points at ANY OpenAI-compatible server, so the exact same
client benchmarks paged-inference-engine, vLLM and TGI. That is what makes the comparison fair.

    python -m bench.client --backend paged-inference-engine --port 8000 \
        --dataset sharegpt --num-prompts 500 --request-rate 8 --out results/ns_r8.json

Methodology rules that make the numbers defensible. Skipping any of them turns the
result into a number nobody should believe, including you:

  * Poisson arrivals, not a fixed batch. A fixed batch measures a different system --
    one with no queueing -- and hides exactly the tail behaviour that matters.
  * ignore_eos + a fixed output length, so every system generates the same token count.
