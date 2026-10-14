"""CPU scheduler simulator with a configurable model-latency estimate.

It drives the real Scheduler and the real BlockManager with a fake model runner whose
latency comes from a cost model. That means you can tune scheduling policy, block size,
token budgets, and chunked-prefill settings against realistic traces in seconds instead
of burning GPU-hours on a rented box.

