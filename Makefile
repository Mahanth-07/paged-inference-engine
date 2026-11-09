.PHONY: help install install-cpu install-gpu test test-cpu test-notorch test-gpu \
        sim sim-sweep bench bench-quick micro profile plot lint clean sync sync-pull serve

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

install:        ## laptop deps (torch is optional: the block manager and scheduler don't need it)
	python -m pip install -e ".[dev]"

install-cpu:    ## laptop deps + torch, so the CPU fixture-model tests run too
	python -m pip install -e ".[dev,cpu]"

install-gpu:    ## rented box deps
	python -m pip install -e ".[dev,gpu,bench]"

test:           ## all tests (GPU ones auto-skip without CUDA)
	python -m pytest tests/ -q

test-cpu:       ## everything that runs on the MacBook (GPU tests auto-skip)
	python -m pytest tests/ -q

test-notorch:   ## the pure-Python core: block manager + scheduler, no torch at all
	python -m pytest tests/test_block_manager.py tests/test_scheduler.py -q

test-gpu:       ## the GPU-only oracle: real weights vs HuggingFace
	python -m pytest tests/test_correctness.py tests/test_paged_attention.py -v

sim:            ## scheduler simulator, single rate
	python -m bench.scheduler_sim --num-requests 300 --request-rate 8

sim-sweep:      ## scheduler simulator, rate sweep -- run this on your laptop, free
	python -m bench.scheduler_sim --sweep --num-requests 300

serve:          ## start the OpenAI-compatible server (GPU box)
	python -m paged_inference_engine.entrypoints.api_server --model $(MODEL) --port 8000

bench:          ## full GPU benchmark sweep (rented box only, ~2h)
	bash bench/run_sweep.sh 2>&1 | tee results/sweep.log

bench-quick:    ## short sweep for sanity-checking the harness before spending 2h
	RATES="4 16" NUM_PROMPTS=100 REPEATS=1 bash bench/run_sweep.sh

micro:          ## attention + fused-kernel microbenchmarks (GPU)
	python -m bench.microbench_attention
	python -m bench.microbench_kernels

profile:        ## step-time breakdown: where the launch overhead is (GPU)
	python scripts/profile.py breakdown

plot:           ## regenerate figures from existing results/
	python -m bench.plot --results results --out docs/figures

lint:
	ruff check paged_inference_engine bench tests scripts

sync:           ## rsync this tree to $BOX
	bash scripts/sync.sh

sync-pull:      ## pull results/ and docs/figures/ back from $BOX
	bash scripts/sync.sh pull

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; rm -rf .pytest_cache build *.egg-info

MODEL ?= meta-llama/Llama-3.2-1B-Instruct
