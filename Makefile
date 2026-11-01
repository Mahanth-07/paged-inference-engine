.PHONY: help install install-cpu install-gpu test test-cpu test-notorch test-gpu \
        sim sim-sweep bench bench-quick micro profile plot lint clean sync sync-pull serve

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

install:        ## laptop deps (torch is optional: the block manager and scheduler don't need it)
	pip install -e ".[dev]"

install-cpu:    ## laptop deps + torch, so the CPU fixture-model tests run too
	pip install -e ".[dev,cpu]"

install-gpu:    ## rented box deps
	pip install -e ".[dev,gpu,bench]"
