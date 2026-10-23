"""Turn results/*.json into the figures that go at the top of the README.

The headline figure is a THROUGHPUT vs P99-LATENCY curve, one line per system, points
labelled by request rate. Not a bar chart. A bar chart of "tokens/sec" hides the
operating point and invites the one question you cannot then answer: "at what latency?"

    python -m bench.plot --results results --out docs/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

FIGURES = {
    "throughput_vs_latency": "Output throughput (tok/s) vs p99 TTFT (ms), per system",
    "itl_cdf": "ITL distribution at a fixed rate -- chunked prefill's effect on stalls",
    "block_size_ablation": "Block size vs throughput and KV-cache utilization",
    "cuda_graph_ablation": "Step latency vs batch size, graphs on/off",
    "batching_ablation": "Continuous vs static batching at matched output-length variance",
}

STYLE = {
    "paged-inference-engine": dict(color="#1f77b4", marker="o"),
    "vllm": dict(color="#d62728", marker="s"),
    "hf_static": dict(color="#7f7f7f", marker="^"),
    "hf_compile": dict(color="#bcbd22", marker="v"),
    "paged-inference-engine-nopaging": dict(color="#9467bd", marker="D"),
}


def load_results(results_dir: str) -> dict:
    """Group results/{system}_rate{r}_rep{n}.json by system and request rate."""
    grouped: dict = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        m = re.match(r"(?P<sys>[a-z_0-9-]+)_rate(?P<rate>[\d.]+|inf)_rep(?P<rep>\d+)\.json",
                     os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            payload = json.load(f)
        if "error" in payload:
            continue
        grouped[m["sys"]][m["rate"]].append(payload)
    return grouped


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def plot_throughput_vs_latency(grouped, out_dir, slo_ms=500.0):
