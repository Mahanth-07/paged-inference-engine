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
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for system, by_rate in sorted(grouped.items()):
        rates = sorted(by_rate, key=lambda r: float(r))
        xs, ys, lo, hi = [], [], [], []
        for rate in rates:
            reps = by_rate[rate]
            tp = [r["output_throughput_tok_s"] for r in reps]
            lat = [r["p99_ttft_ms"] for r in reps]
            xs.append(_median(tp))
            ys.append(_median(lat))
            lo.append(_median(lat) - min(lat))
            hi.append(max(lat) - _median(lat))
        st = STYLE.get(system, {})
        ax.errorbar(xs, ys, yerr=[lo, hi], label=system, linewidth=2.5,
                    markersize=9, capsize=4, **st)
        for x, y, rate in zip(xs, ys, rates):
            ax.annotate(f"{rate}", (x, y), textcoords="offset points",
                        xytext=(6, -12), fontsize=9, alpha=0.75)

    ax.axhline(slo_ms, color="k", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.text(ax.get_xlim()[0], slo_ms * 1.06, f" p99 TTFT SLO = {slo_ms:.0f} ms",
            fontsize=11, alpha=0.8)
    # Log y: latency spans orders of magnitude as the system approaches saturation,
    # and the interesting part -- the knee -- is invisible on a linear axis.
    ax.set_yscale("log")
    ax.set_xlabel("Output throughput (tokens/s)", fontsize=14)
    ax.set_ylabel("p99 TTFT (ms, log scale)", fontsize=14)
    ax.set_title("Throughput vs tail latency\n(points labelled with request rate, req/s)",
                 fontsize=15)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "throughput_vs_latency.png")
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")

    # The headline sentence: where each curve crosses the SLO line.
    for system, by_rate in sorted(grouped.items()):
        best = 0.0
        for rate, reps in by_rate.items():
            if _median([r["p99_ttft_ms"] for r in reps]) <= slo_ms:
                best = max(best, _median([r["output_throughput_tok_s"] for r in reps]))
        print(f"  {system:<20} sustains {best:8.1f} tok/s under a "
              f"{slo_ms:.0f} ms p99 TTFT SLO")


def plot_itl_cdf(grouped, out_dir):
    """Percentile curve of ITL. The mean hides exactly the stalls chunked prefill fixes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    pcts = [50, 75, 90, 95, 99]
    drew = False
    for system, by_rate in sorted(grouped.items()):
        mid = sorted(by_rate, key=lambda r: float(r))
        if not mid:
            continue
        rate = mid[len(mid) // 2]
        reps = by_rate[rate]
        ys = [_median([r[f"p{p}_itl_ms"] for r in reps if f"p{p}_itl_ms" in r] or [float("nan")])
              for p in pcts]
        if all(y != y for y in ys):   # all NaN
            continue
        ax.plot(pcts, ys, label=f"{system} @ {rate} req/s", linewidth=2.5,
                markersize=9, **STYLE.get(system, {}))
        drew = True
    if not drew:
        return
    ax.set_xlabel("Percentile", fontsize=14)
    ax.set_ylabel("Inter-token latency (ms)", fontsize=14)
    ax.set_title("ITL by percentile -- the tail is what users feel", fontsize=15)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "itl_cdf.png")
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def plot_ablation(results_dir, out_dir, prefix, xlabel, title, name, cast=str):
    import matplotlib.pyplot as plt

    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, f"ablation_{prefix}*.json"))):
        key = os.path.basename(path)[len(f"ablation_{prefix}"):-5]
        with open(path) as f:
            payload = json.load(f)
        if "error" in payload:
            continue
        rows.append((key, payload))
    if not rows:
        return
    try:
        rows.sort(key=lambda kv: float(kv[0]))
    except ValueError:
        rows.sort()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    labels = [k for k, _ in rows]
    a1.bar(labels, [p["output_throughput_tok_s"] for _, p in rows], color="#1f77b4")
    a1.set_ylabel("Output throughput (tok/s)", fontsize=13)
    a1.set_xlabel(xlabel, fontsize=13)
    a2.bar(labels, [p["p99_itl_ms"] for _, p in rows], color="#d62728")
    a2.set_ylabel("p99 ITL (ms)", fontsize=13)
    a2.set_xlabel(xlabel, fontsize=13)
    for ax in (a1, a2):
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(labelsize=11)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def write_table(grouped, out_path):
    """Markdown table for BENCHMARKS.md. The graph is the headline; this is the receipt."""
    cols = ["output_throughput_tok_s", "request_throughput_rps", "goodput_rps",
            "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_itl_ms"]
    head = ("| system | rate | " + " | ".join(c.replace("_", " ") for c in cols) + " |\n"
            "|---" * (len(cols) + 2) + "|\n")
    lines = [head]
    for system, by_rate in sorted(grouped.items()):
        for rate in sorted(by_rate, key=lambda r: float(r)):
            reps = by_rate[rate]
            vals = [f"{_median([r[c] for r in reps]):.1f}" for c in cols]
            lines.append(f"| {system} | {rate} | " + " | ".join(vals) + " |\n")
    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--slo-ms", type=float, default=500.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grouped = load_results(args.results)
