"""CPU scheduler simulator with a configurable model-latency estimate.

It drives the real Scheduler and the real BlockManager with a fake model runner whose
latency comes from a cost model. That means you can tune scheduling policy, block size,
token budgets, and chunked-prefill settings against realistic traces in seconds instead
of burning GPU-hours on a rented box.

    python -m bench.scheduler_sim --num-requests 500 --request-rate 8

Sanity-check the cost model against real measurements once you have a GPU engine
working, then trust it for policy exploration.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

from paged_inference_engine.block_manager import BlockManager
from paged_inference_engine.config import SchedulerConfig
from paged_inference_engine.sampling import SamplingParams
from paged_inference_engine.scheduler import Scheduler
from paged_inference_engine.sequence import Sequence
from paged_inference_engine.utils.metrics import MetricsCollector


@dataclass
class CostModel:
    """Crude but directionally correct. Calibrate against a real box in week 4.

    prefill: compute-bound, ~linear in tokens
    decode:  memory-bound, ~constant per step (weight sweep) + small per-seq term
             (KV sweep) -- which is exactly why batching helps decode so much
    """

    prefill_us_per_token: float = 22.0
    decode_base_us: float = 3800.0
    decode_us_per_seq: float = 21.0
    decode_us_per_kv_token: float = 0.004

    def step_time(self, prefill_tokens: int, decode_seqs: int, decode_kv_tokens: int) -> float:
        t = self.prefill_us_per_token * prefill_tokens
        if decode_seqs:
            t += (
                self.decode_base_us
                + self.decode_us_per_seq * decode_seqs
                + self.decode_us_per_kv_token * decode_kv_tokens
            )
        return t / 1e6


def make_trace(n: int, rate: float, rng: random.Random) -> list[tuple[float, int, int]]:
    """(arrival_time, prompt_len, output_len), Poisson arrivals, lognormal lengths.

    Lognormal is the right shape: real chat traffic has a heavy tail of long prompts,
    and it is exactly that tail that makes chunked prefill and preemption matter.
    Uniform lengths make your scheduler look better than it is.
    """
    trace = []
    t = 0.0
    for _ in range(n):
        t += rng.expovariate(rate) if rate > 0 else 0.0
        prompt_len = max(8, min(4000, int(rng.lognormvariate(6.0, 0.9))))
        output_len = max(1, min(1024, int(rng.lognormvariate(4.8, 0.8))))
        trace.append((t, prompt_len, output_len))
    return trace


def simulate(trace, sched_cfg: SchedulerConfig, num_blocks: int, block_size: int,
             cost: CostModel) -> dict:
    bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
    scheduler = Scheduler(sched_cfg, bm)
    metrics = MetricsCollector()

    pending = list(trace)
    now = 0.0
    step = 0
    max_steps = 2_000_000

    while (pending or scheduler.has_unfinished) and step < max_steps:
        step += 1
        while pending and pending[0][0] <= now:
            arrival, plen, olen = pending.pop(0)
            seq = Sequence(
                prompt_token_ids=[0] * plen,
                sampling_params=SamplingParams(max_tokens=olen, ignore_eos=True),
                arrival_time=arrival,
            )
            scheduler.add_seq(seq)

        out = scheduler.schedule()
        if out.is_empty:
            if pending:
                now = pending[0][0]
                continue
            break

        prefill_tokens = sum(s.num_tokens for s in out.prefills)
        decode_kv = sum(s.seq.total_len for s in out.decodes)
        dt = cost.step_time(prefill_tokens, len(out.decodes), decode_kv)
        now += dt

        for s in out.prefills:
            s.seq.num_computed_tokens += s.num_tokens
            if s.seq.num_computed_tokens >= s.seq.total_len:
                s.seq.append_token(0, now=now)
        for s in out.decodes:
            s.seq.append_token(0, now=now)

        metrics.record_step(
            num_batched_tokens=out.num_batched_tokens,
            num_prefill_seqs=len(out.prefills),
            num_decode_seqs=len(out.decodes),
            step_latency=dt,
            kv_utilization=bm.utilization,
            num_preempted=len(out.preempted),
        )
        for seq in scheduler.free_finished(eos_token_id=None):
            seq.finish_time = now
            metrics.record_request(seq)

    s = metrics.summary(duration=now)
    s["goodput_rps"] = metrics.goodput(duration=now)
    s.update(scheduler.stats())
    # Report what did NOT complete. Without this, a run where the scheduler rejected or
    # stalled most of the trace looks merely "slow" instead of "dropped the workload",
    # which is a very different thing.
    s["num_submitted"] = len(trace)
    s["num_completed"] = len(metrics.requests)
    s["num_unfinished"] = len(trace) - len(metrics.requests) - len(pending)
    s["num_never_admitted"] = len(pending)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-requests", type=int, default=300)
    ap.add_argument("--request-rate", type=float, default=8.0)
    ap.add_argument("--num-blocks", type=int, default=8192)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--no-chunked-prefill", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="Sweep request rate and print a throughput/latency table.")
    args = ap.parse_args()

    cfg = SchedulerConfig(
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=not args.no_chunked_prefill,
    )
    cost = CostModel()

    rates = [1, 2, 4, 8, 12, 16, 24, 32] if args.sweep else [args.request_rate]
    rows = []
    for rate in rates:
        rng = random.Random(args.seed)
        trace = make_trace(args.num_requests, rate, rng)
        s = simulate(trace, cfg, args.num_blocks, args.block_size, cost)
        rows.append((rate, s))

    if args.sweep:
        hdr = (f"{'rate':>6} {'out tok/s':>10} {'p50 TTFT':>10} {'p99 TTFT':>10} "
               f"{'p99 ITL':>9} {'preempt':>8} {'kv%':>6} {'done':>9}")
        print(hdr)
        print("-" * len(hdr))
        for rate, s in rows:
            print(f"{rate:>6.1f} {s['output_throughput_tok_s']:>10.1f} "
                  f"{s['median_ttft_ms']:>9.1f}m {s['p99_ttft_ms']:>9.1f}m "
