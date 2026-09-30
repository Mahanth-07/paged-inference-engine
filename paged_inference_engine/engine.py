"""The engine loop. Everything else is policy; this is the crank you turn.

    engine = LLMEngine(EngineConfig())
    engine.add_request(prompt="hello world", sampling_params=SamplingParams(max_tokens=32))
    while engine.has_unfinished_requests():
        for out in engine.step():
            print(out.delta_text, end="")

One `step()` == one forward pass == one scheduling decision. That is what makes this
continuous batching rather than static batching: there is no notion of "a batch"
that has to drain before the next one starts.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

from .block_manager import BlockManager
from .config import EngineConfig
from .sampling import SamplingParams
from .scheduler import Scheduler, SchedulerOutput
from .sequence import RequestOutput, Sequence, SequenceStatus
from .utils.metrics import MetricsCollector


class Detokenizer:
    """Incremental detokenization for streaming.

    A token boundary is not a character boundary: a multi-byte UTF-8 codepoint can
    span two tokens, and emitting the first half produces a replacement character in
    the user's terminal. So we hold back any trailing partial codepoint until the
    next token completes it.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._text: dict[int, str] = {}

    def decode_delta(self, seq: Sequence) -> tuple[str, str]:
        if self.tokenizer is None:
            return "", ""
        full = self.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True)
        prev = self._text.get(seq.seq_id, "")
        if full.endswith("�"):
            return prev, ""            # incomplete codepoint: wait for the next token
        self._text[seq.seq_id] = full
        return full, full[len(prev):]

    def finish(self, seq: Sequence) -> str:
        text = self._text.pop(seq.seq_id, "")
        return text


class LLMEngine:
    def __init__(self, config: EngineConfig, model_runner=None):
        self.config = config
        self.metrics = MetricsCollector()

        # model_runner is injectable so bench/scheduler_sim.py can pass a fake one and
        # scheduling policy can be tuned on a laptop with no GPU.
        if model_runner is None:
            from .model_runner import ModelRunner

            model_runner = ModelRunner(config)
        self.model_runner = model_runner

        # profile_num_blocks() both sizes AND allocates the cache -- the two must agree,
        # so they are deliberately not separate calls the caller can get out of sync.
        num_blocks = self.model_runner.profile_num_blocks()

        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=config.cache.block_size,
            enable_prefix_caching=config.cache.enable_prefix_caching,
        )
        self.scheduler = Scheduler(config.scheduler, self.block_manager)
        self.tokenizer = getattr(model_runner, "tokenizer", None)
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        self.detokenizer = Detokenizer(self.tokenizer)
        self.max_model_len = config.model.max_model_len

        if not config.enforce_eager and hasattr(self.model_runner, "capture_cuda_graphs"):
            self.model_runner.capture_cuda_graphs()

    # ---- public API -------------------------------------------------------

    def add_request(
        self,
        prompt: str | None = None,
        sampling_params: SamplingParams | None = None,
        prompt_token_ids: Optional[list[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> int:
        if prompt_token_ids is None:
            assert prompt is not None and self.tokenizer is not None, \
                "pass prompt_token_ids when the runner has no tokenizer"
            prompt_token_ids = self.tokenizer.encode(prompt)
        sp = sampling_params or SamplingParams()
        # Clamp against the cache geometry so a request can never be admitted and then
        # get stuck: the scheduler's AllocStatus.NEVER path would abort it silently.
        budget = self.max_model_len - len(prompt_token_ids)
        if budget <= 0:
            raise ValueError(
                f"prompt of {len(prompt_token_ids)} tokens exceeds max_model_len "
                f"{self.max_model_len}")
        if sp.max_tokens > budget:
            sp = SamplingParams(**{**sp.__dict__, "max_tokens": budget})
        seq = Sequence(
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sp,
            arrival_time=arrival_time if arrival_time is not None else time.perf_counter(),
        )
        self.scheduler.add_seq(seq)
        return seq.seq_id

    def abort_request(self, seq_id: int) -> None:
        self.scheduler.abort(seq_id)

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished

    def step(self) -> list[RequestOutput]:
        """One forward pass' worth of work.

        Returns an output for every sequence that produced a token this step, plus a
        final one for every sequence that finished. Emitting per-step rather than
        per-request is what makes TTFT and ITL measurable at all.
        """
        sched: SchedulerOutput = self.scheduler.schedule()

        # Rejected requests get a real response. Dropping them on the floor makes an
        # HTTP client hang until it times out, which is a far worse failure than an
        # error and much harder to debug.
        rejected = [self._make_output(seq, finished=True)
                    for seq in self.scheduler.pop_ignored()]
        if sched.is_empty:
            return rejected

        t0 = time.perf_counter()
        sampled = self.model_runner.execute_model(sched, self.block_manager)
        t1 = time.perf_counter()

        for scheduled in sched.prefills:
            scheduled.seq.num_computed_tokens += scheduled.num_tokens
            # Publish prefix-cache hashes only now that the KV genuinely exists.
            self.block_manager.mark_computed_blocks(scheduled.seq)
        # Only sequences whose prefill COMPLETED this step produce a token.
        for seq, token_id in sampled.items():
            seq.append_token(token_id, now=t1)

        outputs = rejected + [self._make_output(seq, finished=False) for seq in sampled]

        finished = self.scheduler.free_finished(self.eos_token_id)
        self.metrics.record_step(
            num_batched_tokens=sched.num_batched_tokens,
            num_prefill_seqs=len(sched.prefills),
            num_decode_seqs=len(sched.decodes),
            step_latency=t1 - t0,
            kv_utilization=self.block_manager.utilization,
            num_preempted=len(sched.preempted),
        )
        by_id = {o.seq_id: o for o in outputs}
        for seq in finished:
            seq.finish_time = t1
            self.metrics.record_request(seq)
            out = by_id.get(seq.seq_id)
            if out is None:
                out = self._make_output(seq, finished=True)
                outputs.append(out)
            out.finished = True
            out.finish_reason = _finish_reason(seq.status)
            out.text = self.detokenizer.finish(seq) or out.text
        return outputs

    def generate(
        self,
        prompts: Iterable[str] | None = None,
        sampling_params: SamplingParams | None = None,
        prompt_token_ids: Optional[list[list[int]]] = None,
    ) -> list[RequestOutput]:
        """Offline batch API, for throughput benchmarks. Returns one output per request,
        in submission order."""
        sp = sampling_params or SamplingParams()
        order: list[int] = []
        if prompt_token_ids is not None:
            for ids in prompt_token_ids:
                order.append(self.add_request(prompt_token_ids=ids, sampling_params=sp))
        else:
            for p in prompts or []:
                order.append(self.add_request(prompt=p, sampling_params=sp))

        final: dict[int, RequestOutput] = {}
        while self.has_unfinished_requests():
            for out in self.step():
                if out.finished:
                    final[out.seq_id] = out
        return [final[i] for i in order if i in final]

    def stats(self) -> dict:
        s = self.scheduler.stats()
        s["num_gpu_blocks"] = self.block_manager.allocator.num_blocks
        s["num_free_blocks"] = self.block_manager.num_free_blocks
        if self.block_manager.num_cache_query_blocks:
            s["prefix_cache_hit_rate"] = (
                self.block_manager.num_cache_hit_blocks
                / self.block_manager.num_cache_query_blocks
            )
        return s

    # ---- internals --------------------------------------------------------

    def _make_output(self, seq: Sequence, finished: bool) -> RequestOutput:
        text, delta = self.detokenizer.decode_delta(seq)
        return RequestOutput(
            seq_id=seq.seq_id,
            prompt_token_ids=seq.prompt_token_ids,
            token_ids=list(seq.output_token_ids),
            text=text,
            delta_text=delta,
            new_token_ids=seq.output_token_ids[-1:],
            finished=finished,
            finish_reason=_finish_reason(seq.status) if finished else None,
        )


def _finish_reason(status: SequenceStatus) -> str:
    return {
        SequenceStatus.FINISHED_STOPPED: "stop",
        SequenceStatus.FINISHED_LENGTH: "length",
        SequenceStatus.FINISHED_ABORTED: "abort",
        SequenceStatus.FINISHED_IGNORED: "ignored",
    }.get(status, "stop")
