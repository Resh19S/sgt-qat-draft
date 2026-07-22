"""Shared speculative-decoding benchmark harness.

Used by notebooks/02_baseline_eagle3.ipynb and notebooks/03_sgt_qat_drafter_bench.ipynb
so both the EAGLE-3 baseline and the SGT-QAT drafter experiment are measured with
identical logic. Meant to run inside a Colab Pro GPU session (imports vllm/torch).

Adapts the config + metrics pattern from vendor/vllm's
examples/features/speculative_decoding/spec_decode_offline.py, extended with wall-clock
timing and peak-memory measurement (which that example doesn't cover).
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch


@dataclass
class BenchResult:
    timestamp: str
    run_name: str
    target_model: str
    speculative_config: dict | None
    num_prompts: int
    max_tokens: int
    wall_clock_seconds: float
    tokens_per_second: float
    peak_memory_bytes: int
    num_drafts: int | None = None
    num_draft_tokens: int | None = None
    num_accepted_tokens: int | None = None
    mean_acceptance_length: float | None = None
    acceptance_rate_by_position: list[float] = field(default_factory=list)
    notes: str = ""


def extract_spec_decode_metrics(llm) -> dict:
    """Pull vLLM's built-in spec-decode counters off an LLM instance.

    Returns zeros/empty when speculative_config was None (no-spec run) since these
    counters won't exist in that case.
    """
    from vllm.v1.metrics.reader import Counter, Vector

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts: list[float] = []

    for metric in llm.get_metrics():
        if metric.name == "vllm:spec_decode_num_drafts" and isinstance(
            metric, Counter
        ):
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens" and isinstance(
            metric, Counter
        ):
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens" and isinstance(
            metric, Counter
        ):
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos" and (
            isinstance(metric, Vector)
        ):
            if not acceptance_counts:
                acceptance_counts = [0.0] * len(metric.values)
            for pos, val in enumerate(metric.values):
                acceptance_counts[pos] += val

    mean_acceptance_length = (
        1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else None
    )
    acceptance_rate_by_position = (
        [c / num_drafts for c in acceptance_counts] if num_drafts > 0 else []
    )

    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "mean_acceptance_length": mean_acceptance_length,
        "acceptance_rate_by_position": acceptance_rate_by_position,
    }


def run_benchmark(
    run_name: str,
    target_model: str,
    prompts: list[str],
    speculative_config: dict | None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    llm_kwargs: dict | None = None,
) -> BenchResult:
    """Build an LLM with the given speculative_config, run generation once, and
    return acceptance/timing/memory metrics.

    speculative_config=None -> no speculative decoding (the control run).
    speculative_config={"method": "eagle3", "model": "Tengyunw/qwen3_8b_eagle3", ...}
        -> EAGLE-3 baseline.
    speculative_config={"method": "draft_model", "model": "<path to SGT-QAT
        checkpoint>", ...} -> our own drafter.
    """
    from vllm import LLM, SamplingParams

    llm_kwargs = llm_kwargs or {}
    torch.cuda.reset_peak_memory_stats()

    llm = LLM(
        model=target_model,
        trust_remote_code=True,
        speculative_config=speculative_config,
        disable_log_stats=False,
        **llm_kwargs,
    )
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    elapsed = time.perf_counter() - start

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    peak_memory_bytes = torch.cuda.max_memory_allocated()

    spec_metrics = (
        extract_spec_decode_metrics(llm)
        if speculative_config is not None
        else {
            "num_drafts": None,
            "num_draft_tokens": None,
            "num_accepted_tokens": None,
            "mean_acceptance_length": None,
            "acceptance_rate_by_position": [],
        }
    )

    result = BenchResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        run_name=run_name,
        target_model=target_model,
        speculative_config=speculative_config,
        num_prompts=len(prompts),
        max_tokens=max_tokens,
        wall_clock_seconds=elapsed,
        tokens_per_second=total_output_tokens / elapsed if elapsed > 0 else 0.0,
        peak_memory_bytes=peak_memory_bytes,
        **spec_metrics,
    )

    # Free GPU memory so a subsequent run_benchmark() call in the same session
    # (e.g. no-spec vs eagle3 vs draft_model, back to back) isn't skewed by the
    # previous LLM instance still holding VRAM.
    del llm
    torch.cuda.empty_cache()

    return result


def save_result(result: BenchResult, results_dir: str | Path = "results") -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = result.timestamp.replace(":", "-")
    fname = f"{result.run_name}_{ts}.json"
    path = results_dir / fname
    path.write_text(json.dumps(asdict(result), indent=2))
    return path


def summarize(results: list[BenchResult]) -> None:
    """Print a quick human-readable comparison table (speedup relative to the
    first result, assumed to be the no-spec-decode control)."""
    if not results:
        return
    baseline_tps = results[0].tokens_per_second
    print(f"{'run':<20} {'tok/s':>10} {'speedup':>10} {'mean AL':>10} {'peak MB':>10}")
    for r in results:
        speedup = r.tokens_per_second / baseline_tps if baseline_tps > 0 else float("nan")
        al = f"{r.mean_acceptance_length:.2f}" if r.mean_acceptance_length else "-"
        mem_mb = r.peak_memory_bytes / (1024 ** 2)
        print(f"{r.run_name:<20} {r.tokens_per_second:>10.1f} {speedup:>9.2f}x {al:>10} {mem_mb:>10.0f}")
