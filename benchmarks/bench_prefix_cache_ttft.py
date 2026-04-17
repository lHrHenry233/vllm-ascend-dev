#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Layer 1 Micro Benchmark: Measure R2 TTFT for all-mode vs align-mode.

Tests the core value proposition of all-mode prefix caching:
  R1: full_prefix + question_A → fills cache (warm-up)
  R2: full_prefix + question_B → cache hit (measure this)

Compares:
  - all-mode: R2 reads cached blocks → less compute → faster
  - align-mode: R2 recomputes all blocks → baseline speed

Usage (on NPU server):
    # Default: align-mode uses AscendC conv1d
    python benchmarks/bench_prefix_cache_ttft.py

    # Fair comparison: both modes use Triton conv1d
    GDN_ALIGN_TRITON_CONV1D=1 python benchmarks/bench_prefix_cache_ttft.py

    # With debug logging
    GDN_DEBUG=1 python benchmarks/bench_prefix_cache_ttft.py

    # Custom iterations
    python benchmarks/bench_prefix_cache_ttft.py --num-iters 10

    # Skip warm-up (if cache is already primed from previous run)
    python benchmarks/bench_prefix_cache_ttft.py --skip-warmup
"""

import argparse
import os
import statistics
import sys
import time

# ────────────────────────────────────────────────────────────
# Prompt data — import all scenarios from the test file
# ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_SCENARIOS = {}

try:
    from tests.e2e.singlecard.test_qwen3_5_cache_scenarios import (
        SURVEY_DOCUMENT, SURVEY_QUESTION_A, SURVEY_QUESTION_B,
        AGENT_PROMPT_A, AGENT_PROMPT_B,
        DIALOG_PROMPT_A, DIALOG_PROMPT_B,
    )
    _SCENARIOS["survey"] = {
        "desc": "Research survey paper (~5500 tokens prefix, 5+ blocks)",
        "prompt_a": SURVEY_DOCUMENT + SURVEY_QUESTION_A,
        "prompt_b": SURVEY_DOCUMENT + SURVEY_QUESTION_B,
    }
    _SCENARIOS["agent"] = {
        "desc": "REST API documentation (~3500 tokens prefix, 3+ blocks)",
        "prompt_a": AGENT_PROMPT_A,
        "prompt_b": AGENT_PROMPT_B,
    }
    _SCENARIOS["dialog"] = {
        "desc": "Technical support dialog (~4300 tokens prefix, 4+ blocks)",
        "prompt_a": DIALOG_PROMPT_A,
        "prompt_b": DIALOG_PROMPT_B,
    }
except ImportError as e:
    print(f"WARNING: Could not import test prompts ({e}). Only 'placeholder' available.")
    _SCENARIOS["placeholder"] = {
        "desc": "Placeholder prefix (not real content)",
        "prompt_a": "This is a placeholder prefix. " * 500 + "\n\nQ: topic?\nA: ",
        "prompt_b": "This is a placeholder prefix. " * 500 + "\n\nQ: conclusions?\nA: ",
    }

MODEL = "/shared/models/Qwen3.5-0.8B-ms"

# ────────────────────────────────────────────────────────────
# Engine configuration
# ────────────────────────────────────────────────────────────
ENGINE_KWARGS = dict(
    tensor_parallel_size=1,
    enforce_eager=True,
    gpu_memory_utilization=0.7,
    enable_prefix_caching=True,
    max_model_len=8192,
    max_num_batched_tokens=4096,
)


def run_benchmark(
    mode: str,
    prompt_a: str,
    prompt_b: str,
    scenario_name: str = "survey",
    num_iters: int = 5,
    max_tokens: int = 10,
    skip_warmup: bool = False,
) -> dict:
    """Run R1→R2 benchmark for a given cache mode.

    Returns dict with timing results.
    """
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"  MODE: {mode}  |  SCENARIO: {scenario_name}")
    print(f"  Prompt prefix: {len(prompt_a)} chars → {len(prompt_b)} chars")
    print(f"  Iterations: {num_iters}, max_tokens: {max_tokens}")
    print(f"  GDN_ALIGN_TRITON_CONV1D={os.environ.get('GDN_ALIGN_TRITON_CONV1D', '(not set)')}")
    print(f"{'='*60}")

    llm = LLM(
        model=MODEL,
        **ENGINE_KWARGS,
        additional_config={"mamba_cache_mode": mode},
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)

    r1_times = []
    r2_times = []

    for i in range(num_iters):
        if not skip_warmup or i == 0:
            # R1: Fill cache with prompt_A
            t0 = time.perf_counter()
            r1_out = llm.generate([prompt_a], sampling_params)
            t1 = time.perf_counter()
            r1_ms = (t1 - t0) * 1000
            r1_times.append(r1_ms)
            r1_text = r1_out[0].outputs[0].text[:60]
        else:
            r1_ms = r1_times[-1]  # reuse last R1 time
            r1_text = "(skipped)"

        # R2: Cache hit with prompt_B (same prefix, different question)
        t2 = time.perf_counter()
        r2_out = llm.generate([prompt_b], sampling_params)
        t3 = time.perf_counter()
        r2_ms = (t3 - t2) * 1000
        r2_times.append(r2_ms)
        r2_text = r2_out[0].outputs[0].text[:60]

        print(f"  iter {i+1}/{num_iters}: R1={r1_ms:8.1f}ms  R2={r2_ms:8.1f}ms")
        if i == 0:
            print(f"    R1 output: {r1_text!r}")
            print(f"    R2 output: {r2_text!r}")

    # Exclude first iteration (cold start / compilation)
    r2_warm = r2_times[1:] if len(r2_times) > 1 else r2_times

    result = {
        "mode": mode,
        "num_iters": num_iters,
        "r1_mean_ms": statistics.mean(r1_times),
        "r2_all_ms": r2_times,
        "r2_mean_ms": statistics.mean(r2_warm),
        "r2_median_ms": statistics.median(r2_warm),
        "r2_stdev_ms": statistics.stdev(r2_warm) if len(r2_warm) > 1 else 0,
        "r2_min_ms": min(r2_warm),
        "r2_max_ms": max(r2_warm),
    }

    del llm  # free GPU memory

    return result


def print_comparison(all_result: dict, align_result: dict):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS: all-mode vs align-mode")
    print("=" * 70)
    triton_flag = os.environ.get("GDN_ALIGN_TRITON_CONV1D", "")
    if triton_flag:
        print("  [FAIR MODE] GDN_ALIGN_TRITON_CONV1D=1 → both use Triton conv1d")
    else:
        print("  [DEFAULT] align=AscendC conv1d, all=Triton conv1d")
    print("-" * 70)

    fmt = "  {:30s} {:>12s} {:>12s} {:>10s}"
    print(fmt.format("Metric", "all-mode", "align-mode", "Speedup"))
    print("-" * 70)

    def row(label, all_val, align_val):
        speedup = align_val / all_val if all_val > 0 else float('inf')
        print(fmt.format(
            label,
            f"{all_val:.1f} ms",
            f"{align_val:.1f} ms",
            f"{speedup:.2f}x",
        ))

    row("R1 mean (cache fill)", all_result["r1_mean_ms"], align_result["r1_mean_ms"])
    row("R2 mean (cache hit)", all_result["r2_mean_ms"], align_result["r2_mean_ms"])
    row("R2 median", all_result["r2_median_ms"], align_result["r2_median_ms"])
    row("R2 min", all_result["r2_min_ms"], align_result["r2_min_ms"])
    row("R2 max", all_result["r2_max_ms"], align_result["r2_max_ms"])

    print("-" * 70)
    speedup = align_result["r2_mean_ms"] / all_result["r2_mean_ms"] if all_result["r2_mean_ms"] > 0 else 0
    saved_ms = align_result["r2_mean_ms"] - all_result["r2_mean_ms"]
    print(f"  >>> all-mode R2 is {speedup:.2f}x faster ({saved_ms:.1f}ms saved)")
    print(f"  >>> R2 stdev: all={all_result['r2_stdev_ms']:.1f}ms, "
          f"align={align_result['r2_stdev_ms']:.1f}ms")
    print("=" * 70)


def main():
    available = list(_SCENARIOS.keys())
    parser = argparse.ArgumentParser(
        description="Layer 1 Micro Benchmark: all-mode vs align-mode R2 TTFT")
    parser.add_argument("--num-iters", type=int, default=5,
                        help="Number of R1→R2 iterations per mode (default: 5)")
    parser.add_argument("--max-tokens", type=int, default=10,
                        help="Max tokens to generate (default: 10)")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Only run R1 on first iteration (reuse cache)")
    parser.add_argument("--mode", choices=["all", "align", "both"], default="both",
                        help="Which mode(s) to benchmark (default: both)")
    parser.add_argument("--scenario", choices=available + ["all"],
                        default=available[0],
                        help=f"Prompt scenario (default: {available[0]})")
    args = parser.parse_args()

    # Resolve scenarios to run
    if args.scenario == "all":
        scenarios = available
    else:
        scenarios = [args.scenario]

    for scenario_name in scenarios:
        sc = _SCENARIOS[scenario_name]
        prompt_a = sc["prompt_a"]
        prompt_b = sc["prompt_b"]

        print(f"\n{'#'*70}")
        print(f"# SCENARIO: {scenario_name}")
        print(f"# {sc['desc']}")
        print(f"# prompt_a: {len(prompt_a)} chars, prompt_b: {len(prompt_b)} chars")
        print(f"{'#'*70}")

        results = {}

        if args.mode in ("all", "both"):
            results["all"] = run_benchmark(
                "all", prompt_a, prompt_b, scenario_name,
                args.num_iters, args.max_tokens, args.skip_warmup)

        if args.mode in ("align", "both"):
            results["align"] = run_benchmark(
                "align", prompt_a, prompt_b, scenario_name,
                args.num_iters, args.max_tokens, args.skip_warmup)

        if "all" in results and "align" in results:
            print_comparison(results["all"], results["align"])
        elif len(results) == 1:
            mode, r = next(iter(results.items()))
            print(f"\n{'='*60}")
            print(f"  {mode}-mode R2 mean: {r['r2_mean_ms']:.1f}ms "
                  f"(median: {r['r2_median_ms']:.1f}ms)")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
