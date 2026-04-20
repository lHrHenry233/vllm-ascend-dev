#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline profiling: all-mode vs align-mode with Ascend PyTorch Profiler.

Captures operator-level traces for both modes, plus TTFT via output.metrics.

Usage (on NPU server):
    # Profile both modes (default)
    python benchmarks/profile_all_vs_align.py

    # Profile only all-mode
    python benchmarks/profile_all_vs_align.py --mode all

    # Fair comparison (both use Triton conv1d)
    GDN_ALIGN_TRITON_CONV1D=1 python benchmarks/profile_all_vs_align.py

    # Custom output dir
    python benchmarks/profile_all_vs_align.py --output-dir ./my_profiles

After run:
    # Analyse the ascend_pt folder
    python -c "
    from torch_npu.profiler.profiler import analyse
    analyse('./vllm_profile_all/<hostname>_*_ascend_pt/')
    analyse('./vllm_profile_align/<hostname>_*_ascend_pt/')
    "
    # Then check ASCEND_PROFILER_OUTPUT/operator_details.csv
"""

import argparse
import os
import sys
import time

# Import prompt data
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from tests.e2e.singlecard.test_qwen3_5_cache_scenarios import (
        SURVEY_DOCUMENT,
        SURVEY_QUESTION_A,
        SURVEY_QUESTION_B,
    )
except ImportError:
    print("WARNING: Could not import test prompts. Using inline placeholder.")
    SURVEY_DOCUMENT = "This is a placeholder prefix. " * 500
    SURVEY_QUESTION_A = "\n\nQuestion: What is the main topic?\n\nAnswer: "
    SURVEY_QUESTION_B = "\n\nQuestion: What are the conclusions?\n\nAnswer: "

PROMPT_A = SURVEY_DOCUMENT + SURVEY_QUESTION_A
PROMPT_B = SURVEY_DOCUMENT + SURVEY_QUESTION_B

MODEL = "/shared/models/Qwen3.5-0.8B-ms"

ENGINE_KWARGS = dict(
    tensor_parallel_size=1,
    enforce_eager=True,
    gpu_memory_utilization=0.7,
    enable_prefix_caching=True,
    max_model_len=8192,
    max_num_batched_tokens=4096,
)


def extract_ttft(output) -> float:
    """Extract TTFT (ms) from a single RequestOutput.
    
    vllm 0.19 RequestStateStats has `first_token_latency` (seconds)
    which is the time from request arrival to first token generation.
    """
    m = output.metrics
    if m is None:
        return float("nan")
    latency = getattr(m, 'first_token_latency', None)
    if latency is not None and isinstance(latency, (int, float)):
        return latency * 1000  # seconds → ms
    return float("nan")


def run_profiled(mode: str, output_dir: str, max_tokens: int = 10,
                 warmup_iters: int = 2, profile_iters: int = 3):
    """Run warm-up, then profile R1→R2 iterations."""
    import torch_npu
    from vllm import LLM, SamplingParams

    profile_dir = os.path.join(output_dir, f"vllm_profile_{mode}")
    os.makedirs(profile_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  MODE: {mode}")
    print(f"  Profile dir: {profile_dir}")
    print(f"  Warmup: {warmup_iters} iters, Profile: {profile_iters} iters")
    print(f"  max_tokens: {max_tokens}")
    print(f"  GDN_ALIGN_TRITON_CONV1D="
          f"{os.environ.get('GDN_ALIGN_TRITON_CONV1D', '(not set)')}")
    print(f"{'='*70}")

    llm = LLM(
        model=MODEL,
        **ENGINE_KWARGS,
        mamba_cache_mode=mode,
        disable_log_stats=False,  # Enable output.metrics for TTFT
    )
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)

    # ── Warm-up (no profiling) ──
    print(f"\n[{mode}] Warm-up ({warmup_iters} iterations)...")
    for i in range(warmup_iters):
        r1 = llm.generate([PROMPT_A], sp)
        r2 = llm.generate([PROMPT_B], sp)
        r1_ttft = extract_ttft(r1[0])
        r2_ttft = extract_ttft(r2[0])
        print(f"  warmup {i+1}: R1_TTFT={r1_ttft:.1f}ms  R2_TTFT={r2_ttft:.1f}ms")

    # ── Profiled iterations ──
    print(f"\n[{mode}] Profiling ({profile_iters} iterations)...")

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
        gc_detect_threshold=None,
    )

    ttft_results = {"r1": [], "r2": [], "r1_wall": [], "r2_wall": []}

    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        with_stack=False,
        profile_memory=False,
        with_modules=False,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            profile_dir, worker_name=f"{mode}_profile",
        ),
    )

    profiler.start()
    try:
        for i in range(profile_iters):
            # R1: cache fill
            t0 = time.perf_counter()
            r1 = llm.generate([PROMPT_A], sp)
            t1 = time.perf_counter()
            r1_ttft = extract_ttft(r1[0])
            r1_wall = (t1 - t0) * 1000

            # R2: cache hit
            t2 = time.perf_counter()
            r2 = llm.generate([PROMPT_B], sp)
            t3 = time.perf_counter()
            r2_ttft = extract_ttft(r2[0])
            r2_wall = (t3 - t2) * 1000

            ttft_results["r1"].append(r1_ttft)
            ttft_results["r2"].append(r2_ttft)
            ttft_results["r1_wall"].append(r1_wall)
            ttft_results["r2_wall"].append(r2_wall)

            print(f"  iter {i+1}/{profile_iters}: "
                  f"R1_TTFT={r1_ttft:.1f}ms R1_wall={r1_wall:.1f}ms | "
                  f"R2_TTFT={r2_ttft:.1f}ms R2_wall={r2_wall:.1f}ms")

            if i == 0:
                print(f"    R1: {r1[0].outputs[0].text[:80]!r}")
                print(f"    R2: {r2[0].outputs[0].text[:80]!r}")
    finally:
        profiler.stop()
        print(f"\n[{mode}] Profiler stopped. Trace saved to: {profile_dir}")

    # ── Summary ──
    import math, statistics
    r2_ttfts = ttft_results["r2"]
    r1_ttfts = ttft_results["r1"]
    r1_valid = [x for x in r1_ttfts if not math.isnan(x)]
    r2_valid = [x for x in r2_ttfts if not math.isnan(x)]
    print(f"\n[{mode}] TTFT Summary:")
    if r1_valid:
        print(f"  R1 mean TTFT: {statistics.mean(r1_valid):.1f}ms")
    else:
        print(f"  R1 mean TTFT: N/A (metrics unavailable, use wall time)")
    if r2_valid:
        print(f"  R2 mean TTFT: {statistics.mean(r2_valid):.1f}ms")
        if len(r2_valid) > 1:
            print(f"  R2 stdev:     {statistics.stdev(r2_valid):.1f}ms")
    else:
        print(f"  R2 mean TTFT: N/A (metrics unavailable, use wall time)")
    # Wall time summary
    r1_walls = ttft_results.get("r1_wall", [])
    r2_walls = ttft_results.get("r2_wall", [])
    if r1_walls:
        print(f"  R1 mean wall: {statistics.mean(r1_walls):.1f}ms")
    if r2_walls:
        print(f"  R2 mean wall: {statistics.mean(r2_walls):.1f}ms")

    del llm
    import gc
    gc.collect()

    # Save results to JSON for subprocess communication
    import json
    summary_path = os.path.join(output_dir, f"{mode}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(ttft_results, f)

    return ttft_results


def main():
    parser = argparse.ArgumentParser(
        description="Offline profiling: all-mode vs align-mode on NPU")
    parser.add_argument("--mode", choices=["all", "align", "both"],
                        default="both", help="Mode(s) to profile (default: both)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Base output directory (default: current dir)")
    parser.add_argument("--max-tokens", type=int, default=10,
                        help="Max tokens to generate (default: 10)")
    parser.add_argument("--warmup-iters", type=int, default=2,
                        help="Warm-up iterations before profiling (default: 2)")
    parser.add_argument("--profile-iters", type=int, default=3,
                        help="Profiled iterations (default: 3)")
    args = parser.parse_args()

    print(f"Model: {MODEL}")
    print(f"Prefix: ~2500 tokens (~3 blocks @ block_size=1024)")
    print(f"Output dir: {args.output_dir}")

    results = {}

    if args.mode in ("all", "both"):
        results["all"] = run_profiled(
            "all", args.output_dir, args.max_tokens,
            args.warmup_iters, args.profile_iters)

    if args.mode in ("align", "both"):
        if "all" in results:
            # NPU cannot be re-initialized in forked subprocesses.
            # Must run align mode in a separate process.
            import subprocess, sys, json
            print("\n[NOTE] Spawning separate process for align mode "
                  "(NPU cannot re-init in same process)...")
            cmd = [sys.executable, __file__,
                   "--mode", "align",
                   "--output-dir", args.output_dir,
                   "--max-tokens", str(args.max_tokens),
                   "--warmup-iters", str(args.warmup_iters),
                   "--profile-iters", str(args.profile_iters)]
            env = dict(os.environ)
            proc = subprocess.run(cmd, env=env)
            if proc.returncode != 0:
                print(f"[ERROR] Align mode subprocess exited with code {proc.returncode}")
            # Can't easily retrieve results from subprocess, so read from saved file
            align_summary = os.path.join(args.output_dir, "align_summary.json")
            if os.path.exists(align_summary):
                with open(align_summary) as f:
                    results["align"] = json.load(f)
        else:
            results["align"] = run_profiled(
                "align", args.output_dir, args.max_tokens,
                args.warmup_iters, args.profile_iters)

    # ── Comparison ──
    if "all" in results and "align" in results:
        import math, statistics

        def safe_mean(lst):
            valid = [x for x in lst if not math.isnan(x)]
            return statistics.mean(valid) if valid else float("nan")

        print(f"\n{'='*70}")
        print(f"  COMPARISON (R2 = cache hit)")
        print(f"{'='*70}")

        # TTFT comparison
        all_r2_ttft = safe_mean(results["all"]["r2"])
        align_r2_ttft = safe_mean(results["align"]["r2"])
        if not math.isnan(all_r2_ttft) and not math.isnan(align_r2_ttft):
            speedup = align_r2_ttft / all_r2_ttft if all_r2_ttft > 0 else float("inf")
            print(f"  [TTFT]  all={all_r2_ttft:.1f}ms  align={align_r2_ttft:.1f}ms  "
                  f"speedup={speedup:.2f}x")

        # Wall time comparison (more reliable)
        all_r2_wall = safe_mean(results["all"].get("r2_wall", []))
        align_r2_wall = safe_mean(results["align"].get("r2_wall", []))
        if not math.isnan(all_r2_wall) and not math.isnan(align_r2_wall):
            speedup_wall = align_r2_wall / all_r2_wall if all_r2_wall > 0 else float("inf")
            print(f"  [Wall]  all={all_r2_wall:.1f}ms  align={align_r2_wall:.1f}ms  "
                  f"speedup={speedup_wall:.2f}x")
        print(f"{'='*70}")

    print("\n[NEXT STEPS]")
    print("  1. Find the *_ascend_pt folder in your profile dirs")
    print("  2. Run analysis:")
    print("     python -c \"from torch_npu.profiler.profiler import analyse; "
          "analyse('<path_to_ascend_pt_folder>')\"")
    print("  3. Check ASCEND_PROFILER_OUTPUT/operator_details.csv")
    print("  4. Open trace_view.json in MindStudio Insight or chrome://tracing")


if __name__ == "__main__":
    main()
