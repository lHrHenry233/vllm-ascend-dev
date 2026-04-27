#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Synthetic online benchmark for ALL-vs-ALIGN prefix caching.

This benchmark creates two long prompts with exact tokenizer-controlled lengths:

- R1 total prompt tokens
- R2 total prompt tokens
- exact shared prefix tokens between R1 and R2

It launches `vllm serve` for all/align, sends R1 -> R2 through the OpenAI
completion API, and parses a server-side log line carrying actual
`num_cached_tokens`.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import bench_all_vs_align_sharegpt as offline_base
import bench_all_vs_align_sharegpt_online as online_base

_CACHE_LOG_RE = re.compile(
    r"\[GDN_CACHE_HIT\] mode=(?P<mode>\S+) request_id=(?P<request_id>\S+) "
    r"prompt_tokens=(?P<prompt_tokens>\d+) total_tokens=(?P<total_tokens>\d+) "
    r"cached_tokens=(?P<cached_tokens>\d+) hit_rate=(?P<hit_rate>\d+(?:\.\d+)?) "
    r"finished=(?P<finished>\S+)"
)

_SHARED_SEED = (
    "Shared prefix benchmark paragraph about cache reuse, repeated scheduling, "
    "and deterministic tokenizer boundaries. "
)
_R1_SUFFIX_SEED = (
    "Route alpha suffix discussing all mode behavior, chunked prefill, and final latency. "
)
_R2_SUFFIX_SEED = (
    "Route beta suffix discussing align mode behavior, recompute cost, and cache hit geometry. "
)


def _decode(tokenizer, ids):
    return tokenizer.decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _repeat_encode(tokenizer, seed_text: str, min_tokens: int):
    text = seed_text
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < min_tokens:
        text += "\n" + seed_text
        ids = tokenizer.encode(text, add_special_tokens=False)
    return ids


def _common_prefix_len(a_ids, b_ids):
    limit = min(len(a_ids), len(b_ids))
    for i in range(limit):
        if a_ids[i] != b_ids[i]:
            return i
    return limit


def build_exact_synthetic_prompts(tokenizer, shared_prefix_tokens: int, r1_total_tokens: int, r2_total_tokens: int):
    if shared_prefix_tokens <= 0:
        raise ValueError("shared_prefix_tokens must be > 0")
    if r1_total_tokens <= shared_prefix_tokens or r2_total_tokens <= shared_prefix_tokens:
        raise ValueError("total prompt tokens must be greater than shared_prefix_tokens")

    suffix_a_tokens = r1_total_tokens - shared_prefix_tokens
    suffix_b_tokens = r2_total_tokens - shared_prefix_tokens

    shared_pool = _repeat_encode(tokenizer, _SHARED_SEED, shared_prefix_tokens + 128)
    suffix_a_pool = _repeat_encode(tokenizer, _R1_SUFFIX_SEED, suffix_a_tokens + 128)
    suffix_b_pool = _repeat_encode(tokenizer, _R2_SUFFIX_SEED, suffix_b_tokens + 128)

    for shared_offset in range(8):
        shared_ids = shared_pool[shared_offset : shared_offset + shared_prefix_tokens]
        if len(shared_ids) < shared_prefix_tokens:
            continue
        for suffix_a_offset in range(8):
            r1_ids = shared_ids + suffix_a_pool[suffix_a_offset : suffix_a_offset + suffix_a_tokens]
            if len(r1_ids) != r1_total_tokens:
                continue
            for suffix_b_offset in range(8):
                r2_ids = shared_ids + suffix_b_pool[suffix_b_offset : suffix_b_offset + suffix_b_tokens]
                if len(r2_ids) != r2_total_tokens:
                    continue
                r1_prompt = _decode(tokenizer, r1_ids)
                r2_prompt = _decode(tokenizer, r2_ids)
                r1_actual = tokenizer.encode(r1_prompt, add_special_tokens=False)
                r2_actual = tokenizer.encode(r2_prompt, add_special_tokens=False)
                if len(r1_actual) != r1_total_tokens or len(r2_actual) != r2_total_tokens:
                    continue
                shared_actual = _common_prefix_len(r1_actual, r2_actual)
                if shared_actual != shared_prefix_tokens:
                    continue
                return {
                    "r1_prompt": r1_prompt,
                    "r2_prompt": r2_prompt,
                    "shared_prefix_tokens": shared_actual,
                    "r1_total_tokens": len(r1_actual),
                    "r2_total_tokens": len(r2_actual),
                    "suffix_a_tokens": len(r1_actual) - shared_actual,
                    "suffix_b_tokens": len(r2_actual) - shared_actual,
                }

    raise RuntimeError(
        "Failed to build exact synthetic prompts with the requested token geometry. "
        "Try changing seed text or search window."
    )


def wait_for_cache_log(log_path: Path, start_offset: int, expected_mode: str, timeout: float):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if log_path.exists():
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(start_offset)
                chunk = f.read()
                if chunk:
                    for line in chunk.splitlines():
                        match = _CACHE_LOG_RE.search(line)
                        if not match:
                            continue
                        info = match.groupdict()
                        if info["mode"] != expected_mode:
                            continue
                        info["prompt_tokens"] = int(info["prompt_tokens"])
                        info["total_tokens"] = int(info["total_tokens"])
                        info["cached_tokens"] = int(info["cached_tokens"])
                        info["hit_rate"] = float(info["hit_rate"])
                        info["finished"] = info["finished"] == "True"
                        return info
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for cache-hit log in {log_path}")


def _send_one_request(server: online_base.LocalOpenAIServer, prompt: str, max_tokens: int):
    start_offset = server.log_path.stat().st_size if server.log_path.exists() else 0
    t0 = time.perf_counter()
    text = server.generate(prompt, max_tokens)
    t1 = time.perf_counter()
    cache_info = wait_for_cache_log(server.log_path, start_offset, server.mode, timeout=15.0)
    return {
        "text": text,
        "ms": (t1 - t0) * 1000,
        "cache": cache_info,
    }


def _warmup_mode(mode: str, model_path: str, args: argparse.Namespace, results_dir: Path, prompts: dict):
    warmup_dir = results_dir / f"{mode}_warmup"
    print(f"\n  Warmup ({mode}): disposable server")
    with online_base.LocalOpenAIServer(mode, model_path, args, warmup_dir) as server:
        _send_one_request(server, prompts["r1_prompt"], args.max_tokens)


def run_one_mode(mode: str, prompts: dict, args: argparse.Namespace, results_dir: Path):
    expected_hit_tokens = prompts["shared_prefix_tokens"] if mode == "all" else 0
    expected_hit_rate = expected_hit_tokens / prompts["r2_total_tokens"]

    print(f"\n{'═' * 76}")
    print(f"  MODE: {mode} (online synthetic)")
    print(f"  Prompt geometry: shared={prompts['shared_prefix_tokens']} "
          f"r1_total={prompts['r1_total_tokens']} r2_total={prompts['r2_total_tokens']}")
    print(f"  Expected R2 hits: {expected_hit_tokens} tokens "
          f"({expected_hit_rate:.4%})")
    print(f"{'═' * 76}")

    if not args.skip_warmup:
        _warmup_mode(mode, args.model, args, results_dir, prompts)

    results = []
    for repeat_idx in range(args.num_repeats):
        repeat_dir = results_dir / f"{mode}_repeat{repeat_idx + 1}"
        print(f"\n  [Repeat {repeat_idx + 1}/{args.num_repeats}] Starting fresh server")
        with online_base.LocalOpenAIServer(mode, args.model, args, repeat_dir) as server:
            r1 = _send_one_request(server, prompts["r1_prompt"], args.max_tokens)
            r2 = _send_one_request(server, prompts["r2_prompt"], args.max_tokens)
            results.append({"r1": r1, "r2": r2})

            print(
                "    R1: "
                f"{r1['ms']:.1f} ms, cached={r1['cache']['cached_tokens']} "
                f"({r1['cache']['hit_rate']:.4%})"
            )
            print(
                "    R2: "
                f"{r2['ms']:.1f} ms, cached={r2['cache']['cached_tokens']} "
                f"({r2['cache']['hit_rate']:.4%})"
            )

    r1_mean = statistics.mean(item["r1"]["ms"] for item in results)
    r2_mean = statistics.mean(item["r2"]["ms"] for item in results)
    actual_cached_mean = statistics.mean(item["r2"]["cache"]["cached_tokens"] for item in results)
    actual_hit_rate_mean = statistics.mean(item["r2"]["cache"]["hit_rate"] for item in results)

    return {
        "mode": mode,
        "expected_hit_tokens": expected_hit_tokens,
        "expected_hit_rate": expected_hit_rate,
        "r1_mean_ms": r1_mean,
        "r2_mean_ms": r2_mean,
        "actual_cached_tokens_mean": actual_cached_mean,
        "actual_hit_rate_mean": actual_hit_rate_mean,
        "r1_output": results[0]["r1"]["text"],
        "r2_output": results[0]["r2"]["text"],
    }


def print_summary(prompts: dict, all_result: dict | None, align_result: dict | None):
    print(f"\n{'═' * 80}")
    print("  SYNTHETIC PREFIX CACHE REPORT")
    print(f"{'═' * 80}")
    print(
        f"  shared_prefix_tokens={prompts['shared_prefix_tokens']} "
        f"({prompts['shared_prefix_tokens'] // 4096} blocks), "
        f"r1_total={prompts['r1_total_tokens']}, "
        f"r2_total={prompts['r2_total_tokens']}"
    )
    print(
        f"  suffix_A={prompts['suffix_a_tokens']} tokens, "
        f"suffix_B={prompts['suffix_b_tokens']} tokens"
    )

    if all_result:
        print(
            f"\n  ALL  : R1={all_result['r1_mean_ms']:.1f} ms  "
            f"R2={all_result['r2_mean_ms']:.1f} ms  "
            f"expected_cached={all_result['expected_hit_tokens']}  "
            f"actual_cached≈{all_result['actual_cached_tokens_mean']:.1f}  "
            f"actual_hit_rate≈{all_result['actual_hit_rate_mean']:.4%}"
        )
    if align_result:
        print(
            f"  ALIGN: R1={align_result['r1_mean_ms']:.1f} ms  "
            f"R2={align_result['r2_mean_ms']:.1f} ms  "
            f"expected_cached={align_result['expected_hit_tokens']}  "
            f"actual_cached≈{align_result['actual_cached_tokens_mean']:.1f}  "
            f"actual_hit_rate≈{align_result['actual_hit_rate_mean']:.4%}"
        )
    if all_result and align_result:
        speedup = align_result["r2_mean_ms"] / max(all_result["r2_mean_ms"], 0.1)
        print(f"\n  >>> ALL R2 speedup vs ALIGN R2: {speedup:.2f}x")
        if round(all_result["actual_cached_tokens_mean"]) != all_result["expected_hit_tokens"]:
            print("  ⚠ ALL actual cached tokens differ from expected geometry")
        if round(align_result["actual_cached_tokens_mean"]) != align_result["expected_hit_tokens"]:
            print("  ⚠ ALIGN actual cached tokens differ from expected geometry")


def main():
    parser = argparse.ArgumentParser(
        description="Online synthetic ALL-vs-ALIGN prefix cache benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="/data/Qwen3.5-9B")
    parser.add_argument("--shared-prefix-tokens", type=int, default=12288)
    parser.add_argument("--r1-total-tokens", type=int, default=17000)
    parser.add_argument("--r2-total-tokens", type=int, default=17000)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--mode", choices=["all", "align", "both"], default="both")
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--align-triton-conv1d", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--served-model-name", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--log-cached-tokens", action="store_true", default=True)
    parser.add_argument("--no-log-cached-tokens", action="store_false", dest="log_cached_tokens")
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    args = parser.parse_args()

    print(f"\nLoading tokenizer from {args.model} ...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"✓ Tokenizer loaded from {getattr(tokenizer, 'name_or_path', args.model)} "
          f"(vocab_size={tokenizer.vocab_size})")

    prompts = build_exact_synthetic_prompts(
        tokenizer,
        shared_prefix_tokens=args.shared_prefix_tokens,
        r1_total_tokens=args.r1_total_tokens,
        r2_total_tokens=args.r2_total_tokens,
    )

    shared_blocks = prompts["shared_prefix_tokens"] // args.block_size
    align_step_blocks = args.max_num_batched_tokens // args.block_size
    print(f"\n  Prompt geometry")
    print(f"    shared_prefix_tokens = {prompts['shared_prefix_tokens']}")
    print(f"    shared_prefix_blocks = {shared_blocks}")
    print(f"    r1_total_tokens      = {prompts['r1_total_tokens']}")
    print(f"    r2_total_tokens      = {prompts['r2_total_tokens']}")
    print(f"    suffix_A_tokens      = {prompts['suffix_a_tokens']}")
    print(f"    suffix_B_tokens      = {prompts['suffix_b_tokens']}")
    print(f"    ALIGN step blocks    = {align_step_blocks}")
    print(f"    ALL expected hits    = {shared_blocks}")
    print(f"    ALIGN expected hits  = {(shared_blocks // align_step_blocks) * align_step_blocks}")

    if args.dry_run:
        print("\n  [DRY RUN] Prompt construction complete.")
        return

    results_dir = Path(
        args.results_dir
        or f"benchmarks/results/synthetic_prefix_online_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    all_result = None
    align_result = None
    if args.mode in ("all", "both"):
        all_result = run_one_mode("all", prompts, args, results_dir)
    if args.mode in ("align", "both"):
        align_result = run_one_mode("align", prompts, args, results_dir)

    print_summary(prompts, all_result, align_result)
    print(f"\n  Results directory: {results_dir}")


if __name__ == "__main__":
    main()
