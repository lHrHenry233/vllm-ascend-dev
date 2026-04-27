#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ShareGPT-based prefix caching benchmark: all-mode vs align-mode.

Downloads ShareGPT conversations and measures cache hit performance
across multiple prefix lengths (measured in blocks). For each prefix
length group, 5 conversations are selected and tested:

  R1: prefix + suffix_A  → compute from scratch, fill cache
  R2: prefix + suffix_B  → same prefix triggers cache hit

  ████████████████████████████ + ▲▲▲▲▲▲▲▲  R1
  ████████████████████████████ + ■■■■■■■■  R2
  ←── shared prefix (~N blocks) ──→ ← suffix →
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  cache hit target (exact same tokens)

Compares:
  all-mode:   R2 hits ALL prefix blocks → minimal recompute
  align-mode: R2 hits only step-boundary blocks → more recompute

With max_num_batched_tokens=4096 and block_size=1024 (alignment=4):
  Group  3 blocks: ALL hits 3, ALIGN hits 0 → 3072 tokens extra recompute
  Group  5 blocks: ALL hits 5, ALIGN hits 4 → 1024 tokens extra recompute
  Group  7 blocks: ALL hits 7, ALIGN hits 4 → 3072 tokens extra recompute
  Group 10 blocks: ALL hits 10, ALIGN hits 8 → 2048 tokens extra recompute

Usage:
  # Full benchmark (default: 4 groups, 5 convs each, 1 measured repeat)
  python benchmarks/bench_all_vs_align_sharegpt.py

  # Specific groups
  python benchmarks/bench_all_vs_align_sharegpt.py --groups 3 5

  # More conversations / measured repeats
  python benchmarks/bench_all_vs_align_sharegpt.py --convs-per-group 10 --num-repeats 3

  # Dry run (construct prompts, print stats, no engine needed)
  python benchmarks/bench_all_vs_align_sharegpt.py --dry-run

  # Pre-downloaded ShareGPT file
  python benchmarks/bench_all_vs_align_sharegpt.py --data-path /path/to/sharegpt.json

  # Pre-built prompt pairs from PR14 workflow
  python benchmarks/bench_all_vs_align_sharegpt.py \
    --load-prompts /path/to/sharegpt_prompts.json \
    --model /data/Qwen3.5-9B

  # Fair ALIGN Triton comparison with TTFT-friendly decoding
  python benchmarks/bench_all_vs_align_sharegpt.py \
    --mode both \
    --align-triton-conv1d \
    --max-tokens 1 \
    --load-prompts /path/to/sharegpt_prompts.json

  # Single mode
  python benchmarks/bench_all_vs_align_sharegpt.py --mode all
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

# ── Configuration ──────────────────────────────────────────────────────

# Download source selection: "hf" (HuggingFace) or "modelscope" (Alibaba)
_DOWNLOAD_SOURCE = os.environ.get("SHAREGPT_SOURCE", "hf")
_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

SHAREGPT_URLS = {
    "hf": (
        f"{_HF_ENDPOINT}/datasets/anon8231489123/"
        "ShareGPT_Vicuna_unfiltered/resolve/main/"
        "ShareGPT_V3_unfiltered_cleaned_split.json"
    ),
    "modelscope": (
        "https://www.modelscope.cn/api/v1/datasets/"
        "otavia/ShareGPT_Vicuna_unfiltered/repo?"
        "Revision=master&FilePath=ShareGPT_V3_unfiltered_cleaned_split.json"
    ),
}
SHAREGPT_URL = SHAREGPT_URLS.get(_DOWNLOAD_SOURCE, SHAREGPT_URLS["hf"])
SHAREGPT_CACHE_DIR = "/tmp"
SHAREGPT_FILENAME = "sharegpt_v3_cleaned.json"

MODEL = os.environ.get("MODEL", "/data/Qwen3.5-9B")
BLOCK_SIZE = 1024
SUFFIX_TARGET_TOKENS = 300
MAX_GEN_TOKENS = 1

ENGINE_KWARGS = dict(
    tensor_parallel_size=1,
    enforce_eager=True,
    gpu_memory_utilization=0.7,
    enable_prefix_caching=True,
    max_model_len=16384,
    max_num_batched_tokens=4096,
)

# Block counts NOT multiples of 4 → maximizes ALL vs ALIGN gap
DEFAULT_GROUPS = [3, 5, 7, 10]


# ── Data loading ───────────────────────────────────────────────────────

def download_sharegpt(cache_dir=SHAREGPT_CACHE_DIR):
    """Download ShareGPT dataset if not already cached."""
    cache_path = os.path.join(cache_dir, SHAREGPT_FILENAME)
    if os.path.exists(cache_path):
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"✓ Using cached ShareGPT: {cache_path} ({size_mb:.0f} MB)")
        return cache_path

    print(f"Downloading ShareGPT (~700 MB) ...")
    print(f"  URL: {SHAREGPT_URL}")
    os.makedirs(cache_dir, exist_ok=True)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.0f} MB / {total_size/(1024*1024):.0f} MB ({pct:.0f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(SHAREGPT_URL, cache_path, reporthook=_progress)
    print()  # newline after progress
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"✓ Downloaded to {cache_path} ({size_mb:.0f} MB)")
    return cache_path


def load_conversations(path):
    """Load ShareGPT JSON → list of conversation turn lists."""
    print(f"Loading conversations from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    convs = []
    for item in data:
        turns = item.get("conversations", [])
        # Need ≥2 turns, first must be human
        if len(turns) >= 2 and turns[0].get("from") == "human":
            convs.append(turns)

    print(f"✓ Loaded {len(convs)} valid conversations "
          f"(from {len(data)} total entries)")
    return convs


# ── Prompt construction ────────────────────────────────────────────────

def _tokenize_conversations(conversations, tokenizer, min_tokens,
                             max_candidates=200):
    """Tokenize conversations, pre-filtering by char length.

    Only tokenizes conversations likely long enough (using ~3 chars/token
    heuristic), and stops early once max_candidates are found.
    Returns list of (conv_idx, text, token_ids).
    """
    CHARS_PER_TOKEN = 3  # conservative estimate for mixed-language text
    min_chars = min_tokens * CHARS_PER_TOKEN

    candidates = []
    skipped = 0
    for i, turns in enumerate(conversations):
        text = "\n\n".join(t.get("value", "") for t in turns)
        if len(text) < min_chars:
            skipped += 1
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= min_tokens:
            candidates.append((i, text, ids))
            if len(candidates) >= max_candidates:
                break

    print(f"  Tokenized {len(candidates) + skipped} convs "
          f"(skipped {skipped} short, kept {len(candidates)})")
    return candidates


def build_prompt_pairs(scored, tokenizer, target_blocks,
                       num_pairs=5, suffix_tokens=300, block_size=1024):
    """Build (R1, R2) prompt pairs with precise prefix block count.

    Args:
      scored: pre-tokenized list of (conv_idx, text, token_ids)
              from _tokenize_conversations()

    For each pair:
      prefix:   first (target_blocks * block_size) tokens of a conversation
      suffix_A: next ~suffix_tokens from the SAME conversation
      suffix_B: ~suffix_tokens from a DIFFERENT conversation

    Returns list of dicts with: prefix, suffix_A, suffix_B,
      r1_prompt, r2_prompt, prefix_tokens, suffix_A_tokens, suffix_B_tokens.
    """
    target_tokens = target_blocks * block_size
    min_tokens = target_tokens + suffix_tokens

    # Filter to those long enough for this group
    valid = [(i, text, ids) for i, text, ids in scored
             if len(ids) >= min_tokens]
    print(f"  Found {len(valid)} conversations with ≥ {min_tokens} tokens")

    # If not enough single conversations, concatenate shorter ones
    if len(valid) < num_pairs + 1:
        print(f"  ⚠ Not enough single conversations; concatenating ...")
        extra = _concatenate_conversations(
            [(text, ids) for _, text, ids in scored],
            tokenizer, target_tokens, suffix_tokens,
            num_needed=num_pairs + 5,
        )
        valid.extend(extra)

    if len(valid) < 2:
        print(f"  ✗ Cannot build pairs for {target_blocks}-block prefix")
        return []

    # Build a suffix pool from all candidates (for suffix_B)
    suffix_pool = []
    for conv_idx, _, ids in valid:
        s_ids = ids[target_tokens: target_tokens + suffix_tokens]
        if len(s_ids) >= suffix_tokens // 2:
            suffix_pool.append({
                "conv_idx": conv_idx,
                "suffix_ids": s_ids,
            })

    if not suffix_pool:
        print(f"  ✗ No valid suffixes found for {target_blocks}-block prefix")
        return []

    def _decode(ids):
        return tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _common_prefix_len(a_ids, b_ids):
        limit = min(len(a_ids), len(b_ids))
        for i in range(limit):
            if a_ids[i] != b_ids[i]:
                return i
        return limit

    def _build_exact_pair(conv_idx, prefix_ids, sa_ids, sb_ids):
        prefix_text = _decode(prefix_ids)
        suffix_a = _decode(sa_ids)
        suffix_b = _decode(sb_ids)
        r1_prompt = _decode(prefix_ids + sa_ids)
        r2_prompt = _decode(prefix_ids + sb_ids)

        r1_full_ids = tokenizer.encode(r1_prompt, add_special_tokens=False)
        r2_full_ids = tokenizer.encode(r2_prompt, add_special_tokens=False)

        if len(r1_full_ids) < target_tokens or len(r2_full_ids) < target_tokens:
            return None
        if r1_full_ids[:target_tokens] != prefix_ids:
            return None
        if r2_full_ids[:target_tokens] != prefix_ids:
            return None

        shared_prefix_tokens = _common_prefix_len(r1_full_ids, r2_full_ids)
        if shared_prefix_tokens != target_tokens:
            return None

        prefix_actual = target_tokens
        sa_actual = len(r1_full_ids) - shared_prefix_tokens
        sb_actual = len(r2_full_ids) - shared_prefix_tokens
        if sa_actual < suffix_tokens // 2 or sb_actual < suffix_tokens // 2:
            return None

        return {
            "conv_idx": conv_idx,
            "prefix": prefix_text,
            "suffix_A": suffix_a,
            "suffix_B": suffix_b,
            "prefix_tokens": prefix_actual,
            "prefix_blocks": (prefix_actual + block_size - 1) // block_size,
            "suffix_A_tokens": sa_actual,
            "suffix_B_tokens": sb_actual,
            "shared_prefix_tokens": shared_prefix_tokens,
            "r1_prompt": r1_prompt,
            "r2_prompt": r2_prompt,
        }

    # Construct pairs
    pairs = []
    for idx, (conv_i, _, ids) in enumerate(valid):
        if len(pairs) >= num_pairs:
            break

        prefix_ids = ids[:target_tokens]
        sa_ids = ids[target_tokens: target_tokens + suffix_tokens]
        if len(sa_ids) < suffix_tokens // 2:
            continue

        exact_pair = None
        for offset in range(1, len(suffix_pool) + 1):
            sb_entry = suffix_pool[(idx + offset) % len(suffix_pool)]
            if sb_entry["conv_idx"] == conv_i:
                continue
            exact_pair = _build_exact_pair(
                conv_i,
                prefix_ids,
                sa_ids,
                sb_entry["suffix_ids"],
            )
            if exact_pair is not None:
                break

        if exact_pair is None:
            continue

        pairs.append(exact_pair)

    return pairs


def _concatenate_conversations(text_ids_list, tokenizer,
                                target_tokens, suffix_tokens,
                                num_needed=10):
    """Concatenate shorter conversations to reach target prefix length.

    Args:
      text_ids_list: list of (text, token_ids) from pre-tokenized data
    """
    min_tokens = target_tokens + suffix_tokens
    results = []
    buffer_text = ""
    buffer_len = 0

    for i, (text, ids) in enumerate(text_ids_list):
        piece_len = len(ids)

        if buffer_text:
            new_text = buffer_text + "\n\n---\n\n" + text
        else:
            new_text = text
        new_len = buffer_len + piece_len + 5  # approx for separator

        if new_len >= min_tokens:
            full_ids = tokenizer.encode(new_text, add_special_tokens=False)
            results.append((i, new_text, full_ids))
            buffer_text = ""
            buffer_len = 0
            if len(results) >= num_needed:
                break
        else:
            buffer_text = new_text
            buffer_len = new_len

    print(f"  Built {len(results)} concatenated texts")
    return results


# ── Benchmark engine ───────────────────────────────────────────────────


def _align_triton_enabled() -> bool:
    return bool(os.environ.get("GDN_ALIGN_TRITON_CONV1D", ""))


def _build_llm(mode, model_path):
    from vllm import LLM

    return LLM(
        model=model_path,
        **ENGINE_KWARGS,
        mamba_cache_mode=mode,
    )


def _build_sampling_params(max_gen_tokens):
    from vllm import SamplingParams

    return SamplingParams(temperature=0, max_tokens=max_gen_tokens)


def _first_pair(all_groups):
    first_group = min(all_groups.keys())
    return all_groups[first_group][0]


def _measure_pair(llm, sp, pair):
    t0 = time.perf_counter()
    r1_out = llm.generate([pair["r1_prompt"]], sp)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    r2_out = llm.generate([pair["r2_prompt"]], sp)
    t3 = time.perf_counter()

    return {
        "r1_ms": (t1 - t0) * 1000,
        "r2_ms": (t3 - t2) * 1000,
        "r1_output": r1_out[0].outputs[0].text,
        "r2_output": r2_out[0].outputs[0].text,
    }


def _warmup_mode(mode, all_groups, model_path, max_gen_tokens):
    pair = _first_pair(all_groups)
    print(f"\n  Warmup ({mode}): compile on one prompt pair using a disposable engine")
    llm = _build_llm(mode, model_path)
    sp = _build_sampling_params(max_gen_tokens)
    llm.generate([pair["r1_prompt"]], sp)
    llm.generate([pair["r2_prompt"]], sp)
    del llm


def run_one_mode(mode, all_groups, model_path, max_gen_tokens=1, num_repeats=1, do_warmup=True):
    """Run benchmark for one cache mode across all groups.

    Each measured repeat uses a fresh engine so R1 keeps the intended
    "fill cache" meaning and is not polluted by a previous repeat's cache.
    """

    print(f"\n{'═' * 70}")
    print(f"  MODE: {mode}")
    print(f"  Groups: {sorted(all_groups.keys())} blocks")
    print(f"  Measured repeats (fresh engine): {num_repeats}")
    print(f"  Warmup engine: {'on' if do_warmup else 'off'}")
    print(f"  ALIGN Triton conv1d: {'on' if _align_triton_enabled() else 'off'}")
    print(f"  max_tokens = {max_gen_tokens}")
    print(f"{'═' * 70}")

    if do_warmup:
        _warmup_mode(mode, all_groups, model_path, max_gen_tokens)

    results = {}
    for target_blocks in sorted(all_groups.keys()):
        pairs = all_groups[target_blocks]
        group_results = []
        for pair in pairs:
            group_results.append(
                {
                    "conv_idx": pair["conv_idx"],
                    "prefix_tokens": pair["prefix_tokens"],
                    "prefix_blocks": pair["prefix_blocks"],
                    "r1_all_ms": [],
                    "r2_all_ms": [],
                    "r1_output": None,
                    "r2_output": None,
                }
            )
        results[target_blocks] = group_results

    for repeat_idx in range(num_repeats):
        print(f"\n  [Repeat {repeat_idx + 1}/{num_repeats}] Starting fresh engine")
        llm = _build_llm(mode, model_path)
        sp = _build_sampling_params(max_gen_tokens)

        for target_blocks in sorted(all_groups.keys()):
            pairs = all_groups[target_blocks]
            for p_idx, pair in enumerate(pairs):
                measured = _measure_pair(llm, sp, pair)
                conv_result = results[target_blocks][p_idx]
                conv_result["r1_all_ms"].append(measured["r1_ms"])
                conv_result["r2_all_ms"].append(measured["r2_ms"])
                if repeat_idx == 0:
                    conv_result["r1_output"] = measured["r1_output"]
                    conv_result["r2_output"] = measured["r2_output"]

        del llm

    for target_blocks in sorted(results.keys()):
        print(f"\n  ── Group: {target_blocks} blocks "
              f"({target_blocks * BLOCK_SIZE} tokens) "
              f"── {len(results[target_blocks])} conversations ──")
        for p_idx, conv_result in enumerate(results[target_blocks]):
            conv_result["r1_mean_ms"] = statistics.mean(conv_result["r1_all_ms"])
            conv_result["r2_mean_ms"] = statistics.mean(conv_result["r2_all_ms"])
            speedup = conv_result["r1_mean_ms"] / max(conv_result["r2_mean_ms"], 0.1)
            print(f"    conv[{p_idx}]: prefix={conv_result['prefix_tokens']}tok "
                  f"R1={conv_result['r1_mean_ms']:.0f}ms "
                  f"R2={conv_result['r2_mean_ms']:.0f}ms "
                  f"(R1/R2={speedup:.1f}x)")

    return results


# ── Report ─────────────────────────────────────────────────────────────

def print_report(all_results, align_results, groups, model_path=MODEL, max_gen_tokens=MAX_GEN_TOKENS):
    """Print side-by-side comparison report."""
    alignment_step = ENGINE_KWARGS["max_num_batched_tokens"] // BLOCK_SIZE

    print(f"\n{'═' * 80}")
    print("  BENCHMARK REPORT: ShareGPT Prefix Cache — all-mode vs align-mode")
    print(f"{'═' * 80}")
    print(f"  Model: {model_path}")
    print(f"  block_size={BLOCK_SIZE}, "
          f"max_num_batched_tokens={ENGINE_KWARGS['max_num_batched_tokens']}")
    print(f"  alignment step = {alignment_step} blocks")
    print(f"  suffix ≈ {SUFFIX_TARGET_TOKENS} tokens, "
          f"max_gen_tokens = {max_gen_tokens}")

    summary_rows = []

    for target_blocks in groups:
        all_group = all_results.get(target_blocks, [])
        align_group = align_results.get(target_blocks, [])

        if not all_group or not align_group:
            print(f"\n  Group {target_blocks} blocks: SKIPPED (no data)")
            continue

        # Expected cache hits
        all_expected = target_blocks
        align_expected = (target_blocks // alignment_step) * alignment_step
        extra_recompute = (all_expected - align_expected) * BLOCK_SIZE

        print(f"\n  {'─' * 76}")
        print(f"  Group: {target_blocks} blocks ({target_blocks * BLOCK_SIZE} tokens)")
        print(f"  Expected cache hits: ALL={all_expected}, ALIGN={align_expected}")
        print(f"  ALIGN extra recompute: {extra_recompute} tokens "
              f"({extra_recompute / BLOCK_SIZE:.0f} blocks)")

        # Aggregate R2 times
        all_r2 = [r["r2_mean_ms"] for r in all_group]
        align_r2 = [r["r2_mean_ms"] for r in align_group]
        all_r1 = [r["r1_mean_ms"] for r in all_group]
        align_r1 = [r["r1_mean_ms"] for r in align_group]

        all_r2_avg = statistics.mean(all_r2)
        align_r2_avg = statistics.mean(align_r2)
        all_r1_avg = statistics.mean(all_r1)
        align_r1_avg = statistics.mean(align_r1)
        speedup = align_r2_avg / all_r2_avg if all_r2_avg > 0 else float("inf")

        hdr = "    {:32s} {:>12s} {:>12s} {:>10s}"
        row = "    {:32s} {:>12s} {:>12s} {:>10s}"
        print()
        print(hdr.format("Metric", "all-mode", "align-mode", "Ratio"))
        print(f"    {'─' * 68}")
        print(row.format(
            "R1 mean (fill cache) ms",
            f"{all_r1_avg:.1f}", f"{align_r1_avg:.1f}",
            f"{align_r1_avg / max(all_r1_avg, 0.1):.2f}x"))
        print(row.format(
            "R2 mean (cache hit) ms",
            f"{all_r2_avg:.1f}", f"{align_r2_avg:.1f}",
            f"{speedup:.2f}x"))
        print(row.format(
            "R2 min ms",
            f"{min(all_r2):.1f}", f"{min(align_r2):.1f}", ""))
        print(row.format(
            "R2 max ms",
            f"{max(all_r2):.1f}", f"{max(align_r2):.1f}", ""))

        print(f"\n    >>> ALL R2 is {speedup:.2f}x faster than ALIGN R2 "
              f"(saved {align_r2_avg - all_r2_avg:.0f} ms avg)")

        # Per-conversation details
        print(f"\n    Per-conversation R2 (ms):")
        for i in range(min(len(all_group), len(align_group))):
            a = all_group[i]
            g = align_group[i]
            local_sp = g["r2_mean_ms"] / max(a["r2_mean_ms"], 0.1)
            print(f"      conv[{i}]: ALL={a['r2_mean_ms']:.0f}  "
                  f"ALIGN={g['r2_mean_ms']:.0f}  ratio={local_sp:.2f}x")

        # Output comparison
        if all_group and align_group:
            print(f"\n    Output samples (conv[0]):")
            print(f"      ALL  R2: {all_group[0]['r2_output'][:80]!r}")
            print(f"      ALIGN R2: {align_group[0]['r2_output'][:80]!r}")
            match = all_group[0]["r2_output"] == align_group[0]["r2_output"]
            print(f"      Match: {'✓' if match else '✗'}")

        summary_rows.append((target_blocks, all_r2_avg, align_r2_avg, speedup))

    # Summary table
    print(f"\n  {'═' * 76}")
    print("  SUMMARY")
    print(f"  {'═' * 76}")
    hdr = "    {:>8s} {:>12s} {:>14s} {:>14s} {:>10s}"
    print(hdr.format("Blocks", "Extra tokens", "ALL R2 (ms)", "ALIGN R2 (ms)", "Speedup"))
    print(f"    {'─' * 60}")
    for blk, a_r2, g_r2, sp in summary_rows:
        extra = (blk - (blk // alignment_step) * alignment_step) * BLOCK_SIZE
        print(hdr.format(
            str(blk), str(extra), f"{a_r2:.1f}", f"{g_r2:.1f}", f"{sp:.2f}x"))
    print(f"  {'═' * 76}")


def print_data_summary(all_groups):
    """Print prompt construction summary (for dry-run or pre-run check)."""
    print(f"\n{'═' * 70}")
    print("  DATA SUMMARY")
    print(f"{'═' * 70}")
    alignment_step = ENGINE_KWARGS["max_num_batched_tokens"] // BLOCK_SIZE

    for target_blocks in sorted(all_groups.keys()):
        pairs = all_groups[target_blocks]
        all_expected = target_blocks
        align_expected = (target_blocks // alignment_step) * alignment_step

        print(f"\n  Group: {target_blocks} blocks "
              f"({target_blocks * BLOCK_SIZE} tokens)")
        print(f"  Expected hits: ALL={all_expected}, ALIGN={align_expected}, "
              f"extra recompute={all_expected - align_expected} blocks")
        print(f"  Conversations: {len(pairs)}")
        for i, p in enumerate(pairs):
            print(f"    [{i}] prefix={p['prefix_tokens']}tok "
                  f"({p['prefix_blocks']} blocks)  "
                  f"sA={p['suffix_A_tokens']}tok  sB={p['suffix_B_tokens']}tok  "
                  f"shared={p.get('shared_prefix_tokens', p['prefix_tokens'])}tok  "
                  f"r1_total={p['prefix_tokens']+p['suffix_A_tokens']}tok  "
                  f"r2_total={p['prefix_tokens']+p['suffix_B_tokens']}tok")


def _load_prompt_file(path):
    with open(path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    if isinstance(saved, dict) and isinstance(saved.get("groups"), dict):
        return saved["groups"], saved.get("_meta")
    return saved, None


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ShareGPT prefix caching benchmark: all-mode vs align-mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--groups", type=int, nargs="+", default=DEFAULT_GROUPS,
        help=f"Prefix block counts to test (default: {DEFAULT_GROUPS})")
    parser.add_argument(
        "--convs-per-group", type=int, default=5,
        help="Conversations per group (default: 5)")
    parser.add_argument(
        "--num-repeats", "--num-iters", dest="num_repeats", type=int, default=1,
        help="Measured repeats with a fresh engine per repeat (default: 1)")
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_GEN_TOKENS,
        help=f"Max tokens to generate (default: {MAX_GEN_TOKENS})")
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to pre-downloaded ShareGPT JSON")
    parser.add_argument(
        "--mode", choices=["all", "align", "both"], default="both",
        help="Cache mode(s) to benchmark (default: both)")
    parser.add_argument(
        "--model", type=str, default=MODEL,
        help=f"Model path (default: {MODEL})")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only construct prompts and print stats (no engine)")
    parser.add_argument(
        "--hf-mirror", type=str, default=None,
        help="HuggingFace mirror URL (e.g. https://hf-mirror.com)")
    parser.add_argument(
        "--source", choices=["hf", "modelscope"], default=None,
        help="Download source: hf (HuggingFace) or modelscope (Alibaba Cloud)")
    parser.add_argument(
        "--export-prompts", type=str, default=None,
        help="Export constructed prompt pairs to JSON (skip engine run)")
    parser.add_argument(
        "--load-prompts", type=str, default=None,
        help="Load pre-built prompt pairs from JSON (skip download+tokenize)")
    parser.add_argument(
        "--align-triton-conv1d",
        action="store_true",
        help="Set GDN_ALIGN_TRITON_CONV1D=1 before importing vLLM so ALIGN uses Triton conv1d.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the disposable warmup engine that compiles kernels before measured repeats.",
    )
    args = parser.parse_args()

    if args.align_triton_conv1d:
        os.environ["GDN_ALIGN_TRITON_CONV1D"] = "1"

    # Apply source/mirror overrides
    global SHAREGPT_URL
    if args.source:
        SHAREGPT_URL = SHAREGPT_URLS[args.source]
    elif args.hf_mirror:
        endpoint = args.hf_mirror.rstrip("/")
        SHAREGPT_URL = (
            f"{endpoint}/datasets/anon8231489123/"
            "ShareGPT_Vicuna_unfiltered/resolve/main/"
            "ShareGPT_V3_unfiltered_cleaned_split.json"
        )

    model_path = args.model
    max_gen_tokens = args.max_tokens

    t_start = time.time()

    # ── Load pre-built prompts or build from scratch ──
    prompt_meta = None
    if args.load_prompts:
        print(f"\nLoading pre-built prompts from {args.load_prompts} ...")
        saved_groups, prompt_meta = _load_prompt_file(args.load_prompts)
        all_groups = {int(k): v for k, v in saved_groups.items()}
        # Filter to requested groups
        if args.groups != DEFAULT_GROUPS:
            all_groups = {k: v for k, v in all_groups.items()
                          if k in args.groups}
        print(f"✓ Loaded {sum(len(v) for v in all_groups.values())} "
              f"prompt pairs across {len(all_groups)} groups")
        if prompt_meta:
            built_tokenizer = prompt_meta.get("tokenizer_name_or_path")
            built_model = prompt_meta.get("model_path")
            print(f"  Prompt meta: model={built_model}, tokenizer={built_tokenizer}")
            if built_tokenizer and os.path.abspath(str(built_tokenizer)) != os.path.abspath(model_path):
                print("  ⚠ Prompt JSON tokenizer path does not match current --model path.")
    else:
        # ── Step 1: Load data ──
        data_path = args.data_path or download_sharegpt()
        conversations = load_conversations(data_path)

        # ── Step 2: Tokenizer ──
        print(f"\nLoading tokenizer from {model_path} ...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        prompt_meta = {
            "model_path": model_path,
            "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", model_path),
            "block_size": BLOCK_SIZE,
            "suffix_target_tokens": SUFFIX_TARGET_TOKENS,
        }
        print(f"✓ Tokenizer loaded from {prompt_meta['tokenizer_name_or_path']} "
              f"(vocab_size={tokenizer.vocab_size})")

        # ── Step 3: Tokenize once (with smallest group's threshold) ──
        min_blocks = min(args.groups)
        min_min_tokens = min_blocks * BLOCK_SIZE + SUFFIX_TARGET_TOKENS
        print(f"\nPre-tokenizing conversations "
              f"(need ≥ {min_min_tokens} tokens for {min_blocks}-block group) ...")
        scored = _tokenize_conversations(
            conversations, tokenizer, min_min_tokens,
            max_candidates=500,
        )

        # ── Step 4: Construct prompts from cached tokenization ──
        all_groups = {}
        for target_blocks in args.groups:
            print(f"\nBuilding {args.convs_per_group} prompt pairs "
                  f"for {target_blocks}-block prefix "
                  f"({target_blocks * BLOCK_SIZE} tokens) ...")
            pairs = build_prompt_pairs(
                scored, tokenizer, target_blocks,
                num_pairs=args.convs_per_group,
                suffix_tokens=SUFFIX_TARGET_TOKENS,
                block_size=BLOCK_SIZE,
            )
            if pairs:
                all_groups[target_blocks] = pairs
            else:
                print(f"  ✗ Skipping {target_blocks}-block group "
                      f"(insufficient data)")

    if not all_groups:
        print("\n✗ No prompt pairs could be constructed. Exiting.")
        return

    # Print data summary
    print_data_summary(all_groups)
    t_data = time.time() - t_start
    print(f"\n  Data preparation took {t_data:.1f}s")

    # Export prompts if requested
    if args.export_prompts:
        export_path = args.export_prompts
        export_parent = os.path.dirname(export_path)
        if export_parent:
            os.makedirs(export_parent, exist_ok=True)
        payload = {
            "_meta": prompt_meta or {
                "model_path": model_path,
                "tokenizer_name_or_path": model_path,
                "block_size": BLOCK_SIZE,
                "suffix_target_tokens": SUFFIX_TARGET_TOKENS,
            },
            "groups": {str(k): v for k, v in all_groups.items()},
        }
        with open(export_path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        size_kb = os.path.getsize(export_path) / 1024
        print(f"\n✓ Exported prompts to {export_path} ({size_kb:.0f} KB)")
        if args.dry_run:
            print("  [DRY RUN + EXPORT] Done.")
            return

    if args.dry_run:
        print("\n  [DRY RUN] Skipping engine runs.")
        return

    # ── Step 4: Run benchmarks ──
    all_results = {}
    align_results = {}

    if args.mode in ("all", "both"):
        all_results = run_one_mode(
            "all", all_groups, model_path, max_gen_tokens, args.num_repeats, not args.skip_warmup)

    if args.mode in ("align", "both"):
        align_results = run_one_mode(
            "align", all_groups, model_path, max_gen_tokens, args.num_repeats, not args.skip_warmup)

    # ── Step 5: Report ──
    if all_results and align_results:
        print_report(all_results, align_results, args.groups, model_path, max_gen_tokens)
    elif all_results:
        print("\n  Only all-mode results available.")
        for blk, group in all_results.items():
            r2_avg = statistics.mean([r["r2_mean_ms"] for r in group])
            print(f"    {blk} blocks: R2 mean = {r2_avg:.1f} ms")
    elif align_results:
        print("\n  Only align-mode results available.")
        for blk, group in align_results.items():
            r2_avg = statistics.mean([r["r2_mean_ms"] for r in group])
            print(f"    {blk} blocks: R2 mean = {r2_avg:.1f} ms")

    total = time.time() - t_start
    print(f"\n  Total benchmark time: {total:.0f}s")


if __name__ == "__main__":
    main()
