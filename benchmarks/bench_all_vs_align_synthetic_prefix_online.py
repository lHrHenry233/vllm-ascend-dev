#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Synthetic online benchmark for manual ALL-vs-ALIGN prefix-cache testing.

This benchmark creates two long prompts with exact tokenizer-controlled lengths:

- R1 total prompt tokens
- R2 total prompt tokens
- exact shared prefix tokens between R1 and R2

Unlike the earlier version, this script does not launch `vllm serve`.
You must start the server manually in the foreground so runtime failures and
stack traces stay visible in your terminal. The script only sends R1 -> R2
through the OpenAI-compatible HTTP API and optionally follows a tee'd server
log to read actual `[GDN_CACHE_HIT]` lines.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import bench_all_vs_align_sharegpt as offline_base

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
        shared_ids = shared_pool[shared_offset: shared_offset + shared_prefix_tokens]
        if len(shared_ids) < shared_prefix_tokens:
            continue
        for suffix_a_offset in range(8):
            r1_ids = shared_ids + suffix_a_pool[suffix_a_offset: suffix_a_offset + suffix_a_tokens]
            if len(r1_ids) != r1_total_tokens:
                continue
            for suffix_b_offset in range(8):
                r2_ids = shared_ids + suffix_b_pool[suffix_b_offset: suffix_b_offset + suffix_b_tokens]
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


def _urlopen_json(url: str, data: dict | None = None, timeout: float = 30.0) -> dict:
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}\n{detail}") from exc


class ExternalOpenAIClient:
    def __init__(
        self,
        mode: str,
        base_url: str,
        request_timeout: float,
        model_id: str | None = None,
    ) -> None:
        self.mode = mode
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.model_id = model_id

    def ensure_model_id(self) -> str:
        if self.model_id is None:
            models = _urlopen_json(f"{self.base_url}/v1/models", timeout=5.0)
            self.model_id = models["data"][0]["id"]
        return self.model_id

    def generate(self, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.ensure_model_id(),
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        data = _urlopen_json(
            f"{self.base_url}/v1/completions",
            data=payload,
            timeout=self.request_timeout,
        )
        return data["choices"][0]["text"]


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


def _send_one_request(
    client: ExternalOpenAIClient,
    log_path: Path,
    prompt: str,
    max_tokens: int,
    cache_log_timeout: float,
):
    start_offset = log_path.stat().st_size if log_path.exists() else 0
    t0 = time.perf_counter()
    text = client.generate(prompt, max_tokens)
    t1 = time.perf_counter()
    cache_info = wait_for_cache_log(log_path, start_offset, client.mode, timeout=cache_log_timeout)
    return {
        "text": text,
        "ms": (t1 - t0) * 1000,
        "cache": cache_info,
    }


def run_one_mode(mode: str, prompts: dict, args: argparse.Namespace):
    expected_hit_tokens = prompts["shared_prefix_tokens"] if mode == "all" else 0
    expected_hit_rate = expected_hit_tokens / prompts["r2_total_tokens"]
    log_path = Path(args.server_log_path)
    client = ExternalOpenAIClient(
        mode=mode,
        base_url=args.base_url,
        request_timeout=args.request_timeout,
        model_id=args.model_id,
    )

    print(f"\n{'=' * 76}")
    print(f"  MODE: {mode} (manual online synthetic)")
    print(f"  Base URL: {args.base_url}")
    print(f"  Server log: {log_path}")
    print(
        f"  Prompt geometry: shared={prompts['shared_prefix_tokens']} "
        f"r1_total={prompts['r1_total_tokens']} r2_total={prompts['r2_total_tokens']}"
    )
    print(f"  Expected R2 hits: {expected_hit_tokens} tokens ({expected_hit_rate:.4%})")
    print(f"{'=' * 76}")

    model_id = client.ensure_model_id()
    print(f"  Connected model_id={model_id}")
    r1 = _send_one_request(client, log_path, prompts["r1_prompt"], args.max_tokens, args.cache_log_timeout)
    r2 = _send_one_request(client, log_path, prompts["r2_prompt"], args.max_tokens, args.cache_log_timeout)

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

    return {
        "mode": mode,
        "expected_hit_tokens": expected_hit_tokens,
        "expected_hit_rate": expected_hit_rate,
        "r1_ms": r1["ms"],
        "r2_ms": r2["ms"],
        "actual_cached_tokens": r2["cache"]["cached_tokens"],
        "actual_hit_rate": r2["cache"]["hit_rate"],
        "r1_output": r1["text"],
        "r2_output": r2["text"],
    }


def print_summary(prompts: dict, args: argparse.Namespace, result: dict):
    print(f"\n{'=' * 80}")
    print("  SYNTHETIC PREFIX CACHE REPORT")
    print(f"{'=' * 80}")
    print(
        f"  mode={result['mode']}, "
        f"shared_prefix_tokens={prompts['shared_prefix_tokens']} "
        f"({prompts['shared_prefix_tokens'] // args.block_size} blocks), "
        f"r1_total={prompts['r1_total_tokens']}, "
        f"r2_total={prompts['r2_total_tokens']}"
    )
    print(
        f"  suffix_A={prompts['suffix_a_tokens']} tokens, "
        f"suffix_B={prompts['suffix_b_tokens']} tokens"
    )
    print(
        f"  R1={result['r1_ms']:.1f} ms  "
        f"R2={result['r2_ms']:.1f} ms  "
        f"expected_cached={result['expected_hit_tokens']}  "
        f"actual_cached={result['actual_cached_tokens']}  "
        f"actual_hit_rate={result['actual_hit_rate']:.4%}"
    )
    if result["actual_cached_tokens"] != result["expected_hit_tokens"]:
        print("  WARNING: actual cached tokens differ from expected geometry")


def export_prompts(prompts: dict, args: argparse.Namespace) -> None:
    export_dir = Path(args.export_prompts_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    r1_txt = export_dir / "r1.txt"
    r2_txt = export_dir / "r2.txt"
    r1_json = export_dir / "r1.json"
    r2_json = export_dir / "r2.json"

    r1_txt.write_text(prompts["r1_prompt"], encoding="utf-8")
    r2_txt.write_text(prompts["r2_prompt"], encoding="utf-8")

    model_name = args.model_id or args.model
    r1_json.write_text(
        json.dumps(
            {
                "model": model_name,
                "prompt": prompts["r1_prompt"],
                "max_tokens": args.max_tokens,
                "temperature": 0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    r2_json.write_text(
        json.dumps(
            {
                "model": model_name,
                "prompt": prompts["r2_prompt"],
                "max_tokens": args.max_tokens,
                "temperature": 0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n  Exported prompt artifacts")
    print(f"    {r1_txt}")
    print(f"    {r2_txt}")
    print(f"    {r1_json}")
    print(f"    {r2_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Manual online synthetic ALL-vs-ALIGN prefix cache benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="/data/Qwen3.5-9B")
    parser.add_argument("--shared-prefix-tokens", type=int, default=12288)
    parser.add_argument("--r1-total-tokens", type=int, default=17000)
    parser.add_argument("--r2-total-tokens", type=int, default=17000)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--mode", choices=["all", "align"], required=True)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8100")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--export-prompts-dir", type=str, default=None)
    parser.add_argument("--server-log-path", type=str, default=None)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--cache-log-timeout", type=float, default=30.0)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.num_repeats != 1:
        parser.error(
            "manual mode only supports --num-repeats 1 because the server cache must "
            "start fresh for each measured run; restart the server and rerun instead"
        )
    if not args.dry_run and not args.server_log_path:
        parser.error("--server-log-path is required unless --dry-run is used")

    print(f"\nLoading tokenizer from {args.model} ...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(
        f"✓ Tokenizer loaded from {getattr(tokenizer, 'name_or_path', args.model)} "
        f"(vocab_size={tokenizer.vocab_size})"
    )

    print("\nBuilding exact synthetic prompts ...")
    prompts = build_exact_synthetic_prompts(
        tokenizer,
        shared_prefix_tokens=args.shared_prefix_tokens,
        r1_total_tokens=args.r1_total_tokens,
        r2_total_tokens=args.r2_total_tokens,
    )
    print("✓ Synthetic prompts constructed")

    shared_blocks = prompts["shared_prefix_tokens"] // args.block_size
    align_step_blocks = args.max_num_batched_tokens // args.block_size
    print("\n  Prompt geometry")
    print(f"    shared_prefix_tokens = {prompts['shared_prefix_tokens']}")
    print(f"    shared_prefix_blocks = {shared_blocks}")
    print(f"    r1_total_tokens      = {prompts['r1_total_tokens']}")
    print(f"    r2_total_tokens      = {prompts['r2_total_tokens']}")
    print(f"    suffix_A_tokens      = {prompts['suffix_a_tokens']}")
    print(f"    suffix_B_tokens      = {prompts['suffix_b_tokens']}")
    print(f"    ALIGN step blocks    = {align_step_blocks}")
    print(f"    ALL expected hits    = {shared_blocks}")
    print(f"    ALIGN expected hits  = {(shared_blocks // align_step_blocks) * align_step_blocks}")

    if args.export_prompts_dir:
        print(f"\nExporting prompt artifacts to {args.export_prompts_dir} ...")
        export_prompts(prompts, args)

    if args.dry_run:
        print("\n  [DRY RUN] Prompt construction complete.")
        return

    print("\n  Manual serve requirements")
    print("    1. Start vllm serve yourself in the foreground.")
    print("    2. Export GDN_BENCH_LOG_CACHED_TOKENS=1 before serve.")
    print("    3. Pipe stdout/stderr through tee and pass the same file to --server-log-path.")

    result = run_one_mode(args.mode, prompts, args)
    print_summary(prompts, args, result)


if __name__ == "__main__":
    main()
