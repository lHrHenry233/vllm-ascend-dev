#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Online ShareGPT-based prefix caching benchmark: all-mode vs align-mode.

Uses the same ShareGPT prompt-pair data and report format as
bench_all_vs_align_sharegpt.py, but runs through `vllm serve` and the
OpenAI-compatible HTTP API instead of `LLM.generate()`.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import bench_all_vs_align_sharegpt as offline_base


def _urlopen_json(url: str, data: dict | None = None, timeout: float = 30.0) -> dict:
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class LocalOpenAIServer:
    def __init__(self, mode: str, model_path: str, args: argparse.Namespace, results_dir: Path) -> None:
        self.mode = mode
        self.model_path = model_path
        self.args = args
        self.results_dir = results_dir
        self.port = args.port
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None
        self.model_id: str | None = None
        self.log_path = self.results_dir / f"server_{mode}.log"
        self.log_file = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def _build_command(self) -> list[str]:
        cmd = [
            "vllm",
            "serve",
            self.model_path,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(offline_base.ENGINE_KWARGS["tensor_parallel_size"]),
            "--enforce-eager",
            "--gpu-memory-utilization",
            str(self.args.gpu_memory_utilization),
            "--enable-prefix-caching",
            "--mamba-cache-mode",
            self.mode,
            "--max-model-len",
            str(self.args.max_model_len),
            "--max-num-batched-tokens",
            str(self.args.max_num_batched_tokens),
        ]
        if self.args.served_model_name:
            cmd.extend(["--served-model-name", self.args.served_model_name])
        return cmd

    def _log_tail(self, max_lines: int = 80) -> str:
        if not self.log_path.exists():
            return "<log file missing>"
        lines = self.log_path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])

    def start(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w")
        cmd = self._build_command()
        print(f"\n>>> Starting online server for {self.mode}: {' '.join(cmd)}")
        env = os.environ.copy()
        if self.args.align_triton_conv1d:
            env["GDN_ALIGN_TRITON_CONV1D"] = "1"
        self.proc = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

        deadline = time.time() + self.args.startup_timeout
        last_error = None
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited early for mode={self.mode}.\n"
                    f"--- server log tail ---\n{self._log_tail()}"
                )
            try:
                models = _urlopen_json(f"{self.base_url}/v1/models", timeout=5.0)
                self.model_id = models["data"][0]["id"]
                print(f"    Server ready, model_id={self.model_id}")
                return
            except Exception as exc:  # readiness polling only
                last_error = exc
                time.sleep(5)

        raise RuntimeError(
            f"Timed out waiting for mode={self.mode} server readiness: {last_error}\n"
            f"--- server log tail ---\n{self._log_tail()}"
        )

    def stop(self) -> None:
        if self.proc is None:
            return
        print(f">>> Stopping online server for {self.mode}...")
        self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        self.proc = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def generate(self, prompt: str, max_tokens: int) -> str:
        if self.model_id is None:
            raise RuntimeError("Server not ready")
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        data = _urlopen_json(
            f"{self.base_url}/v1/completions",
            data=payload,
            timeout=self.args.request_timeout,
        )
        return data["choices"][0]["text"]


def load_or_build_groups(args: argparse.Namespace, model_path: str):
    if args.source:
        offline_base.SHAREGPT_URL = offline_base.SHAREGPT_URLS[args.source]
    elif args.hf_mirror:
        endpoint = args.hf_mirror.rstrip("/")
        offline_base.SHAREGPT_URL = (
            f"{endpoint}/datasets/anon8231489123/"
            "ShareGPT_Vicuna_unfiltered/resolve/main/"
            "ShareGPT_V3_unfiltered_cleaned_split.json"
        )

    if args.load_prompts:
        print(f"\nLoading pre-built prompts from {args.load_prompts} ...")
        with open(args.load_prompts) as f:
            saved = json.load(f)
        all_groups = {int(k): v for k, v in saved.items()}
        if args.groups != offline_base.DEFAULT_GROUPS:
            all_groups = {k: v for k, v in all_groups.items() if k in args.groups}
        print(f"✓ Loaded {sum(len(v) for v in all_groups.values())} prompt pairs across {len(all_groups)} groups")
    else:
        data_path = args.data_path or offline_base.download_sharegpt()
        conversations = offline_base.load_conversations(data_path)

        print(f"\nLoading tokenizer from {model_path} ...")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✓ Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

        min_blocks = min(args.groups)
        min_min_tokens = min_blocks * offline_base.BLOCK_SIZE + offline_base.SUFFIX_TARGET_TOKENS
        print(
            f"\nPre-tokenizing conversations (need ≥ {min_min_tokens} tokens for {min_blocks}-block group) ..."
        )
        scored = offline_base._tokenize_conversations(
            conversations,
            tokenizer,
            min_min_tokens,
            max_candidates=500,
        )

        all_groups = {}
        for target_blocks in args.groups:
            print(
                f"\nBuilding {args.convs_per_group} prompt pairs for {target_blocks}-block prefix "
                f"({target_blocks * offline_base.BLOCK_SIZE} tokens) ..."
            )
            pairs = offline_base.build_prompt_pairs(
                scored,
                tokenizer,
                target_blocks,
                num_pairs=args.convs_per_group,
                suffix_tokens=offline_base.SUFFIX_TARGET_TOKENS,
                block_size=offline_base.BLOCK_SIZE,
            )
            if pairs:
                all_groups[target_blocks] = pairs
            else:
                print(f"  ✗ Skipping {target_blocks}-block group (insufficient data)")

    if not all_groups:
        raise RuntimeError("No prompt pairs could be constructed")

    offline_base.print_data_summary(all_groups)

    if args.export_prompts:
        export_path = args.export_prompts
        export_parent = os.path.dirname(export_path)
        if export_parent:
            os.makedirs(export_parent, exist_ok=True)
        with open(export_path, "w") as f:
            json.dump({str(k): v for k, v in all_groups.items()}, f, ensure_ascii=False, indent=2)
        size_kb = os.path.getsize(export_path) / 1024
        print(f"\n✓ Exported prompts to {export_path} ({size_kb:.0f} KB)")

    return all_groups


def _measure_pair_online(server: LocalOpenAIServer, pair: dict, max_gen_tokens: int):
    t0 = time.perf_counter()
    r1_text = server.generate(pair["r1_prompt"], max_gen_tokens)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    r2_text = server.generate(pair["r2_prompt"], max_gen_tokens)
    t3 = time.perf_counter()

    return {
        "r1_ms": (t1 - t0) * 1000,
        "r2_ms": (t3 - t2) * 1000,
        "r1_output": r1_text,
        "r2_output": r2_text,
    }


def _warmup_mode_online(mode: str, all_groups: dict[int, list[dict]], model_path: str, max_gen_tokens: int,
                        args: argparse.Namespace, results_dir: Path):
    pair = offline_base._first_pair(all_groups)
    warmup_dir = results_dir / f"{mode}_warmup"
    print(f"\n  Warmup ({mode}, online): compile on one prompt pair using a disposable server")
    with LocalOpenAIServer(mode, model_path, args, warmup_dir) as server:
        server.generate(pair["r1_prompt"], max_gen_tokens)
        server.generate(pair["r2_prompt"], max_gen_tokens)


def run_one_mode_online(
    mode: str,
    all_groups: dict[int, list[dict]],
    model_path: str,
    max_gen_tokens: int,
    args: argparse.Namespace,
    results_dir: Path,
):
    print(f"\n{'═' * 70}")
    print(f"  MODE: {mode} (online)")
    print(f"  Groups: {sorted(all_groups.keys())} blocks")
    print(f"  Measured repeats (fresh server): {args.num_repeats}")
    print(f"  Warmup server: {'on' if not args.skip_warmup else 'off'}")
    print(f"  ALIGN Triton conv1d: {'on' if args.align_triton_conv1d else 'off'}")
    print(f"  max_tokens = {max_gen_tokens}")
    print(f"{'═' * 70}")

    if not args.skip_warmup:
        _warmup_mode_online(mode, all_groups, model_path, max_gen_tokens, args, results_dir)

    results = {}
    for target_blocks in sorted(all_groups.keys()):
        pairs = all_groups[target_blocks]
        results[target_blocks] = [
            {
                "conv_idx": pair["conv_idx"],
                "prefix_tokens": pair["prefix_tokens"],
                "prefix_blocks": pair["prefix_blocks"],
                "r1_all_ms": [],
                "r2_all_ms": [],
                "r1_output": None,
                "r2_output": None,
            }
            for pair in pairs
        ]

    for repeat_idx in range(args.num_repeats):
        print(f"\n  [Repeat {repeat_idx + 1}/{args.num_repeats}] Starting fresh server")
        repeat_dir = results_dir / f"{mode}_repeat{repeat_idx + 1}"
        with LocalOpenAIServer(mode, model_path, args, repeat_dir) as server:
            for target_blocks in sorted(all_groups.keys()):
                pairs = all_groups[target_blocks]
                for p_idx, pair in enumerate(pairs):
                    measured = _measure_pair_online(server, pair, max_gen_tokens)
                    conv_result = results[target_blocks][p_idx]
                    conv_result["r1_all_ms"].append(measured["r1_ms"])
                    conv_result["r2_all_ms"].append(measured["r2_ms"])
                    if repeat_idx == 0:
                        conv_result["r1_output"] = measured["r1_output"]
                        conv_result["r2_output"] = measured["r2_output"]

    for target_blocks in sorted(results.keys()):
        print(
            f"\n  ── Group: {target_blocks} blocks "
            f"({target_blocks * offline_base.BLOCK_SIZE} tokens) "
            f"── {len(results[target_blocks])} conversations ──"
        )
        for p_idx, conv_result in enumerate(results[target_blocks]):
            conv_result["r1_mean_ms"] = statistics.mean(conv_result["r1_all_ms"])
            conv_result["r2_mean_ms"] = statistics.mean(conv_result["r2_all_ms"])
            speedup = conv_result["r1_mean_ms"] / max(conv_result["r2_mean_ms"], 0.1)
            print(
                f"    conv[{p_idx}]: prefix={conv_result['prefix_tokens']}tok "
                f"R1={conv_result['r1_mean_ms']:.0f}ms "
                f"R2={conv_result['r2_mean_ms']:.0f}ms "
                f"(R1/R2={speedup:.1f}x)"
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Online ShareGPT prefix caching benchmark: all-mode vs align-mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--groups", type=int, nargs="+", default=offline_base.DEFAULT_GROUPS)
    parser.add_argument("--convs-per-group", type=int, default=5)
    parser.add_argument("--num-repeats", "--num-iters", dest="num_repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=offline_base.MAX_GEN_TOKENS)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--mode", choices=["all", "align", "both"], default="both")
    parser.add_argument("--model", type=str, default=offline_base.MODEL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hf-mirror", type=str, default=None)
    parser.add_argument("--source", choices=["hf", "modelscope"], default=None)
    parser.add_argument("--export-prompts", type=str, default=None)
    parser.add_argument("--load-prompts", type=str, default=None)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=offline_base.ENGINE_KWARGS["gpu_memory_utilization"],
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=offline_base.ENGINE_KWARGS["max_model_len"],
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=offline_base.ENGINE_KWARGS["max_num_batched_tokens"],
    )
    parser.add_argument("--served-model-name", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument(
        "--align-triton-conv1d",
        action="store_true",
        help="Set GDN_ALIGN_TRITON_CONV1D=1 for the server process so ALIGN uses the Triton conv1d path.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the disposable warmup server that compiles kernels before measured repeats.",
    )
    args = parser.parse_args()

    model_path = args.model
    max_gen_tokens = args.max_tokens
    t_start = time.time()

    all_groups = load_or_build_groups(args, model_path)
    t_data = time.time() - t_start
    print(f"\n  Data preparation took {t_data:.1f}s")

    if args.dry_run:
        print("\n  [DRY RUN] Skipping online engine runs.")
        return

    results_dir = Path(
        args.results_dir
        or f"benchmarks/results/sharegpt_online_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    align_results = {}

    if args.mode in ("all", "both"):
        all_results = run_one_mode_online("all", all_groups, model_path, max_gen_tokens, args, results_dir)

    if args.mode in ("align", "both"):
        align_results = run_one_mode_online("align", all_groups, model_path, max_gen_tokens, args, results_dir)

    if all_results and align_results:
        offline_base.print_report(all_results, align_results, args.groups, model_path, max_gen_tokens)
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

    print(f"\n  Results directory: {results_dir}")
    total = time.time() - t_start
    print(f"\n  Total benchmark time: {total:.0f}s")


if __name__ == "__main__":
    main()
