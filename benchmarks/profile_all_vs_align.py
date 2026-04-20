#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Online profiling: all-mode vs align-mode with vLLM Server + Ascend Profiler.

Starts vLLM server with profiling enabled, sends requests to trigger prefix
caching (R1=fill, R2=cache hit), then stops profiling. The generated
trace_view.json can be viewed in MindStudio Insight or chrome://tracing.

Usage (on NPU server):
    # Profile ALL mode
    python benchmarks/profile_all_vs_align.py --mode all

    # Profile ALIGN mode
    python benchmarks/profile_all_vs_align.py --mode align

    # Profile both (sequentially, separate server instances)
    python benchmarks/profile_all_vs_align.py --mode both

    # Fair comparison (both use Triton conv1d)
    GDN_ALIGN_TRITON_CONV1D=1 python benchmarks/profile_all_vs_align.py

After run:
    1. Find trace in ./vllm_profile_{mode}/*_ascend_pt/ASCEND_PROFILER_OUTPUT/
    2. Download trace_view.json to local machine
    3. Open in MindStudio Insight or chrome://tracing or https://ui.perfetto.dev/
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL = "/shared/models/Qwen3.5-0.8B-ms"
PORT = 8100
HOST = "127.0.0.1"
BASE_URL = f"http://{HOST}:{PORT}"

# Prompt data for prefix caching test
SURVEY_DOCUMENT = None  # Loaded at runtime

# ─── Prompt Setup ────────────────────────────────────────────────────────────

def load_prompts():
    """Load test prompts. Try importing from test file, fall back to inline."""
    global SURVEY_DOCUMENT
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from tests.e2e.singlecard.test_qwen3_5_cache_scenarios import (
            SURVEY_DOCUMENT as _doc,
            SURVEY_QUESTION_A,
            SURVEY_QUESTION_B,
        )
        SURVEY_DOCUMENT = _doc
        return _doc + SURVEY_QUESTION_A, _doc + SURVEY_QUESTION_B
    except ImportError:
        print("WARNING: Could not import test prompts. Using inline placeholder.")
        prefix = "This is a placeholder prefix document. " * 500
        SURVEY_DOCUMENT = prefix
        q_a = "\n\nQuestion: What is the main topic?\n\nAnswer: "
        q_b = "\n\nQuestion: What are the conclusions?\n\nAnswer: "
        return prefix + q_a, prefix + q_b


# ─── HTTP Helpers ────────────────────────────────────────────────────────────

def http_post(url: str, data: dict | None = None, timeout: int = 30) -> dict | None:
    """Simple HTTP POST using urllib (no external deps)."""
    body = json.dumps(data).encode() if data else b""
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content = resp.read().decode()
        if content:
            return json.loads(content)
        return None


def http_get(url: str, timeout: int = 5) -> int:
    """Simple HTTP GET, returns status code."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def wait_for_server(timeout: int = 300) -> bool:
    """Poll server health endpoint until ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            status = http_get(f"{BASE_URL}/health")
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


# ─── Chat Completion Request ─────────────────────────────────────────────────

def send_chat_request(prompt: str, max_tokens: int = 10) -> dict:
    """Send a chat completion request and return the response."""
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    return http_post(f"{BASE_URL}/v1/chat/completions", data, timeout=120)


# ─── Server Management ───────────────────────────────────────────────────────

def start_server(mode: str, profile_dir: str) -> subprocess.Popen:
    """Start vLLM server with profiling enabled."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(PORT),
        "--host", HOST,
        "--tensor-parallel-size", "1",
        "--enforce-eager",
        "--gpu-memory-utilization", "0.7",
        "--enable-prefix-caching",
        "--max-model-len", "8192",
        "--max-num-batched-tokens", "4096",
        "--mamba-cache-mode", mode,
        # Profiler config (dotted path syntax per vllm-ascend convention)
        "--profiler-config.profiler", "torch",
        "--profiler-config.torch_profiler_dir", profile_dir,
    ]
    
    env = dict(os.environ)
    # Ensure GDN_ALIGN_TRITON_CONV1D propagates for fair comparison
    if "GDN_ALIGN_TRITON_CONV1D" not in env and mode == "align":
        env["GDN_ALIGN_TRITON_CONV1D"] = "1"

    print(f"\n[{mode}] Starting vLLM server...")
    print(f"  CMD: {' '.join(cmd[:10])}...")
    print(f"  Profile dir: {profile_dir}")
    
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    return proc


def stop_server(proc: subprocess.Popen):
    """Gracefully stop the server process."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ─── Profiling Workflow ──────────────────────────────────────────────────────

def run_profile_session(mode: str, output_dir: str, max_tokens: int = 10,
                        warmup_reqs: int = 4, profile_rounds: int = 3):
    """Complete profiling workflow for one mode."""
    profile_dir = os.path.join(output_dir, f"vllm_profile_{mode}")
    os.makedirs(profile_dir, exist_ok=True)
    
    prompt_a, prompt_b = load_prompts()
    
    print(f"\n{'='*70}")
    print(f"  MODE: {mode}")
    print(f"  Profile dir: {profile_dir}")
    print(f"  Warmup requests: {warmup_reqs}")
    print(f"  Profile rounds (R1+R2 each): {profile_rounds}")
    print(f"  max_tokens: {max_tokens}")
    print(f"  GDN_ALIGN_TRITON_CONV1D="
          f"{os.environ.get('GDN_ALIGN_TRITON_CONV1D', '(not set)')}")
    print(f"{'='*70}")
    
    # Start server
    proc = start_server(mode, profile_dir)
    
    try:
        # Wait for server ready
        print(f"\n[{mode}] Waiting for server to be ready...")
        if not wait_for_server(timeout=300):
            print(f"[ERROR] Server failed to start within 300s")
            # Print server output for debugging
            proc.kill()
            stdout, _ = proc.communicate(timeout=10)
            print(f"Server output (last 2000 chars):\n{stdout[-2000:]}")
            return None
        print(f"[{mode}] Server ready!")
        
        # Warmup (without profiling)
        print(f"\n[{mode}] Sending {warmup_reqs} warmup requests...")
        for i in range(warmup_reqs):
            prompt = prompt_a if i % 2 == 0 else prompt_b
            t0 = time.perf_counter()
            resp = send_chat_request(prompt, max_tokens)
            elapsed = (time.perf_counter() - t0) * 1000
            content = resp["choices"][0]["message"]["content"][:60]
            print(f"  warmup {i+1}: {elapsed:.0f}ms | {content!r}")
        
        # Start profiling
        print(f"\n[{mode}] Starting profiler...")
        try:
            http_post(f"{BASE_URL}/start_profile", timeout=30)
            print(f"[{mode}] Profiler started!")
        except Exception as e:
            print(f"[ERROR] Failed to start profiler: {e}")
            return None
        
        # Profiled requests: R1 (fill cache) → R2 (cache hit)
        results = {"r1_wall": [], "r2_wall": [], "rounds": []}
        print(f"\n[{mode}] Profiling {profile_rounds} rounds...")
        for i in range(profile_rounds):
            # R1: new prompt fills cache
            t0 = time.perf_counter()
            r1_resp = send_chat_request(prompt_a, max_tokens)
            r1_wall = (time.perf_counter() - t0) * 1000
            
            # R2: shared prefix → cache hit
            t0 = time.perf_counter()
            r2_resp = send_chat_request(prompt_b, max_tokens)
            r2_wall = (time.perf_counter() - t0) * 1000
            
            results["r1_wall"].append(r1_wall)
            results["r2_wall"].append(r2_wall)
            
            r1_text = r1_resp["choices"][0]["message"]["content"][:50]
            r2_text = r2_resp["choices"][0]["message"]["content"][:50]
            results["rounds"].append({
                "r1_wall_ms": r1_wall, "r2_wall_ms": r2_wall,
                "r1_text": r1_text, "r2_text": r2_text,
            })
            
            print(f"  round {i+1}/{profile_rounds}: "
                  f"R1={r1_wall:.0f}ms R2={r2_wall:.0f}ms | "
                  f"R2: {r2_text!r}")
        
        # Stop profiling
        print(f"\n[{mode}] Stopping profiler...")
        try:
            http_post(f"{BASE_URL}/stop_profile", timeout=60)
            print(f"[{mode}] Profiler stopped. Traces flushing to disk...")
            time.sleep(5)  # Give profiler time to flush
        except Exception as e:
            print(f"[WARNING] stop_profile failed: {e}")
        
        # Summary
        import statistics
        r1_mean = statistics.mean(results["r1_wall"])
        r2_mean = statistics.mean(results["r2_wall"])
        print(f"\n[{mode}] Summary:")
        print(f"  R1 (cache fill) mean: {r1_mean:.0f}ms")
        print(f"  R2 (cache hit)  mean: {r2_mean:.0f}ms")
        if len(results["r2_wall"]) > 1:
            print(f"  R2 stdev: {statistics.stdev(results['r2_wall']):.0f}ms")
        
        # Save results
        summary_path = os.path.join(output_dir, f"{mode}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
        
    finally:
        print(f"\n[{mode}] Shutting down server...")
        stop_server(proc)
        # Wait a moment for port to be released
        time.sleep(3)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Online profiling: all-mode vs align-mode via vLLM server")
    parser.add_argument("--mode", choices=["all", "align", "both"],
                        default="both",
                        help="Mode(s) to profile (default: both)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Base output directory (default: current dir)")
    parser.add_argument("--max-tokens", type=int, default=10,
                        help="Max tokens per request (default: 10)")
    parser.add_argument("--warmup-reqs", type=int, default=4,
                        help="Warmup requests before profiling (default: 4)")
    parser.add_argument("--profile-rounds", type=int, default=3,
                        help="R1+R2 rounds to profile (default: 3)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path (default: use script constant)")
    parser.add_argument("--port", type=int, default=8100,
                        help="Server port (default: 8100)")
    args = parser.parse_args()
    
    global PORT, BASE_URL, MODEL
    PORT = args.port
    BASE_URL = f"http://{HOST}:{PORT}"
    if args.model:
        MODEL = args.model

    print(f"Model: {MODEL}")
    print(f"Server: {BASE_URL}")
    print(f"Output dir: {args.output_dir}")
    print(f"Prefix: ~2500 tokens (~3 blocks @ block_size=1024)")

    results = {}

    if args.mode in ("all", "both"):
        results["all"] = run_profile_session(
            "all", args.output_dir, args.max_tokens,
            args.warmup_reqs, args.profile_rounds)

    if args.mode in ("align", "both"):
        results["align"] = run_profile_session(
            "align", args.output_dir, args.max_tokens,
            args.warmup_reqs, args.profile_rounds)

    # ── Comparison ──
    if results.get("all") and results.get("align"):
        import statistics

        all_r2 = statistics.mean(results["all"]["r2_wall"])
        align_r2 = statistics.mean(results["align"]["r2_wall"])
        speedup = align_r2 / all_r2 if all_r2 > 0 else float("inf")

        print(f"\n{'='*70}")
        print(f"  COMPARISON (R2 = cache hit, lower is better)")
        print(f"{'='*70}")
        print(f"  ALL   R2 mean: {all_r2:.0f}ms")
        print(f"  ALIGN R2 mean: {align_r2:.0f}ms")
        print(f"  Speedup (align/all): {speedup:.2f}x")
        print(f"{'='*70}")

    # ── Next Steps ──
    print(f"\n{'='*70}")
    print("  NEXT STEPS")
    print(f"{'='*70}")
    print("  1. Find trace files:")
    print(f"     ls {args.output_dir}/vllm_profile_*/*_ascend_pt/ASCEND_PROFILER_OUTPUT/")
    print("  2. Download trace_view.json to local machine")
    print("  3. Open in MindStudio Insight (recommended for Ascend)")
    print("     Or: chrome://tracing / https://ui.perfetto.dev/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
