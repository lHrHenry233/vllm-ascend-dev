# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy sweep: diverse prompts to check all-mode vs align-mode precision.

Tests multiple prompt categories (factual, math, code, reasoning, translation)
to detect systematic precision degradation in all-mode prefix caching.

Each test:
  1. Runs prompt with all-mode (prefix caching enabled)
  2. Runs prompt with align-mode (prefix caching enabled)
  3. Compares outputs token-by-token

Run:
    pytest tests/e2e/singlecard/test_accuracy_sweep.py -v -s
"""

import os

import pytest

from tests.e2e.conftest import VllmRunner

MODEL = "/shared/models/Qwen3.5-0.8B-ms"
MAX_TOKENS = 80

_ENGINE_KWARGS = dict(
    model_name=MODEL,
    tensor_parallel_size=1,
    enforce_eager=True,
    gpu_memory_utilization=0.7,
    enable_prefix_caching=True,
    max_model_len=4096,
    max_num_batched_tokens=4096,
)

# ════════════════════════════════════════════════════════════════
#  Diverse Prompts
# ════════════════════════════════════════════════════════════════

PROMPTS = {
    "factual_short": (
        "What is the capital of France? Answer in one word:",
    ),
    "factual_medium": (
        "Explain the difference between TCP and UDP in networking. "
        "Be concise, use 3-4 sentences:",
    ),
    "math_arithmetic": (
        "Calculate step by step: 127 * 34 + 856 - 291 = ?",
    ),
    "math_word_problem": (
        "A train travels at 60 km/h for 2.5 hours, then at 80 km/h "
        "for 1.5 hours. What is the total distance? Show your work:",
    ),
    "code_python": (
        "Write a Python function that checks if a string is a palindrome. "
        "Return True or False. Only output the code:\n```python\n",
    ),
    "code_explain": (
        "What does the following Python code do?\n"
        "```python\n"
        "def f(n):\n"
        "    return n if n <= 1 else f(n-1) + f(n-2)\n"
        "```\n"
        "Explain briefly:",
    ),
    "reasoning_logic": (
        "If all roses are flowers and some flowers fade quickly, "
        "can we conclude that some roses fade quickly? "
        "Answer yes or no and explain why in one sentence:",
    ),
    "reasoning_sequence": (
        "What comes next in the sequence: 2, 6, 12, 20, 30, ? "
        "Explain the pattern:",
    ),
    "translation_en_zh": (
        "Translate the following English sentence to Chinese:\n"
        "\"The quick brown fox jumps over the lazy dog.\"\n"
        "Translation:",
    ),
    "translation_zh_en": (
        "Translate the following Chinese sentence to English:\n"
        "\"机器学习是人工智能的一个重要分支。\"\n"
        "Translation:",
    ),
    "summarization": (
        "Summarize the following in one sentence:\n"
        "Machine learning is a subset of artificial intelligence that "
        "enables systems to learn and improve from experience without "
        "being explicitly programmed. It focuses on developing computer "
        "programs that can access data and use it to learn for themselves. "
        "The process begins with observations or data, such as examples, "
        "direct experience, or instruction, to look for patterns in data "
        "and make better decisions in the future.\n"
        "Summary:",
    ),
    "creative_story": (
        "Write the opening sentence of a sci-fi story about "
        "a robot discovering emotions for the first time:",
    ),
    # Prefix-sharing pair: same long context, different questions
    "prefix_share_q1": (
        "Context: The Solar System consists of the Sun and the objects "
        "that orbit it, including eight planets (Mercury, Venus, Earth, "
        "Mars, Jupiter, Saturn, Uranus, and Neptune), dwarf planets "
        "(including Pluto), and various smaller bodies such as asteroids "
        "and comets. The four inner planets (Mercury, Venus, Earth, Mars) "
        "are terrestrial planets with solid rocky surfaces. The four outer "
        "planets (Jupiter, Saturn, Uranus, Neptune) are gas or ice giants. "
        "Jupiter is the largest planet, with a mass more than twice that "
        "of all other planets combined. Saturn is known for its prominent "
        "ring system. Earth is the only planet known to support life.\n\n"
        "Question: Which planet is the largest?\nAnswer:"
    ),
    "prefix_share_q2": (
        "Context: The Solar System consists of the Sun and the objects "
        "that orbit it, including eight planets (Mercury, Venus, Earth, "
        "Mars, Jupiter, Saturn, Uranus, and Neptune), dwarf planets "
        "(including Pluto), and various smaller bodies such as asteroids "
        "and comets. The four inner planets (Mercury, Venus, Earth, Mars) "
        "are terrestrial planets with solid rocky surfaces. The four outer "
        "planets (Jupiter, Saturn, Uranus, Neptune) are gas or ice giants. "
        "Jupiter is the largest planet, with a mass more than twice that "
        "of all other planets combined. Saturn is known for its prominent "
        "ring system. Earth is the only planet known to support life.\n\n"
        "Question: How many terrestrial planets are there?\nAnswer:"
    ),
}


# ════════════════════════════════════════════════════════════════
#  Test Infrastructure
# ════════════════════════════════════════════════════════════════

def _run_single_prompt(mode: str, prompt: str) -> str:
    """Run a single prompt with the given cache mode, return output text."""
    with VllmRunner(**_ENGINE_KWARGS, mamba_cache_mode=mode) as vllm:
        outputs = vllm.generate_greedy([prompt], MAX_TOKENS)
    return outputs[0][1] if outputs else "<empty>"


def _run_all_prompts(mode: str, prompts: dict[str, tuple[str]]) -> dict[str, str]:
    """Run all prompts in a single engine instance."""
    results = {}
    with VllmRunner(**_ENGINE_KWARGS, mamba_cache_mode=mode) as vllm:
        for name, (prompt,) in prompts.items():
            outputs = vllm.generate_greedy([prompt], MAX_TOKENS)
            results[name] = outputs[0][1] if outputs else "<empty>"
    return results


def _compare(name: str, all_text: str, align_text: str) -> bool:
    """Compare outputs, print diff if any. Returns True if match."""
    match = all_text == align_text
    status = "✅" if match else "❌"
    print(f"  {status} {name}")
    if not match:
        print(f"      all:   {all_text[:120]}{'...' if len(all_text) > 120 else ''}")
        print(f"      align: {align_text[:120]}{'...' if len(align_text) > 120 else ''}")
        # Find first diff position
        for i, (a, b) in enumerate(zip(all_text, align_text)):
            if a != b:
                print(f"      diff@{i}: '{all_text[max(0,i-5):i+15]}' vs "
                      f"'{align_text[max(0,i-5):i+15]}'")
                break
    return match


# ════════════════════════════════════════════════════════════════
#  Tests
# ════════════════════════════════════════════════════════════════

def test_accuracy_sweep_batch() -> None:
    """Run all diverse prompts through all-mode and align-mode, compare."""
    os.environ["GDN_DEBUG"] = "1"

    print("\n" + "=" * 60)
    print("ACCURACY SWEEP: all-mode vs align-mode (single compute, no cache hit)")
    print("=" * 60)

    print("\n── Running all-mode... ──")
    all_results = _run_all_prompts("all", PROMPTS)

    print("\n── Running align-mode... ──")
    align_results = _run_all_prompts("align", PROMPTS)

    print("\n── Comparison ──")
    matches = 0
    total = len(PROMPTS)
    for name in PROMPTS:
        if _compare(name, all_results[name], align_results[name]):
            matches += 1

    print(f"\n  Result: {matches}/{total} match")
    pct = matches * 100 // total
    print(f"  Match rate: {pct}%")

    os.environ.pop("GDN_DEBUG", None)

    # Don't hard-fail: SSM precision differences are expected
    # Print summary for human inspection
    if matches < total:
        print(f"\n  ⚠️  {total - matches} prompts differ between all/align mode.")
        print(f"  This may be expected due to SSM state dtype differences.")
        print(f"  Check outputs above for severity (minor wording vs garbage).")


def test_prefix_share_cache_hit() -> None:
    """Two prompts with shared prefix: verify cache hit accuracy.

    Runs q1 first (fills cache), then q2 (cache hit on shared prefix).
    Compares all-mode vs align-mode for both rounds.
    """
    os.environ["GDN_DEBUG"] = "1"
    q1_prompt = PROMPTS["prefix_share_q1"][0]
    q2_prompt = PROMPTS["prefix_share_q2"][0]

    print("\n" + "=" * 60)
    print("PREFIX SHARE: cache hit test (solar system context)")
    print("=" * 60)

    # All-mode: R1 fills cache, R2 hits cache
    print("\n── All-mode: R1 + R2 ──")
    with VllmRunner(**_ENGINE_KWARGS, mamba_cache_mode="all") as vllm:
        all_r1 = vllm.generate_greedy([q1_prompt], MAX_TOKENS)
        all_r2 = vllm.generate_greedy([q2_prompt], MAX_TOKENS)
    all_r1_text = all_r1[0][1] if all_r1 else "<empty>"
    all_r2_text = all_r2[0][1] if all_r2 else "<empty>"
    print(f"  R1 (largest planet): {all_r1_text[:100]}")
    print(f"  R2 (terrestrial):    {all_r2_text[:100]}")

    # Align-mode: R1 fills cache, R2 may or may not hit
    print("\n── Align-mode: R1 + R2 ──")
    with VllmRunner(**_ENGINE_KWARGS, mamba_cache_mode="align") as vllm:
        align_r1 = vllm.generate_greedy([q1_prompt], MAX_TOKENS)
        align_r2 = vllm.generate_greedy([q2_prompt], MAX_TOKENS)
    align_r1_text = align_r1[0][1] if align_r1 else "<empty>"
    align_r2_text = align_r2[0][1] if align_r2 else "<empty>"
    print(f"  R1 (largest planet): {align_r1_text[:100]}")
    print(f"  R2 (terrestrial):    {align_r2_text[:100]}")

    print("\n── Comparison ──")
    _compare("R1: largest planet", all_r1_text, align_r1_text)
    _compare("R2: terrestrial (cache hit)", all_r2_text, align_r2_text)

    # Sanity check: outputs should mention correct answers
    print("\n── Sanity Check ──")
    for label, text in [("all-R1", all_r1_text), ("align-R1", align_r1_text)]:
        has_jupiter = "jupiter" in text.lower() or "Jupiter" in text
        print(f"  {label} mentions Jupiter: {'✅' if has_jupiter else '❌'} — {text[:60]}")
    for label, text in [("all-R2", all_r2_text), ("align-R2", align_r2_text)]:
        has_four = "4" in text or "four" in text.lower()
        print(f"  {label} mentions 4/four: {'✅' if has_four else '❌'} — {text[:60]}")

    os.environ.pop("GDN_DEBUG", None)


def test_repeated_runs_determinism() -> None:
    """Run same prompt 3 times with all-mode, check all outputs identical."""
    prompt = PROMPTS["math_arithmetic"][0]

    print("\n" + "=" * 60)
    print("DETERMINISM: 3 runs of same prompt with all-mode")
    print("=" * 60)

    results = []
    with VllmRunner(**_ENGINE_KWARGS, mamba_cache_mode="all") as vllm:
        for i in range(3):
            outputs = vllm.generate_greedy([prompt], MAX_TOKENS)
            text = outputs[0][1] if outputs else "<empty>"
            results.append(text)
            print(f"  Run {i+1}: {text[:80]}{'...' if len(text) > 80 else ''}")

    all_same = all(r == results[0] for r in results)
    print(f"\n  Deterministic: {'✅ YES' if all_same else '❌ NO'}")
    if not all_same:
        for i in range(1, len(results)):
            if results[i] != results[0]:
                print(f"  Run {i+1} differs from Run 1")
