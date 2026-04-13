# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for GDN mamba_cache_mode="all" prefix caching on Qwen3.5.

Tests use a two-round pattern to verify cross-batch cache hits:
  R1: prompt_A fills cache (shared prefix computed from scratch)
  R2: prompt_B with same prefix triggers cache hit (skips prefix computation)

Three max_model_len scenarios cover different block layout behaviors:
  S1 (4096): prefix spans 3+ blocks → multi-block scatter + read
  S2 (2048): prefix spans 1-2 blocks → single boundary
  S3 (1024): prefix fits in 1 block → degenerate case (no cache hit)

Run:
    pytest tests/e2e/singlecard/test_qwen3_5_prefix_cache.py -v -s
"""

import pytest

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODEL = "/shared/models/Qwen3.5-0.8B-ms"
MAX_TOKENS = 50

# ─── Shared table header ──────────────────────────────────────────────
# ruff: noqa: E501
_TABLE_HEADER = (
    "You are a helpful assistant in recognizes the content of tables in "
    "markdown format. Here is a table as follows.\n# Table\n\n"
    "| ID  | Name          | Age | Occupation    | Country       | Email                  | Phone Number   | Address                       |\n"
    "|-----|---------------|-----|---------------|---------------|------------------------|----------------|-------------------------------|\n"
)

# ─── Table rows (60 total) ────────────────────────────────────────────
_TABLE_ROWS = """\
| 1   | John Doe      | 29  | Engineer      | USA           | john.doe@example.com   | 555-1234       | 123 Elm St, Springfield, IL   |
| 2   | Jane Smith    | 34  | Doctor        | Canada        | jane.smith@example.com | 555-5678       | 456 Oak St, Toronto, ON       |
| 3   | Alice Johnson | 27  | Teacher       | UK            | alice.j@example.com    | 555-8765       | 789 Pine St, London, UK       |
| 4   | Bob Brown     | 45  | Artist        | Australia     | bob.b@example.com      | 555-4321       | 321 Maple St, Sydney, NSW     |
| 5   | Carol White   | 31  | Scientist     | New Zealand   | carol.w@example.com    | 555-6789       | 654 Birch St, Wellington, NZ  |
| 6   | Dave Green    | 28  | Lawyer        | Ireland       | dave.g@example.com     | 555-3456       | 987 Cedar St, Dublin, IE      |
| 7   | Emma Black    | 40  | Musician      | USA           | emma.b@example.com     | 555-1111       | 246 Ash St, New York, NY      |
| 8   | Frank Blue    | 37  | Chef          | Canada        | frank.b@example.com    | 555-2222       | 135 Spruce St, Vancouver, BC  |
| 9   | Grace Yellow  | 50  | Engineer      | UK            | grace.y@example.com    | 555-3333       | 864 Fir St, Manchester, UK    |
| 10  | Henry Violet  | 32  | Artist        | Australia     | henry.v@example.com    | 555-4444       | 753 Willow St, Melbourne, VIC |
| 11  | Irene Orange  | 26  | Scientist     | New Zealand   | irene.o@example.com    | 555-5555       | 912 Poplar St, Auckland, NZ   |
| 12  | Jack Indigo   | 38  | Teacher       | Ireland       | jack.i@example.com     | 555-6666       | 159 Elm St, Cork, IE          |
| 13  | Karen Red     | 41  | Lawyer        | USA           | karen.r@example.com    | 555-7777       | 357 Cedar St, Boston, MA      |
| 14  | Leo Brown     | 30  | Chef          | Canada        | leo.b@example.com      | 555-8888       | 246 Oak St, Calgary, AB       |
| 15  | Mia Green     | 33  | Musician      | UK            | mia.g@example.com      | 555-9999       | 975 Pine St, Edinburgh, UK    |
| 16  | Noah Yellow   | 29  | Doctor        | Australia     | noah.y@example.com     | 555-0000       | 864 Birch St, Brisbane, QLD   |
| 17  | Olivia Blue   | 35  | Engineer      | New Zealand   | olivia.b@example.com   | 555-1212       | 753 Maple St, Hamilton, NZ    |
| 18  | Peter Black   | 42  | Artist        | Ireland       | peter.b@example.com    | 555-3434       | 912 Fir St, Limerick, IE      |
| 19  | Quinn White   | 28  | Scientist     | USA           | quinn.w@example.com    | 555-5656       | 159 Willow St, Seattle, WA    |
| 20  | Rachel Red    | 31  | Teacher       | Canada        | rachel.r@example.com   | 555-7878       | 357 Poplar St, Ottawa, ON     |
| 21  | Steve Green   | 44  | Lawyer        | UK            | steve.g@example.com    | 555-9090       | 753 Elm St, Birmingham, UK    |
| 22  | Tina Blue     | 36  | Musician      | Australia     | tina.b@example.com     | 555-1213       | 864 Cedar St, Perth, WA       |
| 23  | Umar Black    | 39  | Chef          | New Zealand   | umar.b@example.com     | 555-3435       | 975 Spruce St, Christchurch, NZ|
| 24  | Victor Yellow | 43  | Engineer      | Ireland       | victor.y@example.com   | 555-5657       | 246 Willow St, Galway, IE     |
| 25  | Wendy Orange  | 27  | Artist        | USA           | wendy.o@example.com    | 555-7879       | 135 Elm St, Denver, CO        |
| 26  | Xavier Green  | 34  | Scientist     | Canada        | xavier.g@example.com   | 555-9091       | 357 Oak St, Montreal, QC      |
| 27  | Yara Red      | 41  | Teacher       | UK            | yara.r@example.com     | 555-1214       | 975 Pine St, Leeds, UK        |
| 28  | Zack Blue     | 30  | Lawyer        | Australia     | zack.b@example.com     | 555-3436       | 135 Birch St, Adelaide, SA    |
| 29  | Amy White     | 33  | Musician      | New Zealand   | amy.w@example.com      | 555-5658       | 159 Maple St, Wellington, NZ  |
| 30  | Ben Black     | 38  | Chef          | Ireland       | ben.b@example.com      | 555-7870       | 246 Fir St, Waterford, IE     |
| 31  | Adam Silver   | 26  | Engineer      | USA           | adam.s@example.com     | 555-1031       | 123 Oak Ave, Austin, TX       |
| 32  | Beth Gold     | 33  | Doctor        | Canada        | beth.g@example.com     | 555-1032       | 456 Elm Rd, Edmonton, AB      |
| 33  | Carl Copper   | 44  | Teacher       | UK            | carl.c@example.com     | 555-1033       | 789 Ash Ln, Bristol, UK       |
| 34  | Dana Iron     | 29  | Artist        | Australia     | dana.i@example.com     | 555-1034       | 321 Pine Dr, Darwin, NT       |
| 35  | Evan Steel    | 37  | Scientist     | New Zealand   | evan.s@example.com     | 555-1035       | 654 Fir Way, Dunedin, NZ      |
| 36  | Fiona Frost   | 31  | Lawyer        | Ireland       | fiona.f@example.com    | 555-1036       | 987 Yew St, Belfast, NI       |
| 37  | Greg Storm    | 48  | Musician      | Germany       | greg.s@example.com     | 555-1037       | 246 Bay Ct, Munich, DE        |
| 38  | Helen Blaze   | 27  | Chef          | France        | helen.b@example.com    | 555-1038       | 135 Rue St, Paris, FR         |
| 39  | Ivan Stone    | 35  | Writer        | Japan         | ivan.s@example.com     | 555-1039       | 864 Yen St, Tokyo, JP         |
| 40  | Julia River   | 40  | Pilot         | Brazil        | julia.r@example.com    | 555-1040       | 753 Sol Av, Sao Paulo, BR     |
| 41  | Kurt Woods    | 30  | Engineer      | USA           | kurt.w@example.com     | 555-1041       | 912 Via St, Portland, OR      |
| 42  | Laura Cloud   | 42  | Doctor        | Canada        | laura.c@example.com    | 555-1042       | 159 Lk Rd, Winnipeg, MB       |
| 43  | Mark Marsh    | 38  | Teacher       | UK            | mark.m@example.com     | 555-1043       | 357 Hill St, Glasgow, UK      |
| 44  | Nina Peak     | 25  | Artist        | Australia     | nina.p@example.com     | 555-1044       | 246 Bay Rd, Hobart, TAS       |
| 45  | Oscar Vale    | 46  | Scientist     | New Zealand   | oscar.v@example.com    | 555-1045       | 135 Mt St, Nelson, NZ         |
| 46  | Paula Rose    | 34  | Lawyer        | Ireland       | paula.r@example.com    | 555-1046       | 975 Dam St, Shannon, IE       |
| 47  | Rick Cruz     | 39  | Musician      | Germany       | rick.c@example.com     | 555-1047       | 864 Berg St, Berlin, DE       |
| 48  | Sara Nash     | 28  | Chef          | France        | sara.n@example.com     | 555-1048       | 753 Lac St, Lyon, FR          |
| 49  | Tim Hart      | 43  | Writer        | Japan         | tim.h@example.com      | 555-1049       | 912 Sakura, Osaka, JP         |
| 50  | Uma Cole      | 36  | Pilot         | Brazil        | uma.c@example.com      | 555-1050       | 159 Rio St, Rio de Janeiro, BR|
| 51  | Vince Reid    | 32  | Engineer      | USA           | vince.r@example.com    | 555-1051       | 357 Sun St, Miami, FL         |
| 52  | Wanda Moss    | 41  | Doctor        | Canada        | wanda.m@example.com    | 555-1052       | 246 Ice Rd, Quebec, QC        |
| 53  | Xander Sage   | 29  | Teacher       | UK            | xander.s@example.com   | 555-1053       | 135 Fog Ln, Cardiff, UK       |
| 54  | Yuki Flynn    | 47  | Artist        | Australia     | yuki.f@example.com     | 555-1054       | 975 Reef Dr, Cairns, QLD      |
| 55  | Zara Brook    | 33  | Scientist     | New Zealand   | zara.b@example.com     | 555-1055       | 864 Bay St, Napier, NZ        |
| 56  | Aaron Crane   | 38  | Lawyer        | Ireland       | aaron.c@example.com    | 555-1056       | 753 Bog Rd, Kilkenny, IE      |
| 57  | Bianca Wolfe  | 30  | Musician      | Germany       | bianca.w@example.com   | 555-1057       | 912 Wald St, Hamburg, DE      |
| 58  | Caleb Price   | 45  | Chef          | France        | caleb.p@example.com    | 555-1058       | 159 Mer Av, Marseille, FR     |
| 59  | Diana Grant   | 27  | Writer        | Japan         | diana.g@example.com    | 555-1059       | 357 Fuji Ln, Kyoto, JP        |
| 60  | Ethan Lane    | 35  | Pilot         | Brazil        | ethan.l@example.com    | 555-1060       | 246 Palm Av, Salvador, BR     |
"""

# Split into individual row lines for building scenario prefixes
_ROW_LINES = [ln for ln in _TABLE_ROWS.strip().split("\n") if ln.startswith("|")]


def _build_prefix(num_rows: int) -> str:
    """Build a table prefix with the given number of rows."""
    return _TABLE_HEADER + "\n".join(_ROW_LINES[:num_rows]) + "\n"


# ─── Scenario prefixes ────────────────────────────────────────────────
# S1: 60 rows → ~2500 tokens → 3+ blocks (block_size=1024)
S1_PREFIX = _build_prefix(60)
# S2: 30 rows → ~1645 tokens → 1-2 blocks
S2_PREFIX = _build_prefix(30)
# S3: 10 rows → ~500 tokens → fits in 1 block (degenerate: no cache hit)
S3_PREFIX = _build_prefix(10)

# Question suffixes — different per prompt to trigger prefix cache hit
_Q_JOHN = "Question: what is the age of John Doe? Your answer: The age of John Doe is "
_Q_ZACK = "Question: what is the age of Zack Blue? Your answer: The age of Zack Blue is "
_Q_BOB = "Question: what is the age of Bob Brown? Your answer: The age of Bob Brown is "

# ─── Scenario prompt pairs [prompt_A, prompt_B] ───────────────────────
S1_PROMPTS = [S1_PREFIX + _Q_JOHN, S1_PREFIX + _Q_ZACK]
S2_PROMPTS = [S2_PREFIX + _Q_JOHN, S2_PREFIX + _Q_ZACK]
S3_PROMPTS = [S3_PREFIX + _Q_JOHN, S3_PREFIX + _Q_BOB]

# Common VllmRunner kwargs
_COMMON_KWARGS = dict(
    model=MODEL,
    tensor_parallel_size=1,
    enforce_eager=True,
    gpu_memory_utilization=0.7,
    enable_prefix_caching=True,
)


def _print_scenario_info(name: str, prefix: str, prompts: list[str]) -> None:
    """Print scenario configuration at test start."""
    block_size = 1024
    # Rough token estimate: ~3.5 chars per token for English + markdown table
    prefix_tokens_est = len(prefix) // 4
    n_blocks = (prefix_tokens_est + block_size - 1) // block_size
    n_boundaries = max(0, n_blocks - 1)
    print(f"\n{'='*60}")
    print(f"[SCENARIO {name}]")
    print(f"  prefix chars    = {len(prefix)}")
    print(f"  prefix tokens   ≈ {prefix_tokens_est} (estimated)")
    print(f"  prefix blocks   ≈ {n_blocks} (block_size={block_size})")
    print(f"  scatter boundaries ≈ {n_boundaries}")
    print(f"  prompt_A chars  = {len(prompts[0])}")
    print(f"  prompt_B chars  = {len(prompts[1])}")
    print(f"  R1: prompt_A → compute all, fill cache")
    print(f"  R2: prompt_B → cache hit (shared prefix)")
    print(f"{'='*60}")


def _run_two_round_test(
    prompts: list[str],
    max_model_len: int,
    scenario_name: str,
) -> None:
    """Two-round cache hit test: R1 fills cache, R2 reads cached state.

    Compares all-mode vs align-mode outputs for both rounds.
    """
    prompt_a, prompt_b = prompts

    _print_scenario_info(scenario_name, prompts[0].rsplit("Question:", 1)[0],
                         prompts)

    # === ALL-MODE ===
    print(f"\n--- {scenario_name}: ALL-MODE ---")
    with VllmRunner(**_COMMON_KWARGS, mamba_cache_mode="all",
                    max_model_len=max_model_len) as vllm:
        all_r1 = vllm.generate_greedy([prompt_a], MAX_TOKENS)
        print(f"  R1 output: {all_r1[0][1][:80]}...")
        all_r2 = vllm.generate_greedy([prompt_b], MAX_TOKENS)
        print(f"  R2 output: {all_r2[0][1][:80]}...")

    # === ALIGN-MODE ===
    print(f"\n--- {scenario_name}: ALIGN-MODE ---")
    with VllmRunner(**_COMMON_KWARGS, mamba_cache_mode="align",
                    max_model_len=max_model_len) as vllm:
        align_r1 = vllm.generate_greedy([prompt_a], MAX_TOKENS)
        print(f"  R1 output: {align_r1[0][1][:80]}...")
        align_r2 = vllm.generate_greedy([prompt_b], MAX_TOKENS)
        print(f"  R2 output: {align_r2[0][1][:80]}...")

    # === ASSERTIONS ===
    print(f"\n--- {scenario_name}: ASSERTIONS ---")
    # R1: first computation — both modes compute from scratch
    check_outputs_equal(
        outputs_0_lst=align_r1,
        outputs_1_lst=all_r1,
        name_0=f"{scenario_name}-align-R1",
        name_1=f"{scenario_name}-all-R1",
    )
    print(f"  ✅ R1 match (first computation)")

    # R2: cache hit — core validation
    check_outputs_equal(
        outputs_0_lst=align_r2,
        outputs_1_lst=all_r2,
        name_0=f"{scenario_name}-align-R2",
        name_1=f"{scenario_name}-all-R2",
    )
    print(f"  ✅ R2 match (cache hit)")


# ─── Test functions ───────────────────────────────────────────────────

def test_cache_hit_multi_block() -> None:
    """S1: prefix >2 blocks → multi-block scatter + cross-batch cache read."""
    _run_two_round_test(S1_PROMPTS, max_model_len=4096,
                        scenario_name="S1-multi-block")


def test_cache_hit_two_blocks() -> None:
    """S2: prefix 1-2 blocks → single boundary scatter + cache read."""
    _run_two_round_test(S2_PROMPTS, max_model_len=2048,
                        scenario_name="S2-two-blocks")


def test_cache_hit_single_block() -> None:
    """S3: prefix ≤1 block → degenerate case (partial block, no cache hit).

    vLLM only caches full blocks. A prefix that doesn't fill a complete block
    won't be cached, so R2 computes everything from scratch (same as R1).
    This test verifies all-mode doesn't crash in this degenerate scenario.
    """
    _run_two_round_test(S3_PROMPTS, max_model_len=1024,
                        scenario_name="S3-single-block")


def test_deterministic() -> None:
    """Two runs of all-mode prefix caching should produce identical outputs."""
    with VllmRunner(**_COMMON_KWARGS, mamba_cache_mode="all",
                    max_model_len=2048, seed=42) as vllm:
        run1 = vllm.generate_greedy(S2_PROMPTS, MAX_TOKENS)

    with VllmRunner(**_COMMON_KWARGS, mamba_cache_mode="all",
                    max_model_len=2048, seed=42) as vllm:
        run2 = vllm.generate_greedy(S2_PROMPTS, MAX_TOKENS)

    check_outputs_equal(
        outputs_0_lst=run1,
        outputs_1_lst=run2,
        name_0="run1",
        name_1="run2",
    )
