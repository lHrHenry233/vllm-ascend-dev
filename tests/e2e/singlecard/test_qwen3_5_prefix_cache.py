# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test: GDN (Qwen3.5) all-mode prefix caching on NPU.

Compare outputs with ``mamba_cache_mode="all"`` + ``enable_prefix_caching=True``
against a baseline without prefix caching.  Uses shared-prefix prompts to
exercise cache hits.

Requires:
- 1 NPU card (Qwen3.5-9B with TP=1)
- Model weights accessible to the runner

Run: pytest tests/e2e/singlecard/test_qwen3_5_prefix_cache.py -v
"""

import pytest

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODEL = "Qwen/Qwen3.5-9B"

# Shared long prefix → different suffixes to trigger prefix cache hits
SHARED_PREFIX = (
    "You are a helpful assistant. Below is a detailed specification document.\n"
    "# System Architecture Overview\n"
    "The system consists of multiple interconnected modules responsible for "
    "data ingestion, processing, transformation, and serving. Each module "
    "communicates via a well-defined API interface using gRPC for internal "
    "services and REST for external-facing endpoints.\n"
    "## Module 1: Data Ingestion\n"
    "This module handles data collection from various sources including "
    "databases, message queues, file systems, and real-time streams. "
    "It supports multiple protocols: JDBC, AMQP, S3, and Kafka. "
    "The ingestion pipeline is designed to handle up to 100K events/second "
    "with configurable batching and backpressure mechanisms.\n"
    "## Module 2: Processing Engine\n"
    "The processing engine applies transformation rules, validation checks, "
    "and enrichment logic to incoming data. It uses a directed acyclic graph "
    "(DAG) execution model where each node represents a processing step. "
    "The engine supports both batch and streaming modes with exactly-once "
    "semantics guaranteed through checkpoint-based recovery.\n"
    "## Module 3: Storage Layer\n"
    "Processed data is stored in a distributed storage system that supports "
    "both row-oriented and column-oriented access patterns. The storage layer "
    "implements automatic data partitioning, replication, and compaction. "
    "It provides ACID transactions within a partition and eventual consistency "
    "across partitions with configurable consistency levels.\n"
    "## Module 4: Serving Interface\n"
    "The serving layer exposes processed data through multiple interfaces: "
    "a query engine supporting SQL-like syntax, a key-value lookup API for "
    "point queries, and a streaming API for real-time data subscriptions. "
    "All interfaces support authentication, rate limiting, and request routing.\n"
)

INPUT_PROMPTS = [
    SHARED_PREFIX + "Question: How many events per second can the ingestion module handle? Answer:",
    SHARED_PREFIX + "Question: What execution model does the processing engine use? Answer:",
    SHARED_PREFIX + "Question: What type of transactions does the storage layer support? Answer:",
]


@pytest.mark.parametrize("max_tokens", [30])
def test_qwen3_5_all_mode_prefix_cache(max_tokens: int) -> None:
    """All-mode prefix caching should produce identical outputs to no-cache."""
    with VllmRunner(
        MODEL,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enforce_eager=True,
        mamba_cache_mode="all",
    ) as vllm_model:
        all_mode_output = vllm_model.generate_greedy(INPUT_PROMPTS, max_tokens)

    with VllmRunner(
        MODEL,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        enforce_eager=True,
    ) as vllm_model:
        baseline_output = vllm_model.generate_greedy(INPUT_PROMPTS, max_tokens)

    check_outputs_equal(
        outputs_0_lst=baseline_output,
        outputs_1_lst=all_mode_output,
        name_0="baseline (no cache)",
        name_1="all-mode prefix cache",
    )


@pytest.mark.parametrize("max_tokens", [20])
def test_qwen3_5_all_mode_cache_hit_second_run(max_tokens: int) -> None:
    """Run the same prompts twice in a single engine → second run hits cache."""
    with VllmRunner(
        MODEL,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enforce_eager=True,
        mamba_cache_mode="all",
    ) as vllm_model:
        first_output = vllm_model.generate_greedy(INPUT_PROMPTS, max_tokens)
        # Second run: shared prefix blocks should be cache hits
        second_output = vllm_model.generate_greedy(INPUT_PROMPTS, max_tokens)

    check_outputs_equal(
        outputs_0_lst=first_output,
        outputs_1_lst=second_output,
        name_0="first run",
        name_1="second run (cache hit)",
    )
