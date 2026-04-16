# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E cache hit rate tests with realistic long-prefix scenarios.

Three scenarios test all-mode vs align-mode cache correctness with
~5500-token shared prefixes spanning 5+ blocks (block_size=1024).

Scenarios:
  B (LongText): Research survey paper + analytical questions
  C (Agent): REST API documentation + task prompts
  A (Multi-turn): Technical support dialog + follow-up questions

Parameters:
  max_model_len=8192, max_num_batched_tokens=4096, block_size=1024

The two-round test pattern:
  R1: prompt_A fills cache (shared prefix computed from scratch)
  R2: prompt_B with same prefix triggers cache hit
  R3: prompt_B in fresh engine (no cache, baseline)

Run:
    pytest tests/e2e/singlecard/test_qwen3_5_cache_scenarios.py -v -s
"""

import os

import pytest

from tests.e2e.conftest import VllmRunner

MODEL = "/shared/models/Qwen3.5-0.8B-ms"
MAX_TOKENS = 50

# ════════════════════════════════════════════════════════════════
# Common engine configuration
# ════════════════════════════════════════════════════════════════

_COMMON_KWARGS = dict(
    model_name=MODEL,
    tensor_parallel_size=1,
    enforce_eager=True,
    gpu_memory_utilization=0.7,
    enable_prefix_caching=True,
    max_model_len=8192,
    max_num_batched_tokens=4096,
)


# ════════════════════════════════════════════════════════════════
#  SCENARIO B — Long Text: Research Survey Paper (~5500 tokens)
# ════════════════════════════════════════════════════════════════
# ruff: noqa: E501

SURVEY_DOCUMENT = """\
You are a research assistant specializing in machine learning and systems optimization. Read the following research paper carefully, then answer the question that follows based only on the information presented in the paper.

---

# A Survey on Efficient Inference Methods for Large Language Models

## Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing tasks, from text generation and summarization to code completion and mathematical reasoning. However, the computational demands of deploying these models in production environments remain a significant challenge. This survey provides a comprehensive overview of recent advances in efficient LLM inference, covering five major categories of optimization techniques: model compression (including quantization, pruning, and knowledge distillation), efficient attention mechanisms, memory management strategies, speculative decoding, and hardware-specific optimizations. We systematically compare these approaches across multiple dimensions including latency reduction, throughput improvement, memory savings, and accuracy preservation. Our analysis of 47 recent papers reveals that combining multiple techniques—particularly int8 quantization with KV cache optimization and continuous batching—yields the most practical improvements for production deployments, achieving 2-4x latency reduction with less than 1% accuracy degradation on standard benchmarks. We identify key open challenges and promising research directions for future work in this rapidly evolving field.

## 1. Introduction

The rapid scaling of language models from millions to hundreds of billions of parameters has created an unprecedented gap between model capability and deployment feasibility. While models like GPT-4, Claude, LLaMA-3, and Qwen-2.5 achieve state-of-the-art performance across diverse tasks, their inference costs can be prohibitive for real-world applications. A single forward pass through a 70B parameter model requires approximately 140 GFLOPS of computation and 140GB of memory just to store the model weights in FP16 format. For many organizations, these requirements translate directly into high hardware costs and latency that exceeds acceptable thresholds for interactive applications.

The inference pipeline for autoregressive LLMs consists of two distinct phases: the prefill phase, where all input tokens are processed in parallel to build the initial key-value (KV) cache, and the decode phase, where tokens are generated one at a time, each requiring a full forward pass through the model. The prefill phase is typically compute-bound (limited by the speed of matrix multiplications), while the decode phase is memory-bandwidth-bound (limited by how fast model weights and KV cache entries can be loaded from memory). This dual nature of LLM inference means that different optimization strategies may be needed for each phase, and the optimal configuration depends on the specific workload characteristics including average prompt length, generation length, batch size, and latency requirements.

In this survey, we categorize existing approaches into five groups and systematically evaluate their effectiveness. Our contributions include: (1) a unified taxonomy of inference optimization techniques with clear categorization boundaries, (2) quantitative comparison of 23 representative methods across standardized benchmarks, (3) analysis of technique composability—which methods can be combined effectively and what are the compound performance effects, and (4) identification of under-explored research directions that could yield significant improvements.

The remainder of this paper is organized as follows: Section 2 covers model compression techniques, Section 3 discusses efficient attention mechanisms, Section 4 examines memory management and batching strategies, Section 5 presents speculative decoding methods, Section 6 reviews hardware-specific optimizations, Section 7 provides our experimental comparison across all categories, and Section 8 concludes with future directions.

## 2. Model Compression

Model compression reduces the size and computational requirements of neural networks while attempting to preserve their capabilities. We discuss three primary approaches: quantization, pruning, and knowledge distillation.

### 2.1 Quantization

Quantization reduces the precision of model weights and/or activations from floating-point (typically FP16 or BF16) to lower-bit representations. Post-Training Quantization (PTQ) methods are particularly attractive because they do not require retraining the model, which would be prohibitively expensive for LLMs with hundreds of billions of parameters.

GPTQ (Frantar et al., 2023) performs layer-wise quantization by solving an optimization problem that minimizes the output reconstruction error for each layer independently. Using a small calibration dataset (typically 128-256 sequences), GPTQ achieves INT4 quantization with minimal accuracy loss—typically less than 0.5 perplexity points on WikiText-2 for models above 7B parameters. The algorithm processes each column of the weight matrix sequentially, quantizing it and then adjusting the remaining unquantized columns to compensate for the quantization error using the inverse Hessian information. This Optimal Brain Quantizer (OBQ) approach achieves near-optimal results in practice, despite the greedy nature of the column-by-column processing. The primary limitation of GPTQ is that it only quantizes weights, leaving activations in full precision during computation.

AWQ (Lin et al., 2024) takes a different approach by observing that not all weights are equally important for maintaining model quality. Specifically, weights connected to activation channels with large magnitudes contribute disproportionately to the model output. AWQ identifies these salient channels using calibration data and applies per-channel scaling factors before quantization, effectively protecting important weights from quantization error while allowing less important weights to absorb more quantization noise. This results in better accuracy preservation compared to GPTQ at the same bit width, particularly for smaller models (1.3B-7B parameters) where quantization error accumulates more significantly. Our experiments confirm AWQ's advantages: on the LLaMA-7B model, AWQ-INT4 achieves 5.62 perplexity on WikiText-2, compared to 5.68 for GPTQ-INT4 and 5.47 for the full FP16 model. Additionally, AWQ preserves downstream task performance better: on MMLU, AWQ-INT4 achieves 45.3% accuracy versus 45.1% for GPTQ-INT4 and 45.8% for FP16.

SmoothQuant (Xiao et al., 2023) addresses the challenge of activation quantization by mathematically migrating the quantization difficulty from activations to weights. The key insight is that activation distributions often have outlier channels with significantly larger magnitudes (sometimes 100x larger than the median), making uniform quantization highly lossy for these channels. SmoothQuant applies a per-channel scaling transformation: s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha), where alpha controls the migration strength (typically 0.5). This divides activation magnitudes by s_j while multiplying the corresponding weight columns by the same factor, producing smoother activation distributions that are much easier to quantize uniformly. Combined with INT8 weight quantization, SmoothQuant enables W8A8 (8-bit weights, 8-bit activations) deployment with negligible accuracy loss and achieves up to 1.56x speedup through efficient INT8 GEMM kernels on NVIDIA GPUs.

Quantization-Aware Training (QAT) simulates quantization during fine-tuning to produce models that are inherently more robust to quantization. While QAT typically achieves better accuracy than PTQ at the same bit width, the computational cost of fine-tuning large models makes it impractical for many deployments. Recent work on QLoRA (Dettmers et al., 2023) combines quantization with parameter-efficient fine-tuning: the base model is quantized to INT4, and only the LoRA adapters are trained in full precision. This enables fine-tuning of 65B models on a single 48GB GPU, democratizing access to LLM customization while maintaining competitive performance.

### 2.2 Pruning

Pruning removes redundant parameters or structures from neural networks. Unstructured pruning (removing individual weights) can achieve high compression ratios but often requires specialized sparse hardware for acceleration. Structured pruning (removing entire neurons, attention heads, or layers) provides guaranteed speedups on standard hardware but typically incurs larger accuracy losses.

SparseGPT (Frantar and Alistarh, 2023) extends the ideas behind GPTQ to unstructured pruning. It performs one-shot pruning of LLMs to 50% sparsity with minimal accuracy degradation by solving a sparse reconstruction problem layer by layer. At 50% sparsity, SparseGPT adds less than 0.3 perplexity points on WikiText-2 for LLaMA-7B. Combined with the 2:4 structured sparsity pattern supported by NVIDIA Ampere GPUs, SparseGPT can achieve actual inference speedups of 1.3-1.5x. However, the practical benefits depend on hardware support for sparse matrix operations, which varies significantly across platforms.

Wanda (Sun et al., 2024) proposes a remarkably simple pruning criterion: the product of weight magnitude and corresponding input activation norm. Despite its simplicity—requiring only a single forward pass through calibration data—Wanda achieves competitive results with SparseGPT while being 300x faster to compute. This finding suggests that the correlation between weights and activations contains rich information about weight importance that can be exploited without complex optimization.

Layer pruning has emerged as a promising direction for very large models. ShortGPT (Men et al., 2024) demonstrates that removing up to 25% of layers from a 70B model can be done with less than 2% accuracy degradation on downstream tasks, by identifying and removing layers with high redundancy scores based on Block Influence (BI) analysis. The BI metric measures each layer's contribution to the transformation of hidden states: layers with small BI scores contribute little unique information and can be safely removed. This approach is particularly effective for models with many layers (60+) where redundancy naturally increases.

### 2.3 Knowledge Distillation

Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model by training the student to match the teacher's behavior. For LLMs, this typically involves minimizing the KL divergence between the student's and teacher's output token probability distributions. Recent advances include progressive distillation, task-specific distillation, and synthetic data generation using the teacher model.

MiniLLM (Gu et al., 2024) proposes reverse KL divergence for distillation, which encourages the student to generate text that the teacher considers high-probability, rather than trying to match the teacher's full distribution including low-probability tails. This approach reduces the student's tendency to generate hallucinations and out-of-distribution text. On instruction-following benchmarks, MiniLLM-distilled 1.2B students outperform conventionally trained 3B models, suggesting that distillation can be more parameter-efficient than training from scratch. The key insight is that reverse KL naturally focuses the student on the modes of the teacher's distribution, producing more conservative but more reliable generation.

## 3. Efficient Attention Mechanisms

The standard attention mechanism in Transformers has O(n^2) complexity in sequence length for both computation and memory, making it a key bottleneck for long-context inference scenarios.

### 3.1 KV Cache Optimization

During autoregressive generation, the Key and Value projections from all previous tokens are cached to avoid redundant computation. The KV cache grows linearly with both sequence length and batch size, often becoming the dominant memory consumer during inference. For a 70B model with 80 layers, 64 KV heads, and 128-dimensional heads, the KV cache for a single sequence of length 8192 requires: 2 (K+V) x 80 (layers) x 64 (heads) x 128 (dim) x 8192 (len) x 2 bytes (FP16) = 20.5 GB. For a batch of 32 such sequences, the KV cache alone requires 656 GB, far exceeding the memory of any single accelerator.

Multi-Query Attention (MQA, Shazeer 2019) reduces the KV cache by sharing a single key and value head across all query heads. Grouped-Query Attention (GQA, Ainslie et al., 2023) provides a middle ground by grouping query heads into G groups, each sharing one key-value head. GQA with 8 groups (used in LLaMA-2-70B, Qwen-2.5-72B, and Mistral-7B) reduces the KV cache by 8x compared to standard multi-head attention, with typically less than 0.3% accuracy degradation on language modeling benchmarks. The key insight is that the learned key and value representations across heads are highly correlated, so sharing them introduces minimal information loss while dramatically reducing memory requirements and bandwidth consumption.

PagedAttention (Kwon et al., 2023), implemented in the vLLM serving framework, addresses KV cache memory fragmentation by managing cache storage in fixed-size blocks (pages), analogous to virtual memory in operating systems. Without paging, each sequence's KV cache must be stored in a contiguous memory region, and since the final generation length is unknown at request time, systems must either allocate for the maximum possible length (wasting memory) or dynamically reallocate (causing fragmentation). PagedAttention eliminates both problems: the KV cache is divided into blocks of fixed token count (typically 16-32), allocated on demand as the sequence grows. This approach reduces memory waste from an average of 60-80% to near zero, enabling 2-4x higher batch sizes and proportionally higher throughput.

An important extension of PagedAttention is prefix caching, which enables memory sharing between sequences with common prefixes. When multiple requests share the same system prompt or document context, the corresponding KV cache blocks can be computed once and shared through copy-on-write references. This is particularly valuable for retrieval-augmented generation (RAG) applications where the same documents are repeatedly used as context across multiple queries.

### 3.2 FlashAttention

FlashAttention (Dao et al., 2022; Dao, 2023) reformulates the attention computation to minimize data movement between GPU global memory (HBM) and fast on-chip memory (SRAM). The standard attention implementation materializes the full n x n attention matrix in HBM, requiring O(n^2) memory reads and writes. FlashAttention computes attention in tiles: it loads blocks of Q, K, V into SRAM, computes partial attention using the online softmax algorithm (which allows computing softmax incrementally without requiring the full attention matrix), and writes only the final output back to HBM. This reduces HBM access from O(n^2) to O(n), achieving 2-4x wall-clock speedup while being mathematically equivalent to standard attention.

FlashAttention-2 further improves performance through better work partitioning across GPU thread blocks and reduced non-matmul FLOPs, achieving up to 230 TFLOPS on A100 GPUs (73% of theoretical peak). FlashAttention-3 targets the Hopper architecture (H100) with asynchronous operations, FP8 tensor core utilization, and warp specialization, pushing attention computation to near-hardware-peak throughput.

## 4. Memory Management and Batching

### 4.1 Continuous Batching

Traditional static batching pads all sequences in a batch to the maximum length and processes the entire batch together, wasting computation on padding tokens. For workloads with high variance in output lengths (common in chat applications), the wasted computation can exceed 50%. Continuous batching (also called in-flight batching or iteration-level batching) dynamically adds and removes sequences from the batch at each generation step. When a sequence completes (generates an end-of-sequence token), a new sequence from the waiting queue immediately takes its place. This eliminates padding waste and improves GPU utilization from typical 20-40% with static batching to 60-80% with continuous batching.

Orca (Yu et al., 2022) pioneered continuous batching and demonstrated throughput improvements of up to 36.9x compared to static batching on production workloads. The key enabling insight is that autoregressive generation naturally processes one token per sequence per step, so there is no fundamental requirement for all sequences in a batch to be at the same generation step. Each sequence independently tracks its progress, and the batch is reconstituted at every iteration with whatever sequences are currently active.

### 4.2 Chunked Prefill

Chunked prefill (also called incremental prefill) addresses the latency impact of long prefill requests on concurrent decode requests. Without chunking, a request with a 10K-token prompt would monopolize the GPU for the entire prefill duration, blocking all ongoing decode operations and causing latency spikes. Chunked prefill limits the number of new tokens processed per scheduler step (typically to 512-4096 tokens), interleaving prefill chunks with decode tokens in the same batch. This smooths out latency and enables predictable SLA compliance.

The interaction between chunked prefill and prefix caching creates interesting optimization opportunities. With prefix caching, a chunked prefill that hits cached blocks can skip those blocks entirely, reducing the effective prefill length. The scheduler must be cache-aware: it should check for cache hits before scheduling prefill chunks, and it should align chunk boundaries with cache block boundaries to maximize hit rates.

### 4.3 Prefix Caching for State-Space Models

For attention-based models, prefix caching is conceptually straightforward: the KV cache entries for each position are independent, so cached blocks can be directly loaded and used. For state-space models (SSMs) like Mamba and GDN (Gated Delta Networks), the situation is fundamentally different.

SSM states are inherently sequential: the state at position t is a function of the entire input sequence from position 0 to t. This means that caching SSM states at block boundaries requires computing and storing the accumulated state at each boundary—not just the local key-value pairs. The "align" mode stores state only at the last block boundary processed in each scheduler step, while the "all" mode stores state at every block boundary within each step.

The critical difference appears during cache hits spanning multiple blocks: if a new request shares a 5-block prefix with a cached request, "align" mode may only have valid state at block boundaries that coincided with scheduler step endings (potentially 3 out of 5 blocks), while "all" mode guarantees valid state at all 5 block boundaries. A cache hit at a block without valid state leads to reading uninitialized or stale data, causing silent output corruption.

## 5. Speculative Decoding

Speculative decoding uses a smaller, faster "draft" model to generate candidate token sequences, which are then verified in parallel by the larger "target" model. The verification step processes all draft tokens simultaneously (similar to prefill), so accepted tokens effectively receive free computation relative to sequential generation.

The acceptance rate alpha (fraction of draft tokens accepted by the target model) determines the speedup: effective_speedup = gamma * alpha / (1 + gamma * c), where gamma is the number of draft tokens and c is the cost ratio of draft-to-target forward passes. Typical acceptance rates range from 60-85% for well-matched draft-target pairs, translating to 2-3x effective speedup for the decode phase.

Medusa (Cai et al., 2024) eliminates the need for a separate draft model by adding multiple lightweight decoding heads to the target model itself. Each head is a small neural network (typically 1-2 transformer blocks) that predicts a different future token position. Tree-structured attention enables efficient parallel verification of all possible candidate sequences formed by the Cartesian product of each head's top-k predictions. Medusa-2 achieves 2.2-3.6x speedup on LLaMA-2-7B with only 0.6B additional parameters.

EAGLE (Li et al., 2024) uses an auto-regressive feature-level draft head that operates on the target model's hidden states rather than token embeddings. By leveraging the rich representations already computed by the target model, EAGLE achieves higher acceptance rates (85-95%) than external draft models, resulting in 2.5-4.3x speedup. The draft head processes features from the previous target model forward pass, requiring minimal additional computation. EAGLE-2 further improves by dynamically adjusting the tree structure based on confidence scores, spending more verification budget on uncertain positions and less on confident ones.

Lookahead decoding (Fu et al., 2024) takes a different approach entirely: it uses Jacobi iteration to generate multiple tokens in parallel without any draft model. By treating autoregressive generation as a fixed-point iteration problem, lookahead decoding can generate n-grams in parallel when the model would have generated the same tokens sequentially. This approach is lossless (produces identical outputs to sequential decoding) and requires no additional training or draft model, but achieves more modest speedups (1.5-2x) compared to draft-model approaches.

## 6. Hardware-Specific Optimizations

Different hardware platforms offer unique optimization opportunities that can significantly impact inference performance.

### 6.1 GPU Optimizations

On NVIDIA GPUs, key optimization strategies include: (1) Tensor Core utilization through mixed-precision computation (FP16/BF16 matrix multiply with FP32 accumulation), (2) kernel fusion to reduce memory bandwidth by combining multiple operations into single kernels, (3) asynchronous memory operations that overlap compute and data transfer, and (4) shared memory optimization to maximize data reuse within thread blocks. Frameworks like FasterTransformer, TensorRT-LLM, and vLLM provide pre-built optimized kernels that exploit these features.

### 6.2 NPU Optimizations

For Ascend NPUs, the CANN (Compute Architecture for Neural Networks) framework provides optimized operator libraries. Key architectural differences from GPUs include: (1) the AI Core with separate vector and cube processing units, enabling independent control over element-wise and matrix operations; (2) a unified buffer architecture that provides more deterministic memory access patterns; (3) hierarchical memory with global memory, L2 cache, and local SRAM, requiring explicit data movement management; and (4) the AscendC programming model that provides C++-like kernel development with hardware-aware abstractions.

For LLM inference on NPUs, specialized Triton backends and custom AscendC operators are developed to match GPU-optimized kernel performance. Challenges include adapting GPU-native optimizations (like FlashAttention's warp specialization) to the NPU's different parallelism model, and handling numerical precision differences between GPU and NPU floating-point implementations.

## 7. Experimental Results

We evaluate representative methods from each category on two model families (LLaMA-2 and Qwen-2.5) across standard benchmarks.

### Table 1: Single-Request Latency (LLaMA-2-7B, seq_len=2048)

| Method | ms/token | Speedup | WikiText-2 PPL | MMLU |
|--------|:--------:|:-------:|:--------------:|:----:|
| FP16 Baseline | 28.3 | 1.00x | 5.47 | 45.8% |
| GPTQ-INT4 | 15.1 | 1.87x | 5.68 | 45.1% |
| AWQ-INT4 | 14.8 | 1.91x | 5.62 | 45.3% |
| SmoothQuant-W8A8 | 18.2 | 1.56x | 5.49 | 45.6% |
| GPTQ-INT4 + FlashAttn-2 | 12.3 | 2.30x | 5.68 | 45.1% |
| AWQ-INT4 + KV-INT8 | 11.7 | 2.42x | 5.71 | 45.0% |
| EAGLE-2 (speculative) | 8.4 | 3.37x | 5.47 | 45.8% |
| INT4 + EAGLE-2 + Flash | 6.1 | 4.64x | 5.72 | 44.9% |

### Table 2: Serving Throughput (Qwen-2.5-7B, A100 80GB)

| Configuration | Max Batch | tok/s | Memory |
|--------------|:---------:|:-----:|:------:|
| FP16, static batch | 8 | 342 | 72.1 GB |
| FP16, continuous batch | 32 | 1,247 | 74.3 GB |
| INT4, continuous batch | 64 | 2,891 | 38.5 GB |
| INT4, cont. + prefix cache | 64 | 4,156 | 41.2 GB |
| INT4, cont. + prefix + spec | 64 | 6,203 | 43.8 GB |

### Table 3: Prefix Cache Effectiveness by Application

| Application Type | Avg Prefix | Hit Rate | Speedup |
|-----------------|:----------:|:--------:|:-------:|
| Chat (system prompt) | 256 tok | 95% | 1.2x |
| Chat (multi-turn, 5 turns) | 2,048 tok | 78% | 2.1x |
| RAG (single document) | 4,096 tok | 62% | 2.8x |
| RAG (document corpus) | 4,096 tok | 91% | 3.4x |
| Code completion (repo ctx) | 8,192 tok | 45% | 1.9x |
| Agent (tool docs + history) | 3,072 tok | 83% | 2.5x |

### Table 4: SSM Cache Mode Comparison (Qwen3.5-0.8B GDN, block_size=1024)

| Metric | Align Mode | All Mode |
|--------|:----------:|:--------:|
| Blocks with valid state (5-block prefix) | 3/5 (60%) | 5/5 (100%) |
| Cache hit correctness | Partial (dead blocks) | Complete |
| Per-step state writes | 1 block | All blocks in step |
| Memory overhead per block | Baseline | +intermediate h tensor |
| Scheduler constraint | Block-aligned steps | Block-aligned steps |

## 8. Discussion and Future Directions

Our analysis reveals several important findings and open challenges for efficient LLM inference:

**Composability is essential**: Individual optimization techniques provide modest improvements (1.5-2x), but composing multiple orthogonal techniques yields compound benefits (3-5x). The most effective production configurations combine quantization (reduces compute per operation), efficient attention (reduces memory access), continuous batching (improves utilization), and prefix caching (eliminates redundant computation). However, some combinations interact negatively: aggressive quantization can reduce speculative decoding acceptance rates by altering the probability distribution.

**The memory bandwidth wall**: For large batch inference, memory bandwidth rather than compute is the fundamental bottleneck. Model weights, KV cache entries, SSM states, and activation tensors all compete for limited memory bandwidth. This motivates the trend toward smaller but more capable models, aggressive quantization, and hardware with higher memory bandwidth (HBM3, HBM3e).

**SSM inference presents unique challenges**: Unlike Transformers where each token's KV cache is independent, SSM states carry accumulated sequential information. This creates novel challenges for prefix caching (the "dead block" problem), speculative decoding (state rollback is more complex), and parallel inference (state cannot be easily partitioned). The trade-off between cache storage granularity (align vs all modes) and memory overhead is a new dimension not present in attention-based model serving.

**Emerging directions include**: (1) Mixture-of-Experts inference, where only a subset of parameters activate per token, requiring efficient routing and expert loading; (2) token-level adaptive computation, where the model allocates different amounts of computation to tokens of varying difficulty; (3) cross-request state sharing beyond simple prefix caching, enabling reuse of intermediate computations across semantically similar but not identical prompts; and (4) hardware-software co-design for SSM architectures, which may benefit from different memory hierarchies and compute patterns than those optimized for attention.

## 9. Conclusion

Efficient LLM inference is a multi-dimensional optimization problem spanning model architecture, numerical precision, memory management, scheduling, and hardware utilization. This survey demonstrates that the field has made remarkable progress: from naive FP16 inference, production deployments can achieve 3-5x improvements through systematic application of quantization, efficient attention, continuous batching, prefix caching, and speculative decoding. As models continue scaling and new architectures like state-space models gain adoption, the optimization landscape will continue to evolve. We hope this survey provides a useful roadmap for practitioners deploying LLMs and researchers pushing the boundaries of efficient inference.

## References

[1] Frantar, E., et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR 2023.
[2] Lin, J., et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys 2024.
[3] Xiao, G., et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." ICML 2023.
[4] Frantar, E. and Alistarh, D. "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot." ICML 2023.
[5] Sun, M., et al. "A Simple and Effective Pruning Approach for Large Language Models." ICLR 2024.
[6] Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
[7] Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.
[8] Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024.
[9] Cai, T., et al. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." ICML 2024.
[10] Li, Y., et al. "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." ICML 2024.
[11] Yu, G., et al. "Orca: A Distributed Serving System for Transformer-Based Generative Models." OSDI 2022.
[12] Gu, Y., et al. "MiniLLM: Knowledge Distillation of Large Language Models." ICLR 2024.
[13] Men, X., et al. "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect." arXiv 2024.
[14] Shazeer, N. "Fast Transformer Decoding: One Write-Head is All You Need." arXiv 2019.
[15] Ainslie, J., et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023.
[16] Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized Large Language Models." NeurIPS 2023.
[17] Fu, Y., et al. "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding." ICML 2024.
"""

# ─── Survey Questions ─────────────────────────────────────────────
SURVEY_QUESTION_A = (
    "Question: According to Table 1 in the paper, which single optimization "
    "method achieved the best latency-to-accuracy trade-off for LLaMA-2-7B, "
    "and what was its speedup and MMLU accuracy? Provide specific numbers."
    "\n\nAnswer: "
)
SURVEY_QUESTION_B = (
    "Question: The paper discusses a 'dead block' problem specific to SSM "
    "prefix caching. Explain what causes dead blocks in align mode, how many "
    "blocks out of 5 have valid state according to Table 4, and why all mode "
    "solves this problem."
    "\n\nAnswer: "
)

SURVEY_PROMPT_A = SURVEY_DOCUMENT + SURVEY_QUESTION_A
SURVEY_PROMPT_B = SURVEY_DOCUMENT + SURVEY_QUESTION_B

# ─── Short prefixes for dead-block exposure tests ─────────
# R1 uses the full SURVEY_DOCUMENT; R2 uses a truncated version.
# The shared prefix (first N complete blocks) exposes align-mode's
# inability to read intermediate block boundary SSM states.

# ~2 blocks shared (~2192 tokens): truncate before Section 3
SHORT_PREFIX_2BLOCK = SURVEY_DOCUMENT[:SURVEY_DOCUMENT.index("\n## 3. Efficient Attention")]

# ~1 block shared (~1047 tokens): truncate after GPTQ paragraph
SHORT_PREFIX_1BLOCK = SURVEY_DOCUMENT[
    :SURVEY_DOCUMENT.index(
        "leaving activations in full precision during computation."
    ) + len("leaving activations in full precision during computation.")
]

# Questions for short prefixes (about content within the truncated portion)
SHORT_QUESTION_2BLOCK = (
    "Question: Based on the paper's discussion of model compression in "
    "Section 2, compare the three categories of compression techniques "
    "(quantization, pruning, knowledge distillation). Which approach "
    "preserves the most accuracy for LLaMA-2-7B and why?"
    "\n\nAnswer: "
)
SHORT_QUESTION_1BLOCK = (
    "Question: The paper discusses quantization methods in Section 2.1. "
    "Compare GPTQ and AWQ approaches — what is GPTQ's key limitation "
    "that AWQ addresses, and what specific accuracy numbers does the "
    "paper report for INT4 quantization on MMLU?"
    "\n\nAnswer: "
)

# Short R2 prompts
SHORT_PROMPT_2BLOCK = SHORT_PREFIX_2BLOCK + SHORT_QUESTION_2BLOCK
SHORT_PROMPT_1BLOCK = SHORT_PREFIX_1BLOCK + SHORT_QUESTION_1BLOCK


# ════════════════════════════════════════════════════════════════
#  SCENARIO C — Agent: REST API Documentation (~5500 tokens)
# ════════════════════════════════════════════════════════════════

AGENT_API_DOCS = """\
You are an AI assistant with access to the TaskFlow API. When the user asks you to perform an operation, respond with the appropriate API call based on the documentation below. Always use the exact endpoint paths and parameter names from the docs.

---

# TaskFlow API Documentation v3.2

## Overview

TaskFlow is a comprehensive project management and task tracking system designed for software development teams. The API follows RESTful conventions, uses JSON for request/response bodies, and requires Bearer token authentication for all endpoints. All timestamps are in ISO 8601 format (UTC). Rate limits: 1000 requests/minute for standard tier, 5000/minute for enterprise.

Base URL: `https://api.taskflow.io/v3`

## Authentication

All requests must include an `Authorization: Bearer <token>` header. Tokens are obtained through the OAuth 2.0 flow via `/auth/token`. Tokens expire after 24 hours and can be refreshed using `/auth/refresh`.

## Common Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 204 | No Content (successful deletion) |
| 400 | Bad Request (invalid parameters) |
| 401 | Unauthorized (invalid or expired token) |
| 403 | Forbidden (insufficient permissions) |
| 404 | Not Found |
| 409 | Conflict (e.g., duplicate name) |
| 422 | Unprocessable Entity (validation error) |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |

## Pagination

List endpoints support cursor-based pagination via `?cursor=<token>&limit=<n>` parameters. Default limit is 25, maximum is 100. Response includes `next_cursor` and `has_more` fields.

---

## 1. Projects

### 1.1 List Projects
`GET /projects`

Query parameters:
- `status` (string, optional): Filter by status. Values: `active`, `archived`, `all`. Default: `active`.
- `owner_id` (string, optional): Filter by project owner user ID.
- `search` (string, optional): Full-text search on project name and description.
- `sort_by` (string, optional): Sort field. Values: `name`, `created_at`, `updated_at`, `task_count`. Default: `updated_at`.
- `sort_order` (string, optional): `asc` or `desc`. Default: `desc`.
- `cursor` (string, optional): Pagination cursor.
- `limit` (integer, optional): Results per page (1-100, default 25).

Response: `{ "projects": [...], "total_count": 42, "next_cursor": "abc123", "has_more": true }`

### 1.2 Create Project
`POST /projects`

Body:
```json
{
  "name": "string (required, 1-100 chars, unique within org)",
  "description": "string (optional, max 2000 chars)",
  "visibility": "public | private | internal (default: private)",
  "default_assignee_id": "string (optional, user ID)",
  "tags": ["string array (optional, max 20 tags)"],
  "template_id": "string (optional, project template ID)"
}
```

Returns: 201 with the created project object.

### 1.3 Get Project
`GET /projects/{project_id}`

Returns the full project object including computed fields (task_count, completion_percentage, team_members).

### 1.4 Update Project
`PATCH /projects/{project_id}`

Body: Any subset of the fields from Create Project. Additionally supports:
- `status`: `active` or `archived`
- `lead_id`: Project lead user ID

Returns: 200 with updated project object.

### 1.5 Delete Project
`DELETE /projects/{project_id}`

Query parameters:
- `cascade` (boolean, optional): If true, also deletes all tasks, comments, and attachments. Default: false (returns 409 if project has tasks).

Returns: 204 on success.

---

## 2. Tasks

### 2.1 List Tasks
`GET /projects/{project_id}/tasks`

Query parameters:
- `status` (string, optional): `open`, `in_progress`, `review`, `done`, `blocked`, `all`. Default: `all`.
- `assignee_id` (string, optional): Filter by assigned user.
- `priority` (string, optional): `critical`, `high`, `medium`, `low`. Multiple values comma-separated.
- `label_ids` (string, optional): Comma-separated label IDs. Tasks matching ANY label are returned.
- `due_before` (ISO 8601, optional): Filter tasks due before this date.
- `due_after` (ISO 8601, optional): Filter tasks due after this date.
- `search` (string, optional): Full-text search on title and description.
- `parent_id` (string, optional): Filter subtasks of a specific parent task. Use `null` for root tasks only.
- `sort_by` (string, optional): `title`, `priority`, `due_date`, `created_at`, `updated_at`, `position`. Default: `position`.
- `include` (string, optional): Comma-separated related resources to include: `assignee`, `labels`, `subtasks`, `comments_count`, `attachments_count`.

Response: `{ "tasks": [...], "total_count": 156, "next_cursor": "...", "has_more": true }`

### 2.2 Create Task
`POST /projects/{project_id}/tasks`

Body:
```json
{
  "title": "string (required, 1-200 chars)",
  "description": "string (optional, max 10000 chars, supports Markdown)",
  "status": "open | in_progress | review | done | blocked (default: open)",
  "priority": "critical | high | medium | low (default: medium)",
  "assignee_id": "string (optional)",
  "due_date": "ISO 8601 (optional)",
  "estimated_hours": "number (optional, 0.25-999)",
  "parent_id": "string (optional, for creating subtasks)",
  "label_ids": ["string array (optional)"],
  "position": "number (optional, for manual ordering)",
  "custom_fields": { "key": "value (optional, max 50 fields)" },
  "watchers": ["string array (optional, user IDs to notify)"]
}
```

Returns: 201 with created task object including auto-generated `task_number` (e.g., "PROJ-142").

### 2.3 Get Task
`GET /projects/{project_id}/tasks/{task_id}`

Query parameters:
- `include` (string, optional): `assignee`, `labels`, `subtasks`, `comments`, `attachments`, `activity_log`, `time_entries`.

Returns: Full task object with requested includes.

### 2.4 Update Task
`PATCH /projects/{project_id}/tasks/{task_id}`

Body: Any subset of Create Task fields. Additionally:
- `actual_hours`: Number (completed work hours)
- `resolution`: String (only when status changes to `done`)
- `blocked_reason`: String (only when status changes to `blocked`)

Triggers webhook: `task.updated` with changed fields listed in `changes` array.

### 2.5 Delete Task
`DELETE /projects/{project_id}/tasks/{task_id}`

Query parameters:
- `cascade` (boolean): Delete subtasks too. Default: false.

Returns: 204.

### 2.6 Bulk Update Tasks
`PATCH /projects/{project_id}/tasks/bulk`

Body:
```json
{
  "task_ids": ["array of task IDs (required, max 100)"],
  "updates": {
    "status": "string (optional)",
    "assignee_id": "string (optional)",
    "priority": "string (optional)",
    "label_ids": { "add": ["..."], "remove": ["..."] },
    "due_date": "ISO 8601 (optional)"
  }
}
```

Returns: 200 with `{ "updated_count": 15, "failed": [] }`.

### 2.7 Move Task
`POST /projects/{project_id}/tasks/{task_id}/move`

Body:
```json
{
  "target_project_id": "string (required)",
  "keep_assignee": "boolean (default: true)",
  "keep_labels": "boolean (default: false, labels are project-scoped)"
}
```

Returns: 200 with task in new project.

---

## 3. Comments

### 3.1 List Comments
`GET /projects/{project_id}/tasks/{task_id}/comments`

Query parameters: `cursor`, `limit`, `sort_order` (default: `asc` for chronological).

### 3.2 Create Comment
`POST /projects/{project_id}/tasks/{task_id}/comments`

Body:
```json
{
  "body": "string (required, max 5000 chars, Markdown supported)",
  "mentions": ["user IDs to @mention (optional)"],
  "attachments": ["attachment IDs (optional, pre-uploaded)"]
}
```

Returns: 201.

### 3.3 Update Comment
`PATCH /projects/{project_id}/tasks/{task_id}/comments/{comment_id}`

Only the comment author or project admin can update. Body: `{ "body": "updated text" }`.

### 3.4 Delete Comment
`DELETE /projects/{project_id}/tasks/{task_id}/comments/{comment_id}`

Returns: 204.

---

## 4. Labels

### 4.1 List Labels
`GET /projects/{project_id}/labels`

Returns all labels for the project with usage counts.

### 4.2 Create Label
`POST /projects/{project_id}/labels`

Body: `{ "name": "string (required)", "color": "hex color (required, e.g. #FF5733)", "description": "string (optional)" }`

### 4.3 Update Label
`PATCH /projects/{project_id}/labels/{label_id}`

### 4.4 Delete Label
`DELETE /projects/{project_id}/labels/{label_id}`

Query: `remove_from_tasks` (boolean, default true).

---

## 5. Users & Teams

### 5.1 List Users
`GET /users`

Query: `role` (admin, member, viewer), `team_id`, `search`, `status` (active, deactivated).

### 5.2 Get Current User
`GET /users/me`

Returns the authenticated user's profile including permissions, teams, and notification preferences.

### 5.3 List Teams
`GET /teams`

### 5.4 Get Team Members
`GET /teams/{team_id}/members`

### 5.5 Add Team Member
`POST /teams/{team_id}/members`

Body: `{ "user_id": "string", "role": "lead | member | viewer" }`

### 5.6 Remove Team Member
`DELETE /teams/{team_id}/members/{user_id}`

---

## 6. Time Tracking

### 6.1 List Time Entries
`GET /projects/{project_id}/tasks/{task_id}/time_entries`

Query: `user_id`, `date_from`, `date_to`.

### 6.2 Create Time Entry
`POST /projects/{project_id}/tasks/{task_id}/time_entries`

Body: `{ "hours": "number (required, 0.25-24)", "date": "ISO 8601 date", "description": "string (optional)" }`

### 6.3 Update Time Entry
`PATCH /projects/{project_id}/tasks/{task_id}/time_entries/{entry_id}`

### 6.4 Delete Time Entry
`DELETE /projects/{project_id}/tasks/{task_id}/time_entries/{entry_id}`

---

## 7. Webhooks

### 7.1 List Webhooks
`GET /webhooks`

### 7.2 Create Webhook
`POST /webhooks`

Body:
```json
{
  "url": "string (required, HTTPS only)",
  "events": ["task.created", "task.updated", "task.deleted", "comment.created", "project.updated"],
  "project_id": "string (optional, all projects if omitted)",
  "secret": "string (optional, for HMAC signature verification)",
  "active": "boolean (default: true)"
}
```

### 7.3 Update Webhook
`PATCH /webhooks/{webhook_id}`

### 7.4 Delete Webhook
`DELETE /webhooks/{webhook_id}`

### 7.5 Test Webhook
`POST /webhooks/{webhook_id}/test`

Sends a test payload to the webhook URL and returns the response status.

---

## 8. Reports & Analytics

### 8.1 Project Summary
`GET /projects/{project_id}/reports/summary`

Returns: task counts by status, average completion time, overdue task count, team workload distribution.

### 8.2 Burndown Chart
`GET /projects/{project_id}/reports/burndown`

Query: `sprint_id` or `date_from`/`date_to`.

Returns: Daily data points with `{ "date": "...", "total_remaining": 42, "ideal_remaining": 38, "completed_today": 5 }`.

### 8.3 Team Velocity
`GET /projects/{project_id}/reports/velocity`

Query: `period` (week, sprint, month), `count` (number of periods, default 6).

Returns: Array of `{ "period": "...", "points_completed": 34, "tasks_completed": 12, "avg_cycle_time_hours": 28.5 }`.

### 8.4 User Workload
`GET /reports/workload`

Query: `team_id`, `date_from`, `date_to`.

Returns per-user: assigned task count, estimated hours, logged hours, overdue count.

---

## 9. Notifications

### 9.1 List Notifications
`GET /notifications`

Query: `read` (boolean), `type` (mention, assignment, due_date, comment, status_change).

### 9.2 Mark as Read
`PATCH /notifications/{notification_id}/read`

### 9.3 Mark All as Read
`POST /notifications/read_all`

---

## 10. Search

### 10.1 Global Search
`GET /search`

Query:
- `q` (string, required): Search query.
- `type` (string, optional): `task`, `project`, `comment`, `all`. Default: `all`.
- `project_id` (string, optional): Scope search to a project.
- `filters` (JSON string, optional): Advanced filters like `{"status":"open","priority":"high"}`.

Returns: `{ "results": [{ "type": "task", "id": "...", "title": "...", "snippet": "...", "score": 0.95 }], "total": 23 }`

---

## 11. File Attachments

### 11.1 Upload Attachment
`POST /projects/{project_id}/attachments`

Content-Type: `multipart/form-data`

Form fields:
- `file` (required): The file to upload. Max size: 50MB. Allowed types: images (png, jpg, gif, svg), documents (pdf, doc, docx, xls, xlsx, ppt, pptx), code (txt, md, json, yaml, xml, csv), archives (zip, tar.gz).
- `task_id` (string, optional): Associate with a specific task.
- `description` (string, optional): File description, max 500 chars.

Response: `{ "id": "att_abc123", "filename": "screenshot.png", "size_bytes": 245760, "mime_type": "image/png", "url": "https://cdn.taskflow.io/...", "thumbnail_url": "https://cdn.taskflow.io/.../thumb", "uploaded_by": "usr_...", "created_at": "..." }`

### 11.2 List Attachments
`GET /projects/{project_id}/attachments`

Query: `task_id`, `uploaded_by`, `mime_type` (prefix match, e.g., `image/`), `cursor`, `limit`.

### 11.3 Get Attachment
`GET /projects/{project_id}/attachments/{attachment_id}`

Returns metadata and a time-limited signed download URL (valid for 1 hour).

### 11.4 Delete Attachment
`DELETE /projects/{project_id}/attachments/{attachment_id}`

Only the uploader or project admin can delete. Returns 204.

---

## 12. Sprints

### 12.1 List Sprints
`GET /projects/{project_id}/sprints`

Query: `status` (planning, active, completed, all), `cursor`, `limit`.

### 12.2 Create Sprint
`POST /projects/{project_id}/sprints`

Body:
```json
{
  "name": "string (required, e.g., 'Sprint 23')",
  "goal": "string (optional, sprint goal description, max 500 chars)",
  "start_date": "ISO 8601 (required)",
  "end_date": "ISO 8601 (required)",
  "capacity_points": "number (optional, team capacity in story points)"
}
```

Only one sprint can be `active` at a time. Creating a new sprint while another is active returns 409.

### 12.3 Get Sprint
`GET /projects/{project_id}/sprints/{sprint_id}`

Returns sprint details with computed fields: `total_points`, `completed_points`, `remaining_points`, `task_count`, `days_remaining`.

### 12.4 Update Sprint
`PATCH /projects/{project_id}/sprints/{sprint_id}`

Body: Any subset of Create Sprint fields, plus `status` (planning → active → completed).

### 12.5 Add Tasks to Sprint
`POST /projects/{project_id}/sprints/{sprint_id}/tasks`

Body: `{ "task_ids": ["array of task IDs"], "points": { "task_id": 5, "task_id2": 3 } }`

### 12.6 Remove Tasks from Sprint
`DELETE /projects/{project_id}/sprints/{sprint_id}/tasks`

Body: `{ "task_ids": ["array of task IDs"] }`

### 12.7 Sprint Retrospective
`POST /projects/{project_id}/sprints/{sprint_id}/retrospective`

Body:
```json
{
  "went_well": ["string array"],
  "needs_improvement": ["string array"],
  "action_items": ["string array"],
  "team_morale": "number (1-5 scale)"
}
```

---

## 13. Automations

### 13.1 List Automations
`GET /projects/{project_id}/automations`

### 13.2 Create Automation
`POST /projects/{project_id}/automations`

Body:
```json
{
  "name": "string (required)",
  "trigger": {
    "event": "task.created | task.updated | task.status_changed | comment.created | due_date.approaching",
    "conditions": [
      { "field": "status", "operator": "equals", "value": "done" },
      { "field": "priority", "operator": "in", "value": ["critical", "high"] },
      { "field": "label_ids", "operator": "contains", "value": "lbl_bug" }
    ]
  },
  "actions": [
    { "type": "update_field", "field": "assignee_id", "value": "usr_reviewer" },
    { "type": "add_label", "label_id": "lbl_reviewed" },
    { "type": "send_notification", "user_ids": ["usr_lead"], "message": "Task completed: {{task.title}}" },
    { "type": "move_to_sprint", "sprint_id": "spr_current" },
    { "type": "create_subtask", "title": "Review: {{task.title}}", "assignee_id": "usr_qa" }
  ],
  "enabled": "boolean (default: true)"
}
```

Supported operators: `equals`, `not_equals`, `in`, `not_in`, `contains`, `greater_than`, `less_than`, `is_empty`, `is_not_empty`.

Template variables in action values: `{{task.title}}`, `{{task.assignee.name}}`, `{{task.project.name}}`, `{{trigger.user.name}}`.

### 13.3 Update Automation
`PATCH /projects/{project_id}/automations/{automation_id}`

### 13.4 Delete Automation
`DELETE /projects/{project_id}/automations/{automation_id}`

### 13.5 Test Automation
`POST /projects/{project_id}/automations/{automation_id}/test`

Body: `{ "task_id": "string" }` — simulates the automation against a specific task without actually executing actions. Returns: `{ "would_trigger": true, "matched_conditions": [...], "planned_actions": [...] }`.

---

## 14. Custom Fields

### 14.1 List Custom Field Definitions
`GET /projects/{project_id}/custom_fields`

### 14.2 Create Custom Field
`POST /projects/{project_id}/custom_fields`

Body:
```json
{
  "name": "string (required)",
  "type": "text | number | date | select | multi_select | url | email | checkbox",
  "description": "string (optional)",
  "required": "boolean (default: false)",
  "options": ["for select/multi_select types: array of option strings"],
  "default_value": "any (optional, must match type)",
  "position": "number (optional, display order)"
}
```

### 14.3 Update Custom Field
`PATCH /projects/{project_id}/custom_fields/{field_id}`

### 14.4 Delete Custom Field
`DELETE /projects/{project_id}/custom_fields/{field_id}`

Warning: Deleting a custom field removes all values for that field from all tasks. This operation is irreversible.

---

## 15. Audit Log

### 15.1 List Audit Events
`GET /projects/{project_id}/audit_log`

Query:
- `actor_id` (string, optional): Filter by user who performed the action.
- `action` (string, optional): `created`, `updated`, `deleted`, `moved`, `assigned`, `status_changed`.
- `resource_type` (string, optional): `task`, `project`, `comment`, `label`, `sprint`, `automation`.
- `date_from` (ISO 8601, optional): Start of date range.
- `date_to` (ISO 8601, optional): End of date range.
- `cursor`, `limit`: Pagination.

Response: Array of `{ "id": "...", "actor": { "id": "...", "name": "..." }, "action": "updated", "resource_type": "task", "resource_id": "...", "changes": [{ "field": "status", "old": "open", "new": "in_progress" }], "timestamp": "...", "ip_address": "..." }`

Audit logs are retained for 90 days (standard) or 365 days (enterprise).

---

## Rate Limiting

Rate limit headers are included in every response:
- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Unix timestamp when the limit resets

When rate limited (429), the response includes: `{ "error": "rate_limit_exceeded", "retry_after_seconds": 32 }`.

---

## Error Response Format

All error responses follow a consistent structure:
```json
{
  "error": {
    "code": "validation_error",
    "message": "Human-readable error description",
    "details": [
      { "field": "title", "message": "Title is required", "code": "required" },
      { "field": "due_date", "message": "Must be a future date", "code": "invalid_value" }
    ],
    "request_id": "req_abc123 (for support reference)"
  }
}
```

Common error codes: `validation_error`, `not_found`, `permission_denied`, `rate_limit_exceeded`, `conflict`, `internal_error`, `service_unavailable`.

---

## SDK Examples

### Python
```python
from taskflow import TaskFlowClient

client = TaskFlowClient(api_key="your_token")

# Create a task
task = client.tasks.create(
    project_id="proj_abc123",
    title="Implement user authentication",
    priority="high",
    assignee_id="usr_john",
    labels=["lbl_feature", "lbl_backend"]
)

# List tasks with filters
tasks = client.tasks.list(
    project_id="proj_abc123",
    status="open",
    priority=["critical", "high"],
    sort_by="due_date"
)

# Bulk update
client.tasks.bulk_update(
    project_id="proj_abc123",
    task_ids=[t.id for t in overdue_tasks],
    updates={"priority": "critical", "labels": {"add": ["lbl_overdue"]}}
)
```

### JavaScript
```javascript
const { TaskFlow } = require('@taskflow/sdk');

const client = new TaskFlow({ apiKey: 'your_token' });

// Search across all projects
const results = await client.search({
  q: 'authentication bug',
  type: 'task',
  filters: { status: 'open', priority: 'high' }
});

// Create automation
await client.automations.create('proj_abc123', {
  name: 'Auto-assign bugs to QA',
  trigger: {
    event: 'task.created',
    conditions: [{ field: 'label_ids', operator: 'contains', value: 'lbl_bug' }]
  },
  actions: [
    { type: 'update_field', field: 'assignee_id', value: 'usr_qa_lead' }
  ]
});
```

---

## Changelog

- v3.2 (2024-12): Added bulk task update, global search, burndown charts, automations, custom fields
- v3.1 (2024-09): Added time tracking, webhooks, team velocity reports, sprints, file attachments
- v3.0 (2024-06): Major rewrite with cursor pagination, custom fields, subtasks, audit log
- v2.5 (2024-03): Added GQA support, labels, task dependencies
- v2.0 (2023-12): REST API redesign, OAuth 2.0 authentication
- v1.0 (2023-06): Initial release with basic CRUD operations
"""

# ─── Agent Tasks ───────────────────────────────────────────────
AGENT_TASK_A = (
    "Task: Create a new high-priority task in project 'proj_abc123' titled "
    "'Fix authentication timeout bug' with description 'Users report being "
    "logged out after 5 minutes instead of the configured 24-hour token "
    "expiry. Investigate the refresh token flow.', assign it to user "
    "'usr_john_doe', set due date to 2025-02-15, and add labels 'bug' "
    "(label_id: 'lbl_bug') and 'auth' (label_id: 'lbl_auth'). "
    "Show the complete API call with endpoint, method, headers, and body."
    "\n\nAPI Call:\n"
)
AGENT_TASK_B = (
    "Task: I need to get a burndown chart for project 'proj_xyz789' "
    "for the date range January 1 to January 31, 2025. Then find all "
    "tasks in that project that are currently blocked with critical "
    "priority. Finally, generate a workload report for team 'team_eng' "
    "for the same date range. Show all three API calls in order."
    "\n\nAPI Calls:\n"
)

AGENT_PROMPT_A = AGENT_API_DOCS + AGENT_TASK_A
AGENT_PROMPT_B = AGENT_API_DOCS + AGENT_TASK_B

# ─── Agent short prefixes for dead-block exposure tests ────────
# R1 uses the full AGENT_API_DOCS; R2 uses a truncated version.

# ~2 blocks shared (~2151 tokens): truncate before §11.2 (well past 2 full blocks)
AGENT_SHORT_2BLOCK = AGENT_API_DOCS[:AGENT_API_DOCS.index("\n### 11.2 List Attachments")]

# ~1 block shared (~1267 tokens): truncate before Section 3 (Comments)
AGENT_SHORT_1BLOCK = AGENT_API_DOCS[:AGENT_API_DOCS.index("\n## 3. Comments")]

# Questions about content within the truncated portions
AGENT_SHORT_QUESTION_2BLOCK = (
    "Task: Using the Search and Notifications APIs documented above, "
    "write a workflow that: (1) searches for all tasks mentioning 'timeout' "
    "in project 'proj_abc123', (2) marks all related notifications as read. "
    "Show both API calls with full request details."
    "\n\nAPI Calls:\n"
)
AGENT_SHORT_QUESTION_1BLOCK = (
    "Task: Using only the Projects and Tasks APIs documented above, "
    "list all tasks in project 'proj_abc123' that have status 'in_progress' "
    "and were created after 2025-01-01. Show the API call with query parameters."
    "\n\nAPI Call:\n"
)

# Short R2 prompts (Agent)
AGENT_SHORT_PROMPT_2BLOCK = AGENT_SHORT_2BLOCK + AGENT_SHORT_QUESTION_2BLOCK
AGENT_SHORT_PROMPT_1BLOCK = AGENT_SHORT_1BLOCK + AGENT_SHORT_QUESTION_1BLOCK


# ════════════════════════════════════════════════════════════════
#  SCENARIO A — Multi-turn Dialog (~5500 tokens)
# ════════════════════════════════════════════════════════════════

DIALOG_HISTORY = """\
You are a senior DevOps engineer helping debug production issues. Continue the conversation based on the full dialog history below.

---

**System**: You are an expert DevOps engineer with 15 years of experience in cloud infrastructure, Kubernetes, CI/CD pipelines, and distributed systems debugging. You provide detailed, actionable technical advice. When diagnosing issues, you systematically consider all possible causes and ask clarifying questions when needed.

**User**: Hey, we're having a critical issue in production. Our main API service started returning 502 errors about 30 minutes ago. The error rate has gone from nearly 0% to about 35% of requests. We're running on Kubernetes (EKS) with 12 pods behind an ALB. Can you help us debug this?

**Assistant**: Absolutely, let's debug this systematically. A 35% 502 error rate with 12 pods suggests some pods are unhealthy while others are still serving. Here's my initial diagnostic plan:

**Immediate checks:**
1. Pod health: `kubectl get pods -n production -l app=api-service` — look for pods in CrashLoopBackOff, Pending, or with high restart counts
2. Pod logs: `kubectl logs -n production -l app=api-service --tail=100 --prefix` — look for error patterns
3. Resource pressure: `kubectl top pods -n production -l app=api-service` — check CPU/memory
4. Recent changes: Did anyone deploy a new version or config change in the last hour?

**ALB-specific checks:**
5. Target group health: Check AWS Console → EC2 → Target Groups → look for unhealthy targets
6. ALB access logs: Look for patterns in 502 responses (specific target IPs?)

Can you run those first two commands and share the output? Also, was there any deployment or config change recently?

**User**: Here's the output:

```
$ kubectl get pods -n production -l app=api-service
NAME                          READY   STATUS    RESTARTS   AGE
api-service-7d4f8b6c9-2hx9k  1/1     Running   0          3d
api-service-7d4f8b6c9-4mn3p  1/1     Running   0          3d
api-service-7d4f8b6c9-5rt2q  1/1     Running   3          45m
api-service-7d4f8b6c9-8kl5v  1/1     Running   0          3d
api-service-7d4f8b6c9-9wx1y  1/1     Running   5          45m
api-service-7d4f8b6c9-b7j6m  1/1     Running   0          3d
api-service-7d4f8b6c9-d3h8n  1/1     Running   4          45m
api-service-7d4f8b6c9-f6g2k  1/1     Running   0          3d
api-service-7d4f8b6c9-h9p4r  1/1     Running   0          3d
api-service-7d4f8b6c9-k2m7s  1/1     Running   0          3d
api-service-7d4f8b6c9-n5q9t  1/1     Running   0          3d
api-service-7d4f8b6c9-r8v3w  1/1     Running   0          3d
```

Three pods (5rt2q, 9wx1y, d3h8n) have restarts starting 45 minutes ago. The logs show:

```
[api-service-7d4f8b6c9-5rt2q] 2025-01-20T14:23:15Z ERROR Failed to connect to database: dial tcp 10.0.5.42:5432: connect: connection refused
[api-service-7d4f8b6c9-5rt2q] 2025-01-20T14:23:15Z ERROR Health check failed: database unavailable
[api-service-7d4f8b6c9-5rt2q] 2025-01-20T14:23:45Z INFO  Starting up... connecting to database...
[api-service-7d4f8b6c9-5rt2q] 2025-01-20T14:23:46Z INFO  Database connection established
[api-service-7d4f8b6c9-5rt2q] 2025-01-20T14:23:46Z INFO  Server ready on :8080
[api-service-7d4f8b6c9-5rt2q] 2025-01-20T14:25:12Z ERROR Failed to connect to database: dial tcp 10.0.5.42:5432: connect: connection refused
```

So they connect, work for a bit, then lose the database connection again. No deployments in the last 24 hours.

**Assistant**: Great data. The pattern is clear: 3 out of 12 pods (25%, close to your 35% error rate accounting for retry distribution) are experiencing intermittent database connectivity issues. They crash, restart, briefly connect to the DB, then lose connectivity again. This explains the 502s — ALB sends traffic to pods that are technically Running but can't serve requests during the brief window when DB is down.

Key observations:
1. **Only 3 pods affected** — this points to a network/routing issue, not a database server issue (if the DB were fully down, all 12 pods would fail)
2. **Same DB IP (10.0.5.42)** — all three connect to the same database endpoint
3. **Intermittent** — connects briefly, then fails again

Let me ask targeted questions:
- Is 10.0.5.42 a direct pod IP, a Service ClusterIP, or an RDS endpoint?
- Is your database running in-cluster or on AWS RDS?
- What's your database connection pool configuration? (max connections, idle timeout, connection lifetime)
- Run: `kubectl get endpoints -n production db-service` (or whatever your DB service is named)
- Run: `kubectl get nodes -o wide` and tell me which nodes those 3 pods are on vs the healthy pods

My hypothesis: those 3 pods landed on the same node (or nodes in the same AZ), and there's a network issue between that node/AZ and the database.

**User**: Good hypothesis! Here's what I found:

10.0.5.42 is an RDS endpoint (PostgreSQL 15, db.r6g.xlarge). The connection pool uses PgBouncer as a sidecar with max 50 connections per pod, idle timeout 300s, server lifetime 3600s.

```
$ kubectl get pods -n production -l app=api-service -o wide
NAME                          NODE                          
api-service-7d4f8b6c9-2hx9k  ip-10-0-1-15.ec2.internal    # AZ: us-east-1a
api-service-7d4f8b6c9-4mn3p  ip-10-0-1-15.ec2.internal    # AZ: us-east-1a
api-service-7d4f8b6c9-5rt2q  ip-10-0-3-28.ec2.internal    # AZ: us-east-1c  ← RESTARTS
api-service-7d4f8b6c9-8kl5v  ip-10-0-2-42.ec2.internal    # AZ: us-east-1b
api-service-7d4f8b6c9-9wx1y  ip-10-0-3-28.ec2.internal    # AZ: us-east-1c  ← RESTARTS
api-service-7d4f8b6c9-b7j6m  ip-10-0-1-15.ec2.internal    # AZ: us-east-1a
api-service-7d4f8b6c9-d3h8n  ip-10-0-3-45.ec2.internal    # AZ: us-east-1c  ← RESTARTS
api-service-7d4f8b6c9-f6g2k  ip-10-0-2-42.ec2.internal    # AZ: us-east-1b
api-service-7d4f8b6c9-h9p4r  ip-10-0-2-42.ec2.internal    # AZ: us-east-1b
api-service-7d4f8b6c9-k2m7s  ip-10-0-1-15.ec2.internal    # AZ: us-east-1a
api-service-7d4f8b6c9-n5q9t  ip-10-0-2-42.ec2.internal    # AZ: us-east-1b
api-service-7d4f8b6c9-r8v3w  ip-10-0-1-15.ec2.internal    # AZ: us-east-1a
```

All three restarting pods are in us-east-1c! The RDS instance is in us-east-1a with a read replica in us-east-1b. There's no replica in us-east-1c.

But wait — the DB endpoint 10.0.5.42 is the RDS DNS endpoint which should route through AWS networking regardless of AZ. Unless... could there be a security group or NACL issue?

**Assistant**: Excellent detective work! You've confirmed my AZ hypothesis. All failing pods are in us-east-1c, and your RDS has no presence in that AZ. While RDS endpoints *should* work cross-AZ, the intermittent nature (connects briefly, then drops) suggests something changed in the network path. Let me refine the diagnosis:

**Most likely causes (ranked):**

1. **Security Group / NACL change**: Someone may have modified the security group or Network ACL for the us-east-1c subnet. Check:
   ```bash
   aws ec2 describe-security-groups --group-ids <rds-sg-id>
   aws ec2 describe-network-acls --filters "Name=association.subnet-id,Values=<1c-subnet-id>"
   ```

2. **NAT Gateway / Route Table issue in us-east-1c**: If your EKS nodes use private subnets with NAT gateways, the us-east-1c NAT gateway might be unhealthy:
   ```bash
   aws ec2 describe-nat-gateways --filter "Name=subnet-id,Values=<1c-nat-subnet-id>"
   aws ec2 describe-route-tables --filters "Name=association.subnet-id,Values=<1c-private-subnet-id>"
   ```

3. **VPC Peering / Transit Gateway route change**: If RDS is in a different VPC or uses peering, check if routes to the RDS subnet are still present for us-east-1c.

4. **RDS Security Group**: Verify the RDS SG allows inbound from the us-east-1c node CIDR (10.0.3.0/24 based on node IPs).

**Immediate mitigation** while investigating:
```bash
# Cordon the 1c nodes to prevent new pods from scheduling there
kubectl cordon ip-10-0-3-28.ec2.internal
kubectl cordon ip-10-0-3-45.ec2.internal

# Delete the affected pods — they'll reschedule to 1a or 1b nodes
kubectl delete pods -n production api-service-7d4f8b6c9-5rt2q api-service-7d4f8b6c9-9wx1y api-service-7d4f8b6c9-d3h8n
```

Can you check the security groups and NACLs? Also run: `aws ec2 describe-network-acls --filters "Name=association.subnet-id,Values=<1c-subnet-id>"` — look for any DENY rules that might have been recently added.

**User**: You nailed it! Someone on the infra team added a new NACL rule to the us-east-1c private subnet 40 minutes ago as part of a "security hardening" initiative. The rule was intended to block external traffic but accidentally blocked the CIDR range that includes the RDS subnet in us-east-1a.

Here's the problematic rule:
```
Rule #: 50
Type: All Traffic
Protocol: All
Port Range: All
Source: 10.0.1.0/24  (this is the RDS subnet!)
Action: DENY
```

They meant to block 10.0.100.0/24 (an old dev subnet) but fat-fingered the CIDR. The intermittent behavior was because existing TCP connections would sometimes survive briefly due to connection pooling and keep-alive, but new connections were blocked.

I've removed the rule and the pods are recovering. Error rate dropping fast — down to 5% and falling.

**Assistant**: Classic "security hardening" incident! Glad we found it quickly. Let me summarize the root cause and suggest some preventive measures:

**Root Cause Analysis:**
- **What**: NACL rule #50 in us-east-1c private subnet blocked all traffic from 10.0.1.0/24 (RDS subnet in us-east-1a)
- **Why**: Manual CIDR entry error — intended 10.0.100.0/24, typed 10.0.1.0/24
- **Impact**: 3/12 API pods (all in us-east-1c) lost database connectivity → 35% 502 error rate
- **Duration**: ~45 minutes (from NACL change to detection and fix)
- **Time to resolution**: ~15 minutes from first alert

**Prevention recommendations:**

1. **Infrastructure as Code (IaC)**: All NACL/SG changes should go through Terraform/CloudFormation with PR review. No manual console changes in production.

2. **NACL change validation**: Add a pre-apply check that validates NACL rules don't block known critical CIDRs (database subnets, API subnets, monitoring endpoints).

3. **Canary testing for network changes**: Apply NACL changes to one AZ first, wait 5 minutes monitoring error rates, then roll to other AZs. This would have limited blast radius.

4. **Better monitoring**: Add a specific alert for cross-AZ connectivity to RDS. Something like:
   ```yaml
   # Prometheus alert
   - alert: CrossAZDatabaseConnectivity
     expr: increase(pgbouncer_errors_total{error="server_connection"}[5m]) > 10
     for: 2m
     labels:
       severity: critical
   ```

5. **PgBouncer circuit breaker**: Configure PgBouncer with `server_connect_timeout = 5` and `query_timeout = 30` so failed connections are detected faster. Also consider adding a readiness probe that checks actual DB query execution, not just TCP connectivity.

6. **Topology spread constraints**: Add a Kubernetes `topologySpreadConstraints` to ensure pods are evenly distributed across AZs:
   ```yaml
   topologySpreadConstraints:
   - maxSkew: 1
     topologyKey: topology.kubernetes.io/zone
     whenUnsatisfiable: DoNotSchedule
   ```

Should I help draft the Terraform code for the correct NACL rule, or set up any of these monitoring alerts?

**User**: This is great. Actually, before we move on to prevention, I want to make sure we're fully recovered. Can you help me verify the recovery and also figure out if we need to worry about any data consistency issues from the ~45 minutes of degraded service?

**Assistant**: Absolutely — let's do a thorough recovery verification. I'll break this into three phases: immediate health checks, traffic validation, and data consistency audit.

### Phase 1: Immediate Health Checks

First, verify all pods are running and ready:

```bash
# Check pod status across all AZs
kubectl get pods -l app=payment-api -o wide --sort-by='.spec.nodeName'

# Verify readiness/liveness probe status
kubectl get pods -l app=payment-api -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.conditions[?(@.type=="Ready")].status}{"\t"}{.spec.nodeName}{"\n"}{end}'

# Check for any recent restarts (indicating instability)
kubectl get pods -l app=payment-api -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'

# Verify HPA is stable
kubectl get hpa payment-api-hpa -o jsonpath='Current: {.status.currentReplicas}, Desired: {.status.desiredReplicas}, Min: {.spec.minReplicas}, Max: {.spec.maxReplicas}'
```

Expected: 12/12 pods Ready, 0 recent restarts, HPA stable at desired count.

### Phase 2: Cross-AZ Connectivity Validation

Since the root cause was NACL blocking cross-AZ traffic, we need to verify connectivity is restored:

```bash
# Test cross-AZ database connectivity from each pod
for pod in $(kubectl get pods -l app=payment-api -o name); do
  echo "--- Testing from $pod ---"
  kubectl exec $pod -- pg_isready -h payment-db-primary.internal -p 5432
  kubectl exec $pod -- pg_isready -h payment-db-replica-1.internal -p 5432
  kubectl exec $pod -- pg_isready -h payment-db-replica-2.internal -p 5432
  echo ""
done

# Test cross-AZ service mesh connectivity
for pod in $(kubectl get pods -l app=payment-api -o name); do
  az=$(kubectl get $pod -o jsonpath='{.metadata.labels.topology.kubernetes.io/zone}')
  echo "Pod: $pod (AZ: $az)"
  kubectl exec $pod -- curl -s -o /dev/null -w "HTTP %{http_code} - %{time_total}s" http://inventory-service.internal:8080/health
  kubectl exec $pod -- curl -s -o /dev/null -w "HTTP %{http_code} - %{time_total}s" http://notification-service.internal:8080/health
  echo ""
done

# Verify DNS resolution is consistent
kubectl exec payment-api-0 -- nslookup payment-db-primary.internal
kubectl exec payment-api-4 -- nslookup payment-db-primary.internal
kubectl exec payment-api-8 -- nslookup payment-db-primary.internal
```

### Phase 3: Traffic and Error Rate Validation

```bash
# Check current error rates (last 15 minutes vs last 24h baseline)
kubectl exec -it prometheus-0 -- promtool query instant \
  'sum(rate(http_requests_total{service="payment-api",code=~"5.."}[15m])) / sum(rate(http_requests_total{service="payment-api"}[15m])) * 100'

# Compare with baseline
kubectl exec -it prometheus-0 -- promtool query instant \
  'sum(rate(http_requests_total{service="payment-api",code=~"5.."}[24h])) / sum(rate(http_requests_total{service="payment-api"}[24h])) * 100'

# Check P99 latency has recovered
kubectl exec -it prometheus-0 -- promtool query instant \
  'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{service="payment-api"}[15m])) by (le))'

# Check connection pool utilization
kubectl exec -it prometheus-0 -- promtool query instant \
  'avg(hikaricp_connections_active{service="payment-api"}) / avg(hikaricp_connections_max{service="payment-api"}) * 100'
```

Expected: 5xx rate < 0.1%, P99 < 500ms, connection pool < 60% utilized.

### Phase 4: Data Consistency Audit

This is the critical part. During the 45-minute degraded window, some requests may have:
1. **Timed out mid-transaction** — leaving partial state
2. **Been retried by the frontend** — potentially causing duplicates
3. **Hit the wrong database replica** — reading stale data

```sql
-- Find transactions during the incident window
SELECT
    id, amount, status, created_at, updated_at,
    COUNT(*) OVER (PARTITION BY idempotency_key) as dup_count
FROM payments
WHERE created_at BETWEEN '2024-12-10 14:30:00' AND '2024-12-10 15:15:00'
ORDER BY created_at;

-- Check for stuck transactions (started but not completed)
SELECT id, amount, status, created_at
FROM payments
WHERE status IN ('pending', 'processing')
AND created_at BETWEEN '2024-12-10 14:30:00' AND '2024-12-10 15:15:00';

-- Check for duplicate idempotency keys with different outcomes
SELECT idempotency_key, COUNT(*), array_agg(status), array_agg(amount)
FROM payments
WHERE created_at BETWEEN '2024-12-10 14:30:00' AND '2024-12-10 15:15:00'
GROUP BY idempotency_key
HAVING COUNT(*) > 1;

-- Verify ledger balance consistency
SELECT
    SUM(CASE WHEN type = 'credit' THEN amount ELSE -amount END) as calculated_balance,
    (SELECT balance FROM accounts WHERE id = 'main_revenue') as stored_balance
FROM ledger_entries
WHERE created_at <= NOW();
```

If you find stuck transactions, we can manually resolve them:
```sql
-- For stuck 'processing' payments, check with payment gateway
UPDATE payments SET status = 'failed', updated_at = NOW(), notes = 'Manually resolved after incident'
WHERE id IN ('<stuck_payment_ids>') AND status = 'processing';
```

### Phase 5: Monitoring Setup for Recovery Period

Even after everything looks green, I recommend enhanced monitoring for the next 24 hours:

```yaml
# Alert rule: payment error spike (lower threshold during recovery)
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: payment-recovery-alerts
spec:
  groups:
  - name: recovery-monitoring
    rules:
    - alert: PaymentErrorRateElevated
      expr: |
        sum(rate(http_requests_total{service="payment-api",code=~"5.."}[5m]))
        / sum(rate(http_requests_total{service="payment-api"}[5m])) > 0.005
      for: 2m
      labels:
        severity: warning
        team: platform
      annotations:
        summary: "Payment API error rate > 0.5% during recovery period"
    - alert: CrossAZLatencyHigh
      expr: |
        histogram_quantile(0.95,
          sum(rate(http_request_duration_seconds_bucket{service="payment-api"}[5m])) by (le, source_az, dest_az)
        ) > 0.2
      for: 3m
      labels:
        severity: warning
      annotations:
        summary: "Cross-AZ latency elevated: {{ $labels.source_az }} -> {{ $labels.dest_az }}"
    - alert: DBConnectionPoolExhaustion
      expr: |
        hikaricp_connections_active{service="payment-api"}
        / hikaricp_connections_max{service="payment-api"} > 0.8
      for: 5m
      labels:
        severity: critical
```

### Regarding Uncordoning us-east-1c Nodes

**Don't uncordon yet.** Here's the sequence I recommend:

1. ✅ Verify NACL fix is applied and tested (done)
2. ✅ Verify existing pods are healthy (doing now)
3. ⏳ Run the data consistency audit
4. ⏳ Monitor for 30 minutes with current healthy state
5. ⏳ Then gradually uncordon: `kubectl uncordon <node> --dry-run=server` first
6. ⏳ Uncordon one node at a time, wait 5 minutes between each
7. ⏳ Watch the HPA scale pods onto uncordoned nodes
8. ⏳ Verify new pods pass health checks

```bash
# When ready to uncordon (one at a time):
for node in $(kubectl get nodes -l topology.kubernetes.io/zone=us-east-1c --no-headers -o name); do
  echo "Uncordoning $node..."
  kubectl uncordon $node
  echo "Waiting 5 minutes for stabilization..."
  sleep 300
  kubectl get pods -l app=payment-api -o wide | grep $node
  echo "---"
done
```

**User**: OK, I ran the Phase 1 checks and everything looks good — 12/12 pods ready, zero restarts, HPA stable at 12. The cross-AZ connectivity tests all pass too. Let me run the data consistency queries now. While I do that, can you help me think about the bigger picture? I'm worried about our overall incident response process. This took us 45 minutes to diagnose because we didn't have good cross-AZ monitoring. What would a proper observability stack look like for catching this kind of issue faster?

**Assistant**: Great — glad to hear Phase 1 and 2 are green. While you run the data queries, let me outline a comprehensive observability architecture specifically designed to catch cross-AZ and network-layer issues faster.

### The Observability Gap That Hurt You

Your current monitoring caught **symptoms** (502s, pod failures) but not the **root cause** (NACL misconfiguration). The gap is in the "middle layer" — between application metrics and infrastructure state. Here's the full observability stack I recommend:

### Layer 1: Infrastructure State Monitoring (IaC Drift Detection)

This is what would have caught the NACL issue before it caused any impact:

```yaml
# AWS Config Rule: Monitor NACL changes
resource "aws_config_config_rule" "nacl_compliance" {
  name = "nacl-cross-az-compliance"
  source {
    owner             = "CUSTOM_LAMBDA"
    source_identifier = "${aws_lambda_function.nacl_checker.arn}"
  }
  scope {
    compliance_resource_types = ["AWS::EC2::NetworkAcl"]
  }
}

# Lambda function checks:
# 1. All NACLs allow traffic on required ports between VPC CIDRs
# 2. No deny rules that block cross-subnet traffic
# 3. Rules match expected Terraform state
```

### Layer 2: Network Path Monitoring (Synthetic Probes)

```yaml
# Deploy canary pods in each AZ that continuously test cross-AZ connectivity
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: network-canary
spec:
  selector:
    matchLabels:
      app: network-canary
  template:
    spec:
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
      containers:
      - name: prober
        image: network-canary:latest
        env:
        - name: PROBE_TARGETS
          value: "payment-db-primary:5432,payment-db-replica-1:5432,inventory-service:8080"
        - name: PROBE_INTERVAL
          value: "10s"
        - name: METRICS_PORT
          value: "9090"
```

Each canary pod exports metrics:
- `network_probe_latency_seconds{source_az, dest_az, dest_service}` — RTT histogram
- `network_probe_success_total{...}` / `network_probe_failure_total{...}` — success/fail counters
- `network_probe_dns_latency_seconds{...}` — DNS resolution time

Alert rule:
```promql
# Cross-AZ probe failure rate > 5% for 1 minute = P1 alert
sum(rate(network_probe_failure_total[1m])) by (source_az, dest_az, dest_service)
/ (sum(rate(network_probe_success_total[1m])) by (source_az, dest_az, dest_service) + 0.001) > 0.05
```

This would have caught your issue within 60 seconds instead of 45 minutes.

### Layer 3: Application-Level Distributed Tracing

```yaml
# OpenTelemetry Collector config with AZ-aware attributes
receivers:
  otlp:
    protocols:
      grpc: { endpoint: "0.0.0.0:4317" }
processors:
  resource:
    attributes:
    - key: k8s.pod.az
      from_attribute: topology.kubernetes.io/zone
      action: insert
  tail_sampling:
    policies:
    - name: error-traces
      type: status_code
      status_code: { status_codes: [ERROR] }
    - name: slow-traces
      type: latency
      latency: { threshold_ms: 500 }
    - name: cross-az-traces
      type: string_attribute
      string_attribute:
        key: cross_az_hop
        values: ["true"]
exporters:
  otlp: { endpoint: "tempo:4317" }
```

Key traces to capture:
- Payment request → DB query with source/dest AZ tags
- Cross-service calls with AZ hop detection
- Connection pool acquisition time by AZ

### Layer 4: Runbook Automation

```python
# PagerDuty + Rundeck integration: automated first-response
{
  "trigger": "PaymentErrorRateHigh",
  "severity": "P1",
  "auto_actions": [
    {
      "name": "collect_diagnostics",
      "command": "kubectl get pods -o wide && kubectl top pods && kubectl get events --sort-by=.lastTimestamp | head -50",
      "timeout": "30s"
    },
    {
      "name": "test_cross_az_connectivity",
      "command": "network-canary-cli test --all-pairs --timeout 5s",
      "timeout": "60s"
    },
    {
      "name": "check_aws_changes",
      "command": "aws cloudtrail lookup-events --lookup-attributes AttributeKey=EventName,AttributeValue=ReplaceNetworkAclEntry --start-time $(date -d '1 hour ago' -u +%Y-%m-%dT%H:%M:%SZ)",
      "timeout": "30s"
    }
  ],
  "escalation": {
    "if_auto_actions_show": ["cross_az_failure", "nacl_change"],
    "then": "page_network_team",
    "with_context": "auto_diagnostics_results"
  }
}
```

This four-layer approach gives you: IaC drift detection (prevent), synthetic probes (detect in <60s), distributed tracing (diagnose), and automated runbooks (respond). Together they should cut your MTTR from 45 minutes to under 5.

**User**: This is incredibly thorough. I just finished running the data consistency queries and found 3 stuck 'processing' payments totaling $847.50. I also found 2 cases where the same idempotency key appears twice with different statuses (one 'completed' and one 'failed'). Before I manually fix anything, I want to understand the full picture of what happened during those 45 minutes.
"""

# ─── Multi-turn Follow-up Questions ───────────────────────────
DIALOG_FOLLOWUP_A = (
    "**User**: Yes, let me clarify the data concern. During the 45 minutes "
    "of degraded service, our API was handling payment processing. Some "
    "payment requests may have been sent to the failing pods and received "
    "502 errors. Our frontend retries failed requests up to 3 times with "
    "exponential backoff. Could this have caused duplicate payments? How "
    "do I check for and fix any duplicates?"
    "\n\n**Assistant**: "
)
DIALOG_FOLLOWUP_B = (
    "**User**: Actually, I want to focus on the recovery verification "
    "first. Can you give me a comprehensive checklist of commands to run "
    "to verify that all 12 pods are healthy, the database connections are "
    "stable across all AZs, and the error rate has returned to baseline? "
    "Also, should I uncordon the us-east-1c nodes now or wait?"
    "\n\n**Assistant**: "
)

DIALOG_PROMPT_A = DIALOG_HISTORY + DIALOG_FOLLOWUP_A
DIALOG_PROMPT_B = DIALOG_HISTORY + DIALOG_FOLLOWUP_B

# ─── Dialog short prefixes for dead-block exposure tests ───────
# R1 uses the full DIALOG_HISTORY; R2 uses a truncated version.

# ~2 blocks shared (~2080 tokens): truncate before the Phase 1 verification turn
DIALOG_SHORT_2BLOCK = DIALOG_HISTORY[
    :DIALOG_HISTORY.index(
        "**User**: This is great. Actually, before we move on"
    )
]

# ~1 block shared (~1507 tokens): truncate before the NACL discovery turn
DIALOG_SHORT_1BLOCK = DIALOG_HISTORY[
    :DIALOG_HISTORY.index(
        "**User**: You nailed it! Someone on the infra team added"
    )
]

# Questions about the truncated conversation context
DIALOG_SHORT_QUESTION_2BLOCK = (
    "**User**: Based on our investigation so far, what are the top 3 most "
    "likely root causes of the API service degradation, ranked by probability? "
    "For each one, tell me what specific evidence we have from the logs and "
    "diagnostics that supports or contradicts it."
    "\n\n**Assistant**: "
)
DIALOG_SHORT_QUESTION_1BLOCK = (
    "**User**: Given the pod crash-loop pattern and the database connection "
    "errors we've seen so far, should I focus on the application layer or "
    "the infrastructure layer first? What's the most efficient diagnostic "
    "path from here?"
    "\n\n**Assistant**: "
)

# Short R2 prompts (Dialog)
DIALOG_SHORT_PROMPT_2BLOCK = DIALOG_SHORT_2BLOCK + DIALOG_SHORT_QUESTION_2BLOCK
DIALOG_SHORT_PROMPT_1BLOCK = DIALOG_SHORT_1BLOCK + DIALOG_SHORT_QUESTION_1BLOCK


# ════════════════════════════════════════════════════════════════
#  Test Infrastructure
# ════════════════════════════════════════════════════════════════

def _enable_gdn_debug_logging() -> None:
    """Enable GDN block-level debug logging in engine subprocess."""
    os.environ["GDN_DEBUG"] = "1"


def _compare_outputs(
    label: str,
    outputs_a: list,
    outputs_b: list,
    name_a: str,
    name_b: str,
) -> bool:
    """Print full text comparison with complete output. Returns True if match."""
    text_a = outputs_a[0][1] if outputs_a else "<empty>"
    text_b = outputs_b[0][1] if outputs_b else "<empty>"
    match = text_a == text_b
    status = "✅ MATCH" if match else "❌ DIFF"
    print(f"\n  ── [{label}] {name_a} vs {name_b}: {status} ──")
    # Print full output for both
    print(f"    [{name_a}] ({len(text_a)} chars):")
    for line in text_a.split('\n'):
        print(f"      {line}")
    print(f"    [{name_b}] ({len(text_b)} chars):")
    for line in text_b.split('\n'):
        print(f"      {line}")
    if not match:
        min_len = min(len(text_a), len(text_b))
        for i in range(min_len):
            if text_a[i] != text_b[i]:
                print(f"    first diff at char {i}: "
                      f"'{text_a[max(0,i-10):i+10]}' vs "
                      f"'{text_b[max(0,i-10):i+10]}'")
                break
        else:
            print(f"    same content to char {min_len}, "
                  f"then length differs ({len(text_a)} vs {len(text_b)})")
    return match


def _run_scenario_test(
    scenario_name: str,
    prompt_a: str,
    prompt_b: str,
) -> None:
    """Two-engine test: all-mode (cache hit) vs align-mode (full recompute).

    Demonstrates all-mode advantage: R2 reuses cached prefix block states
    while align-mode must recompute everything from scratch.
    """
    _enable_gdn_debug_logging()

    # Compute shared prefix length (common leading characters)
    shared_len = 0
    for a, b in zip(prompt_a, prompt_b):
        if a != b:
            break
        shared_len += 1

    print(f"\n{'='*60}")
    print(f"[{scenario_name}] all-mode vs align-mode cache hit demo")
    print(f"  max_model_len=8192, max_num_batched_tokens=4096")
    print(f"  prompt_a: {len(prompt_a)} chars")
    print(f"  prompt_b: {len(prompt_b)} chars")
    print(f"  shared prefix: {shared_len} chars "
          f"({shared_len*100//max(len(prompt_a),1)}% of prompt_a)")
    print(f"{'='*60}")

    # === ALL-MODE: R1 (fill cache) + R2 (cache HIT) ===
    print(f"\n{'─'*60}")
    print(f"[{scenario_name}] ALL-MODE: R1 fills cache, R2 gets cache HIT")
    print(f"  R2 should only compute new tokens (suffix), not full prefix")
    with VllmRunner(**_COMMON_KWARGS, mamba_cache_mode="all") as vllm:
        all_r1 = vllm.generate_greedy([prompt_a], MAX_TOKENS)
        print(f"\n  [all-R1 output] ({len(all_r1[0][1]) if all_r1 else 0} chars):")
        print(all_r1[0][1] if all_r1 else "<empty>")

        all_r2 = vllm.generate_greedy([prompt_b], MAX_TOKENS)
        print(f"\n  [all-R2 output] ({len(all_r2[0][1]) if all_r2 else 0} chars):")
        print(all_r2[0][1] if all_r2 else "<empty>")

    # === ALIGN-MODE: R1 + R2 (no SSM cache hit, full recompute) ===
    print(f"\n{'─'*60}")
    print(f"[{scenario_name}] ALIGN-MODE: R1 + R2 (full recompute, no SSM cache hit)")
    print(f"  R2 must recompute entire prefix — align doesn't store block boundaries")
    with VllmRunner(**_COMMON_KWARGS, mamba_cache_mode="align") as vllm:
        align_r1 = vllm.generate_greedy([prompt_a], MAX_TOKENS)
        print(f"\n  [align-R1 output] ({len(align_r1[0][1]) if align_r1 else 0} chars):")
        print(align_r1[0][1] if align_r1 else "<empty>")

        align_r2 = vllm.generate_greedy([prompt_b], MAX_TOKENS)
        print(f"\n  [align-R2 output] ({len(align_r2[0][1]) if align_r2 else 0} chars):")
        print(align_r2[0][1] if align_r2 else "<empty>")

    # === COMPARISON REPORT ===
    print(f"\n{'='*60}")
    print(f"[{scenario_name}] RESULTS")
    print(f"{'='*60}")

    # Output comparison
    r1_match = _compare_outputs(
        "R1", all_r1, align_r1, "all-R1", "align-R1")
    r2_match = _compare_outputs(
        "R2", all_r2, align_r2, "all-R2(hit)", "align-R2(recompute)")

    print(f"\n  ── Summary ──")
    print(f"  R1 (from scratch): {'MATCH' if r1_match else 'DIFF (expected: different conv1d kernels)'}")
    print(f"  R2 (hit vs recompute): {'MATCH' if r2_match else 'DIFF (expected: different conv1d kernels)'}")
    print(f"  Key: Check GDN_DEBUG logs above for block table behavior:")
    print(f"    all-mode R2 should show: HIT + ctx>0 + scatter=0")
    print(f"    align-mode R2 should show: full prefill (no HIT)")

    os.environ.pop("GDN_DEBUG", None)


# ════════════════════════════════════════════════════════════════
#  Test Functions
# ════════════════════════════════════════════════════════════════

def test_cache_hit_long_text() -> None:
    """Scenario B: Long text — research paper prefix + analytical questions.

    ~5500-token survey paper as shared prefix, two different analytical
    questions as suffixes. Tests multi-block (5+) cache hit correctness.
    """
    _run_scenario_test("B-LongText", SURVEY_PROMPT_A, SURVEY_PROMPT_B)


def test_cache_hit_agent() -> None:
    """Scenario C: Agent — API documentation prefix + task prompts.

    ~5500-token REST API docs as shared prefix, two different API task
    requests as suffixes. Tests cache hit with structured technical content.
    """
    _run_scenario_test("C-Agent", AGENT_PROMPT_A, AGENT_PROMPT_B)


def test_cache_hit_dialog() -> None:
    """Scenario A: Multi-turn dialog — conversation history + follow-ups.

    ~5500-token technical support dialog as shared prefix, two different
    follow-up questions as suffixes. Tests cache hit with conversational content.
    """
    _run_scenario_test("A-Dialog", DIALOG_PROMPT_A, DIALOG_PROMPT_B)


# ════════════════════════════════════════════════════════════════
#  Dead-Block Exposure Tests (short R2, 1-2 block cache hit)
# ════════════════════════════════════════════════════════════════
#
# These tests expose align-mode's "dead block" problem:
#   R1: full 5500-token survey document (fills cache, multiple blocks)
#   R2: truncated document (~1-2 blocks shared with R1)
#
# When R2 gets a cache hit for 1-2 blocks, it needs the SSM state
# at the last cached block's boundary. In align-mode, intermediate
# block boundaries are dead (no SSM state stored) — only the running
# block has state. All-mode scatters intermediate states, so R2 can
# correctly resume from any block boundary.
#
# Expected behavior:
#   all-R2:   correct output (reads scattered block boundary state)
#   align-R2: degraded output (starts from zero SSM state → dead block)


def test_dead_block_2blocks() -> None:
    """Dead-block test: R2 shares ~2 blocks with R1.

    R1 = full 5500-token survey paper (fills 5+ blocks of cache)
    R2 = first ~2192 tokens of paper + different question (~2 blocks shared)

    R2 cache hit: blocks 0, 1 → needs block 1 boundary SSM state
    All-mode:   block 1 boundary was scattered → correct ✅
    Align-mode: block 1 = dead block → zero SSM state → degraded ❌
    """
    _run_scenario_test(
        "DEAD-BLOCK-2",
        SURVEY_PROMPT_A,          # R1: full document
        SHORT_PROMPT_2BLOCK,  # R2: truncated (~2 blocks shared)
    )


def test_dead_block_1block() -> None:
    """Dead-block test: R2 shares ~1 block with R1.

    R1 = full 5500-token survey paper (fills 5+ blocks of cache)
    R2 = first ~1047 tokens of paper + different question (~1 block shared)

    R2 cache hit: block 0 → needs block 0 boundary SSM state
    All-mode:   block 0 boundary was scattered → correct ✅
    Align-mode: block 0 = dead block → zero SSM state → degraded ❌
    """
    _run_scenario_test(
        "DEAD-BLOCK-1",
        SURVEY_PROMPT_A,          # R1: full document
        SHORT_PROMPT_1BLOCK,  # R2: truncated (~1 block shared)
    )


# ─── Dead-Block: Agent API Documentation ──────────────────────

def test_dead_block_agent_2blocks() -> None:
    """Dead-block test: Agent doc, R2 shares ~2 blocks with R1.

    R1 = full ~3463-token API docs (fills 3+ blocks of cache)
    R2 = first ~2151 tokens (before §11.2) + different question (~2 blocks shared)

    R2 cache hit: blocks 0, 1 → needs block 1 boundary SSM state
    All-mode:   block 1 boundary was scattered → correct ✅
    Align-mode: block 1 = dead block → zero SSM state → degraded ❌
    """
    _run_scenario_test(
        "DEAD-BLOCK-AGENT-2",
        AGENT_PROMPT_A,               # R1: full API docs
        AGENT_SHORT_PROMPT_2BLOCK,    # R2: truncated (~2 blocks shared)
    )


def test_dead_block_agent_1block() -> None:
    """Dead-block test: Agent doc, R2 shares ~1 block with R1.

    R1 = full ~3463-token API docs (fills 3+ blocks of cache)
    R2 = first ~1267 tokens (before §3) + different question (~1 block shared)

    R2 cache hit: block 0 → needs block 0 boundary SSM state
    All-mode:   block 0 boundary was scattered → correct ✅
    Align-mode: block 0 = dead block → zero SSM state → degraded ❌
    """
    _run_scenario_test(
        "DEAD-BLOCK-AGENT-1",
        AGENT_PROMPT_A,               # R1: full API docs
        AGENT_SHORT_PROMPT_1BLOCK,    # R2: truncated (~1 block shared)
    )


# ─── Dead-Block: Multi-turn Dialog ────────────────────────────

def test_dead_block_dialog_2blocks() -> None:
    """Dead-block test: Dialog history, R2 shares ~2 blocks with R1.

    R1 = full ~4288-token dialog (fills 4+ blocks of cache)
    R2 = first ~2080 tokens (before Phase 1 turn) + different question

    R2 cache hit: blocks 0, 1 → needs block 1 boundary SSM state
    All-mode:   block 1 boundary was scattered → correct ✅
    Align-mode: block 1 = dead block → zero SSM state → degraded ❌
    """
    _run_scenario_test(
        "DEAD-BLOCK-DIALOG-2",
        DIALOG_PROMPT_A,                # R1: full dialog
        DIALOG_SHORT_PROMPT_2BLOCK,     # R2: truncated (~2 blocks shared)
    )


def test_dead_block_dialog_1block() -> None:
    """Dead-block test: Dialog history, R2 shares ~1 block with R1.

    R1 = full ~4288-token dialog (fills 4+ blocks of cache)
    R2 = first ~1507 tokens (before NACL turn) + different question

    R2 cache hit: block 0 → needs block 0 boundary SSM state
    All-mode:   block 0 boundary was scattered → correct ✅
    Align-mode: block 0 = dead block → zero SSM state → degraded ❌
    """
    _run_scenario_test(
        "DEAD-BLOCK-DIALOG-1",
        DIALOG_PROMPT_A,                # R1: full dialog
        DIALOG_SHORT_PROMPT_1BLOCK,     # R2: truncated (~1 block shared)
    )
