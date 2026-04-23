#!/bin/bash
# Layer 2 & 3 Benchmark: all-mode vs align-mode prefix caching
#
# Usage:
#   # Layer 2: Online serving benchmark (requires vllm serve)
#   bash benchmarks/bench_prefix_cache_serving.sh layer2
#
#   # Layer 3: Offline prefix caching benchmark
#   bash benchmarks/bench_prefix_cache_serving.sh layer3
#
#   # Both layers
#   bash benchmarks/bench_prefix_cache_serving.sh all
#
#   # Fair comparison (Triton conv1d for both modes)
#   GDN_ALIGN_TRITON_CONV1D=1 bash benchmarks/bench_prefix_cache_serving.sh all

set -euo pipefail

MODEL="${MODEL:-/data/Qwen3.5-9B}"
LAYER="${1:-all}"
RESULTS_DIR="benchmarks/results/prefix_cache_$(date +%Y%m%d_%H%M%S)"
PORT=8100
NUM_PROMPTS=100
PREFIX_LEN=2048       # ~2 blocks @ block_size=1024
SUFFIX_LEN=256
NUM_PREFIXES=10       # 10 unique prefixes, each repeated NUM_PROMPTS/10 times
OUTPUT_LEN=10
REQUEST_RATE=inf      # max throughput

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "  Prefix Cache Benchmark"
echo "  Model: $MODEL"
echo "  Results: $RESULTS_DIR"
echo "  GDN_ALIGN_TRITON_CONV1D=${GDN_ALIGN_TRITON_CONV1D:-not set}"
echo "=============================================="

# ─── Helper functions ───────────────────────────────────

start_server() {
    local mode=$1
    echo ""
    echo ">>> Starting vllm server (mamba_cache_mode=$mode) on port $PORT..."
    vllm serve "$MODEL" \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --enforce-eager \
        --gpu-memory-utilization 0.7 \
        --enable-prefix-caching \
        --max-model-len 8192 \
        --max-num-batched-tokens 4096 \
        --override-generation-config '{"mamba_cache_mode": "'$mode'"}' \
        --disable-log-requests \
        --disable-log-stats \
        > "$RESULTS_DIR/server_${mode}.log" 2>&1 &
    SERVER_PID=$!
    echo "    PID=$SERVER_PID, waiting for ready..."

    # Wait for server to be ready (up to 300s)
    for i in $(seq 1 60); do
        if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
            echo "    Server ready after ${i}*5s"
            return 0
        fi
        sleep 5
    done
    echo "    ERROR: Server failed to start"
    kill $SERVER_PID 2>/dev/null || true
    return 1
}

stop_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo ">>> Stopping server (PID=$SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        unset SERVER_PID
        sleep 5
    fi
}

run_layer2() {
    local mode=$1
    local output_file="$RESULTS_DIR/layer2_${mode}.json"

    echo ""
    echo "─── Layer 2: Online serving benchmark ($mode) ───"

    start_server "$mode"

    echo ">>> Running vllm bench serve (prefix_repetition dataset)..."
    vllm bench serve \
        --backend openai \
        --base-url "http://localhost:$PORT" \
        --model "$MODEL" \
        --dataset-name prefix_repetition \
        --prefix-repetition-prefix-len "$PREFIX_LEN" \
        --prefix-repetition-suffix-len "$SUFFIX_LEN" \
        --prefix-repetition-num-prefixes "$NUM_PREFIXES" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$REQUEST_RATE" \
        --output-len "$OUTPUT_LEN" \
        --save-result \
        --result-dir "$RESULTS_DIR" \
        --result-filename "layer2_${mode}.json" \
        2>&1 | tee "$RESULTS_DIR/layer2_${mode}_output.txt"

    stop_server
    echo "    Results saved to $output_file"
}

run_layer3() {
    local mode=$1
    local output_file="$RESULTS_DIR/layer3_${mode}.txt"

    echo ""
    echo "─── Layer 3: Offline prefix caching benchmark ($mode) ───"

    python -m vllm.benchmarks.benchmark_prefix_caching \
        --model "$MODEL" \
        --tensor-parallel-size 1 \
        --enforce-eager \
        --gpu-memory-utilization 0.7 \
        --enable-prefix-caching \
        --max-model-len 8192 \
        --max-num-batched-tokens 4096 \
        --num-prompts 20 \
        --repeat-count 5 \
        --input-length-range "1024:3072" \
        --output-len "$OUTPUT_LEN" \
        --prefix-len "$PREFIX_LEN" \
        2>&1 | tee "$output_file"

    echo "    Results saved to $output_file"
}

# ─── Main ───────────────────────────────────────────────

trap 'stop_server' EXIT

case "$LAYER" in
    layer2)
        run_layer2 "all"
        run_layer2 "align"
        ;;
    layer3)
        run_layer3 "all"
        run_layer3 "align"
        ;;
    all)
        # Layer 2
        run_layer2 "all"
        run_layer2 "align"
        # Layer 3
        run_layer3 "all"
        run_layer3 "align"
        ;;
    *)
        echo "Usage: $0 {layer2|layer3|all}"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "  All benchmarks complete!"
echo "  Results in: $RESULTS_DIR"
echo "=============================================="
ls -la "$RESULTS_DIR/"
