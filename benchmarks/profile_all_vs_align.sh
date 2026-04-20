#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ALL vs ALIGN profiling — 一键完成两轮采集 + analyse
# 用法: bash benchmarks/profile_all_vs_align.sh [--model PATH]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set -e

MODEL="/data/Qwen3.5-9B"
PORT=8100
FLUSH_WAIT=120
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

BASE_URL="http://localhost:${PORT}"

info()  { echo -e "\033[1;32m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

wait_for_server() {
    info "等待 server 启动..."
    for i in $(seq 1 120); do
        if curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
            info "Server 就绪! (${i}s)"
            return 0
        fi
        sleep 1
    done
    error "Server 120s 未启动"
}

run_profiling() {
    local mode=$1
    info "━━━ 预热 (${mode}) ━━━"
    curl -s "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":1,\"temperature\":0}" \
        > /dev/null
    sleep 2

    info "━━━ start_profile ━━━"
    curl -s -X POST "${BASE_URL}/start_profile"
    echo ""

    info "━━━ R1: 填充缓存 ━━━"
    curl -s "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d @"${SCRIPT_DIR}/prompt_a.json"
    echo ""

    info "━━━ R2: cache hit ━━━"
    curl -s "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d @"${SCRIPT_DIR}/prompt_b.json"
    echo ""

    info "━━━ stop_profile ━━━"
    curl -s -X POST "${BASE_URL}/stop_profile"
    echo ""
}

start_server() {
    local mode=$1
    local prof_dir="./prof_${mode}"
    rm -rf "${prof_dir}"

    info "启动 ${mode^^}-mode server (model=${MODEL})"
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --port "${PORT}" \
        --enforce-eager \
        --enable-prefix-caching \
        --mamba-cache-mode "${mode}" \
        --max-model-len 8192 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.9 \
        --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${prof_dir}\",\"torch_profiler_with_stack\":false}" \
        > "${prof_dir}.server.log" 2>&1 &
    SERVER_PID=$!
    info "Server PID: ${SERVER_PID}"
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        info "停止 server (PID=${SERVER_PID})"
        kill "${SERVER_PID}"
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
    sleep 3
}

# ━━━━━━━━━━━━━ MAIN ━━━━━━━━━━━━━

info "Model: ${MODEL}"
info "Port:  ${PORT}"
echo ""

# ━━━ Round 1: ALL mode ━━━
info "══════════════════════════════════════"
info "  ROUND 1: ALL MODE"
info "══════════════════════════════════════"
start_server "all"
wait_for_server
run_profiling "all"
info "等待 flush (${FLUSH_WAIT}s)..."
sleep "${FLUSH_WAIT}"
stop_server

# ━━━ Round 2: ALIGN mode ━━━
info "══════════════════════════════════════"
info "  ROUND 2: ALIGN MODE"
info "══════════════════════════════════════"
start_server "align"
wait_for_server
run_profiling "align"
info "等待 flush (${FLUSH_WAIT}s)..."
sleep "${FLUSH_WAIT}"
stop_server

# ━━━ Analyse ━━━
info "══════════════════════════════════════"
info "  ANALYSE"
info "══════════════════════════════════════"
python3 -c "
import glob, os
from torch_npu.profiler.profiler import analyse

for mode in ['all', 'align']:
    dirs = glob.glob(f'./prof_{mode}/*_ascend_pt/')
    for d in dirs:
        print(f'Analysing {d}...')
        analyse(d)
        output = os.path.join(d, 'ASCEND_PROFILER_OUTPUT')
        if os.path.isdir(output):
            files = os.listdir(output)
            print(f'  Output ({len(files)} files): {files}')
print('Done!')
"

info ""
info "══════════════════════════════════════"
info "  完成!"
info "══════════════════════════════════════"
info "ALL  数据: ./prof_all/"
info "ALIGN数据: ./prof_align/"
info ""
info "对比 CSV:"
info "  diff <(sort prof_all/*_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv) \\"
info "       <(sort prof_align/*_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv)"
