#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# profiling 采集脚本 — 假设 server 已启动在 :8100
# 用法: bash benchmarks/run_profiling.sh
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set -e
PORT=8100
BASE_URL="http://localhost:${PORT}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== [1/4] 预热 (不采集) ==="
curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/Qwen3.5-9B","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"temperature":0}' \
  | python3 -m json.tool | head -5
echo ""

echo "=== [2/4] 开始 profiling ==="
curl -s -X POST "${BASE_URL}/start_profile"
echo ""

echo "=== [3/4] 发请求: R1(填 cache) + R2(cache hit) ==="
echo "--- R1 ---"
curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"${SCRIPT_DIR}/prompt_a.json" \
  | python3 -m json.tool | head -10
echo ""

echo "--- R2 ---"
curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"${SCRIPT_DIR}/prompt_b.json" \
  | python3 -m json.tool | head -10
echo ""

echo "=== [4/4] 停止 profiling ==="
curl -s -X POST "${BASE_URL}/stop_profile"
echo ""

echo "=== 采集完成! ==="
echo "等待 flush (120s)..."
sleep 120

echo "=== 运行 analyse ==="
python3 -c "
import glob, os
from torch_npu.profiler.profiler import analyse
dirs = glob.glob('./prof_*/*_ascend_pt/')
for d in dirs:
    print(f'Analysing {d}...')
    analyse(d)
    output = os.path.join(d, 'ASCEND_PROFILER_OUTPUT')
    if os.path.isdir(output):
        print(f'  Output: {os.listdir(output)}')
print('Done!')
"
