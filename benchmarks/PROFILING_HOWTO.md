# Prefix Cache Profiling 操作手册

## 启动 Server

```bash
# ALL mode
python3 -m vllm.entrypoints.openai.api_server \
  --model /data/Qwen3.5-9B \
  --port 8100 --enforce-eager \
  --enable-prefix-caching --mamba-cache-mode all \
  --max-model-len 8192 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.9 \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./prof_all", "torch_profiler_with_stack": false}'

# ALIGN mode (对比用，换终端)
python3 -m vllm.entrypoints.openai.api_server \
  --model /data/Qwen3.5-9B \
  --port 8100 --enforce-eager \
  --enable-prefix-caching --mamba-cache-mode align \
  --max-model-len 8192 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.9 \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./prof_align", "torch_profiler_with_stack": false}'
```

## Profiling 三步 curl

```bash
# ━━━━ Step 1: 开始采集 ━━━━
curl -X POST http://localhost:8100/start_profile

# ━━━━ Step 2: 发请求 (R1 填 cache, R2 cache hit) ━━━━
# prompt_a.json / prompt_b.json: ~2000 token 共享前缀, max_tokens=1
curl -s http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @benchmarks/prompt_a.json

curl -s http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @benchmarks/prompt_b.json

# (可重复多轮 R1+R2)

# ━━━━ Step 3: 停止采集 ━━━━
curl -X POST http://localhost:8100/stop_profile
```

## 分析数据

```bash
# 等 1-2 分钟让 profiler flush 完成，然后:
python3 -c "
from torch_npu.profiler.profiler import analyse
analyse('./prof_all/*_ascend_pt/')
"

# 分析完的结果在:
ls ./prof_all/*_ascend_pt/ASCEND_PROFILER_OUTPUT/
# → op_statistic.csv, operator_details.csv, kernel_details.csv, trace_view.json
```

## 查看 Timeline

```bash
# 下载 trace_view.json 到本地，用以下工具打开:
# - MindStudio Insight (推荐)
# - chrome://tracing
# - https://ui.perfetto.dev/
```
