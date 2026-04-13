# Full Benchmark Auto-Test Script Design

## Overview

A single-command benchmark script (`run_full_benchmark.py`) that fully automatically tests a remote llama-server (DeepSeek-V3.2-Q4_K_M on 8×A100) across 8 test groups, producing a delivery-grade comparison report.

**8 test groups** = `{thinking, non-thinking}` × `{openqa, longalpaca}` × `{evalscope, native}`

## Target Environment

- Hardware: 8× NVIDIA A100 (80GB each)
- OS: Ubuntu 24.04
- Server: llama-server serving DeepSeek-V3.2-Q4_K_M.gguf on `0.0.0.0:8080`
- Model: DeepSeek-V3.2 (671B MoE, Q4_K_M quantization)

## CLI Interface

```bash
# Full auto — one command, zero config
python3 run_full_benchmark.py --host 10.0.0.1

# With overrides
python3 run_full_benchmark.py --host 10.0.0.1 --port 8080 --max-concurrency 4

# Dry run — show plan, don't execute
python3 run_full_benchmark.py --host 10.0.0.1 --dry-run

# Skip evalscope groups (if not installed)
python3 run_full_benchmark.py --host 10.0.0.1 --native-only
```

Parameters:
- `--host` (required): target machine IP
- `--port` (default 8080): llama-server port
- `--max-concurrency` (optional): override auto-detected parallel slots limit
- `--dry-run`: generate plan and configs only, do not execute
- `--native-only`: skip evalscope groups (run 4 groups instead of 8)
- `--output-dir` (default `./results/benchmark-{timestamp}`): output directory

## Phase 0: Pre-flight

### 0.1 Health Check

Single GET to `/health`. Fail → exit immediately.

### 0.2 Detect Parallel Slots

Strategy (in order, first success wins):

1. **`/slots` endpoint**: If llama-server was started with `--slots`, returns JSON array. `S = len(array)`. Done.

2. **Timing-based detection** (if `/slots` unavailable):

   ```python
   # 测单请求基线延迟
   T = measure_single_request_latency()
   S = 1  # 已知至少 1 个 slot

   for n in [2, 4, 8]:
       W = measure_concurrent_requests(n, prompt="Say OK", max_tokens=8)
       # 如果 n 个请求的 wall time 接近 T，说明它们真的在并行
       # 如果 wall time 接近 T × (n / S)，说明多出来的请求在排队
       expected_if_parallel = T * 1.5      # 允许 50% 波动
       expected_if_queuing  = T * (n / S)  # 基于当前已知 S 推算排队时间
       if W < (expected_if_parallel + expected_if_queuing) / 2:
           S = n  # 这一轮确实是并行的
       else:
           break  # 明显在排队了，停止探测
   ```

3. **Fallback**: If detection is inconclusive, default `S = 1` (safest).

If `--max-concurrency` is provided, `S = min(detected, max_concurrency)`.

### 0.3 Detect Thinking Mode

Send a test request with `extra_args: {chat_template_kwargs: {thinking: true}}` and a prompt "Think about what 2+3 equals".

- If response contains `<think>` tag → thinking via `chat_template_kwargs` works. Record `thinking_method = "chat_template_kwargs"`.
- If not → log warning, skip thinking groups (run 4 groups instead of 8).

Note: system prompt 方式（如 "Think step by step"）不可靠——无法区分模型自发输出 `<think>` 和真正进入 thinking mode，因此不采用。

### 0.4 Check Datasets

- `openqa`: check `datasets/openqa.jsonl` exists. If not, use built-in fallback prompts (already in `dataset.py`).
- `longalpaca`: check `datasets/longalpaca.jsonl` exists. If not, run `datasets/download_longalpaca.py` to download.

### 0.5 Check Evalscope

- Run `evalscope --version` or `python3 -m evalscope --version`.
- If unavailable → log warning, add `--native-only` flag (run 4 groups instead of 8).

### 0.6 Print Pre-flight Summary

```
Pre-flight Results
────────────────────────────────────────
  Server        http://10.0.0.1:8080  OK
  Parallel Slots  4 (timing-based detection)
  Thinking Mode   chat_template_kwargs  OK
  Dataset openqa  OK (5 prompts, fallback)
  Dataset longalpaca  OK (1200 prompts)
  Evalscope       OK (v0.6.2)
```

## Phase 1: Generate Test Plan

### Concurrency / Requests Calculation

Given detected `S` parallel slots:

```
if S == 1:
    concurrency = [1]
    requests    = [10]
elif S <= 4:
    concurrency = [1, S]
    requests    = [5, S * 5]
else:
    concurrency = [1, S // 2, S]
    requests    = [5, S * 3, S * 5]
```

All 8 groups use the same concurrency/requests for fair comparison.

### 8 Test Groups

| Group | Thinking | Dataset | Engine |
|-------|----------|---------|--------|
| 1 | off | openqa | native |
| 2 | off | openqa | evalscope |
| 3 | off | longalpaca | native |
| 4 | off | longalpaca | evalscope |
| 5 | on | openqa | native |
| 6 | on | openqa | evalscope |
| 7 | on | longalpaca | native |
| 8 | on | longalpaca | evalscope |

### Config Generation

For each group, generate a YAML config programmatically (in memory, not written to disk unless `--dry-run`).

Thinking mode config (both engines):
- `extra_args: {chat_template_kwargs: {thinking: true}}`

Engine-specific config:
- **native engine**: 使用现有 `Orchestrator` + `NativeEngine`，传入 `EngineConfig`，Orchestrator 自动处理多 concurrency level 的逐级执行。
- **evalscope engine**: 使用现有 `EvalScopeEngine`，内部调用 `evalscope perf --parallel N --number M --dataset D --stream`。evalscope 的 `--parallel` 对应 concurrency，`--number` 对应 requests。每个 concurrency level 独立调用一次 evalscope CLI。结果通过 `EvalScopeEngine._parse_output()` 映射到统一的 `LevelResult` / `AggregatedMetrics` 结构，与 native engine 输出格式一致，确保横向可比。

### Print Test Plan

```
Test Plan (8 groups)
═══════════════════════════════════════════════════════════
  Concurrency levels: 1 → 4 → 8
  Requests per level: 5 → 24 → 40
  Total requests per group: 69
  Total requests all groups: 552
  Estimated time: ~25 min

  #  Thinking  Dataset      Engine      Requests
  ─────────────────────────────────────────────────
  1  off       openqa       native      69
  2  off       openqa       evalscope   69
  3  off       longalpaca   native      69
  4  off       longalpaca   evalscope   69
  5  on        openqa       native      69
  6  on        openqa       evalscope   69
  7  on        longalpaca   native      69
  8  on        longalpaca   evalscope   69

  Proceed? [Y/n]
```

With `--dry-run`, print and exit.

## Phase 2: Execute 8 Groups

### Execution Loop

```python
for group in groups:
    print_group_header(group)
    try:
        result = run_single_group(group, timeout=600)
        save_group_result(group, result)
        print_group_summary(result)
    except Exception as e:
        record_group_failure(group, e)
        print_group_failure(e)
    sleep(cooldown_seconds)  # 5-10s between groups
```

### Per-Group Execution

Each group invokes the existing `Orchestrator` (for native engine) or `EvalScopeEngine` directly, using the generated config. Results are saved to:

```
results/benchmark-2026-04-13T12-00-00/
├── group_1_off_openqa_native/
│   ├── summary.json
│   ├── raw/
│   └── ...
├── group_2_off_openqa_evalscope/
│   └── ...
├── ...
├── benchmark_summary.json     ← all 8 groups aggregated
├── benchmark_report.md        ← Markdown comparison report
└── benchmark_report.html      ← HTML comparison report
```

### Terminal Output Per Group

```
[3/8] non-thinking × longalpaca × native
════════════════════════════════════════════
  [1/3] concurrency=1  requests=5
  ████████████████████  5/5   OK
  Result:  PASS   TTFT=1.23s  Tok/s=12.5

  [2/3] concurrency=4  requests=24
  ████████████████████ 24/24  OK
  Result:  PASS   TTFT=3.45s  Tok/s=38.2

  [3/3] concurrency=8  requests=40
  ████████████████████ 40/40  OK
  Result:  PASS   TTFT=8.12s  Tok/s=42.1

  Group 3 complete: PASS  Max concurrency=8  Duration=3m12s
  Cooling down 5s ...
```

### Safety Mechanisms

- **Per-level timeout**: 每个 concurrency level 独立超时，`timeout = max(300, requests_count × 120)` 秒。thinking + longalpaca 场景单请求可能 30-60s，此公式确保有足够余量。
- **Per-group timeout**: 所有 level 累计上限 20 分钟，超时杀掉跳下一组。
- **Cooldown**: 5 seconds between groups for KV cache release.
- **Orchestrator circuit breaker**: Existing 3× consecutive 5xx and 80% failure rate abort.
- **Concurrency cap**: Never exceed detected `S` slots.
- **Group isolation**: One group's failure does not stop subsequent groups.

## Phase 3: Comparison Report

### benchmark_summary.json

```json
{
  "meta": {
    "host": "10.0.0.1:8080",
    "model": "DeepSeek-V3.2-Q4_K_M",
    "parallel_slots": 8,
    "timestamp": "2026-04-13T12:00:00Z"
  },
  "groups": [
    {
      "id": 1,
      "thinking": false,
      "dataset": "openqa",
      "engine": "native",
      "status": "completed",
      "levels": [...],
      "best_concurrency": 8,
      "metrics_at_best": {
        "success_rate": 1.0,
        "gen_toks_per_sec": 42.1,
        "avg_ttft": 8.12,
        "avg_tpot": 0.11,
        "p50_latency": 9.5,
        "p99_latency": 15.2
      }
    },
    ...
  ]
}
```

### Markdown Report (benchmark_report.md)

```markdown
# DeepSeek-V3.2-Q4_K_M Benchmark Report

## Environment
| Item | Value |
| ... | ... |

## Overall Comparison (8 groups at max concurrency)
| Group | Thinking | Dataset | Engine | Tok/s | TTFT | P99 | Result |
| ... |

## Dimension Analysis

### Thinking vs Non-Thinking
| Metric | Non-Thinking | Thinking | Delta |
(averaged across dataset/engine, excluding status != completed)

### OpenQA vs LongAlpaca
| Metric | OpenQA | LongAlpaca | Delta |
(averaged across thinking/engine, excluding status != completed)

### Native vs Evalscope
| Metric | Native | Evalscope | Delta |
(averaged across thinking/dataset, excluding status != completed)

聚合策略：仅对 `status=completed` 的 group 计算平均值。如果某维度对比的两侧都无 completed group，该维度显示 "N/A — insufficient data"。

## Per-Group Detail
(full metrics table for each group)

## Conclusion
- Recommended max concurrency: X
- Bottleneck: ...
- Notes: ...
```

### HTML Report (benchmark_report.html)

Same content as Markdown, rendered as a styled HTML page (extending the existing `report/html.py` pattern). Suitable for sharing with clients.

## File Structure

New files:
- `run_full_benchmark.py` — main entry point (~400-500 lines)
- `src/llm_stress_test/benchmark.py` — core benchmark logic (probe, plan, execute, report)

Modified files:
- None. The script composes existing modules (`Orchestrator`, `NativeEngine`, `EvalScopeEngine`, `dataset`, `metrics`) without modifying them.

## Pass Criteria

Same criteria for all 8 groups (ensures comparability):

```yaml
pass_criteria:
  - metric: success_rate
    operator: ">="
    threshold: 0.95
  - metric: avg_ttft
    operator: "<="
    threshold: 30.0
```

## Edge Cases

| Situation | Behavior |
|---|---|
| evalscope not installed | Skip 4 evalscope groups, run 4 native groups, note in report |
| thinking mode not detected | Skip 4 thinking groups, run 4 non-thinking groups, note in report |
| longalpaca download fails | Use openqa for all groups, note in report |
| Server goes down mid-test | Current group fails, cooldown, try next group |
| All groups fail | Report shows all failures with error messages |
| detected S=1 (single slot) | All groups run with concurrency=[1], requests=[10] |
