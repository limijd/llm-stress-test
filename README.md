# LLM Stress Test

LLM 推理服务压力测试工具。支持 OpenAI 兼容 API，双压测内核（evalscope + 自研），自动多并发梯度测试、通过判定、降级探测，一键生成图表和 HTML 报告。

## 功能特性

- **双引擎**：evalscope（行业标准）和自研 asyncio 引擎，按需切换
- **自动降级**：目标并发不通过时，自动逐步降低并发找到最大通过值
- **可配置判定条件**：Success Rate、吞吐量、TTFT 等指标，阈值自由设定
- **完整报告**：JSON/CSV 原始数据 + matplotlib 图表 + HTML 汇总报告
- **密钥脱敏**：所有输出产物自动屏蔽 API Key
- **GUI 配置编辑器**：Tkinter 桌面应用，可视化编辑配置文件
- **支持所有 OpenAI 兼容 API**：DeepSeek、Qwen、GLM、vLLM 等

## 安装

### 环境要求

- Python >= 3.11
- pip

### 安装步骤

```bash
git clone https://github.com/yourname/llm-stress-test.git
cd llm-stress-test

# 安装（含开发依赖）
pip3 install -e ".[dev]"

# 如果需要使用 evalscope 引擎
pip3 install -e ".[evalscope]"
```

安装完成后会注册两个命令行工具：
- `llm-stress-test` — 压测 CLI
- `llm-stress-config-gui` — GUI 配置编辑器

## 快速开始

### 1. 准备配置文件

复制示例配置并修改：

```bash
cp config/example.yaml config/my_test.yaml
```

最小可用配置：

```yaml
target:
  name: "DeepSeek-V3.2-Exp"
  api_url: "https://your-api-endpoint/v1/chat/completions"
  api_key: "${LLM_API_KEY}"    # 从环境变量读取，避免明文
  model: "DeepSeek-V3.2-Exp"

engine: "native"               # "native" 或 "evalscope"

test:
  concurrency: [1, 5, 10, 20, 50]
  requests_per_level: [10, 50, 100, 200, 500]
  dataset: "openqa"

pass_criteria:
  - metric: "success_rate"
    operator: ">="
    threshold: 1.0
  - metric: "gen_toks_per_sec"
    operator: ">="
    threshold: 500
  - metric: "avg_ttft"
    operator: "<="
    threshold: 10.0
```

### 2. 设置 API Key

```bash
export LLM_API_KEY="sk-your-api-key"
```

### 3. 校验配置

```bash
llm-stress-test validate --config config/my_test.yaml
```

### 4. 执行测试

```bash
llm-stress-test run --config config/my_test.yaml
```

终端输出示例：

```
[Engine: native] aiohttp 3.9.5

[1/5] 并发=1  Success Rate: 100.0%  Gen toks/s: 45.2  Avg TTFT: 1.2s  ✓ PASS
[2/5] 并发=5  Success Rate: 100.0%  Gen toks/s: 221.5  Avg TTFT: 2.1s  ✓ PASS
[3/5] 并发=10  Success Rate: 100.0%  Gen toks/s: 438.7  Avg TTFT: 3.5s  ✓ PASS
[4/5] 并发=20  Success Rate: 100.0%  Gen toks/s: 612.3  Avg TTFT: 5.8s  ✓ PASS
[5/5] 并发=50  Success Rate: 98.2%  Gen toks/s: 489.1  Avg TTFT: 12.3s  ✗ FAIL
  └ success_rate: 0.982 < 1.0
  └ gen_toks_per_sec: 489.1 < 500
  └ avg_ttft: 12.3 > 10.0

⚠ 目标并发(50)未通过，启动自动降级...
[降级] 并发=40  Success Rate: 100.0%  Gen toks/s: 578.4  Avg TTFT: 7.2s  ✗ FAIL
[降级] 并发=30  Success Rate: 100.0%  Gen toks/s: 531.2  Avg TTFT: 5.1s  ✓ PASS

============================================================
结论: 最大通过并发数 = 30
建议: 要求赛事主方增加 GPU Server 数，或调整参赛队伍数量
报告已生成: results/2026-04-10T14-30-00_DeepSeek-V3.2-Exp_native/
============================================================
```

### 5. 查看报告

测试完成后在 `results/` 目录下生成完整报告：

```
results/2026-04-10T14-30-00_DeepSeek-V3.2-Exp_native/
├── config_snapshot.yaml          # 测试配置快照（API Key 已脱敏）
├── summary.json                  # 聚合指标 + 判定结果
├── summary.csv                   # 表格形式的摘要
├── report.html                   # HTML 汇总报告（浏览器打开即可）
├── raw/
│   ├── level_01_c1.json          # 每个并发级别的逐请求原始数据
│   ├── level_02_c5.json
│   └── ...
└── charts/
    ├── throughput.png            # 吞吐量 vs 并发数
    ├── ttft.png                  # TTFT vs 并发数
    ├── latency_p50_p99.png       # P50/P99 延迟 vs 并发数
    └── success_rate.png          # 成功率 vs 并发数
```

用浏览器打开 `report.html` 即可看到完整的可视化报告。

## CLI 命令详解

### `llm-stress-test run` — 执行压测

```bash
llm-stress-test run --config config/my_test.yaml [选项]
```

| 选项 | 说明 | 示例 |
|------|------|------|
| `--config` | 配置文件路径（必填） | `--config config/deepseek.yaml` |
| `--engine` | 覆盖引擎选择 | `--engine native` |
| `--concurrency` | 覆盖并发梯度（逗号分隔） | `--concurrency 1,5,10` |
| `--api-url` | 覆盖 API 地址 | `--api-url https://...` |
| `--model` | 覆盖模型名 | `--model gpt-4` |
| `--dataset` | 覆盖数据集 | `--dataset longalpaca` |
| `-v` / `-vv` | 日志详细程度（INFO / DEBUG） | `-vv` |

CLI 参数优先级高于配置文件，配置文件优先级高于内置默认值。

### `llm-stress-test validate` — 校验配置

```bash
llm-stress-test validate --config config/my_test.yaml
```

检查配置文件的语法、必填字段、数组长度一致性等，不执行测试。

### `llm-stress-test report` — 重新生成报告

```bash
llm-stress-test report --result-dir results/2026-04-10_xxx/ --formats html,csv
```

从已有的 `summary.json` 重新生成报告，适用于修改报告模板后需要重新渲染的场景。

## 配置文件完整参考

```yaml
# ===== 测试目标 =====
target:
  name: "DeepSeek-V3.2-Exp"                          # 测试名称，用于报告标题和结果目录命名
  api_url: "https://llmapi.paratera.com/v1/chat/completions"  # OpenAI 兼容 API 端点
  api_key: "${LLM_API_KEY}"                           # API Key，支持 ${ENV_VAR} 环境变量引用
  model: "DeepSeek-V3.2-Exp"                          # 模型名称

# ===== 引擎选择 =====
engine: "evalscope"    # "evalscope" — 调用 evalscope perf CLI
                       # "native"    — 内置 asyncio+aiohttp 引擎

# ===== 请求参数 =====
request:
  stream: true         # 是否使用流式响应（SSE）
  extra_args:          # 透传给 API 的额外参数
    chat_template_kwargs:
      thinking: true

# ===== 测试参数 =====
test:
  concurrency: [1, 5, 10, 20, 50]            # 并发梯度（从低到高）
  requests_per_level: [10, 50, 100, 200, 500] # 每级请求数（与 concurrency 一一对应，长度必须一致）
  dataset: "longalpaca"                        # 数据集："openqa" | "longalpaca" | 自定义 JSONL 文件路径

# ===== 通过条件 =====
pass_criteria:
  - metric: "success_rate"        # 成功率（0.0~1.0）
    operator: ">="
    threshold: 1.0
  - metric: "gen_toks_per_sec"    # 总输出吞吐量（tokens/s）
    operator: ">="
    threshold: 500
  - metric: "avg_ttft"            # 平均首 token 响应时间（秒）
    operator: "<="
    threshold: 10.0

# ===== 降级策略 =====
degradation:
  enabled: true           # true=自动降级  false=仅报告不降级
  step: 10                # 每次降低的并发步长
  min_concurrency: 10     # 最低探测到的并发数

# ===== 输出 =====
output:
  dir: "./results"                    # 结果输出根目录
  formats: ["json", "csv", "html"]   # 输出格式
  charts: true                        # 是否生成图表
```

### 可用的判定指标

| 指标名 | 含义 | 单位 |
|--------|------|------|
| `success_rate` | 请求成功率 | 0.0~1.0 |
| `gen_toks_per_sec` | 总输出吞吐量 | tokens/s |
| `avg_ttft` | 平均首 token 响应时间 | 秒 |
| `avg_tpot` | 平均每 token 输出时间 | 秒 |
| `p50_latency` | P50 延迟（中位数） | 秒 |
| `p99_latency` | P99 延迟 | 秒 |
| `avg_latency` | 平均延迟 | 秒 |

所有指标都可在 `pass_criteria` 中使用，操作符支持 `>=`、`<=`、`>`、`<`、`==`。

## 双引擎说明

### Native 引擎（推荐新手使用）

内置引擎，零外部依赖。直接用 asyncio + aiohttp 向 API 发送并发请求，解析 SSE 流式响应，精确采集每个请求的 TTFT、TPOT、延迟等指标。

```bash
llm-stress-test run --config config/my_test.yaml --engine native
```

### EvalScope 引擎

封装阿里 [EvalScope](https://github.com/modelscope/evalscope) 的 `evalscope perf` 命令。适合需要与 evalscope 历史数据对比的场景。

需要额外安装：

```bash
pip3 install evalscope
```

```bash
llm-stress-test run --config config/my_test.yaml --engine evalscope
```

## 数据集

### 内置数据集

| 名称 | 说明 | Payload 大小 |
|------|------|-------------|
| `openqa` | HC3-Chinese 短文本问答 | < 100 tokens |
| `longalpaca` | LongAlpaca-12k 长文本 | > 6000 tokens |

`openqa` 可直接使用。`longalpaca` 需要先下载：

```bash
python3 datasets/download_longalpaca.py
```

数据集会缓存到 `~/.cache/llm-stress-test/datasets/`，只需下载一次。

### 自定义数据集

创建 JSONL 文件，每行一条：

```jsonl
{"messages": [{"role": "user", "content": "你的 prompt 内容"}]}
```

在配置中指定文件路径：

```yaml
test:
  dataset: "/path/to/your/dataset.jsonl"
```

也支持 `{"question": "..."}` 和 `{"instruction": "..."}` 格式。

## GUI 配置编辑器

提供独立的桌面 GUI 工具，用于可视化创建和编辑配置文件：

```bash
# 创建新配置
llm-stress-config-gui

# 编辑已有配置
llm-stress-config-gui --config config/my_test.yaml
```

功能：
- 表单式编辑所有配置项
- 保存前自动校验配置合法性
- 支持打开/保存/另存为
- 快捷键：Ctrl+O 打开、Ctrl+S 保存

> GUI 只负责编辑配置文件，不执行测试。编辑完成后用 `llm-stress-test run` 执行。

## 降级逻辑说明

当目标并发（配置中的最高并发值）不通过时，工具会自动向下探测：

```
目标并发 50 → 不通过
     ↓ 降级 step=10
   并发 40 → 不通过
     ↓
   并发 30 → 通过 ← 最大通过并发数
```

- 已经在梯度测试中跑过的并发级别会复用结果，不重复测试
- 降级的请求数按比例缩放（如 50 并发/500 请求 → 40 并发/400 请求）
- 设置 `degradation.enabled: false` 可关闭自动降级

## 错误处理

工具区分两类错误：

**性能不足**（继续测试 + 降级）：请求成功但指标不达标。

**系统性故障**（立即终止）：
- API 认证失败 (401/403) → 立即停止
- 网络不通（同轮累计 ≥ 3 次）→ 立即停止
- 连续服务端错误（5xx ≥ 10 次）→ 立即停止

按 Ctrl+C 可随时优雅退出，已完成的测试数据不会丢失。

## 开发

### 运行测试

```bash
pytest tests/ -v
```

### 项目结构

```
src/llm_stress_test/
├── cli.py              # CLI 入口（click）
├── config.py           # 配置加载、校验、脱敏
├── models.py           # 共享数据模型
├── metrics.py          # 指标聚合、通过判定
├── dataset.py          # 数据集加载
├── orchestrator.py     # 测试编排、降级策略
├── engine/
│   ├── base.py         # 引擎抽象基类
│   ├── native.py       # 自研引擎 (asyncio+aiohttp)
│   └── evalscope.py    # EvalScope 引擎 (subprocess)
├── report/
│   ├── exporter.py     # JSON/CSV 导出
│   ├── chart.py        # matplotlib 图表
│   └── html.py         # HTML 报告
└── gui/
    └── app.py          # Tkinter 配置编辑器
```

## License

MIT
