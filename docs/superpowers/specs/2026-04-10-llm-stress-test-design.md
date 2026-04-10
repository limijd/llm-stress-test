# LLM 压力测试工具 — 设计文档

## 1. 背景与目标

为团队提供一个自动化的大模型压力测试工具，用于验证 LLM 推理服务在不同并发负载下的性能表现。工具需要：

- 支持所有 OpenAI 兼容 API 端点（DeepSeek、Qwen、GLM、vLLM 等）
- 提供双压测内核（evalscope + 自研），用户可选择
- 自动执行多并发梯度测试、判定通过/不通过、不通过时自动降级
- 生成结构化数据 + 图表 + HTML 汇总报告
- 提供友好的 CLI 界面和独立的 Tkinter GUI 配置编辑器
- 团队成员可直接使用，无需深入了解底层工具

### 通过条件（默认，可配置）

| 指标 | 条件 | 含义 |
|------|------|------|
| Success Rate | >= 100% | 所有请求必须成功 |
| Gen toks/s | >= 500 | 总输出吞吐 >= 500 tokens/s |
| Avg TTFT | <= 10s | 平均首 token 响应时间 <= 10 秒 |

### 降级逻辑

目标并发（默认 50）不通过时，以 10 为步长向下探测，找到满足所有条件的最大并发数。

## 2. 架构

### 分层插件架构

```
CLI (llm-stress-test) / GUI (llm-stress-config-gui)
       |
  配置层 (YAML 解析 + CLI 参数合并 + 校验)
       |
  编排层 (测试调度、降级策略、通过判定)
       |
  引擎接口 (抽象基类 BaseEngine)
    +-- EvalScope 引擎 (subprocess 调用 evalscope perf)
    +-- 自研引擎 (asyncio + aiohttp 直接压测)
       |
  报告层 (JSON/CSV + matplotlib 图表 + HTML 汇总报告)
```

各层通过明确接口通信，可独立测试和替换。

### 项目结构

```
llm-stress-test/
+-- pyproject.toml              # 项目元数据、依赖、两个入口点
+-- config/
|   +-- example.yaml            # 示例配置文件
+-- src/
|   +-- llm_stress_test/
|       +-- __init__.py
|       +-- cli.py              # CLI 入口 (argparse/click)
|       +-- config.py           # 配置加载：YAML 解析 + CLI 参数合并
|       +-- orchestrator.py     # 编排层：调度测试、降级策略、通过判定
|       +-- engine/
|       |   +-- __init__.py
|       |   +-- base.py         # 引擎抽象基类 (ABC)
|       |   +-- evalscope.py    # EvalScope 引擎实现
|       |   +-- native.py       # 自研引擎实现 (asyncio + aiohttp)
|       +-- metrics.py          # 指标定义、聚合、通过判定
|       +-- report/
|       |   +-- __init__.py
|       |   +-- exporter.py     # JSON/CSV 导出
|       |   +-- chart.py        # matplotlib 图表生成
|       |   +-- html.py         # HTML 汇总报告
|       +-- gui/
|           +-- __init__.py
|           +-- app.py          # Tkinter 配置编辑器
+-- datasets/                   # 内置数据集 + 下载脚本
+-- results/                    # 测试结果输出目录
+-- tests/
```

### 入口点

```toml
[project.scripts]
llm-stress-test = "llm_stress_test.cli:main"
llm-stress-config-gui = "llm_stress_test.gui.app:main"
```

两个工具独立运行。GUI 只依赖 `config.py`，不依赖引擎/编排/报告层。

## 3. 配置体系

### YAML 配置文件

```yaml
# 测试目标
target:
  name: "DeepSeek-V3.2-Exp"
  api_url: "https://llmapi.paratera.com/v1/chat/completions"
  api_key: "${LLM_API_KEY}"           # 支持环境变量引用
  model: "DeepSeek-V3.2-Exp"

# 引擎选择
engine: "evalscope"                   # "evalscope" | "native"

# 请求参数
request:
  stream: true
  extra_args:                         # 模型特定参数，原样透传 API
    chat_template_kwargs:
      thinking: true

# 测试参数
test:
  concurrency: [1, 5, 10, 20, 50]
  requests_per_level: [10, 50, 100, 200, 500]  # 与 concurrency 按索引一一对应
  dataset: "longalpaca"               # "openqa" | "longalpaca" | 自定义路径

# 通过条件
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

# 降级策略
degradation:
  enabled: true
  start_concurrency: 50
  step: 10
  min_concurrency: 10

# 输出
output:
  dir: "./results"
  formats: ["json", "csv", "html"]
  charts: true
```

### 优先级

CLI 参数 > YAML 配置 > 内置默认值

### CLI 命令

```bash
# 执行测试
llm-stress-test run --config config/deepseek.yaml
llm-stress-test run --config config/deepseek.yaml --engine native --concurrency 1,5,10

# 纯 CLI 模式
llm-stress-test run --api-url "https://..." --model "xxx" --engine evalscope --concurrency 1,5,10,20,50 --dataset longalpaca

# 校验配置
llm-stress-test validate --config config/deepseek.yaml

# 从已有结果重新生成报告
llm-stress-test report --result-dir results/2026-04-10_xxx/

# GUI 配置编辑器
llm-stress-config-gui
llm-stress-config-gui --config config/deepseek.yaml
```

## 4. 引擎抽象与双内核

### 引擎接口

```python
@dataclass
class EngineConfig:
    api_url: str
    api_key: str
    model: str
    concurrency: int          # 当次并发数
    num_requests: int         # 当次请求数
    dataset: str              # 数据集名称或路径
    stream: bool
    extra_args: dict          # 模型特定参数

@dataclass
class RequestMetric:
    success: bool
    ttft: float               # Time To First Token (秒)
    total_latency: float      # 总延迟 (秒)
    output_tokens: int
    input_tokens: int
    tpot: float               # Time Per Output Token (秒)
    error: str | None = None

@dataclass
class LevelResult:
    concurrency: int
    num_requests: int
    requests: list[RequestMetric]
    duration: float           # 本轮总耗时 (秒)

class BaseEngine(ABC):
    @abstractmethod
    def run(self, config: EngineConfig) -> LevelResult: ...

    @abstractmethod
    def check_available(self) -> tuple[bool, str]: ...
```

两个引擎输出完全一致的 `LevelResult`，上层无需感知引擎差异。

### EvalScope 引擎

- `check_available()` — 检查 evalscope 安装和版本
- `run()` — 将 EngineConfig 转为 evalscope perf CLI 参数，subprocess 调用，解析输出，转为 LevelResult

不修改 evalscope 源码，只通过 CLI 和配置参数交互。

### 自研引擎

- `check_available()` — 检查 aiohttp（基本总是可用）
- `run()` 核心流程：
  1. 从数据集加载 prompt
  2. asyncio.Semaphore 控制并发
  3. 并发 POST `/v1/chat/completions` (stream=true)
  4. 解析 SSE 响应，记录首 token 时间 (TTFT)、逐 token 时间戳 (TPOT)、完成时间
  5. 汇总为 LevelResult

## 5. 编排层与降级策略

### 主流程

1. 启动前检查（配置校验、引擎可用性、API 连通性、数据集存在）
2. 遍历并发梯度，对每个级别：
   - 调用 engine.run()
   - metrics.aggregate() 聚合指标
   - metrics.judge() 判定通过/不通过
   - 终端实时输出本轮摘要
   - 保存本轮原始数据
3. 最高并发通过 → 生成报告，结束
4. 最高并发不通过 → 进入降级流程

### 降级流程

- 从 start_concurrency 以 step 步长向下探测
- 已跑过的并发级别复用结果，不重复测试
- 降级探测的新并发级别，请求数按 `concurrency[-1]` 对应的 `requests_per_level[-1]` 与并发数等比缩放（如 50 并发对应 500 请求，则 40 并发对应 400 请求）
- 找到最大通过并发数后，输出建议文案，生成完整报告
- `degradation.enabled=false` 时跳过自动降级，仅输出建议

### 指标体系

```python
@dataclass
class AggregatedMetrics:
    success_rate: float
    gen_toks_per_sec: float
    avg_ttft: float
    avg_tpot: float
    p50_latency: float
    p99_latency: float
    avg_latency: float
    total_output_tokens: int
    total_duration: float
```

可扩展：新增指标只需在 aggregate() 中计算新字段，在配置 pass_criteria 中引用，判定逻辑无需修改。

## 6. 报告层

### 输出目录结构

```
results/
+-- 2026-04-10T14-30-00_DeepSeek-V3.2-Exp_evalscope/
    +-- config.yaml              # 测试配置快照
    +-- raw/
    |   +-- level_01_c1.json     # 每个并发级别的原始数据
    |   +-- level_02_c5.json
    |   +-- ...
    +-- summary.json             # 聚合指标 + 判定结果
    +-- summary.csv
    +-- charts/
    |   +-- throughput.png       # 吞吐量 vs 并发数
    |   +-- ttft.png             # TTFT vs 并发数
    |   +-- latency_p50_p99.png  # P50/P99 延迟 vs 并发数
    |   +-- success_rate.png     # 成功率 vs 并发数
    +-- report.html              # HTML 汇总报告
```

目录命名：`{日期时间}_{模型名}_{引擎名}`

### 图表

- 吞吐量柱状图：X=并发数，Y=tokens/s，通过阈值虚线，thinking=true/false 双柱对比
- TTFT 柱状图：X=并发数，Y=秒，阈值虚线
- 延迟分布图：P50 + P99 双柱
- 成功率图：X=并发数，Y=%，阈值线

### HTML 报告

包含：测试结论（最大通过并发数、建议）、通过条件判定表、性能详情表、图表、测试配置快照。

### report 子命令

```bash
llm-stress-test report --result-dir results/xxx/ --formats html,csv
```

从已有原始数据重新生成报告。

## 7. GUI 配置编辑器

### 定位

独立工具 `llm-stress-config-gui`，只负责 YAML 配置文件的创建和编辑，不执行测试。

### 技术选型

Tkinter（Python 标准库），Linux 部署零额外依赖，X11 转发兼容。

### 界面模块

- 测试目标：名称、API 地址、API Key（支持环境变量引用）、模型名
- 引擎与测试参数：引擎选择、并发梯度、请求数、数据集、流式开关、额外参数 JSON 编辑
- 通过条件：可增删行的表格（指标、操作符、阈值）
- 降级策略：启用开关、起始并发、步长、最低并发
- 输出设置：输出目录、格式勾选、图表开关
- 操作：打开配置、保存配置、另存为

保存前调用 `config.py` 的校验逻辑。

## 8. 数据集管理

| 数据集 | 大小 | 条数 | 存放方式 |
|--------|------|------|---------|
| openqa (HC3-Chinese) | 6.5 MB | 3,290 条 | 仓库内置 `datasets/openqa.jsonl` |
| longalpaca (LongAlpaca-12k) | 498 MB | 12,000 条 | 下载脚本 `datasets/download_longalpaca.py`，首次使用自动下载缓存 |

自定义数据集格式：

```jsonl
{"messages": [{"role": "user", "content": "prompt 内容"}]}
```

与 OpenAI API messages 格式一致。

## 9. 错误处理与健壮性

### 启动前检查

依次检查：YAML 语法 → 必填字段 → 引擎可用性 → API 连通性 → 数据集存在。任一失败即报错 + 修复建议，不继续执行。

### 测试中错误处理

| 场景 | 处理 |
|------|------|
| 单请求超时/失败 | 记录 success=false，不中断本轮 |
| 某轮全部失败 | 标记 FAIL，继续下一级别 |
| API 完全不可达 | 连续 10 个请求失败后中止本轮 |
| evalscope 进程崩溃 | 捕获异常，记录 stderr，标记 FAIL |
| Ctrl+C | 优雅退出：保存已完成轮次数据，生成部分报告（标注"测试未完成"）|

### 日志

- `results/xxx/test.log` — 完整日志
- 终端输出 — 精简摘要
- `--verbose` / `-v` 控制日志级别，`-vv` 输出每请求详细计时

## 10. 依赖

### 核心依赖

- Python >= 3.11
- pyyaml — 配置解析
- aiohttp — 自研引擎 HTTP 客户端
- matplotlib — 图表生成
- jinja2 — HTML 报告模板
- click — CLI 框架

### 可选依赖

- evalscope — EvalScope 引擎（用户选择 evalscope 内核时需要）
- tkinter — GUI（系统自带，`python3-tk`）

### 开发依赖

- pytest
- pytest-asyncio
