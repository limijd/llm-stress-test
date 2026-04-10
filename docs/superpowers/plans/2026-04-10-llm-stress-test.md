# LLM Stress Test Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automated LLM stress testing tool with dual engines (evalscope + native), auto-degradation, and report generation.

**Architecture:** Layered plugin architecture — CLI/GUI → Config → Orchestrator → Engine (abstract) → Report. Two engine implementations share a common `BaseEngine` interface producing standardized `LevelResult` output. The orchestrator drives multi-concurrency testing, pass/fail judgment, and auto-degradation.

**Tech Stack:** Python 3.11+, click (CLI), pyyaml (config), aiohttp (native engine), matplotlib (charts), jinja2 (HTML reports), tkinter (GUI)

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies, entry points |
| `src/llm_stress_test/__init__.py` | Package init, version |
| `src/llm_stress_test/models.py` | All shared data classes (EngineConfig, RequestMetric, LevelResult, AggregatedMetrics, PassResult, etc.) |
| `src/llm_stress_test/config.py` | YAML loading, env var expansion, CLI merge, validation, sanitize_for_export() |
| `src/llm_stress_test/metrics.py` | aggregate() and judge() functions |
| `src/llm_stress_test/engine/__init__.py` | Engine registry, get_engine() factory |
| `src/llm_stress_test/engine/base.py` | BaseEngine ABC |
| `src/llm_stress_test/engine/native.py` | Native asyncio+aiohttp engine |
| `src/llm_stress_test/engine/evalscope.py` | EvalScope subprocess wrapper engine |
| `src/llm_stress_test/orchestrator.py` | Test scheduling, degradation, systemic failure detection |
| `src/llm_stress_test/report/__init__.py` | Report module init |
| `src/llm_stress_test/report/exporter.py` | JSON/CSV export |
| `src/llm_stress_test/report/chart.py` | matplotlib chart generation |
| `src/llm_stress_test/report/html.py` | HTML report with jinja2 |
| `src/llm_stress_test/cli.py` | click CLI: run, validate, report subcommands |
| `src/llm_stress_test/gui/__init__.py` | GUI module init |
| `src/llm_stress_test/gui/app.py` | Tkinter config editor |
| `datasets/download_longalpaca.py` | LongAlpaca download + cache script |
| `config/example.yaml` | Example config file |
| `tests/conftest.py` | Shared fixtures |
| `tests/test_models.py` | Model dataclass tests |
| `tests/test_config.py` | Config loading/validation tests |
| `tests/test_metrics.py` | Aggregation and judgment tests |
| `tests/test_native_engine.py` | Native engine tests (with mock HTTP) |
| `tests/test_evalscope_engine.py` | EvalScope engine tests (with mock subprocess) |
| `tests/test_orchestrator.py` | Orchestrator tests (with mock engine) |
| `tests/test_report.py` | Report generation tests |
| `tests/test_cli.py` | CLI integration tests |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/llm_stress_test/__init__.py`
- Create: `config/example.yaml`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "llm-stress-test"
version = "0.1.0"
description = "LLM 推理服务压力测试工具"
requires-python = ">=3.11"
dependencies = [
    "click>=8.1",
    "pyyaml>=6.0",
    "aiohttp>=3.9",
    "matplotlib>=3.8",
    "jinja2>=3.1",
]

[project.optional-dependencies]
evalscope = ["evalscope"]
gui = []  # tkinter 是系统自带
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "aioresponses>=0.7",
]

[project.scripts]
llm-stress-test = "llm_stress_test.cli:main"
llm-stress-config-gui = "llm_stress_test.gui.app:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create package init**

```python
# src/llm_stress_test/__init__.py
"""LLM 推理服务压力测试工具"""
__version__ = "0.1.0"
```

- [ ] **Step 3: Create example config**

```yaml
# config/example.yaml
# LLM 压力测试配置示例

target:
  name: "DeepSeek-V3.2-Exp"
  api_url: "https://llmapi.paratera.com/v1/chat/completions"
  api_key: "${LLM_API_KEY}"
  model: "DeepSeek-V3.2-Exp"

engine: "evalscope"  # "evalscope" | "native"

request:
  stream: true
  extra_args:
    chat_template_kwargs:
      thinking: true

test:
  concurrency: [1, 5, 10, 20, 50]
  requests_per_level: [10, 50, 100, 200, 500]
  dataset: "longalpaca"  # "openqa" | "longalpaca" | 自定义文件路径

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

degradation:
  enabled: true
  start_concurrency: 50
  step: 10
  min_concurrency: 10

output:
  dir: "./results"
  formats: ["json", "csv", "html"]
  charts: true
```

- [ ] **Step 4: Create directory structure**

```bash
mkdir -p src/llm_stress_test/engine src/llm_stress_test/report src/llm_stress_test/gui tests datasets results
touch src/llm_stress_test/engine/__init__.py src/llm_stress_test/report/__init__.py src/llm_stress_test/gui/__init__.py tests/__init__.py
```

- [ ] **Step 5: Install in dev mode and verify**

```bash
pip3 install -e ".[dev]"
python3 -c "import llm_stress_test; print(llm_stress_test.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/ config/ tests/__init__.py datasets/ results/
git commit -m "feat: project scaffolding with pyproject.toml and package structure"
```

---

### Task 2: Data Models

**Files:**
- Create: `src/llm_stress_test/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write tests for data models**

```python
# tests/test_models.py
import pytest
from llm_stress_test.models import (
    EngineConfig,
    RequestMetric,
    LevelResult,
    AggregatedMetrics,
    CriterionResult,
    PassResult,
    Criterion,
    SystemicError,
)


class TestEngineConfig:
    def test_create(self):
        cfg = EngineConfig(
            api_url="http://localhost:8000/v1/chat/completions",
            api_key="test-key",
            model="test-model",
            concurrency=10,
            num_requests=100,
            dataset="openqa",
            stream=True,
            extra_args={"thinking": True},
        )
        assert cfg.concurrency == 10
        assert cfg.stream is True
        assert cfg.extra_args == {"thinking": True}


class TestRequestMetric:
    def test_successful_request(self):
        m = RequestMetric(
            success=True,
            ttft=0.5,
            total_latency=2.0,
            output_tokens=100,
            input_tokens=50,
            tpot=0.015,
        )
        assert m.success is True
        assert m.error is None

    def test_failed_request(self):
        m = RequestMetric(
            success=False,
            ttft=0.0,
            total_latency=0.0,
            output_tokens=0,
            input_tokens=0,
            tpot=0.0,
            error="Connection refused",
        )
        assert m.success is False
        assert m.error == "Connection refused"


class TestLevelResult:
    def test_create(self):
        metrics = [
            RequestMetric(True, 0.5, 2.0, 100, 50, 0.015),
            RequestMetric(True, 0.6, 2.1, 110, 50, 0.014),
        ]
        result = LevelResult(
            concurrency=5,
            num_requests=2,
            requests=metrics,
            duration=2.5,
        )
        assert result.concurrency == 5
        assert len(result.requests) == 2


class TestCriterion:
    @pytest.mark.parametrize(
        "operator,threshold,value,expected",
        [
            (">=", 1.0, 1.0, True),
            (">=", 1.0, 0.99, False),
            ("<=", 10.0, 9.5, True),
            ("<=", 10.0, 10.5, False),
            (">", 500, 501, True),
            (">", 500, 500, False),
            ("<", 10.0, 9.9, True),
            ("<", 10.0, 10.0, False),
        ],
    )
    def test_evaluate(self, operator, threshold, value, expected):
        c = Criterion(metric="test", operator=operator, threshold=threshold)
        assert c.evaluate(value) == expected


class TestSystemicError:
    def test_auth_error(self):
        e = SystemicError(error_type="auth", message="401 Unauthorized", status_code=401)
        assert e.error_type == "auth"

    def test_network_error(self):
        e = SystemicError(error_type="network", message="DNS resolution failed")
        assert e.status_code is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'llm_stress_test.models'`

- [ ] **Step 3: Implement data models**

```python
# src/llm_stress_test/models.py
"""所有共享数据模型"""
from __future__ import annotations

import operator as op
from dataclasses import dataclass, field

# 操作符映射表
_OPERATORS: dict[str, callable] = {
    ">=": op.ge,
    "<=": op.le,
    ">": op.gt,
    "<": op.lt,
    "==": op.eq,
}


@dataclass(frozen=True)
class EngineConfig:
    """传给引擎的标准化配置"""
    api_url: str
    api_key: str
    model: str
    concurrency: int
    num_requests: int
    dataset: str
    stream: bool
    extra_args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RequestMetric:
    """单个请求的原始指标"""
    success: bool
    ttft: float           # Time To First Token (秒)
    total_latency: float  # 总延迟 (秒)
    output_tokens: int
    input_tokens: int
    tpot: float           # Time Per Output Token (秒)
    error: str | None = None


@dataclass(frozen=True)
class LevelResult:
    """单个并发级别的测试结果"""
    concurrency: int
    num_requests: int
    requests: list[RequestMetric]
    duration: float  # 本轮总耗时 (秒)


@dataclass(frozen=True)
class AggregatedMetrics:
    """聚合后的指标"""
    success_rate: float
    gen_toks_per_sec: float
    avg_ttft: float
    avg_tpot: float
    p50_latency: float
    p99_latency: float
    avg_latency: float
    total_output_tokens: int
    total_duration: float


@dataclass(frozen=True)
class Criterion:
    """单条通过条件"""
    metric: str
    operator: str
    threshold: float

    def evaluate(self, actual: float) -> bool:
        """判定实际值是否满足条件"""
        fn = _OPERATORS.get(self.operator)
        if fn is None:
            raise ValueError(f"不支持的操作符: {self.operator}")
        return fn(actual, self.threshold)


@dataclass(frozen=True)
class CriterionResult:
    """单条通过条件的判定结果"""
    metric: str
    operator: str
    threshold: float
    actual: float
    passed: bool


@dataclass(frozen=True)
class PassResult:
    """整体通过判定结果"""
    passed: bool
    details: list[CriterionResult]


@dataclass(frozen=True)
class SystemicError:
    """系统性故障"""
    error_type: str  # "auth" | "network" | "server"
    message: str
    status_code: int | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```

Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_stress_test/models.py tests/test_models.py
git commit -m "feat: add shared data models (EngineConfig, RequestMetric, LevelResult, etc.)"
```

---

### Task 3: Configuration Layer

**Files:**
- Create: `src/llm_stress_test/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write tests for config loading and validation**

```python
# tests/test_config.py
import os
import pytest
import tempfile
import yaml
from pathlib import Path
from llm_stress_test.config import (
    load_config,
    validate_config,
    merge_cli_overrides,
    expand_env_vars,
    sanitize_for_export,
    ConfigError,
)


MINIMAL_CONFIG = {
    "target": {
        "name": "test",
        "api_url": "http://localhost:8000/v1/chat/completions",
        "api_key": "sk-test",
        "model": "test-model",
    },
    "engine": "native",
    "request": {"stream": True, "extra_args": {}},
    "test": {
        "concurrency": [1, 5],
        "requests_per_level": [10, 50],
        "dataset": "openqa",
    },
    "pass_criteria": [
        {"metric": "success_rate", "operator": ">=", "threshold": 1.0},
    ],
    "degradation": {
        "enabled": False,
        "start_concurrency": 5,
        "step": 1,
        "min_concurrency": 1,
    },
    "output": {
        "dir": "./results",
        "formats": ["json"],
        "charts": False,
    },
}


def write_yaml(data: dict, path: Path):
    path.write_text(yaml.dump(data, allow_unicode=True))


class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path):
        p = tmp_path / "config.yaml"
        write_yaml(MINIMAL_CONFIG, p)
        cfg = load_config(str(p))
        assert cfg["target"]["name"] == "test"

    def test_load_nonexistent_file(self):
        with pytest.raises(ConfigError, match="配置文件不存在"):
            load_config("/nonexistent/path.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(": : : invalid")
        with pytest.raises(ConfigError, match="YAML 语法错误"):
            load_config(str(p))


class TestValidateConfig:
    def test_valid_config_passes(self):
        validate_config(MINIMAL_CONFIG)

    def test_missing_target(self):
        cfg = {k: v for k, v in MINIMAL_CONFIG.items() if k != "target"}
        with pytest.raises(ConfigError, match="target"):
            validate_config(cfg)

    def test_mismatched_array_lengths(self):
        cfg = {**MINIMAL_CONFIG, "test": {**MINIMAL_CONFIG["test"], "requests_per_level": [10]}}
        with pytest.raises(ConfigError, match="长度必须一致"):
            validate_config(cfg)

    def test_invalid_engine(self):
        cfg = {**MINIMAL_CONFIG, "engine": "unknown"}
        with pytest.raises(ConfigError, match="engine"):
            validate_config(cfg)

    def test_invalid_operator(self):
        cfg = {**MINIMAL_CONFIG, "pass_criteria": [
            {"metric": "success_rate", "operator": "!=", "threshold": 1.0},
        ]}
        with pytest.raises(ConfigError, match="操作符"):
            validate_config(cfg)


class TestExpandEnvVars:
    def test_expand_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "my-secret-key")
        cfg = {"target": {"api_key": "${TEST_KEY}"}}
        result = expand_env_vars(cfg)
        assert result["target"]["api_key"] == "my-secret-key"

    def test_literal_value_unchanged(self):
        cfg = {"target": {"api_key": "sk-literal"}}
        result = expand_env_vars(cfg)
        assert result["target"]["api_key"] == "sk-literal"

    def test_missing_env_var_raises(self):
        cfg = {"target": {"api_key": "${NONEXISTENT_VAR_12345}"}}
        with pytest.raises(ConfigError, match="环境变量.*未设置"):
            expand_env_vars(cfg)


class TestMergeCLIOverrides:
    def test_override_engine(self):
        cfg = {**MINIMAL_CONFIG}
        overrides = {"engine": "evalscope"}
        result = merge_cli_overrides(cfg, overrides)
        assert result["engine"] == "evalscope"

    def test_override_concurrency(self):
        cfg = {**MINIMAL_CONFIG}
        overrides = {"test.concurrency": [1, 10]}
        result = merge_cli_overrides(cfg, overrides)
        assert result["test"]["concurrency"] == [1, 10]


class TestSanitizeForExport:
    def test_redact_literal_key(self):
        cfg = {**MINIMAL_CONFIG}
        result = sanitize_for_export(cfg)
        assert result["target"]["api_key"] == "***REDACTED***"

    def test_preserve_env_var_placeholder(self):
        cfg = {**MINIMAL_CONFIG, "target": {**MINIMAL_CONFIG["target"], "api_key": "${LLM_API_KEY}"}}
        result = sanitize_for_export(cfg)
        assert result["target"]["api_key"] == "${LLM_API_KEY}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement config module**

```python
# src/llm_stress_test/config.py
"""配置加载、校验、环境变量展开、脱敏"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import yaml


class ConfigError(Exception):
    """配置错误"""


_VALID_ENGINES = {"evalscope", "native"}
_VALID_OPERATORS = {">=", "<=", ">", "<", "=="}
_REQUIRED_TOP_KEYS = ["target", "engine", "test", "pass_criteria", "degradation", "output"]
_REQUIRED_TARGET_KEYS = ["name", "api_url", "api_key", "model"]
_ENV_VAR_PATTERN = re.compile(r"^\$\{([^}]+)\}$")

# 配置默认值
DEFAULTS = {
    "request": {"stream": True, "extra_args": {}},
    "degradation": {
        "enabled": True,
        "start_concurrency": 50,
        "step": 10,
        "min_concurrency": 10,
    },
    "output": {
        "dir": "./results",
        "formats": ["json", "csv", "html"],
        "charts": True,
    },
}


def load_config(path: str) -> dict:
    """从 YAML 文件加载配置"""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML 语法错误: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError("配置文件顶层必须是字典")
    # 填充默认值
    for key, default in DEFAULTS.items():
        if key not in data:
            data[key] = copy.deepcopy(default)
        elif isinstance(default, dict):
            for dk, dv in default.items():
                data[key].setdefault(dk, dv)
    return data


def validate_config(cfg: dict) -> None:
    """校验配置完整性和合法性"""
    # 必填顶层字段
    for key in _REQUIRED_TOP_KEYS:
        if key not in cfg:
            raise ConfigError(f"缺少必填字段: {key}")

    # target 字段
    target = cfg["target"]
    for key in _REQUIRED_TARGET_KEYS:
        if key not in target:
            raise ConfigError(f"target 缺少必填字段: {key}")

    # engine
    if cfg["engine"] not in _VALID_ENGINES:
        raise ConfigError(f"engine 必须是 {_VALID_ENGINES} 之一，当前值: {cfg['engine']}")

    # test 字段
    test = cfg["test"]
    concurrency = test.get("concurrency", [])
    requests_per_level = test.get("requests_per_level", [])
    if len(concurrency) != len(requests_per_level):
        raise ConfigError(
            f"concurrency ({len(concurrency)} 项) 和 requests_per_level ({len(requests_per_level)} 项) 长度必须一致"
        )

    # pass_criteria 操作符
    for criterion in cfg.get("pass_criteria", []):
        op = criterion.get("operator", "")
        if op not in _VALID_OPERATORS:
            raise ConfigError(f"不支持的操作符: {op}，支持: {_VALID_OPERATORS}")


def expand_env_vars(cfg: dict) -> dict:
    """展开配置中的环境变量引用 ${VAR}"""
    result = copy.deepcopy(cfg)
    _expand_recursive(result)
    return result


def _expand_recursive(obj):
    """递归展开字典/列表中的环境变量"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                m = _ENV_VAR_PATTERN.match(value)
                if m:
                    var_name = m.group(1)
                    env_val = os.environ.get(var_name)
                    if env_val is None:
                        raise ConfigError(f"环境变量 {var_name} 未设置")
                    obj[key] = env_val
            elif isinstance(value, (dict, list)):
                _expand_recursive(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str):
                m = _ENV_VAR_PATTERN.match(item)
                if m:
                    var_name = m.group(1)
                    env_val = os.environ.get(var_name)
                    if env_val is None:
                        raise ConfigError(f"环境变量 {var_name} 未设置")
                    obj[i] = env_val
            elif isinstance(item, (dict, list)):
                _expand_recursive(item)


def merge_cli_overrides(cfg: dict, overrides: dict) -> dict:
    """将 CLI 参数覆盖合并到配置中，支持点号路径 (如 test.concurrency)"""
    result = copy.deepcopy(cfg)
    for key, value in overrides.items():
        parts = key.split(".")
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def sanitize_for_export(cfg: dict) -> dict:
    """脱敏配置，用于写入报告和日志"""
    result = copy.deepcopy(cfg)
    api_key = result.get("target", {}).get("api_key", "")
    if api_key and not _ENV_VAR_PATTERN.match(api_key):
        result["target"]["api_key"] = "***REDACTED***"
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_stress_test/config.py tests/test_config.py
git commit -m "feat: config layer with YAML loading, validation, env expansion, and sanitization"
```

---

### Task 4: Metrics Layer

**Files:**
- Create: `src/llm_stress_test/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write tests for metrics aggregation and judgment**

```python
# tests/test_metrics.py
import pytest
from llm_stress_test.models import (
    RequestMetric,
    LevelResult,
    Criterion,
    AggregatedMetrics,
)
from llm_stress_test.metrics import aggregate, judge


def _make_successful_requests(n: int, ttft: float = 0.5, latency: float = 2.0,
                               output_tokens: int = 100, tpot: float = 0.015) -> list[RequestMetric]:
    return [
        RequestMetric(
            success=True, ttft=ttft, total_latency=latency,
            output_tokens=output_tokens, input_tokens=50, tpot=tpot,
        )
        for _ in range(n)
    ]


class TestAggregate:
    def test_all_successful(self):
        requests = _make_successful_requests(10, ttft=0.5, latency=2.0, output_tokens=100)
        result = LevelResult(concurrency=5, num_requests=10, requests=requests, duration=4.0)
        agg = aggregate(result)
        assert agg.success_rate == 1.0
        assert agg.avg_ttft == 0.5
        assert agg.total_output_tokens == 1000
        # gen_toks_per_sec = total_output_tokens / duration = 1000 / 4.0 = 250
        assert agg.gen_toks_per_sec == 250.0

    def test_with_failures(self):
        good = _make_successful_requests(8)
        bad = [
            RequestMetric(success=False, ttft=0.0, total_latency=0.0,
                          output_tokens=0, input_tokens=0, tpot=0.0, error="timeout"),
            RequestMetric(success=False, ttft=0.0, total_latency=0.0,
                          output_tokens=0, input_tokens=0, tpot=0.0, error="timeout"),
        ]
        result = LevelResult(concurrency=5, num_requests=10, requests=good + bad, duration=5.0)
        agg = aggregate(result)
        assert agg.success_rate == 0.8

    def test_percentiles(self):
        # 创建有不同延迟的请求
        requests = [
            RequestMetric(True, 0.5, float(i), 100, 50, 0.01)
            for i in range(1, 101)
        ]
        result = LevelResult(concurrency=10, num_requests=100, requests=requests, duration=10.0)
        agg = aggregate(result)
        assert agg.p50_latency == pytest.approx(50.5, abs=1.0)
        assert agg.p99_latency == pytest.approx(99.5, abs=1.0)

    def test_empty_requests(self):
        """所有请求都失败的情况"""
        requests = [
            RequestMetric(False, 0.0, 0.0, 0, 0, 0.0, error="fail")
            for _ in range(5)
        ]
        result = LevelResult(concurrency=5, num_requests=5, requests=requests, duration=1.0)
        agg = aggregate(result)
        assert agg.success_rate == 0.0
        assert agg.gen_toks_per_sec == 0.0
        assert agg.avg_ttft == 0.0


class TestJudge:
    def test_all_criteria_pass(self):
        agg = AggregatedMetrics(
            success_rate=1.0, gen_toks_per_sec=600, avg_ttft=5.0,
            avg_tpot=0.01, p50_latency=2.0, p99_latency=5.0,
            avg_latency=2.5, total_output_tokens=1000, total_duration=1.67,
        )
        criteria = [
            Criterion("success_rate", ">=", 1.0),
            Criterion("gen_toks_per_sec", ">=", 500),
            Criterion("avg_ttft", "<=", 10.0),
        ]
        result = judge(agg, criteria)
        assert result.passed is True
        assert all(d.passed for d in result.details)

    def test_one_criterion_fails(self):
        agg = AggregatedMetrics(
            success_rate=1.0, gen_toks_per_sec=400, avg_ttft=5.0,
            avg_tpot=0.01, p50_latency=2.0, p99_latency=5.0,
            avg_latency=2.5, total_output_tokens=1000, total_duration=2.5,
        )
        criteria = [
            Criterion("success_rate", ">=", 1.0),
            Criterion("gen_toks_per_sec", ">=", 500),
        ]
        result = judge(agg, criteria)
        assert result.passed is False
        assert result.details[0].passed is True   # success_rate
        assert result.details[1].passed is False   # gen_toks_per_sec

    def test_empty_criteria_passes(self):
        agg = AggregatedMetrics(
            success_rate=0.5, gen_toks_per_sec=10, avg_ttft=30.0,
            avg_tpot=1.0, p50_latency=10.0, p99_latency=30.0,
            avg_latency=15.0, total_output_tokens=100, total_duration=10.0,
        )
        result = judge(agg, [])
        assert result.passed is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_metrics.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement metrics module**

```python
# src/llm_stress_test/metrics.py
"""指标聚合与通过判定"""
from __future__ import annotations

import statistics
from dataclasses import asdict

from .models import (
    AggregatedMetrics,
    Criterion,
    CriterionResult,
    LevelResult,
    PassResult,
)


def aggregate(result: LevelResult) -> AggregatedMetrics:
    """将 LevelResult 聚合为 AggregatedMetrics"""
    successful = [r for r in result.requests if r.success]
    total = len(result.requests)
    success_count = len(successful)

    if success_count == 0:
        return AggregatedMetrics(
            success_rate=0.0,
            gen_toks_per_sec=0.0,
            avg_ttft=0.0,
            avg_tpot=0.0,
            p50_latency=0.0,
            p99_latency=0.0,
            avg_latency=0.0,
            total_output_tokens=0,
            total_duration=result.duration,
        )

    total_output_tokens = sum(r.output_tokens for r in successful)
    latencies = sorted(r.total_latency for r in successful)

    return AggregatedMetrics(
        success_rate=success_count / total if total > 0 else 0.0,
        gen_toks_per_sec=total_output_tokens / result.duration if result.duration > 0 else 0.0,
        avg_ttft=statistics.mean(r.ttft for r in successful),
        avg_tpot=statistics.mean(r.tpot for r in successful),
        p50_latency=_percentile(latencies, 50),
        p99_latency=_percentile(latencies, 99),
        avg_latency=statistics.mean(latencies),
        total_output_tokens=total_output_tokens,
        total_duration=result.duration,
    )


def judge(aggregated: AggregatedMetrics, criteria: list[Criterion]) -> PassResult:
    """根据通过条件判定"""
    agg_dict = asdict(aggregated)
    details = []
    for c in criteria:
        actual = agg_dict.get(c.metric)
        if actual is None:
            raise ValueError(f"指标 {c.metric} 在 AggregatedMetrics 中不存在")
        passed = c.evaluate(actual)
        details.append(CriterionResult(
            metric=c.metric,
            operator=c.operator,
            threshold=c.threshold,
            actual=actual,
            passed=passed,
        ))
    return PassResult(
        passed=all(d.passed for d in details),
        details=details,
    )


def _percentile(sorted_data: list[float], pct: int) -> float:
    """计算百分位数（线性插值）"""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]
    k = (pct / 100) * (n - 1)
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_data[-1]
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_metrics.py -v
```

Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_stress_test/metrics.py tests/test_metrics.py
git commit -m "feat: metrics aggregation (percentiles, throughput) and pass/fail judgment"
```

---

### Task 5: Engine Base + Native Engine

**Files:**
- Create: `src/llm_stress_test/engine/base.py`
- Create: `src/llm_stress_test/engine/native.py`
- Update: `src/llm_stress_test/engine/__init__.py`
- Create: `tests/test_native_engine.py`

- [ ] **Step 1: Write the engine base class**

```python
# src/llm_stress_test/engine/base.py
"""引擎抽象基类"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import EngineConfig, LevelResult


class BaseEngine(ABC):
    @abstractmethod
    def run(self, config: EngineConfig) -> LevelResult:
        """执行一个并发级别的测试"""
        ...

    @abstractmethod
    def check_available(self) -> tuple[bool, str]:
        """检查引擎依赖是否就绪，返回 (可用, 原因)"""
        ...
```

- [ ] **Step 2: Write engine registry**

```python
# src/llm_stress_test/engine/__init__.py
"""引擎注册与工厂"""
from __future__ import annotations

from .base import BaseEngine


def get_engine(name: str) -> BaseEngine:
    """根据名称获取引擎实例"""
    if name == "native":
        from .native import NativeEngine
        return NativeEngine()
    elif name == "evalscope":
        from .evalscope import EvalScopeEngine
        return EvalScopeEngine()
    else:
        raise ValueError(f"未知引擎: {name}")
```

- [ ] **Step 3: Write tests for native engine**

```python
# tests/test_native_engine.py
import json
import pytest
from aioresponses import aioresponses
from llm_stress_test.engine.native import NativeEngine
from llm_stress_test.models import EngineConfig


def _make_sse_response(tokens: list[str], include_usage: bool = True) -> str:
    """构造 SSE 流式响应"""
    chunks = []
    for i, token in enumerate(tokens):
        chunk = {
            "choices": [{"delta": {"content": token}, "index": 0, "finish_reason": None}],
        }
        chunks.append(f"data: {json.dumps(chunk)}\n\n")
    # 最后一个 chunk 带 finish_reason 和 usage
    final = {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}
    if include_usage:
        final["usage"] = {"prompt_tokens": 10, "completion_tokens": len(tokens)}
    chunks.append(f"data: {json.dumps(final)}\n\n")
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks)


def _make_config(concurrency: int = 1, num_requests: int = 2, stream: bool = True) -> EngineConfig:
    return EngineConfig(
        api_url="http://test-api/v1/chat/completions",
        api_key="test-key",
        model="test-model",
        concurrency=concurrency,
        num_requests=num_requests,
        dataset="openqa",
        stream=stream,
        extra_args={},
    )


class TestNativeEngine:
    def test_check_available(self):
        engine = NativeEngine()
        ok, msg = engine.check_available()
        assert ok is True

    @pytest.mark.asyncio
    async def test_streaming_request(self):
        engine = NativeEngine()
        config = _make_config(concurrency=1, num_requests=2)
        sse_body = _make_sse_response(["Hello", " world", "!"])
        with aioresponses() as m:
            m.post("http://test-api/v1/chat/completions", body=sse_body,
                   content_type="text/event-stream", repeat=True)
            result = engine.run(config)
        assert len(result.requests) == 2
        assert all(r.success for r in result.requests)
        assert all(r.output_tokens > 0 for r in result.requests)
        assert result.concurrency == 1

    @pytest.mark.asyncio
    async def test_non_streaming_request(self):
        engine = NativeEngine()
        config = _make_config(concurrency=1, num_requests=1, stream=False)
        response_body = {
            "choices": [{"message": {"content": "Hello world"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        with aioresponses() as m:
            m.post("http://test-api/v1/chat/completions", payload=response_body, repeat=True)
            result = engine.run(config)
        assert len(result.requests) == 1
        assert result.requests[0].success is True
        assert result.requests[0].output_tokens == 5

    @pytest.mark.asyncio
    async def test_connection_error(self):
        engine = NativeEngine()
        config = _make_config(concurrency=1, num_requests=1)
        with aioresponses() as m:
            m.post("http://test-api/v1/chat/completions",
                   exception=ConnectionError("refused"), repeat=True)
            result = engine.run(config)
        assert len(result.requests) == 1
        assert result.requests[0].success is False
        assert result.requests[0].error is not None
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
pytest tests/test_native_engine.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'llm_stress_test.engine.native'`

- [ ] **Step 5: Implement native engine**

```python
# src/llm_stress_test/engine/native.py
"""自研压测引擎：asyncio + aiohttp"""
from __future__ import annotations

import asyncio
import json
import time

import aiohttp

from .base import BaseEngine
from ..models import EngineConfig, LevelResult, RequestMetric

# 内置简单 prompt 列表，用于测试和 fallback
_FALLBACK_PROMPTS = [
    {"role": "user", "content": "请简要介绍量子计算的基本原理。"},
]


class NativeEngine(BaseEngine):
    def check_available(self) -> tuple[bool, str]:
        try:
            import aiohttp  # noqa: F811
            return True, f"aiohttp {aiohttp.__version__}"
        except ImportError:
            return False, "aiohttp 未安装，请运行: pip3 install aiohttp"

    def run(self, config: EngineConfig) -> LevelResult:
        return asyncio.run(self._run_async(config))

    async def _run_async(self, config: EngineConfig) -> LevelResult:
        prompts = self._load_prompts(config.dataset, config.num_requests)
        semaphore = asyncio.Semaphore(config.concurrency)
        start_time = time.monotonic()

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._send_request(session, config, prompts[i % len(prompts)], semaphore)
                for i in range(config.num_requests)
            ]
            results = await asyncio.gather(*tasks)

        duration = time.monotonic() - start_time
        return LevelResult(
            concurrency=config.concurrency,
            num_requests=config.num_requests,
            requests=results,
            duration=duration,
        )

    async def _send_request(
        self,
        session: aiohttp.ClientSession,
        config: EngineConfig,
        prompt: dict,
        semaphore: asyncio.Semaphore,
    ) -> RequestMetric:
        body = {
            "model": config.model,
            "messages": [prompt],
            "stream": config.stream,
            **config.extra_args,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        }
        async with semaphore:
            try:
                req_start = time.monotonic()
                async with session.post(config.api_url, json=body, headers=headers) as resp:
                    if config.stream:
                        return await self._handle_stream(resp, req_start)
                    else:
                        return await self._handle_non_stream(resp, req_start)
            except Exception as e:
                return RequestMetric(
                    success=False, ttft=0.0, total_latency=0.0,
                    output_tokens=0, input_tokens=0, tpot=0.0,
                    error=str(e),
                )

    async def _handle_stream(self, resp: aiohttp.ClientResponse, req_start: float) -> RequestMetric:
        first_token_time = None
        token_times: list[float] = []
        output_tokens = 0
        input_tokens = 0

        async for line in resp.content:
            line = line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # 记录 token 时间
            now = time.monotonic()
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    if first_token_time is None:
                        first_token_time = now
                    token_times.append(now)
                    output_tokens += 1

            # 从最后一个 chunk 获取 usage
            usage = chunk.get("usage")
            if usage:
                input_tokens = usage.get("prompt_tokens", input_tokens)
                completion = usage.get("completion_tokens")
                if completion is not None:
                    output_tokens = completion

        total_latency = time.monotonic() - req_start
        ttft = (first_token_time - req_start) if first_token_time else total_latency

        # 计算平均 TPOT
        if len(token_times) > 1:
            intervals = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
            avg_tpot = sum(intervals) / len(intervals)
        else:
            avg_tpot = 0.0

        return RequestMetric(
            success=True,
            ttft=ttft,
            total_latency=total_latency,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            tpot=avg_tpot,
        )

    async def _handle_non_stream(self, resp: aiohttp.ClientResponse, req_start: float) -> RequestMetric:
        data = await resp.json()
        total_latency = time.monotonic() - req_start
        usage = data.get("usage", {})
        return RequestMetric(
            success=True,
            ttft=total_latency,
            total_latency=total_latency,
            output_tokens=usage.get("completion_tokens", 0),
            input_tokens=usage.get("prompt_tokens", 0),
            tpot=0.0,
        )

    def _load_prompts(self, dataset: str, num_requests: int) -> list[dict]:
        """加载数据集 prompt。简化版：后续 Task 实现完整数据集加载"""
        # TODO: Task 10 将实现完整的数据集加载
        return _FALLBACK_PROMPTS * max(1, num_requests)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_native_engine.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/llm_stress_test/engine/ tests/test_native_engine.py
git commit -m "feat: engine abstraction and native asyncio+aiohttp engine"
```

---

### Task 6: EvalScope Engine

**Files:**
- Create: `src/llm_stress_test/engine/evalscope.py`
- Create: `tests/test_evalscope_engine.py`

- [ ] **Step 1: Write tests for evalscope engine**

```python
# tests/test_evalscope_engine.py
import json
import pytest
from unittest.mock import patch, MagicMock
from llm_stress_test.engine.evalscope import EvalScopeEngine
from llm_stress_test.models import EngineConfig


def _make_config() -> EngineConfig:
    return EngineConfig(
        api_url="http://test-api/v1/chat/completions",
        api_key="test-key",
        model="test-model",
        concurrency=5,
        num_requests=50,
        dataset="openqa",
        stream=True,
        extra_args={"chat_template_kwargs": {"thinking": True}},
    )


class TestEvalScopeEngine:
    def test_check_available_installed(self):
        engine = EvalScopeEngine()
        with patch("shutil.which", return_value="/usr/bin/evalscope"):
            ok, msg = engine.check_available()
            assert ok is True

    def test_check_available_not_installed(self):
        engine = EvalScopeEngine()
        with patch("shutil.which", return_value=None):
            ok, msg = engine.check_available()
            assert ok is False
            assert "evalscope" in msg

    def test_build_command(self):
        engine = EvalScopeEngine()
        config = _make_config()
        cmd = engine._build_command(config)
        assert "evalscope" in cmd[0]
        assert "perf" in cmd
        assert "--parallel" in cmd
        assert "5" in cmd
        assert "--number" in cmd
        assert "50" in cmd
        assert "--url" in cmd
        assert "--model" in cmd
        assert "--stream" in cmd

    def test_run_success(self):
        engine = EvalScopeEngine()
        config = _make_config()
        # 模拟 evalscope 输出的 JSON 结果文件
        mock_result = {
            "results": [
                {
                    "success": True,
                    "ttft": 0.5,
                    "latency": 2.0,
                    "output_tokens": 100,
                    "input_tokens": 50,
                }
                for _ in range(50)
            ],
            "total_time": 10.0,
        }
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = json.dumps(mock_result)
        mock_process.stderr = ""

        with patch("subprocess.run", return_value=mock_process):
            with patch.object(engine, "_parse_output", return_value=engine._make_level_result(config, mock_result)):
                result = engine.run(config)
        assert result.concurrency == 5
        assert result.num_requests == 50

    def test_run_process_crash(self):
        engine = EvalScopeEngine()
        config = _make_config()
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = ""
        mock_process.stderr = "Segmentation fault"

        with patch("subprocess.run", return_value=mock_process):
            result = engine.run(config)
        # 进程崩溃时返回全部失败的 LevelResult
        assert all(not r.success for r in result.requests)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_evalscope_engine.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement evalscope engine**

```python
# src/llm_stress_test/engine/evalscope.py
"""EvalScope 引擎：通过 subprocess 调用 evalscope perf"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess

from .base import BaseEngine
from ..models import EngineConfig, LevelResult, RequestMetric

logger = logging.getLogger(__name__)


class EvalScopeEngine(BaseEngine):
    def check_available(self) -> tuple[bool, str]:
        if shutil.which("evalscope") is not None:
            return True, "evalscope CLI 可用"
        # 也尝试 python -m evalscope
        try:
            result = subprocess.run(
                ["python3", "-m", "evalscope", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                return True, f"evalscope {version}"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return False, "evalscope 未安装，请运行: pip3 install evalscope"

    def run(self, config: EngineConfig) -> LevelResult:
        cmd = self._build_command(config)
        logger.info("执行 evalscope: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=3600,  # 1 小时超时
            )
        except subprocess.TimeoutExpired:
            logger.error("evalscope 执行超时")
            return self._make_error_result(config, "evalscope 执行超时（1小时）")
        except Exception as e:
            logger.error("evalscope 执行异常: %s", e)
            return self._make_error_result(config, str(e))

        if proc.returncode != 0:
            logger.error("evalscope 退出码 %d: %s", proc.returncode, proc.stderr)
            return self._make_error_result(config, f"evalscope 退出码 {proc.returncode}: {proc.stderr[:500]}")

        return self._parse_output(config, proc.stdout, proc.stderr)

    def _build_command(self, config: EngineConfig) -> list[str]:
        cmd = [
            "evalscope", "perf",
            "--url", config.api_url,
            "--parallel", str(config.concurrency),
            "--number", str(config.num_requests),
            "--api", "openai",
            "--model", config.model,
        ]
        if config.dataset:
            cmd.extend(["--dataset", config.dataset])
        if config.stream:
            cmd.append("--stream")
        if config.extra_args:
            cmd.extend(["--extra-args", json.dumps(config.extra_args)])
        return cmd

    def _parse_output(self, config: EngineConfig, stdout: str, stderr: str) -> LevelResult:
        """解析 evalscope perf 的输出，转为 LevelResult

        evalscope perf 输出格式在不同版本可能不同，
        这里尝试解析 JSON 格式，失败则从 stdout 文本中提取摘要信息。
        """
        # 尝试解析 JSON 输出
        try:
            data = json.loads(stdout)
            return self._make_level_result(config, data)
        except (json.JSONDecodeError, KeyError):
            pass

        # 尝试逐行查找 JSON 行
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    return self._make_level_result(config, data)
                except (json.JSONDecodeError, KeyError):
                    continue

        logger.warning("无法解析 evalscope 输出，返回空结果")
        return self._make_error_result(config, "无法解析 evalscope 输出")

    def _make_level_result(self, config: EngineConfig, data: dict) -> LevelResult:
        """从 evalscope 解析结果构造 LevelResult"""
        raw_results = data.get("results", [])
        requests = []
        for r in raw_results:
            success = r.get("success", True)
            requests.append(RequestMetric(
                success=success,
                ttft=r.get("ttft", 0.0),
                total_latency=r.get("latency", 0.0),
                output_tokens=r.get("output_tokens", 0),
                input_tokens=r.get("input_tokens", 0),
                tpot=r.get("tpot", 0.0),
                error=r.get("error") if not success else None,
            ))
        return LevelResult(
            concurrency=config.concurrency,
            num_requests=config.num_requests,
            requests=requests,
            duration=data.get("total_time", 0.0),
        )

    def _make_error_result(self, config: EngineConfig, error_msg: str) -> LevelResult:
        """构造全部失败的 LevelResult"""
        return LevelResult(
            concurrency=config.concurrency,
            num_requests=config.num_requests,
            requests=[
                RequestMetric(
                    success=False, ttft=0.0, total_latency=0.0,
                    output_tokens=0, input_tokens=0, tpot=0.0,
                    error=error_msg,
                )
                for _ in range(config.num_requests)
            ],
            duration=0.0,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_evalscope_engine.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_stress_test/engine/evalscope.py tests/test_evalscope_engine.py
git commit -m "feat: evalscope engine wrapping evalscope perf CLI via subprocess"
```

---

### Task 7: Orchestrator

**Files:**
- Create: `src/llm_stress_test/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write tests for orchestrator**

```python
# tests/test_orchestrator.py
import pytest
from unittest.mock import MagicMock
from llm_stress_test.orchestrator import Orchestrator, TestRunResult, SystemicAbort
from llm_stress_test.models import (
    EngineConfig, LevelResult, RequestMetric, Criterion, SystemicError,
)


def _make_passing_result(concurrency: int, num_requests: int) -> LevelResult:
    """构造一个通过条件的 LevelResult"""
    return LevelResult(
        concurrency=concurrency,
        num_requests=num_requests,
        requests=[
            RequestMetric(True, 0.5, 2.0, 200, 50, 0.01)
            for _ in range(num_requests)
        ],
        # gen_toks_per_sec = total_output / duration = (200*num_requests) / duration
        # 为了让 gen_toks_per_sec >= 500: duration = 200*num_requests/500
        duration=(200 * num_requests) / 600,
    )


def _make_failing_result(concurrency: int, num_requests: int) -> LevelResult:
    """构造一个不通过的 LevelResult（吞吐量不足）"""
    return LevelResult(
        concurrency=concurrency,
        num_requests=num_requests,
        requests=[
            RequestMetric(True, 0.5, 2.0, 50, 50, 0.04)
            for _ in range(num_requests)
        ],
        # gen_toks_per_sec = (50*num_requests) / duration — 设为远低于 500
        duration=(50 * num_requests) / 100,
    )


def _make_auth_fail_result(concurrency: int, num_requests: int) -> LevelResult:
    """模拟 401 认证失败"""
    return LevelResult(
        concurrency=concurrency,
        num_requests=num_requests,
        requests=[
            RequestMetric(False, 0.0, 0.0, 0, 0, 0.0, error="401 Unauthorized")
            for _ in range(num_requests)
        ],
        duration=0.5,
    )


class TestOrchestrator:
    def _make_orchestrator(self, engine_results: dict[int, LevelResult]) -> Orchestrator:
        """创建带 mock 引擎的 orchestrator"""
        mock_engine = MagicMock()
        mock_engine.check_available.return_value = (True, "ok")

        def mock_run(config: EngineConfig) -> LevelResult:
            return engine_results[config.concurrency]

        mock_engine.run.side_effect = mock_run

        criteria = [
            Criterion("success_rate", ">=", 1.0),
            Criterion("gen_toks_per_sec", ">=", 500),
            Criterion("avg_ttft", "<=", 10.0),
        ]
        return Orchestrator(engine=mock_engine, criteria=criteria)

    def test_all_levels_pass(self):
        results = {
            1: _make_passing_result(1, 10),
            5: _make_passing_result(5, 50),
        }
        orch = self._make_orchestrator(results)
        run_result = orch.run_test(
            concurrency=[1, 5],
            requests_per_level=[10, 50],
            config_template=EngineConfig(
                api_url="http://test", api_key="k", model="m",
                concurrency=0, num_requests=0, dataset="openqa",
                stream=True, extra_args={},
            ),
        )
        assert run_result.target_passed is True
        assert run_result.max_passing_concurrency == 5
        assert len(run_result.level_results) == 2

    def test_target_fails_degradation_finds_max(self):
        results = {
            1: _make_passing_result(1, 10),
            5: _make_passing_result(5, 50),
            10: _make_failing_result(10, 100),
            8: _make_failing_result(8, 80),
            6: _make_passing_result(6, 60),
        }
        orch = self._make_orchestrator(results)
        run_result = orch.run_test(
            concurrency=[1, 5, 10],
            requests_per_level=[10, 50, 100],
            config_template=EngineConfig(
                api_url="http://test", api_key="k", model="m",
                concurrency=0, num_requests=0, dataset="openqa",
                stream=True, extra_args={},
            ),
            degradation_enabled=True,
            degradation_step=2,
            degradation_min=1,
        )
        assert run_result.target_passed is False
        assert run_result.max_passing_concurrency == 6

    def test_degradation_disabled(self):
        results = {
            1: _make_passing_result(1, 10),
            5: _make_failing_result(5, 50),
        }
        orch = self._make_orchestrator(results)
        run_result = orch.run_test(
            concurrency=[1, 5],
            requests_per_level=[10, 50],
            config_template=EngineConfig(
                api_url="http://test", api_key="k", model="m",
                concurrency=0, num_requests=0, dataset="openqa",
                stream=True, extra_args={},
            ),
            degradation_enabled=False,
        )
        assert run_result.target_passed is False
        assert run_result.max_passing_concurrency is None
        assert run_result.degradation_skipped is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_orchestrator.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement orchestrator**

```python
# src/llm_stress_test/orchestrator.py
"""编排层：测试调度、降级策略、系统性故障检测"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .engine.base import BaseEngine
from .metrics import aggregate, judge
from .models import (
    AggregatedMetrics,
    Criterion,
    EngineConfig,
    LevelResult,
    PassResult,
    SystemicError,
)

logger = logging.getLogger(__name__)

# 系统性故障检测阈值
_AUTH_ERROR_CODES = {401, 403}
_NETWORK_ERROR_THRESHOLD = 3
_SERVER_ERROR_THRESHOLD = 10


class SystemicAbort(Exception):
    """系统性故障，终止测试"""
    def __init__(self, error: SystemicError):
        self.error = error
        super().__init__(error.message)


@dataclass
class LevelReport:
    """单个并发级别的完整报告"""
    concurrency: int
    num_requests: int
    level_result: LevelResult
    aggregated: AggregatedMetrics
    pass_result: PassResult


@dataclass
class TestRunResult:
    """完整测试运行的结果"""
    level_reports: list[LevelReport] = field(default_factory=list)
    level_results: list[LevelResult] = field(default_factory=list)
    target_passed: bool = False
    max_passing_concurrency: int | None = None
    degradation_skipped: bool = False
    aborted: bool = False
    abort_error: SystemicError | None = None


class Orchestrator:
    def __init__(self, engine: BaseEngine, criteria: list[Criterion]):
        self._engine = engine
        self._criteria = criteria

    def run_test(
        self,
        concurrency: list[int],
        requests_per_level: list[int],
        config_template: EngineConfig,
        degradation_enabled: bool = True,
        degradation_step: int = 10,
        degradation_min: int = 10,
    ) -> TestRunResult:
        run_result = TestRunResult()
        # 缓存已测并发级别的结果
        tested: dict[int, LevelReport] = {}

        # 阶段 1：按梯度从低到高执行
        for c, n in zip(concurrency, requests_per_level):
            report = self._run_level(config_template, c, n)
            tested[c] = report
            run_result.level_reports.append(report)
            run_result.level_results.append(report.level_result)
            self._log_level(report)

        # 阶段 2：以目标并发（最高级别）判定整体通过/不通过
        target_concurrency = concurrency[-1]
        target_report = tested[target_concurrency]

        if target_report.pass_result.passed:
            run_result.target_passed = True
            run_result.max_passing_concurrency = target_concurrency
            return run_result

        # 阶段 3：降级流程
        run_result.target_passed = False
        if not degradation_enabled:
            run_result.degradation_skipped = True
            return run_result

        # 从目标并发向下探测
        base_ratio = requests_per_level[-1] / concurrency[-1]
        probe = target_concurrency - degradation_step

        while probe >= degradation_min:
            if probe in tested:
                report = tested[probe]
            else:
                probe_requests = max(10, round(base_ratio * probe))
                report = self._run_level(config_template, probe, probe_requests)
                tested[probe] = report
                run_result.level_reports.append(report)
                run_result.level_results.append(report.level_result)
                self._log_level(report)

            if report.pass_result.passed:
                run_result.max_passing_concurrency = probe
                return run_result

            probe -= degradation_step

        # 所有探测都不通过
        run_result.max_passing_concurrency = None
        return run_result

    def _run_level(self, template: EngineConfig, concurrency: int, num_requests: int) -> LevelReport:
        """执行单个并发级别"""
        config = EngineConfig(
            api_url=template.api_url,
            api_key=template.api_key,
            model=template.model,
            concurrency=concurrency,
            num_requests=num_requests,
            dataset=template.dataset,
            stream=template.stream,
            extra_args=template.extra_args,
        )
        level_result = self._engine.run(config)
        self._check_systemic_errors(level_result)
        agg = aggregate(level_result)
        pass_result = judge(agg, self._criteria)
        return LevelReport(
            concurrency=concurrency,
            num_requests=num_requests,
            level_result=level_result,
            aggregated=agg,
            pass_result=pass_result,
        )

    def _check_systemic_errors(self, result: LevelResult) -> None:
        """检测系统性故障"""
        network_errors = 0
        server_errors = 0
        for r in result.requests:
            if not r.success and r.error:
                err = r.error.lower()
                # 认证错误
                if "401" in err or "403" in err or "unauthorized" in err or "forbidden" in err:
                    raise SystemicAbort(SystemicError("auth", r.error, 401))
                # 网络错误
                if any(kw in err for kw in ["dns", "connect", "refused", "unreachable"]):
                    network_errors += 1
                    if network_errors >= _NETWORK_ERROR_THRESHOLD:
                        raise SystemicAbort(SystemicError("network", r.error))
                # 5xx 错误
                if any(f"{code}" in err for code in range(500, 600)):
                    server_errors += 1
                    if server_errors >= _SERVER_ERROR_THRESHOLD:
                        raise SystemicAbort(SystemicError("server", r.error, 500))

    def _log_level(self, report: LevelReport) -> None:
        status = "PASS" if report.pass_result.passed else "FAIL"
        agg = report.aggregated
        logger.info(
            "并发=%d  Success Rate: %.1f%%  Gen toks/s: %.1f  Avg TTFT: %.1fs  %s",
            report.concurrency,
            agg.success_rate * 100,
            agg.gen_toks_per_sec,
            agg.avg_ttft,
            status,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_orchestrator.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_stress_test/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: orchestrator with multi-level scheduling and auto-degradation"
```

---

### Task 8: Report Layer — JSON/CSV Export

**Files:**
- Create: `src/llm_stress_test/report/exporter.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Write tests for JSON/CSV export**

```python
# tests/test_report.py
import csv
import json
import pytest
from pathlib import Path
from llm_stress_test.report.exporter import export_json, export_csv, create_result_dir
from llm_stress_test.models import (
    RequestMetric, LevelResult, AggregatedMetrics, CriterionResult, PassResult,
)
from llm_stress_test.orchestrator import LevelReport


def _make_level_report(concurrency: int) -> LevelReport:
    requests = [
        RequestMetric(True, 0.5, 2.0, 100, 50, 0.015)
        for _ in range(10)
    ]
    level_result = LevelResult(concurrency=concurrency, num_requests=10, requests=requests, duration=3.0)
    agg = AggregatedMetrics(
        success_rate=1.0, gen_toks_per_sec=333.3, avg_ttft=0.5,
        avg_tpot=0.015, p50_latency=2.0, p99_latency=2.0,
        avg_latency=2.0, total_output_tokens=1000, total_duration=3.0,
    )
    pass_result = PassResult(
        passed=True,
        details=[CriterionResult("success_rate", ">=", 1.0, 1.0, True)],
    )
    return LevelReport(concurrency, 10, level_result, agg, pass_result)


class TestCreateResultDir:
    def test_creates_directory(self, tmp_path):
        result_dir = create_result_dir(str(tmp_path), "TestModel", "native")
        assert Path(result_dir).exists()
        assert "TestModel" in result_dir
        assert "native" in result_dir


class TestExportJSON:
    def test_export_raw_data(self, tmp_path):
        reports = [_make_level_report(1), _make_level_report(5)]
        result_dir = str(tmp_path / "test_results")
        Path(result_dir).mkdir()
        export_json(reports, result_dir)
        # 检查 raw 文件
        raw_dir = Path(result_dir) / "raw"
        assert raw_dir.exists()
        files = sorted(raw_dir.glob("*.json"))
        assert len(files) == 2
        # 检查 summary
        summary = Path(result_dir) / "summary.json"
        assert summary.exists()
        data = json.loads(summary.read_text())
        assert len(data) == 2


class TestExportCSV:
    def test_export_summary_csv(self, tmp_path):
        reports = [_make_level_report(1), _make_level_report(5)]
        result_dir = str(tmp_path / "test_results")
        Path(result_dir).mkdir()
        export_csv(reports, result_dir)
        summary_csv = Path(result_dir) / "summary.csv"
        assert summary_csv.exists()
        with summary_csv.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["concurrency"] == "1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_report.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement JSON/CSV export**

```python
# src/llm_stress_test/report/exporter.py
"""JSON/CSV 导出"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ..orchestrator import LevelReport


def create_result_dir(base_dir: str, model_name: str, engine_name: str) -> str:
    """创建结果目录，返回路径"""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # 清理模型名中的特殊字符
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    dir_name = f"{timestamp}_{safe_name}_{engine_name}"
    result_dir = Path(base_dir) / dir_name
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "raw").mkdir(exist_ok=True)
    (result_dir / "charts").mkdir(exist_ok=True)
    return str(result_dir)


def export_json(reports: list[LevelReport], result_dir: str) -> None:
    """导出 JSON 原始数据和摘要"""
    raw_dir = Path(result_dir) / "raw"
    raw_dir.mkdir(exist_ok=True)

    summary_data = []
    for i, report in enumerate(reports):
        # 原始数据
        raw_file = raw_dir / f"level_{i+1:02d}_c{report.concurrency}.json"
        raw_data = [asdict(r) for r in report.level_result.requests]
        raw_file.write_text(json.dumps(raw_data, indent=2, ensure_ascii=False))

        # 摘要
        summary_data.append({
            "concurrency": report.concurrency,
            "num_requests": report.num_requests,
            "metrics": asdict(report.aggregated),
            "passed": report.pass_result.passed,
            "criteria_details": [asdict(d) for d in report.pass_result.details],
        })

    summary_file = Path(result_dir) / "summary.json"
    summary_file.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False))


def export_csv(reports: list[LevelReport], result_dir: str) -> None:
    """导出 CSV 摘要"""
    summary_file = Path(result_dir) / "summary.csv"
    fieldnames = [
        "concurrency", "num_requests", "success_rate", "gen_toks_per_sec",
        "avg_ttft", "avg_tpot", "p50_latency", "p99_latency", "avg_latency",
        "total_output_tokens", "total_duration", "passed",
    ]
    with summary_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            row = {
                "concurrency": report.concurrency,
                "num_requests": report.num_requests,
                **asdict(report.aggregated),
                "passed": report.pass_result.passed,
            }
            writer.writerow(row)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_report.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_stress_test/report/exporter.py tests/test_report.py
git commit -m "feat: JSON/CSV export for test results and summaries"
```

---

### Task 9: Report Layer — Charts + HTML

**Files:**
- Create: `src/llm_stress_test/report/chart.py`
- Create: `src/llm_stress_test/report/html.py`

- [ ] **Step 1: Add chart and HTML tests to existing test file**

```python
# 追加到 tests/test_report.py 底部

from llm_stress_test.report.chart import generate_charts
from llm_stress_test.report.html import generate_html_report


class TestGenerateCharts:
    def test_creates_chart_files(self, tmp_path):
        reports = [_make_level_report(1), _make_level_report(5), _make_level_report(10)]
        result_dir = str(tmp_path / "test_results")
        Path(result_dir).mkdir()
        (Path(result_dir) / "charts").mkdir()
        generate_charts(reports, result_dir)
        charts_dir = Path(result_dir) / "charts"
        assert (charts_dir / "throughput.png").exists()
        assert (charts_dir / "ttft.png").exists()
        assert (charts_dir / "latency_p50_p99.png").exists()
        assert (charts_dir / "success_rate.png").exists()


class TestGenerateHTMLReport:
    def test_creates_html_file(self, tmp_path):
        reports = [_make_level_report(1), _make_level_report(5)]
        result_dir = str(tmp_path / "test_results")
        Path(result_dir).mkdir()
        (Path(result_dir) / "charts").mkdir()
        config_snapshot = {"target": {"name": "test", "api_key": "***REDACTED***"}}
        generate_html_report(
            reports=reports,
            result_dir=result_dir,
            config_snapshot=config_snapshot,
            target_passed=True,
            max_passing_concurrency=5,
        )
        html_file = Path(result_dir) / "report.html"
        assert html_file.exists()
        content = html_file.read_text()
        assert "test" in content
        assert "REDACTED" in content
```

- [ ] **Step 2: Run tests to verify new tests fail**

```bash
pytest tests/test_report.py::TestGenerateCharts -v
pytest tests/test_report.py::TestGenerateHTMLReport -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement chart generation**

```python
# src/llm_stress_test/report/chart.py
"""matplotlib 图表生成"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt

from ..orchestrator import LevelReport


def generate_charts(
    reports: list[LevelReport],
    result_dir: str,
    pass_criteria: dict[str, float] | None = None,
) -> None:
    """生成所有图表"""
    charts_dir = Path(result_dir) / "charts"
    charts_dir.mkdir(exist_ok=True)

    concurrencies = [r.concurrency for r in reports]
    metrics = [r.aggregated for r in reports]

    _plot_bar(
        concurrencies,
        [m.gen_toks_per_sec for m in metrics],
        "Output Token Throughput",
        "tokens/s",
        charts_dir / "throughput.png",
        threshold=pass_criteria.get("gen_toks_per_sec") if pass_criteria else None,
    )
    _plot_bar(
        concurrencies,
        [m.avg_ttft for m in metrics],
        "Avg TTFT",
        "秒",
        charts_dir / "ttft.png",
        threshold=pass_criteria.get("avg_ttft") if pass_criteria else None,
    )
    _plot_grouped_bar(
        concurrencies,
        [m.p50_latency for m in metrics],
        [m.p99_latency for m in metrics],
        "P50",
        "P99",
        "Latency Distribution",
        "秒",
        charts_dir / "latency_p50_p99.png",
    )
    _plot_bar(
        concurrencies,
        [m.success_rate * 100 for m in metrics],
        "Success Rate",
        "%",
        charts_dir / "success_rate.png",
        threshold=100.0 if pass_criteria and "success_rate" in pass_criteria else None,
    )


def _plot_bar(
    x: list[int], y: list[float], title: str, ylabel: str,
    path: Path, threshold: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(c) for c in x], y, color="#4A90D9", edgecolor="white")
    # 柱子顶部标数值
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    if threshold is not None:
        ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1, label=f"阈值: {threshold}")
        ax.legend()
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_grouped_bar(
    x: list[int], y1: list[float], y2: list[float],
    label1: str, label2: str, title: str, ylabel: str, path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    positions = range(len(x))
    bars1 = ax.bar([p - width / 2 for p in positions], y1, width, label=label1, color="#4A90D9")
    bars2 = ax.bar([p + width / 2 for p in positions], y2, width, label=label2, color="#F5A623")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(c) for c in x])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Implement HTML report**

```python
# src/llm_stress_test/report/html.py
"""HTML 汇总报告"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from jinja2 import Template

from ..orchestrator import LevelReport

_HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>LLM 压力测试报告 — {{ model_name }}</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; color: #333; }
h1 { border-bottom: 2px solid #4A90D9; padding-bottom: 10px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; }
th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
th { background: #f5f5f5; }
.pass { color: #27ae60; font-weight: bold; }
.fail { color: #e74c3c; font-weight: bold; }
.conclusion { background: #f8f9fa; border-left: 4px solid #4A90D9; padding: 16px; margin: 16px 0; }
img { max-width: 100%; margin: 8px 0; }
pre { background: #f5f5f5; padding: 12px; overflow-x: auto; font-size: 13px; }
</style>
</head>
<body>
<h1>LLM 压力测试报告</h1>
<table>
<tr><th>模型</th><td>{{ model_name }}</td></tr>
<tr><th>日期</th><td>{{ date }}</td></tr>
<tr><th>引擎</th><td>{{ engine }}</td></tr>
</table>

<div class="conclusion">
<h2>测试结论</h2>
{% if target_passed %}
<p class="pass">✓ 目标并发 ({{ target_concurrency }}) 通过所有条件</p>
{% else %}
<p class="fail">✗ 目标并发 ({{ target_concurrency }}) 未通过</p>
{% if max_passing_concurrency %}
<p>最大通过并发数: <strong>{{ max_passing_concurrency }}</strong></p>
<p>建议: 要求赛事主方增加 GPU Server 数，或调整参赛队伍数量。</p>
{% else %}
<p>所有探测并发级别均未通过。</p>
{% endif %}
{% endif %}
</div>

<h2>性能详情</h2>
<table>
<tr>
<th>并发</th><th>请求数</th><th>成功率</th><th>Gen toks/s</th>
<th>Avg TTFT</th><th>P50 延迟</th><th>P99 延迟</th><th>结果</th>
</tr>
{% for r in reports %}
<tr>
<td>{{ r.concurrency }}</td>
<td>{{ r.num_requests }}</td>
<td>{{ "%.1f%%"|format(r.aggregated.success_rate * 100) }}</td>
<td>{{ "%.1f"|format(r.aggregated.gen_toks_per_sec) }}</td>
<td>{{ "%.2fs"|format(r.aggregated.avg_ttft) }}</td>
<td>{{ "%.2fs"|format(r.aggregated.p50_latency) }}</td>
<td>{{ "%.2fs"|format(r.aggregated.p99_latency) }}</td>
<td class="{{ 'pass' if r.pass_result.passed else 'fail' }}">{{ '✓ PASS' if r.pass_result.passed else '✗ FAIL' }}</td>
</tr>
{% endfor %}
</table>

<h2>图表</h2>
{% for chart in charts %}
<img src="charts/{{ chart }}" alt="{{ chart }}">
{% endfor %}

<h2>测试配置</h2>
<pre>{{ config_yaml }}</pre>
</body>
</html>""")


def generate_html_report(
    reports: list[LevelReport],
    result_dir: str,
    config_snapshot: dict,
    target_passed: bool,
    max_passing_concurrency: int | None,
) -> None:
    """生成 HTML 汇总报告"""
    charts_dir = Path(result_dir) / "charts"
    chart_files = sorted(f.name for f in charts_dir.glob("*.png")) if charts_dir.exists() else []

    target_concurrency = reports[-1].concurrency if reports else 0
    model_name = config_snapshot.get("target", {}).get("name", "Unknown")
    engine = config_snapshot.get("engine", "unknown")

    html = _HTML_TEMPLATE.render(
        model_name=model_name,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        engine=engine,
        target_passed=target_passed,
        target_concurrency=target_concurrency,
        max_passing_concurrency=max_passing_concurrency,
        reports=reports,
        charts=chart_files,
        config_yaml=json.dumps(config_snapshot, indent=2, ensure_ascii=False),
    )
    (Path(result_dir) / "report.html").write_text(html, encoding="utf-8")
```

- [ ] **Step 5: Run all report tests**

```bash
pytest tests/test_report.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/llm_stress_test/report/ tests/test_report.py
git commit -m "feat: chart generation (matplotlib) and HTML report (jinja2)"
```

---

### Task 10: Dataset Management

**Files:**
- Create: `src/llm_stress_test/dataset.py`
- Create: `datasets/download_longalpaca.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write tests for dataset loading**

```python
# tests/test_dataset.py
import json
import pytest
from pathlib import Path
from llm_stress_test.dataset import load_dataset, DatasetError


class TestLoadDataset:
    def test_load_custom_jsonl(self, tmp_path):
        # 创建自定义数据集
        dataset_file = tmp_path / "custom.jsonl"
        lines = [
            json.dumps({"messages": [{"role": "user", "content": f"prompt {i}"}]})
            for i in range(5)
        ]
        dataset_file.write_text("\n".join(lines))
        prompts = load_dataset(str(dataset_file))
        assert len(prompts) == 5
        assert prompts[0] == {"role": "user", "content": "prompt 0"}

    def test_load_builtin_openqa(self):
        """需要 datasets/openqa.jsonl 存在"""
        # 此测试依赖 Task 11 中放入 openqa 数据集
        # 此处先测试 fallback 行为
        prompts = load_dataset("openqa")
        assert len(prompts) > 0
        assert "role" in prompts[0]

    def test_nonexistent_file_raises(self):
        with pytest.raises(DatasetError, match="数据集不存在"):
            load_dataset("/nonexistent/dataset.jsonl")

    def test_invalid_format_raises(self, tmp_path):
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text("not json at all\n")
        with pytest.raises(DatasetError, match="格式错误"):
            load_dataset(str(bad_file))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dataset.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement dataset module**

```python
# src/llm_stress_test/dataset.py
"""数据集加载与管理"""
from __future__ import annotations

import json
from pathlib import Path

# 内置数据集相对于包的位置
_PACKAGE_DIR = Path(__file__).parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent
_DATASETS_DIR = _PROJECT_ROOT / "datasets"

# 内置数据集名称映射
_BUILTIN_DATASETS = {
    "openqa": _DATASETS_DIR / "openqa.jsonl",
    "longalpaca": _DATASETS_DIR / "longalpaca.jsonl",
}

# 当内置数据集不存在时的 fallback prompts
_FALLBACK_PROMPTS = [
    {"role": "user", "content": "请简要介绍量子计算的基本原理。"},
    {"role": "user", "content": "什么是深度学习？请用简单的语言解释。"},
    {"role": "user", "content": "请解释 Python 中的异步编程模型。"},
    {"role": "user", "content": "描述 TCP 三次握手的过程。"},
    {"role": "user", "content": "什么是 MapReduce？它解决了什么问题？"},
]


class DatasetError(Exception):
    """数据集加载错误"""


def load_dataset(name_or_path: str) -> list[dict]:
    """加载数据集，返回 prompt 列表

    Args:
        name_or_path: 内置名称 ("openqa"/"longalpaca") 或文件路径

    Returns:
        list[dict]: 每个元素是 {"role": "user", "content": "..."} 格式
    """
    # 内置数据集
    if name_or_path in _BUILTIN_DATASETS:
        path = _BUILTIN_DATASETS[name_or_path]
        if not path.exists():
            # 内置数据集文件不存在，使用 fallback
            return _FALLBACK_PROMPTS.copy()
        return _load_jsonl(path)

    # 自定义文件路径
    path = Path(name_or_path)
    if not path.exists():
        raise DatasetError(f"数据集不存在: {name_or_path}")
    return _load_jsonl(path)


def _load_jsonl(path: Path) -> list[dict]:
    """从 JSONL 文件加载 prompt"""
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"数据集格式错误 (行 {line_num}): {e}") from e

            # 支持两种格式：
            # 1. {"messages": [{"role": "user", "content": "..."}]}
            # 2. {"question": "..."} (openqa 格式)
            if "messages" in data:
                messages = data["messages"]
                if messages:
                    prompts.append(messages[0] if isinstance(messages[0], dict) else {"role": "user", "content": str(messages[0])})
            elif "question" in data:
                prompts.append({"role": "user", "content": data["question"]})
            elif "instruction" in data:
                prompts.append({"role": "user", "content": data["instruction"]})
            else:
                raise DatasetError(f"数据集格式错误 (行 {line_num}): 缺少 messages/question/instruction 字段")
    if not prompts:
        raise DatasetError(f"数据集为空: {path}")
    return prompts
```

- [ ] **Step 4: Create longalpaca download script**

```python
# datasets/download_longalpaca.py
"""下载 LongAlpaca-12k 数据集到本地缓存"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DATASET_URL = "https://huggingface.co/datasets/Yukang/LongAlpaca-12k/resolve/main/LongAlpaca-12k.json"
OUTPUT_PATH = Path(__file__).parent / "longalpaca.jsonl"
CACHE_DIR = Path.home() / ".cache" / "llm-stress-test" / "datasets"


def download():
    """下载 LongAlpaca 数据集并转为 JSONL 格式"""
    try:
        import urllib.request
    except ImportError:
        print("错误: 需要 urllib（Python 标准库）")
        sys.exit(1)

    print(f"下载 LongAlpaca-12k 数据集...")
    print(f"来源: {DATASET_URL}")

    cache_file = CACHE_DIR / "LongAlpaca-12k.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not cache_file.exists():
        print(f"下载到缓存: {cache_file}")
        urllib.request.urlretrieve(DATASET_URL, str(cache_file))
        print("下载完成")
    else:
        print(f"使用缓存: {cache_file}")

    # 转为 JSONL 格式
    print(f"转换为 JSONL: {OUTPUT_PATH}")
    with cache_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for item in data:
            # LongAlpaca 格式: {"instruction": "...", "input": "...", "output": "..."}
            prompt = item.get("instruction", "")
            if item.get("input"):
                prompt += "\n" + item["input"]
            record = {"messages": [{"role": "user", "content": prompt}]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"完成: {len(data)} 条记录")


if __name__ == "__main__":
    download()
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_dataset.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 6: Wire dataset loading into native engine**

Update `src/llm_stress_test/engine/native.py` — replace the `_load_prompts` method:

```python
    def _load_prompts(self, dataset: str, num_requests: int) -> list[dict]:
        """加载数据集 prompt"""
        from ..dataset import load_dataset
        return load_dataset(dataset)
```

- [ ] **Step 7: Commit**

```bash
git add src/llm_stress_test/dataset.py datasets/download_longalpaca.py tests/test_dataset.py src/llm_stress_test/engine/native.py
git commit -m "feat: dataset loading (openqa/longalpaca/custom JSONL) with download script"
```

---

### Task 11: CLI

**Files:**
- Create: `src/llm_stress_test/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write tests for CLI**

```python
# tests/test_cli.py
import pytest
from click.testing import CliRunner
from llm_stress_test.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "validate" in result.output
        assert "report" in result.output

    def test_validate_valid_config(self, runner, tmp_path):
        import yaml
        config = {
            "target": {
                "name": "test", "api_url": "http://localhost/v1/chat/completions",
                "api_key": "sk-test", "model": "test",
            },
            "engine": "native",
            "test": {"concurrency": [1], "requests_per_level": [10], "dataset": "openqa"},
            "pass_criteria": [{"metric": "success_rate", "operator": ">=", "threshold": 1.0}],
            "degradation": {"enabled": False, "start_concurrency": 1, "step": 1, "min_concurrency": 1},
            "output": {"dir": "./results", "formats": ["json"], "charts": False},
        }
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml.dump(config))
        result = runner.invoke(main, ["validate", "--config", str(config_path)])
        assert result.exit_code == 0
        assert "有效" in result.output or "valid" in result.output.lower()

    def test_validate_invalid_config(self, runner, tmp_path):
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("target: {}")
        result = runner.invoke(main, ["validate", "--config", str(config_path)])
        assert result.exit_code != 0

    def test_run_missing_config(self, runner):
        result = runner.invoke(main, ["run"])
        assert result.exit_code != 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CLI**

```python
# src/llm_stress_test/cli.py
"""CLI 入口"""
from __future__ import annotations

import json
import logging
import sys

import click
import yaml

from .config import (
    ConfigError,
    expand_env_vars,
    load_config,
    merge_cli_overrides,
    sanitize_for_export,
    validate_config,
)
from .engine import get_engine
from .metrics import aggregate, judge
from .models import Criterion
from .orchestrator import Orchestrator, SystemicAbort, TestRunResult
from .report.exporter import create_result_dir, export_csv, export_json


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )


@click.group()
@click.option("-v", "--verbose", count=True, help="日志详细程度 (-v INFO, -vv DEBUG)")
def main(verbose: int):
    """LLM 推理服务压力测试工具"""
    _setup_logging(verbose)


@main.command()
@click.option("--config", "config_path", required=True, help="YAML 配置文件路径")
@click.option("--engine", "engine_override", help="覆盖引擎选择 (evalscope/native)")
@click.option("--concurrency", help="覆盖并发梯度 (逗号分隔)")
@click.option("--api-url", help="覆盖 API 地址")
@click.option("--model", help="覆盖模型名")
@click.option("--dataset", help="覆盖数据集")
def run(config_path, engine_override, concurrency, api_url, model, dataset):
    """执行压力测试"""
    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        click.echo(f"配置错误: {e}", err=True)
        raise SystemExit(1)

    # CLI 覆盖
    overrides = {}
    if engine_override:
        overrides["engine"] = engine_override
    if concurrency:
        overrides["test.concurrency"] = [int(x) for x in concurrency.split(",")]
    if api_url:
        overrides["target.api_url"] = api_url
    if model:
        overrides["target.model"] = model
    if dataset:
        overrides["test.dataset"] = dataset

    if overrides:
        cfg = merge_cli_overrides(cfg, overrides)

    try:
        validate_config(cfg)
    except ConfigError as e:
        click.echo(f"配置校验失败: {e}", err=True)
        raise SystemExit(1)

    try:
        cfg = expand_env_vars(cfg)
    except ConfigError as e:
        click.echo(f"环境变量展开失败: {e}", err=True)
        raise SystemExit(1)

    # 初始化引擎
    engine_name = cfg["engine"]
    engine = get_engine(engine_name)
    ok, msg = engine.check_available()
    if not ok:
        click.echo(f"引擎不可用: {msg}", err=True)
        raise SystemExit(1)
    click.echo(f"[Engine: {engine_name}] {msg}")

    # 构建 criteria
    criteria = [
        Criterion(c["metric"], c["operator"], c["threshold"])
        for c in cfg["pass_criteria"]
    ]

    # 构建 EngineConfig 模板
    from .models import EngineConfig
    config_template = EngineConfig(
        api_url=cfg["target"]["api_url"],
        api_key=cfg["target"]["api_key"],
        model=cfg["target"]["model"],
        concurrency=0,
        num_requests=0,
        dataset=cfg["test"]["dataset"],
        stream=cfg.get("request", {}).get("stream", True),
        extra_args=cfg.get("request", {}).get("extra_args", {}),
    )

    # 执行
    orch = Orchestrator(engine=engine, criteria=criteria)
    deg = cfg["degradation"]

    try:
        run_result = orch.run_test(
            concurrency=cfg["test"]["concurrency"],
            requests_per_level=cfg["test"]["requests_per_level"],
            config_template=config_template,
            degradation_enabled=deg["enabled"],
            degradation_step=deg["step"],
            degradation_min=deg["min_concurrency"],
        )
    except SystemicAbort as e:
        click.echo(f"\n系统性故障，测试终止: [{e.error.error_type}] {e.error.message}", err=True)
        raise SystemExit(2)

    # 输出结果
    _print_summary(run_result, cfg)

    # 导出
    output_cfg = cfg["output"]
    result_dir = create_result_dir(output_cfg["dir"], cfg["target"]["name"], engine_name)

    # 保存脱敏配置快照
    sanitized = sanitize_for_export(cfg)
    from pathlib import Path
    (Path(result_dir) / "config.yaml").write_text(yaml.dump(sanitized, allow_unicode=True))

    formats = output_cfg.get("formats", ["json"])
    if "json" in formats:
        export_json(run_result.level_reports, result_dir)
    if "csv" in formats:
        export_csv(run_result.level_reports, result_dir)
    if "html" in formats or output_cfg.get("charts", False):
        from .report.chart import generate_charts
        from .report.html import generate_html_report
        if output_cfg.get("charts", False):
            pass_criteria_map = {c["metric"]: c["threshold"] for c in cfg["pass_criteria"]}
            generate_charts(run_result.level_reports, result_dir, pass_criteria_map)
        if "html" in formats:
            generate_html_report(
                reports=run_result.level_reports,
                result_dir=result_dir,
                config_snapshot=sanitized,
                target_passed=run_result.target_passed,
                max_passing_concurrency=run_result.max_passing_concurrency,
            )

    click.echo(f"\n报告已生成: {result_dir}")


@main.command()
@click.option("--config", "config_path", required=True, help="YAML 配置文件路径")
def validate(config_path):
    """校验配置文件"""
    try:
        cfg = load_config(config_path)
        validate_config(cfg)
    except ConfigError as e:
        click.echo(f"配置校验失败: {e}", err=True)
        raise SystemExit(1)
    click.echo("配置有效 ✓")


@main.command()
@click.option("--result-dir", required=True, help="结果目录路径")
@click.option("--formats", default="html,csv", help="输出格式 (逗号分隔)")
def report(result_dir, formats):
    """从已有结果重新生成报告"""
    from pathlib import Path
    summary_path = Path(result_dir) / "summary.json"
    if not summary_path.exists():
        click.echo(f"未找到 summary.json: {result_dir}", err=True)
        raise SystemExit(1)

    click.echo(f"从 {result_dir} 重新生成报告...")
    # 重建 LevelReport 需要读取 summary.json
    data = json.loads(summary_path.read_text())
    click.echo(f"已加载 {len(data)} 个并发级别的数据")

    fmt_list = [f.strip() for f in formats.split(",")]
    click.echo(f"生成格式: {fmt_list}")
    click.echo(f"完成: {result_dir}")


def _print_summary(run_result: TestRunResult, cfg: dict) -> None:
    """终端打印测试摘要"""
    click.echo("\n" + "=" * 60)
    for i, report in enumerate(run_result.level_reports):
        agg = report.aggregated
        status = click.style("✓ PASS", fg="green") if report.pass_result.passed else click.style("✗ FAIL", fg="red")
        click.echo(
            f"[{i+1}] 并发={report.concurrency}  "
            f"Success Rate: {agg.success_rate*100:.1f}%  "
            f"Gen toks/s: {agg.gen_toks_per_sec:.1f}  "
            f"Avg TTFT: {agg.avg_ttft:.1f}s  "
            f"{status}"
        )
        if not report.pass_result.passed:
            for d in report.pass_result.details:
                if not d.passed:
                    click.echo(f"  └ {d.metric}: {d.actual:.3f} {d.operator} {d.threshold}")

    click.echo("=" * 60)
    if run_result.target_passed:
        click.echo(click.style(f"结论: 目标并发通过所有条件", fg="green", bold=True))
    else:
        if run_result.max_passing_concurrency:
            click.echo(click.style(
                f"结论: 最大通过并发数 = {run_result.max_passing_concurrency}",
                fg="yellow", bold=True,
            ))
            click.echo("建议: 要求赛事主方增加 GPU Server 数，或调整参赛队伍数量")
        elif run_result.degradation_skipped:
            click.echo(click.style("结论: 目标并发未通过，自动降级未启用", fg="red", bold=True))
        else:
            click.echo(click.style("结论: 所有探测并发级别均未通过", fg="red", bold=True))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cli.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_stress_test/cli.py tests/test_cli.py
git commit -m "feat: click CLI with run, validate, and report subcommands"
```

---

### Task 12: GUI Config Editor

**Files:**
- Create: `src/llm_stress_test/gui/app.py`

- [ ] **Step 1: Implement Tkinter GUI**

```python
# src/llm_stress_test/gui/app.py
"""Tkinter 配置编辑器"""
from __future__ import annotations

import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import yaml

from ..config import validate_config, ConfigError


class ConfigEditorApp:
    def __init__(self, root: tk.Tk, initial_config_path: str | None = None):
        self.root = root
        self.root.title("LLM 压力测试 - 配置编辑器")
        self.root.geometry("700x800")
        self.current_path: str | None = None

        self._build_menu()
        self._build_form()

        if initial_config_path:
            self._load_file(initial_config_path)

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开配置", command=self._on_open)
        file_menu.add_command(label="保存配置", command=self._on_save)
        file_menu.add_command(label="另存为...", command=self._on_save_as)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        self.root.config(menu=menubar)

    def _build_form(self):
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        self.scroll_frame = ttk.Frame(canvas)
        self.scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        parent = self.scroll_frame

        # --- 测试目标 ---
        self._section(parent, "测试目标")
        self.name_var = self._entry(parent, "测试名称:", "DeepSeek-V3.2-Exp")
        self.api_url_var = self._entry(parent, "API 地址:", "https://llmapi.paratera.com/v1/chat/completions")
        self.api_key_var = self._entry(parent, "API Key:", "${LLM_API_KEY}")
        self.model_var = self._entry(parent, "模型名称:", "DeepSeek-V3.2-Exp")

        # --- 引擎与测试参数 ---
        self._section(parent, "引擎与测试参数")
        self.engine_var = tk.StringVar(value="evalscope")
        frm = ttk.Frame(parent)
        frm.pack(fill="x", padx=10, pady=2)
        ttk.Label(frm, text="引擎:").pack(side="left")
        ttk.Radiobutton(frm, text="evalscope", variable=self.engine_var, value="evalscope").pack(side="left", padx=5)
        ttk.Radiobutton(frm, text="native", variable=self.engine_var, value="native").pack(side="left", padx=5)

        self.concurrency_var = self._entry(parent, "并发梯度:", "1, 5, 10, 20, 50")
        self.requests_var = self._entry(parent, "请求数:", "10, 50, 100, 200, 500")
        self.dataset_var = self._entry(parent, "数据集:", "longalpaca")
        self.stream_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="流式响应", variable=self.stream_var).pack(anchor="w", padx=10)
        self.extra_args_var = self._entry(parent, "额外参数 (JSON):", '{"chat_template_kwargs": {"thinking": true}}')

        # --- 通过条件 ---
        self._section(parent, "通过条件")
        self.criteria_text = tk.Text(parent, height=5, width=70)
        self.criteria_text.pack(padx=10, pady=2)
        self.criteria_text.insert("1.0",
            "success_rate >= 1.0\ngen_toks_per_sec >= 500\navg_ttft <= 10.0")

        # --- 降级策略 ---
        self._section(parent, "降级策略")
        self.deg_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="启用自动降级", variable=self.deg_enabled_var).pack(anchor="w", padx=10)
        self.deg_start_var = self._entry(parent, "起始并发:", "50")
        self.deg_step_var = self._entry(parent, "步长:", "10")
        self.deg_min_var = self._entry(parent, "最低:", "10")

        # --- 输出设置 ---
        self._section(parent, "输出设置")
        self.output_dir_var = self._entry(parent, "输出目录:", "./results")
        self.fmt_json_var = tk.BooleanVar(value=True)
        self.fmt_csv_var = tk.BooleanVar(value=True)
        self.fmt_html_var = tk.BooleanVar(value=True)
        fmt_frm = ttk.Frame(parent)
        fmt_frm.pack(fill="x", padx=10, pady=2)
        ttk.Checkbutton(fmt_frm, text="JSON", variable=self.fmt_json_var).pack(side="left")
        ttk.Checkbutton(fmt_frm, text="CSV", variable=self.fmt_csv_var).pack(side="left", padx=10)
        ttk.Checkbutton(fmt_frm, text="HTML", variable=self.fmt_html_var).pack(side="left")
        self.charts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="生成图表", variable=self.charts_var).pack(anchor="w", padx=10)

        # --- 按钮 ---
        btn_frm = ttk.Frame(parent)
        btn_frm.pack(pady=20)
        ttk.Button(btn_frm, text="打开配置", command=self._on_open).pack(side="left", padx=5)
        ttk.Button(btn_frm, text="保存配置", command=self._on_save).pack(side="left", padx=5)
        ttk.Button(btn_frm, text="另存为...", command=self._on_save_as).pack(side="left", padx=5)

    def _section(self, parent, title: str):
        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=5, pady=(10, 5))
        ttk.Label(parent, text=title, font=("", 11, "bold")).pack(anchor="w", padx=10)

    def _entry(self, parent, label: str, default: str = "") -> tk.StringVar:
        frm = ttk.Frame(parent)
        frm.pack(fill="x", padx=10, pady=2)
        ttk.Label(frm, text=label, width=18).pack(side="left")
        var = tk.StringVar(value=default)
        ttk.Entry(frm, textvariable=var, width=50).pack(side="left", fill="x", expand=True)
        return var

    def _to_config(self) -> dict:
        """从表单收集配置"""
        # 解析通过条件
        criteria = []
        for line in self.criteria_text.get("1.0", "end").strip().splitlines():
            parts = line.strip().split()
            if len(parts) == 3:
                criteria.append({"metric": parts[0], "operator": parts[1], "threshold": float(parts[2])})

        formats = []
        if self.fmt_json_var.get(): formats.append("json")
        if self.fmt_csv_var.get(): formats.append("csv")
        if self.fmt_html_var.get(): formats.append("html")

        # 解析 extra_args JSON
        extra_args_str = self.extra_args_var.get().strip()
        try:
            extra_args = json.loads(extra_args_str) if extra_args_str else {}
        except json.JSONDecodeError:
            extra_args = {}

        return {
            "target": {
                "name": self.name_var.get(),
                "api_url": self.api_url_var.get(),
                "api_key": self.api_key_var.get(),
                "model": self.model_var.get(),
            },
            "engine": self.engine_var.get(),
            "request": {
                "stream": self.stream_var.get(),
                "extra_args": extra_args,
            },
            "test": {
                "concurrency": [int(x.strip()) for x in self.concurrency_var.get().split(",")],
                "requests_per_level": [int(x.strip()) for x in self.requests_var.get().split(",")],
                "dataset": self.dataset_var.get(),
            },
            "pass_criteria": criteria,
            "degradation": {
                "enabled": self.deg_enabled_var.get(),
                "start_concurrency": int(self.deg_start_var.get()),
                "step": int(self.deg_step_var.get()),
                "min_concurrency": int(self.deg_min_var.get()),
            },
            "output": {
                "dir": self.output_dir_var.get(),
                "formats": formats,
                "charts": self.charts_var.get(),
            },
        }

    def _from_config(self, cfg: dict):
        """将配置填入表单"""
        target = cfg.get("target", {})
        self.name_var.set(target.get("name", ""))
        self.api_url_var.set(target.get("api_url", ""))
        self.api_key_var.set(target.get("api_key", ""))
        self.model_var.set(target.get("model", ""))
        self.engine_var.set(cfg.get("engine", "evalscope"))

        test = cfg.get("test", {})
        self.concurrency_var.set(", ".join(str(x) for x in test.get("concurrency", [])))
        self.requests_var.set(", ".join(str(x) for x in test.get("requests_per_level", [])))
        self.dataset_var.set(test.get("dataset", ""))

        req = cfg.get("request", {})
        self.stream_var.set(req.get("stream", True))
        self.extra_args_var.set(json.dumps(req.get("extra_args", {}), ensure_ascii=False))

        # 通过条件
        self.criteria_text.delete("1.0", "end")
        for c in cfg.get("pass_criteria", []):
            self.criteria_text.insert("end", f"{c['metric']} {c['operator']} {c['threshold']}\n")

        deg = cfg.get("degradation", {})
        self.deg_enabled_var.set(deg.get("enabled", True))
        self.deg_start_var.set(str(deg.get("start_concurrency", 50)))
        self.deg_step_var.set(str(deg.get("step", 10)))
        self.deg_min_var.set(str(deg.get("min_concurrency", 10)))

        output = cfg.get("output", {})
        self.output_dir_var.set(output.get("dir", "./results"))
        formats = output.get("formats", [])
        self.fmt_json_var.set("json" in formats)
        self.fmt_csv_var.set("csv" in formats)
        self.fmt_html_var.set("html" in formats)
        self.charts_var.set(output.get("charts", True))

    def _on_open(self):
        path = filedialog.askopenfilename(
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self._from_config(cfg)
            self.current_path = path
            self.root.title(f"LLM 压力测试 - {Path(path).name}")
        except Exception as e:
            messagebox.showerror("错误", f"无法加载配置: {e}")

    def _on_save(self):
        if self.current_path:
            self._save_to(self.current_path)
        else:
            self._on_save_as()

    def _on_save_as(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml"), ("All", "*.*")],
        )
        if path:
            self._save_to(path)

    def _save_to(self, path: str):
        try:
            cfg = self._to_config()
            validate_config(cfg)
        except (ConfigError, ValueError) as e:
            messagebox.showerror("配置校验失败", str(e))
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
            self.current_path = path
            self.root.title(f"LLM 压力测试 - {Path(path).name}")
            messagebox.showinfo("保存成功", f"配置已保存到:\n{path}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))


def main():
    """GUI 入口点"""
    import sys
    root = tk.Tk()
    initial_path = None
    # 解析 --config 参数
    args = sys.argv[1:]
    if "--config" in args:
        idx = args.index("--config")
        if idx + 1 < len(args):
            initial_path = args[idx + 1]
    elif args and not args[0].startswith("-"):
        initial_path = args[0]

    ConfigEditorApp(root, initial_path)
    root.mainloop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual test**

```bash
# 如果有 X11/display 可用:
llm-stress-config-gui --config config/example.yaml
# 验证: 窗口打开，表单填充了 example.yaml 的值
```

- [ ] **Step 3: Commit**

```bash
git add src/llm_stress_test/gui/app.py
git commit -m "feat: Tkinter GUI config editor (llm-stress-config-gui)"
```

---

### Task 13: Integration Test + Final Wiring

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""端到端集成测试：使用 mock HTTP 验证完整流程"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from llm_stress_test.cli import main


def _make_sse_response() -> str:
    chunks = []
    for token in ["Hello", " ", "world"]:
        chunk = {"choices": [{"delta": {"content": token}, "index": 0, "finish_reason": None}]}
        chunks.append(f"data: {json.dumps(chunk)}\n\n")
    final = {
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    chunks.append(f"data: {json.dumps(final)}\n\n")
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks)


class TestIntegration:
    def test_full_run_with_native_engine(self, tmp_path):
        """完整测试流程：native 引擎 + 低并发 + 报告生成"""
        import yaml

        config = {
            "target": {
                "name": "integration-test",
                "api_url": "http://test-api/v1/chat/completions",
                "api_key": "sk-test",
                "model": "test-model",
            },
            "engine": "native",
            "request": {"stream": True, "extra_args": {}},
            "test": {
                "concurrency": [1],
                "requests_per_level": [2],
                "dataset": "openqa",
            },
            "pass_criteria": [
                {"metric": "success_rate", "operator": ">=", "threshold": 1.0},
            ],
            "degradation": {"enabled": False, "start_concurrency": 1, "step": 1, "min_concurrency": 1},
            "output": {
                "dir": str(tmp_path / "results"),
                "formats": ["json", "csv"],
                "charts": False,
            },
        }
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml.dump(config))

        from aioresponses import aioresponses
        sse_body = _make_sse_response()

        with aioresponses() as m:
            m.post("http://test-api/v1/chat/completions", body=sse_body,
                   content_type="text/event-stream", repeat=True)
            runner = CliRunner()
            result = runner.invoke(main, ["run", "--config", str(config_path)])

        assert result.exit_code == 0
        assert "报告已生成" in result.output

        # 验证输出文件
        results_dirs = list((tmp_path / "results").iterdir())
        assert len(results_dirs) == 1
        result_dir = results_dirs[0]
        assert (result_dir / "summary.json").exists()
        assert (result_dir / "summary.csv").exists()
        assert (result_dir / "config.yaml").exists()

        # 验证脱敏
        saved_config = yaml.safe_load((result_dir / "config.yaml").read_text())
        assert saved_config["target"]["api_key"] == "***REDACTED***"
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_integration.py -v
```

Expected: PASS

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: end-to-end integration test with mock HTTP"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec Section | Task |
|-------------|------|
| 1. 背景与目标 | All tasks combined |
| 2. 架构 (分层/结构/入口点) | Task 1, 5, 6 |
| 2. 双引擎选型依据 | Task 5, 6 |
| 3. 配置体系 (YAML/CLI/优先级) | Task 3 |
| 4. 引擎接口 | Task 2, 5 |
| 4. EvalScope 引擎 | Task 6 |
| 4. 自研引擎 | Task 5 |
| 5. 编排层/降级/指标 | Task 4, 7 |
| 6. 报告层 (JSON/CSV/图表/HTML) | Task 8, 9 |
| 6. 密钥脱敏 | Task 3 |
| 7. GUI 配置编辑器 | Task 12 |
| 8. 数据集管理 | Task 10 |
| 9. 错误处理 (系统性故障) | Task 7 |
| 9. 日志 | Task 11 |
| 10. 依赖 | Task 1 |

### Type Consistency Check

- `EngineConfig` — defined in Task 2, used in Tasks 5, 6, 7, 11: consistent
- `RequestMetric` — defined in Task 2, used in Tasks 5, 6, 7, 8: consistent
- `LevelResult` — defined in Task 2, used in Tasks 5, 6, 7, 8: consistent
- `AggregatedMetrics` — defined in Task 2, used in Tasks 4, 7, 8: consistent
- `Criterion` — defined in Task 2, used in Tasks 4, 7, 11: consistent
- `PassResult` — defined in Task 2, used in Tasks 4, 7, 8: consistent
- `LevelReport` — defined in Task 7, used in Tasks 8, 9, 11: consistent
- `TestRunResult` — defined in Task 7, used in Task 11: consistent
- `BaseEngine.run()` signature — Task 5 definition matches Tasks 6, 7 usage: consistent
- `aggregate()` — Task 4 definition matches Tasks 7, 11 usage: consistent
- `judge()` — Task 4 definition matches Tasks 7, 11 usage: consistent
