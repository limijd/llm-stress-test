"""所有共享数据模型"""
from __future__ import annotations
import operator as op
from dataclasses import dataclass, field

_OPERATORS: dict[str, callable] = {
    ">=": op.ge, "<=": op.le, ">": op.gt, "<": op.lt, "==": op.eq,
}

@dataclass(frozen=True)
class EngineConfig:
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
    success: bool
    ttft: float
    total_latency: float
    output_tokens: int
    input_tokens: int
    tpot: float
    error: str | None = None

@dataclass(frozen=True)
class LevelResult:
    concurrency: int
    num_requests: int
    requests: list[RequestMetric]
    duration: float

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class Criterion:
    metric: str
    operator: str
    threshold: float

    def evaluate(self, actual: float) -> bool:
        fn = _OPERATORS.get(self.operator)
        if fn is None:
            raise ValueError(f"不支持的操作符: {self.operator}")
        return fn(actual, self.threshold)

@dataclass(frozen=True)
class CriterionResult:
    metric: str
    operator: str
    threshold: float
    actual: float
    passed: bool

@dataclass(frozen=True)
class PassResult:
    passed: bool
    details: list[CriterionResult]

@dataclass(frozen=True)
class SystemicError:
    error_type: str
    message: str
    status_code: int | None = None
