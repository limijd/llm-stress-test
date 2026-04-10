"""指标聚合与通过判定"""
from __future__ import annotations
import statistics
from dataclasses import asdict
from .models import AggregatedMetrics, Criterion, CriterionResult, LevelResult, PassResult

def aggregate(result: LevelResult) -> AggregatedMetrics:
    successful = [r for r in result.requests if r.success]
    total = len(result.requests)
    success_count = len(successful)
    if success_count == 0:
        return AggregatedMetrics(success_rate=0.0, gen_toks_per_sec=0.0, avg_ttft=0.0,
                                 avg_tpot=0.0, p50_latency=0.0, p99_latency=0.0,
                                 avg_latency=0.0, total_output_tokens=0, total_duration=result.duration)
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
    agg_dict = asdict(aggregated)
    details = []
    for c in criteria:
        actual = agg_dict.get(c.metric)
        if actual is None:
            raise ValueError(f"指标 {c.metric} 在 AggregatedMetrics 中不存在")
        passed = c.evaluate(actual)
        details.append(CriterionResult(metric=c.metric, operator=c.operator,
                                       threshold=c.threshold, actual=actual, passed=passed))
    return PassResult(passed=all(d.passed for d in details), details=details)

def _percentile(sorted_data: list[float], pct: int) -> float:
    n = len(sorted_data)
    if n == 0: return 0.0
    if n == 1: return sorted_data[0]
    k = (pct / 100) * (n - 1)
    f = int(k)
    c = f + 1
    if c >= n: return sorted_data[-1]
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])
