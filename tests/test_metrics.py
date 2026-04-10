import pytest
from llm_stress_test.models import (
    RequestMetric, LevelResult, Criterion, AggregatedMetrics,
)
from llm_stress_test.metrics import aggregate, judge

def _make_successful_requests(n, ttft=0.5, latency=2.0, output_tokens=100, tpot=0.015):
    return [RequestMetric(success=True, ttft=ttft, total_latency=latency,
                          output_tokens=output_tokens, input_tokens=50, tpot=tpot) for _ in range(n)]

class TestAggregate:
    def test_all_successful(self):
        requests = _make_successful_requests(10, ttft=0.5, latency=2.0, output_tokens=100)
        result = LevelResult(concurrency=5, num_requests=10, requests=requests, duration=4.0)
        agg = aggregate(result)
        assert agg.success_rate == 1.0
        assert agg.avg_ttft == 0.5
        assert agg.total_output_tokens == 1000
        assert agg.gen_toks_per_sec == 250.0  # 1000/4.0

    def test_with_failures(self):
        good = _make_successful_requests(8)
        bad = [RequestMetric(success=False, ttft=0.0, total_latency=0.0,
                             output_tokens=0, input_tokens=0, tpot=0.0, error="timeout") for _ in range(2)]
        result = LevelResult(concurrency=5, num_requests=10, requests=good+bad, duration=5.0)
        agg = aggregate(result)
        assert agg.success_rate == 0.8

    def test_percentiles(self):
        requests = [RequestMetric(True, 0.5, float(i), 100, 50, 0.01) for i in range(1, 101)]
        result = LevelResult(concurrency=10, num_requests=100, requests=requests, duration=10.0)
        agg = aggregate(result)
        assert agg.p50_latency == pytest.approx(50.5, abs=1.0)
        assert agg.p99_latency == pytest.approx(99.5, abs=1.0)

    def test_empty_requests(self):
        requests = [RequestMetric(False, 0.0, 0.0, 0, 0, 0.0, error="fail") for _ in range(5)]
        result = LevelResult(concurrency=5, num_requests=5, requests=requests, duration=1.0)
        agg = aggregate(result)
        assert agg.success_rate == 0.0
        assert agg.gen_toks_per_sec == 0.0
        assert agg.avg_ttft == 0.0

class TestJudge:
    def test_all_criteria_pass(self):
        agg = AggregatedMetrics(success_rate=1.0, gen_toks_per_sec=600, avg_ttft=5.0,
                                avg_tpot=0.01, p50_latency=2.0, p99_latency=5.0,
                                avg_latency=2.5, total_output_tokens=1000, total_duration=1.67)
        criteria = [Criterion("success_rate", ">=", 1.0), Criterion("gen_toks_per_sec", ">=", 500), Criterion("avg_ttft", "<=", 10.0)]
        result = judge(agg, criteria)
        assert result.passed is True
        assert all(d.passed for d in result.details)

    def test_one_criterion_fails(self):
        agg = AggregatedMetrics(success_rate=1.0, gen_toks_per_sec=400, avg_ttft=5.0,
                                avg_tpot=0.01, p50_latency=2.0, p99_latency=5.0,
                                avg_latency=2.5, total_output_tokens=1000, total_duration=2.5)
        criteria = [Criterion("success_rate", ">=", 1.0), Criterion("gen_toks_per_sec", ">=", 500)]
        result = judge(agg, criteria)
        assert result.passed is False
        assert result.details[0].passed is True
        assert result.details[1].passed is False

    def test_empty_criteria_passes(self):
        agg = AggregatedMetrics(success_rate=0.5, gen_toks_per_sec=10, avg_ttft=30.0,
                                avg_tpot=1.0, p50_latency=10.0, p99_latency=30.0,
                                avg_latency=15.0, total_output_tokens=100, total_duration=10.0)
        result = judge(agg, [])
        assert result.passed is True
