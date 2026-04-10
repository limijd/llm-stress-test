import pytest
from llm_stress_test.models import (
    EngineConfig, RequestMetric, LevelResult, AggregatedMetrics,
    CriterionResult, PassResult, Criterion, SystemicError,
)

class TestEngineConfig:
    def test_create(self):
        cfg = EngineConfig(
            api_url="http://localhost:8000/v1/chat/completions",
            api_key="test-key", model="test-model",
            concurrency=10, num_requests=100, dataset="openqa",
            stream=True, extra_args={"thinking": True},
        )
        assert cfg.concurrency == 10
        assert cfg.stream is True
        assert cfg.extra_args == {"thinking": True}

class TestRequestMetric:
    def test_successful_request(self):
        m = RequestMetric(success=True, ttft=0.5, total_latency=2.0,
                          output_tokens=100, input_tokens=50, tpot=0.015)
        assert m.success is True
        assert m.error is None

    def test_failed_request(self):
        m = RequestMetric(success=False, ttft=0.0, total_latency=0.0,
                          output_tokens=0, input_tokens=0, tpot=0.0,
                          error="Connection refused")
        assert m.success is False
        assert m.error == "Connection refused"

class TestLevelResult:
    def test_create(self):
        metrics = [
            RequestMetric(True, 0.5, 2.0, 100, 50, 0.015),
            RequestMetric(True, 0.6, 2.1, 110, 50, 0.014),
        ]
        result = LevelResult(concurrency=5, num_requests=2, requests=metrics, duration=2.5)
        assert result.concurrency == 5
        assert len(result.requests) == 2

class TestCriterion:
    @pytest.mark.parametrize("operator,threshold,value,expected", [
        (">=", 1.0, 1.0, True), (">=", 1.0, 0.99, False),
        ("<=", 10.0, 9.5, True), ("<=", 10.0, 10.5, False),
        (">", 500, 501, True), (">", 500, 500, False),
        ("<", 10.0, 9.9, True), ("<", 10.0, 10.0, False),
    ])
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
