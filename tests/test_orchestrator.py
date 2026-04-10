import pytest
from unittest.mock import MagicMock
from llm_stress_test.orchestrator import Orchestrator, TestRunResult, SystemicAbort
from llm_stress_test.models import EngineConfig, LevelResult, RequestMetric, Criterion, SystemicError

def _make_passing_result(concurrency, num_requests):
    return LevelResult(
        concurrency=concurrency, num_requests=num_requests,
        requests=[RequestMetric(True, 0.5, 2.0, 200, 50, 0.01) for _ in range(num_requests)],
        duration=(200 * num_requests) / 600,  # gen_toks_per_sec = 600
    )

def _make_failing_result(concurrency, num_requests):
    return LevelResult(
        concurrency=concurrency, num_requests=num_requests,
        requests=[RequestMetric(True, 0.5, 2.0, 50, 50, 0.04) for _ in range(num_requests)],
        duration=(50 * num_requests) / 100,  # gen_toks_per_sec = 100 (below 500)
    )

class TestOrchestrator:
    def _make_orchestrator(self, engine_results):
        mock_engine = MagicMock()
        mock_engine.check_available.return_value = (True, "ok")
        mock_engine.run.side_effect = lambda config: engine_results[config.concurrency]
        criteria = [
            Criterion("success_rate", ">=", 1.0),
            Criterion("gen_toks_per_sec", ">=", 500),
            Criterion("avg_ttft", "<=", 10.0),
        ]
        return Orchestrator(engine=mock_engine, criteria=criteria)

    def test_all_levels_pass(self):
        results = {1: _make_passing_result(1, 10), 5: _make_passing_result(5, 50)}
        orch = self._make_orchestrator(results)
        run_result = orch.run_test(
            concurrency=[1, 5], requests_per_level=[10, 50],
            config_template=EngineConfig(api_url="http://test", api_key="k", model="m",
                                         concurrency=0, num_requests=0, dataset="openqa", stream=True, extra_args={}),
        )
        assert run_result.target_passed is True
        assert run_result.max_passing_concurrency == 5
        assert len(run_result.level_results) == 2

    def test_target_fails_degradation_finds_max(self):
        results = {
            1: _make_passing_result(1, 10), 5: _make_passing_result(5, 50),
            10: _make_failing_result(10, 100),
            8: _make_failing_result(8, 80), 6: _make_passing_result(6, 60),
        }
        orch = self._make_orchestrator(results)
        run_result = orch.run_test(
            concurrency=[1, 5, 10], requests_per_level=[10, 50, 100],
            config_template=EngineConfig(api_url="http://test", api_key="k", model="m",
                                         concurrency=0, num_requests=0, dataset="openqa", stream=True, extra_args={}),
            degradation_enabled=True, degradation_step=2, degradation_min=1,
        )
        assert run_result.target_passed is False
        assert run_result.max_passing_concurrency == 6

    def test_degradation_disabled(self):
        results = {1: _make_passing_result(1, 10), 5: _make_failing_result(5, 50)}
        orch = self._make_orchestrator(results)
        run_result = orch.run_test(
            concurrency=[1, 5], requests_per_level=[10, 50],
            config_template=EngineConfig(api_url="http://test", api_key="k", model="m",
                                         concurrency=0, num_requests=0, dataset="openqa", stream=True, extra_args={}),
            degradation_enabled=False,
        )
        assert run_result.target_passed is False
        assert run_result.max_passing_concurrency is None
        assert run_result.degradation_skipped is True
