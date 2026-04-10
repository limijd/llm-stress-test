import csv
import json
import pytest
from pathlib import Path
from llm_stress_test.report.exporter import export_json, export_csv, create_result_dir
from llm_stress_test.report.chart import generate_charts
from llm_stress_test.report.html import generate_html_report
from llm_stress_test.models import (
    RequestMetric, LevelResult, AggregatedMetrics, CriterionResult, PassResult,
)
from llm_stress_test.orchestrator import LevelReport

def _make_level_report(concurrency):
    requests = [RequestMetric(True, 0.5, 2.0, 100, 50, 0.015) for _ in range(10)]
    level_result = LevelResult(concurrency=concurrency, num_requests=10, requests=requests, duration=3.0)
    agg = AggregatedMetrics(success_rate=1.0, gen_toks_per_sec=333.3, avg_ttft=0.5,
                            avg_tpot=0.015, p50_latency=2.0, p99_latency=2.0,
                            avg_latency=2.0, total_output_tokens=1000, total_duration=3.0)
    pass_result = PassResult(passed=True, details=[CriterionResult("success_rate", ">=", 1.0, 1.0, True)])
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
        raw_dir = Path(result_dir) / "raw"
        assert raw_dir.exists()
        files = sorted(raw_dir.glob("*.json"))
        assert len(files) == 2
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
        generate_html_report(reports=reports, result_dir=result_dir, config_snapshot=config_snapshot,
                             target_passed=True, max_passing_concurrency=5)
        html_file = Path(result_dir) / "report.html"
        assert html_file.exists()
        content = html_file.read_text()
        assert "test" in content
        assert "REDACTED" in content
