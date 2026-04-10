"""JSON/CSV 导出"""
from __future__ import annotations
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from ..orchestrator import LevelReport

def create_result_dir(base_dir, model_name, engine_name):
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    dir_name = f"{timestamp}_{safe_name}_{engine_name}"
    result_dir = Path(base_dir) / dir_name
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "raw").mkdir(exist_ok=True)
    (result_dir / "charts").mkdir(exist_ok=True)
    return str(result_dir)

def export_json(reports, result_dir):
    raw_dir = Path(result_dir) / "raw"
    raw_dir.mkdir(exist_ok=True)
    summary_data = []
    for i, report in enumerate(reports):
        raw_file = raw_dir / f"level_{i+1:02d}_c{report.concurrency}.json"
        raw_data = [asdict(r) for r in report.level_result.requests]
        raw_file.write_text(json.dumps(raw_data, indent=2, ensure_ascii=False))
        summary_data.append({
            "concurrency": report.concurrency, "num_requests": report.num_requests,
            "metrics": asdict(report.aggregated),
            "passed": report.pass_result.passed,
            "criteria_details": [asdict(d) for d in report.pass_result.details],
        })
    (Path(result_dir) / "summary.json").write_text(json.dumps(summary_data, indent=2, ensure_ascii=False))

def export_csv(reports, result_dir):
    fieldnames = ["concurrency", "num_requests", "success_rate", "gen_toks_per_sec",
                  "avg_ttft", "avg_tpot", "p50_latency", "p99_latency", "avg_latency",
                  "total_output_tokens", "total_duration", "passed"]
    with (Path(result_dir) / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            writer.writerow({"concurrency": report.concurrency, "num_requests": report.num_requests,
                             **asdict(report.aggregated), "passed": report.pass_result.passed})
