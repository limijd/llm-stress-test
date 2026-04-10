"""HTML 报告生成 — 纯字符串拼接，零外部依赖"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def generate_html_report(
    reports,
    result_dir,
    config_snapshot,
    target_passed,
    max_passing_concurrency,
):
    """渲染并写出 report.html 到 result_dir。"""
    target_cfg = config_snapshot.get("target", {})
    model_name = target_cfg.get("model", target_cfg.get("name", "unknown"))
    engine_name = config_snapshot.get("engine", "unknown")
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_json = json.dumps(config_snapshot, indent=2, ensure_ascii=False)

    # 构建表格行
    rows_html = []
    for r in reports:
        badge_class = "pass" if r.pass_result.passed else "fail"
        badge_text = "PASS" if r.pass_result.passed else "FAIL"
        rows_html.append(
            f"      <tr>"
            f"<td>{r.concurrency}</td>"
            f"<td>{r.num_requests}</td>"
            f"<td>{r.aggregated.success_rate * 100:.1f}%</td>"
            f"<td>{r.aggregated.gen_toks_per_sec:.1f}</td>"
            f"<td>{r.aggregated.avg_ttft:.3f}</td>"
            f"<td>{r.aggregated.p50_latency:.3f}</td>"
            f"<td>{r.aggregated.p99_latency:.3f}</td>"
            f'<td><span class="badge {badge_class}">{badge_text}</span></td>'
            f"</tr>"
        )
    table_rows = "\n".join(rows_html)

    # 结论区块
    conclusion_class = "passed" if target_passed else "failed"
    conclusion_badge_class = "pass" if target_passed else "fail"
    conclusion_badge_text = "PASSED" if target_passed else "FAILED"

    if target_passed:
        conclusion_detail = (
            f"The model meets all pass criteria at the target concurrency level. "
            f"Recommended maximum concurrency: <strong>{max_passing_concurrency}</strong>."
        )
    else:
        if max_passing_concurrency:
            conclusion_detail = (
                f"The model failed to meet pass criteria at the target concurrency level. "
                f"Recommended maximum concurrency: <strong>{max_passing_concurrency}</strong>."
            )
        else:
            conclusion_detail = (
                "The model failed to meet pass criteria at the target concurrency level. "
                "No passing concurrency level was found; consider reducing load or optimizing the service."
            )

    max_conc_line = ""
    if max_passing_concurrency:
        max_conc_line = f"<p><strong>Max passing concurrency:</strong> {max_passing_concurrency}</p>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Stress Test Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
  h1 {{ color: #4A90D9; }}
  h2 {{ color: #555; border-bottom: 2px solid #4A90D9; padding-bottom: 4px; }}
  .header-meta {{ color: #666; font-size: 0.9em; margin-bottom: 24px; }}
  .conclusion {{
    padding: 16px 20px; border-radius: 6px; margin-bottom: 24px;
    border-left: 6px solid;
  }}
  .conclusion.passed {{ background: #e6f4ea; border-color: #34a853; }}
  .conclusion.failed {{ background: #fce8e6; border-color: #ea4335; }}
  .conclusion h3 {{ margin: 0 0 8px; font-size: 1.1em; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  th {{ background: #4A90D9; color: #fff; padding: 10px 12px; text-align: left; font-size: 0.9em; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #eee; font-size: 0.88em; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:nth-child(even) td {{ background: #f9f9f9; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.82em; font-weight: bold; }}
  .badge.pass {{ background: #34a853; color: #fff; }}
  .badge.fail {{ background: #ea4335; color: #fff; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
  .charts img {{ width: 100%; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,.15); }}
  pre {{ background: #fff; padding: 16px; border-radius: 6px; font-size: 0.83em;
        overflow-x: auto; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  section {{ margin-bottom: 32px; }}
</style>
</head>
<body>

<h1>LLM Stress Test Report</h1>
<div class="header-meta">
  <strong>Model:</strong> {_html_escape(model_name)} &nbsp;|&nbsp;
  <strong>Engine:</strong> {_html_escape(engine_name)} &nbsp;|&nbsp;
  <strong>Date:</strong> {report_date}
</div>

<section>
  <h2>Conclusion</h2>
  <div class="conclusion {conclusion_class}">
    <h3>Target: <span class="badge {conclusion_badge_class}">
      {conclusion_badge_text}
    </span></h3>
    {max_conc_line}
    <p>{conclusion_detail}</p>
  </div>
</section>

<section>
  <h2>Performance Details</h2>
  <table>
    <thead>
      <tr>
        <th>Concurrency</th>
        <th>Requests</th>
        <th>Success Rate</th>
        <th>Gen Toks/s</th>
        <th>Avg TTFT (s)</th>
        <th>P50 Latency (s)</th>
        <th>P99 Latency (s)</th>
        <th>Result</th>
      </tr>
    </thead>
    <tbody>
{table_rows}
    </tbody>
  </table>
</section>

<section>
  <h2>Charts</h2>
  <div class="charts">
    <img src="charts/throughput.png" alt="Throughput">
    <img src="charts/ttft.png" alt="TTFT">
    <img src="charts/latency_p50_p99.png" alt="Latency P50/P99">
    <img src="charts/success_rate.png" alt="Success Rate">
  </div>
</section>

<section>
  <h2>Config Snapshot</h2>
  <pre>{_html_escape(config_json)}</pre>
</section>

</body>
</html>
"""

    output_path = Path(result_dir) / "report.html"
    output_path.write_text(html_content, encoding="utf-8")


def _html_escape(text: str) -> str:
    """基础 HTML 转义。"""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
