"""全自动 Benchmark 编排：preflight → plan → execute → report"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .engine import get_engine
from .metrics import aggregate, judge
from .models import AggregatedMetrics, Criterion, EngineConfig, LevelResult
from .orchestrator import LevelReport, Orchestrator, TestRunResult


# ── 输出工具 ──────────────────────────────────────────

_COLORS = {
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "cyan": "\033[36m", "dim": "\033[2m", "bold": "\033[1m", "reset": "\033[0m",
}
_COLOR = sys.stdout.isatty()

def _s(text: str, fg: str | None = None, bold: bool = False) -> str:
    if not _COLOR:
        return text
    p = ""
    if bold: p += _COLORS["bold"]
    if fg and fg in _COLORS: p += _COLORS[fg]
    return f"{p}{text}{_COLORS['reset']}" if p else text

def _echo(msg: str = "", nl: bool = True):
    print(msg, end="\n" if nl else "", flush=True)

def _divider(char: str = "─", w: int = 72) -> str:
    return char * w

def _bar(ratio: float, w: int = 20) -> str:
    f = int(ratio * w)
    return _s("█" * f, fg="green") + _s("░" * (w - f), fg="dim") if _COLOR else "█" * f + "░" * (w - f)


# ── 数据模型 ──────────────────────────────────────────

@dataclass
class GroupConfig:
    id: int
    thinking: bool
    dataset: str
    engine: str   # "native" or "evalscope"
    concurrency: list[int]
    requests_per_level: list[int]
    extra_args: dict = field(default_factory=dict)

@dataclass
class GroupResult:
    config: GroupConfig
    status: str  # "completed" | "failed" | "skipped"
    test_run: TestRunResult | None = None
    error: str | None = None
    duration: float = 0.0

@dataclass
class PreflightResult:
    base_url: str
    healthy: bool
    parallel_slots: int
    slots_method: str  # "endpoint" | "timing" | "fallback"
    thinking_available: bool
    thinking_method: str  # "chat_template_kwargs" | "none"
    dataset_openqa: bool
    dataset_longalpaca: bool
    evalscope_available: bool
    evalscope_version: str


# ── HTTP 工具 ──────────────────────────────────────────

def _api_get(url: str, timeout: int = 10) -> tuple[int, str]:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return 0, ""

def _api_post(url: str, payload: dict, timeout: int = 120) -> tuple[int, dict | None]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return 0, None


# ── Phase 0: Preflight ──────────────────────────────────

def _check_health(base_url: str) -> bool:
    status, body = _api_get(f"{base_url}/health")
    return status == 200


def _detect_slots_endpoint(base_url: str) -> int | None:
    status, body = _api_get(f"{base_url}/slots")
    if status == 200:
        try:
            slots = json.loads(body)
            if isinstance(slots, list):
                return len(slots)
        except json.JSONDecodeError:
            pass
    return None


def _measure_request_latency(base_url: str) -> float:
    t0 = time.monotonic()
    _api_post(f"{base_url}/v1/chat/completions", {
        "model": "default", "max_tokens": 8,
        "messages": [{"role": "user", "content": "Say OK"}],
    }, timeout=120)
    return time.monotonic() - t0


def _measure_concurrent_latency(base_url: str, n: int) -> float:
    def _do(_i):
        _api_post(f"{base_url}/v1/chat/completions", {
            "model": "default", "max_tokens": 8,
            "messages": [{"role": "user", "content": "Say OK"}],
        }, timeout=120)

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_do, i) for i in range(n)]
        for f in as_completed(futures):
            f.result()
    return time.monotonic() - t0


def _detect_slots_timing(base_url: str) -> tuple[int, str]:
    s = _detect_slots_endpoint(base_url)
    if s is not None:
        return s, "endpoint"

    _echo("      测量单请求基线延迟 ...", nl=False)
    T = _measure_request_latency(base_url)
    _echo(f" {T:.2f}s")
    S = 1

    for n in [2, 4, 8]:
        _echo(f"      测量 {n} 并发 ...", nl=False)
        W = _measure_concurrent_latency(base_url, n)
        _echo(f" {W:.2f}s")

        expected_if_parallel = T * 1.5
        expected_if_queuing = T * (n / S)
        threshold = (expected_if_parallel + expected_if_queuing) / 2
        if W < threshold:
            S = n
        else:
            break

    if S == 1:
        return 1, "fallback"
    return S, "timing"


def _detect_thinking(base_url: str) -> tuple[bool, str]:
    status, body = _api_post(f"{base_url}/v1/chat/completions", {
        "model": "default", "max_tokens": 256,
        "messages": [{"role": "user", "content": "Think about what 2+3 equals."}],
        "chat_template_kwargs": {"thinking": True},
    }, timeout=120)
    if status == 200 and body:
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        if "<think>" in content:
            return True, "chat_template_kwargs"
    return False, "none"


def _check_datasets() -> tuple[bool, bool]:
    project_root = Path(__file__).parent.parent.parent
    ds_dir = project_root / "datasets"

    openqa_ok = (ds_dir / "openqa.jsonl").exists()

    longalpaca_ok = (ds_dir / "longalpaca.jsonl").exists()
    if not longalpaca_ok:
        dl_script = ds_dir / "download_longalpaca.py"
        if dl_script.exists():
            try:
                subprocess.run(["python3", str(dl_script)], capture_output=True, timeout=120)
                longalpaca_ok = (ds_dir / "longalpaca.jsonl").exists()
            except Exception:
                pass

    return openqa_ok or True, longalpaca_ok


def _check_evalscope() -> tuple[bool, str]:
    if shutil.which("evalscope"):
        try:
            r = subprocess.run(["evalscope", "--version"], capture_output=True, text=True, timeout=10)
            return True, r.stdout.strip() or "available"
        except Exception:
            return True, "available"
    try:
        r = subprocess.run(["python3", "-m", "evalscope", "--version"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return True, r.stdout.strip() or "available (module)"
    except Exception:
        pass
    return False, "not installed"


def run_preflight(host: str, port: int, max_concurrency: int | None) -> PreflightResult:
    base_url = f"http://{host}:{port}"

    _echo(_s("  Phase 0: Pre-flight", bold=True))
    _echo(_divider("═"))

    _echo("  [1/5] Health check ...", nl=False)
    healthy = _check_health(base_url)
    _echo(f" {_s('OK', fg='green') if healthy else _s('FAIL', fg='red')}")
    if not healthy:
        return PreflightResult(base_url=base_url, healthy=False, parallel_slots=0,
                               slots_method="", thinking_available=False, thinking_method="none",
                               dataset_openqa=False, dataset_longalpaca=False,
                               evalscope_available=False, evalscope_version="")

    _echo("  [2/5] Detect parallel slots ...")
    S, method = _detect_slots_timing(base_url)
    if max_concurrency is not None:
        S = min(S, max_concurrency)
    _echo(f"      => {_s(str(S), bold=True)} slots ({method})")

    _echo("  [3/5] Detect thinking mode ...", nl=False)
    think_ok, think_method = _detect_thinking(base_url)
    _echo(f" {_s('OK', fg='green') if think_ok else _s('not available', fg='yellow')}")

    _echo("  [4/5] Check datasets ...", nl=False)
    oqa, lalp = _check_datasets()
    _echo(f" openqa={'OK' if oqa else 'fallback'}  longalpaca={'OK' if lalp else 'MISSING'}")

    _echo("  [5/5] Check evalscope ...", nl=False)
    es_ok, es_ver = _check_evalscope()
    _echo(f" {_s('OK', fg='green') if es_ok else _s('not installed', fg='yellow')} ({es_ver})")

    _echo()
    return PreflightResult(
        base_url=base_url, healthy=True, parallel_slots=S, slots_method=method,
        thinking_available=think_ok, thinking_method=think_method,
        dataset_openqa=oqa, dataset_longalpaca=lalp,
        evalscope_available=es_ok, evalscope_version=es_ver,
    )


# ── Phase 1: Generate Plan ──────────────────────────────

def generate_plan(pf: PreflightResult, native_only: bool) -> list[GroupConfig]:
    S = pf.parallel_slots

    if S == 1:
        concurrency = [1]
        requests = [10]
    elif S <= 4:
        concurrency = [1, S]
        requests = [5, S * 5]
    else:
        concurrency = [1, S // 2, S]
        requests = [5, S * 3, S * 5]

    thinking_modes = [False]
    if pf.thinking_available:
        thinking_modes.append(True)

    datasets = ["openqa"]
    if pf.dataset_longalpaca:
        datasets.append("longalpaca")

    engines = ["native"]
    if pf.evalscope_available and not native_only:
        engines.append("evalscope")

    groups: list[GroupConfig] = []
    gid = 0
    for thinking in thinking_modes:
        for ds in datasets:
            for eng in engines:
                gid += 1
                extra = {}
                if thinking:
                    extra["chat_template_kwargs"] = {"thinking": True}
                groups.append(GroupConfig(
                    id=gid, thinking=thinking, dataset=ds, engine=eng,
                    concurrency=concurrency, requests_per_level=requests,
                    extra_args=extra,
                ))

    return groups


def print_plan(groups: list[GroupConfig], pf: PreflightResult):
    total_per_group = sum(groups[0].requests_per_level) if groups else 0
    total_all = total_per_group * len(groups)
    conc_str = " -> ".join(f"c{c}" for c in groups[0].concurrency) if groups else ""
    req_str = " -> ".join(str(r) for r in groups[0].requests_per_level) if groups else ""

    _echo(_s("  Phase 1: Test Plan", bold=True))
    _echo(_divider("═"))
    _echo(f"  Parallel slots    {_s(str(pf.parallel_slots), bold=True)} ({pf.slots_method})")
    _echo(f"  Concurrency       {conc_str}")
    _echo(f"  Requests/level    {req_str}")
    _echo(f"  Total/group       {total_per_group}")
    _echo(f"  Groups            {len(groups)}")
    _echo(f"  Total requests    {total_all}")
    _echo()

    _echo(f"  {'#':>3}  {'Thinking':<10} {'Dataset':<12} {'Engine':<10} {'Requests':>8}")
    _echo(f"  {'─'*3}  {'─'*10} {'─'*12} {'─'*10} {'─'*8}")
    for g in groups:
        think_str = _s("ON", fg="cyan") if g.thinking else "off"
        _echo(f"  {g.id:>3}  {think_str:<10} {g.dataset:<12} {g.engine:<10} {total_per_group:>8}")
    _echo()


# ── Phase 2: Execute Groups ──────────────────────────────

PASS_CRITERIA = [
    Criterion(metric="success_rate", operator=">=", threshold=0.95),
    Criterion(metric="avg_ttft", operator="<=", threshold=30.0),
]

COOLDOWN_SECONDS = 5
PER_GROUP_TIMEOUT = 1200


def run_single_group(gc: GroupConfig, base_url: str) -> GroupResult:
    parsed = urllib.parse.urlparse(base_url)
    api_url = f"{base_url}/v1/chat/completions"

    engine = get_engine(gc.engine)
    config_template = EngineConfig(
        api_url=api_url,
        api_key="not-needed",
        model="default",
        concurrency=0,
        num_requests=0,
        dataset=gc.dataset,
        stream=True,
        extra_args=gc.extra_args,
    )

    orchestrator = Orchestrator(engine=engine, criteria=PASS_CRITERIA)

    _level_start = [0.0]
    _level_ok = [0]
    _level_fail = [0]

    def on_progress(completed, total, concurrency, req_metric):
        if req_metric.success:
            _level_ok[0] += 1
            status = _s("OK", fg="green")
        else:
            _level_fail[0] += 1
            status = _s("FAIL", fg="red")
        elapsed = time.monotonic() - _level_start[0]
        pct = completed / total if total else 0
        _echo(f"\r    {_bar(pct)} {completed:>3}/{total}  {elapsed:.0f}s  {status}", nl=False)

    def on_level_start(c, n, idx, total, is_deg):
        _level_start[0] = time.monotonic()
        _level_ok[0] = 0
        _level_fail[0] = 0
        label = f"DEGRADE c={c}" if is_deg else f"[{idx}/{total}] c={c}"
        _echo(f"    {label}  requests={n}")

    def on_level_done(report, idx, total):
        agg = report.aggregated
        badge = _s("PASS", fg="green") if report.pass_result.passed else _s("FAIL", fg="red")
        _echo(f"\n    => {badge}  "
              f"Success={agg.success_rate*100:.0f}%  "
              f"Tok/s={agg.gen_toks_per_sec:.1f}  "
              f"TTFT={agg.avg_ttft:.2f}s")

    if hasattr(engine, "on_progress"):
        engine.on_progress = on_progress
    orchestrator.on_level_start = on_level_start
    orchestrator.on_level_done = on_level_done

    t0 = time.monotonic()
    try:
        test_run = orchestrator.run_test(
            concurrency=gc.concurrency,
            requests_per_level=gc.requests_per_level,
            config_template=config_template,
            degradation_enabled=True,
            degradation_step=1,
            degradation_min=1,
        )
        return GroupResult(config=gc, status="completed", test_run=test_run,
                           duration=time.monotonic() - t0)
    except Exception as e:
        return GroupResult(config=gc, status="failed", error=str(e),
                           duration=time.monotonic() - t0)


def execute_groups(groups: list[GroupConfig], pf: PreflightResult) -> list[GroupResult]:
    _echo(_s("  Phase 2: Execute", bold=True))
    _echo(_divider("═"))

    results: list[GroupResult] = []
    for i, gc in enumerate(groups):
        think_label = _s("thinking", fg="cyan") if gc.thinking else "non-thinking"
        _echo(f"\n  {_s(f'[{i+1}/{len(groups)}]', fg='cyan', bold=True)} "
              f"{think_label} x {gc.dataset} x {gc.engine}")
        _echo(f"  {'─' * 50}")

        result = run_single_group(gc, pf.base_url)
        results.append(result)

        if result.status == "completed" and result.test_run:
            tr = result.test_run
            badge = _s("PASS", fg="green", bold=True) if tr.target_passed else _s("FAIL", fg="red", bold=True)
            max_c = tr.max_passing_concurrency or "N/A"
            _echo(f"\n  Group {gc.id}: {badge}  Max concurrency={max_c}  Duration={result.duration:.0f}s")
        else:
            _echo(f"\n  Group {gc.id}: {_s('ERROR', fg='red', bold=True)}  {result.error}")

        if i < len(groups) - 1:
            _echo(f"  Cooling down {COOLDOWN_SECONDS}s ...")
            time.sleep(COOLDOWN_SECONDS)

    return results


# ── Phase 3: Report ──────────────────────────────────────

def _get_best_metrics(gr: GroupResult) -> dict | None:
    if gr.status != "completed" or gr.test_run is None:
        return None
    reports = gr.test_run.level_reports
    if not reports:
        return None
    last = reports[-1]
    agg = last.aggregated
    return {
        "concurrency": last.concurrency,
        "success_rate": agg.success_rate,
        "gen_toks_per_sec": round(agg.gen_toks_per_sec, 2),
        "avg_ttft": round(agg.avg_ttft, 3),
        "avg_tpot": round(agg.avg_tpot, 3),
        "p50_latency": round(agg.p50_latency, 3),
        "p99_latency": round(agg.p99_latency, 3),
    }


def _dimension_compare(results: list[GroupResult], key: str, val_a, val_b, label_a: str, label_b: str) -> str:
    def _avg_metrics(val):
        completed = [r for r in results if r.status == "completed" and getattr(r.config, key) == val]
        if not completed:
            return None
        metrics_list = [_get_best_metrics(r) for r in completed]
        metrics_list = [m for m in metrics_list if m is not None]
        if not metrics_list:
            return None
        avg = {}
        for k in ["gen_toks_per_sec", "avg_ttft", "avg_tpot", "p50_latency", "p99_latency"]:
            vals = [m[k] for m in metrics_list if k in m]
            avg[k] = round(sum(vals) / len(vals), 3) if vals else 0
        return avg

    ma = _avg_metrics(val_a)
    mb = _avg_metrics(val_b)
    if ma is None and mb is None:
        return f"N/A — insufficient data\n"

    lines = []
    lines.append(f"| Metric | {label_a} | {label_b} | Delta |")
    lines.append(f"| --- | --- | --- | --- |")
    for k, label in [("gen_toks_per_sec", "Tok/s"), ("avg_ttft", "TTFT(s)"), ("avg_tpot", "TPOT(s)"),
                      ("p50_latency", "P50(s)"), ("p99_latency", "P99(s)")]:
        va = ma.get(k, 0) if ma else "N/A"
        vb = mb.get(k, 0) if mb else "N/A"
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va != 0:
            delta = f"{(vb - va) / va * 100:+.1f}%"
        else:
            delta = ""
        lines.append(f"| {label} | {va} | {vb} | {delta} |")
    lines.append("")
    return "\n".join(lines)


def generate_summary_json(results: list[GroupResult], pf: PreflightResult) -> dict:
    groups_data = []
    for gr in results:
        gd = {
            "id": gr.config.id,
            "thinking": gr.config.thinking,
            "dataset": gr.config.dataset,
            "engine": gr.config.engine,
            "status": gr.status,
            "duration_s": round(gr.duration, 2),
            "error": gr.error,
        }
        metrics = _get_best_metrics(gr)
        if metrics:
            gd["best_concurrency"] = metrics["concurrency"]
            gd["metrics"] = metrics
        if gr.test_run:
            gd["target_passed"] = gr.test_run.target_passed
            gd["max_passing_concurrency"] = gr.test_run.max_passing_concurrency
        groups_data.append(gd)

    return {
        "meta": {
            "host": pf.base_url,
            "parallel_slots": pf.parallel_slots,
            "slots_method": pf.slots_method,
            "thinking_available": pf.thinking_available,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "groups": groups_data,
    }


def generate_markdown_report(results: list[GroupResult], pf: PreflightResult) -> str:
    lines: list[str] = []

    lines.append("# LLM Benchmark Report\n")

    lines.append("## Environment\n")
    lines.append("| Item | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Target | `{pf.base_url}` |")
    lines.append(f"| Parallel Slots | {pf.parallel_slots} ({pf.slots_method}) |")
    lines.append(f"| Thinking Mode | {'available' if pf.thinking_available else 'not available'} |")
    lines.append(f"| Evalscope | {pf.evalscope_version} |")
    lines.append(f"| Timestamp | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} |")
    lines.append("")

    lines.append("## Overall Comparison\n")
    lines.append("| # | Thinking | Dataset | Engine | Tok/s | TTFT(s) | P99(s) | Result |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for gr in results:
        m = _get_best_metrics(gr)
        if m:
            passed = gr.test_run.target_passed if gr.test_run else False
            badge = "PASS" if passed else "FAIL"
            lines.append(
                f"| {gr.config.id} "
                f"| {'ON' if gr.config.thinking else 'off'} "
                f"| {gr.config.dataset} "
                f"| {gr.config.engine} "
                f"| {m['gen_toks_per_sec']} "
                f"| {m['avg_ttft']} "
                f"| {m['p99_latency']} "
                f"| {badge} |"
            )
        else:
            lines.append(
                f"| {gr.config.id} "
                f"| {'ON' if gr.config.thinking else 'off'} "
                f"| {gr.config.dataset} "
                f"| {gr.config.engine} "
                f"| — | — | — "
                f"| {gr.status.upper()} |"
            )
    lines.append("")

    lines.append("## Dimension Analysis\n")

    lines.append("### Thinking vs Non-Thinking\n")
    lines.append(_dimension_compare(results, "thinking", False, True, "Non-Thinking", "Thinking"))

    lines.append("### OpenQA vs LongAlpaca\n")
    lines.append(_dimension_compare(results, "dataset", "openqa", "longalpaca", "OpenQA", "LongAlpaca"))

    lines.append("### Native vs Evalscope\n")
    lines.append(_dimension_compare(results, "engine", "native", "evalscope", "Native", "Evalscope"))

    lines.append("## Per-Group Detail\n")
    for gr in results:
        think_label = "thinking" if gr.config.thinking else "non-thinking"
        lines.append(f"### Group {gr.config.id}: {think_label} x {gr.config.dataset} x {gr.config.engine}\n")
        if gr.status != "completed" or gr.test_run is None:
            lines.append(f"Status: **{gr.status}** {gr.error or ''}\n")
            continue

        lines.append("| Conc | Reqs | Success% | Tok/s | TTFT(s) | P50(s) | P99(s) | Result |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for lr in gr.test_run.level_reports:
            a = lr.aggregated
            badge = "PASS" if lr.pass_result.passed else "FAIL"
            lines.append(
                f"| {lr.concurrency} | {lr.num_requests} "
                f"| {a.success_rate*100:.1f}% "
                f"| {a.gen_toks_per_sec:.1f} "
                f"| {a.avg_ttft:.3f} "
                f"| {a.p50_latency:.2f} "
                f"| {a.p99_latency:.2f} "
                f"| {badge} |"
            )
        max_c = gr.test_run.max_passing_concurrency
        if max_c:
            lines.append(f"\nMax stable concurrency: **{max_c}**\n")
        lines.append("")

    lines.append("## Conclusion\n")
    completed = [r for r in results if r.status == "completed" and r.test_run]
    if completed:
        max_concs = [r.test_run.max_passing_concurrency for r in completed if r.test_run and r.test_run.max_passing_concurrency]
        if max_concs:
            lines.append(f"- Recommended max concurrency: **{min(max_concs)}** (conservative, across all groups)")
        failing_metrics: dict[str, int] = {}
        for r in completed:
            if r.test_run:
                for lr in r.test_run.level_reports:
                    for d in lr.pass_result.details:
                        if not d.passed:
                            failing_metrics[d.metric] = failing_metrics.get(d.metric, 0) + 1
        if failing_metrics:
            bottleneck = max(failing_metrics, key=failing_metrics.get)
            lines.append(f"- Primary bottleneck: **{bottleneck}**")
    else:
        lines.append("- All groups failed. Check server health and configuration.")

    failed = [r for r in results if r.status == "failed"]
    if failed:
        lines.append(f"- {len(failed)} group(s) failed during execution")
    lines.append("")

    return "\n".join(lines)


def generate_html_report(md_content: str, pf: PreflightResult) -> str:
    html_body = ""
    in_table = False
    for line in md_content.splitlines():
        if line.startswith("# "):
            if in_table:
                html_body += "</tbody></table>\n"
                in_table = False
            html_body += f"<h1>{_html_esc(line[2:])}</h1>\n"
        elif line.startswith("## "):
            if in_table:
                html_body += "</tbody></table>\n"
                in_table = False
            html_body += f"<h2>{_html_esc(line[3:])}</h2>\n"
        elif line.startswith("### "):
            if in_table:
                html_body += "</tbody></table>\n"
                in_table = False
            html_body += f"<h3>{_html_esc(line[4:])}</h3>\n"
        elif line.startswith("| ") and "---" in line:
            continue
        elif line.startswith("| "):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if not in_table:
                html_body += "<table><thead><tr>"
                html_body += "".join(f"<th>{_html_esc(c)}</th>" for c in cells)
                html_body += "</tr></thead><tbody>\n"
                in_table = True
            else:
                html_body += "<tr>"
                html_body += "".join(f"<td>{_html_esc(c)}</td>" for c in cells)
                html_body += "</tr>\n"
        elif line.startswith("- "):
            if in_table:
                html_body += "</tbody></table>\n"
                in_table = False
            content = line[2:]
            content = content.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
            html_body += f"<p>{content}</p>\n"
        elif line.strip() == "":
            if in_table:
                html_body += "</tbody></table>\n"
                in_table = False
        else:
            html_body += f"<p>{_html_esc(line)}</p>\n"

    if in_table:
        html_body += "</tbody></table>\n"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Benchmark Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
         margin: 0; padding: 24px 40px; background: #f8f9fa; color: #1a1a2e; max-width: 1100px; margin: 0 auto; }}
  h1 {{ color: #2d3436; border-bottom: 3px solid #4A90D9; padding-bottom: 8px; }}
  h2 {{ color: #4A90D9; margin-top: 32px; }}
  h3 {{ color: #636e72; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; background: #fff;
           border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  th {{ background: #4A90D9; color: #fff; padding: 10px 14px; text-align: left; font-size: .9em; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #eee; font-size: .88em; }}
  tr:nth-child(even) td {{ background: #f8f9fa; }}
  p {{ line-height: 1.6; }}
</style>
</head>
<body>
{html_body}
<hr>
<p style="color:#999;font-size:.8em">Generated by llm-stress-test benchmark</p>
</body>
</html>"""


def _html_esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def save_reports(results: list[GroupResult], pf: PreflightResult, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = generate_summary_json(results, pf)
    (out / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    md = generate_markdown_report(results, pf)
    (out / "benchmark_report.md").write_text(md)

    html = generate_html_report(md, pf)
    (out / "benchmark_report.html").write_text(html)

    for gr in results:
        think = "on" if gr.config.thinking else "off"
        dirname = f"group_{gr.config.id}_{think}_{gr.config.dataset}_{gr.config.engine}"
        gdir = out / dirname
        gdir.mkdir(exist_ok=True)
        if gr.test_run and gr.test_run.level_reports:
            level_data = []
            for lr in gr.test_run.level_reports:
                level_data.append({
                    "concurrency": lr.concurrency,
                    "num_requests": lr.num_requests,
                    "metrics": asdict(lr.aggregated),
                    "passed": lr.pass_result.passed,
                })
            (gdir / "summary.json").write_text(
                json.dumps(level_data, indent=2, ensure_ascii=False))
        elif gr.error:
            (gdir / "error.txt").write_text(gr.error)

    return out, md


# ── 主入口 ──────────────────────────────────────────────

def run_benchmark(
    host: str,
    port: int = 8080,
    max_concurrency: int | None = None,
    dry_run: bool = False,
    native_only: bool = False,
    output_dir: str | None = None,
) -> int:
    _echo()
    _echo(_s("╔══════════════════════════════════════════════════════════════════════╗", fg="cyan"))
    _echo(_s("║                  LLM Full Benchmark                                ║", fg="cyan"))
    _echo(_s("╚══════════════════════════════════════════════════════════════════════╝", fg="cyan"))
    _echo()

    # Phase 0
    pf = run_preflight(host, port, max_concurrency)
    if not pf.healthy:
        _echo(_s("  服务不可达，终止。", fg="red"))
        return 1

    # Phase 1
    groups = generate_plan(pf, native_only)
    if not groups:
        _echo(_s("  无可用测试组（thinking/dataset/engine 均不可用）", fg="red"))
        return 1
    print_plan(groups, pf)

    if dry_run:
        _echo(_s("  --dry-run 模式，不执行测试。", fg="yellow"))
        return 0

    # Phase 2
    results = execute_groups(groups, pf)

    # Phase 3
    _echo()
    _echo(_s("  Phase 3: Report", bold=True))
    _echo(_divider("═"))

    if output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = f"./results/benchmark-{ts}"

    out_path, md_content = save_reports(results, pf, output_dir)

    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]
    _echo(f"  Completed: {_s(str(len(completed)), fg='green')}/{len(results)}")
    if failed:
        _echo(f"  Failed:    {_s(str(len(failed)), fg='red')}/{len(results)}")
    _echo()
    _echo(_s("  Output", bold=True))
    _echo(_divider())
    _echo(f"  {out_path}/benchmark_summary.json")
    _echo(f"  {out_path}/benchmark_report.md")
    _echo(f"  {out_path}/benchmark_report.html")
    _echo()

    return 0
