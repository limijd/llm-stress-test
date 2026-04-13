#!/usr/bin/env python3
"""
LLM 模型能力 & 服务配置探测脚本
探测 llama-server 的服务端配置、模型元信息、能力边界和基线性能。
用于压测前确定合理参数，压测后作为基线对照。
零依赖，只用 Python 标准库。

用法:
    python3 probe_model.py                          # 探测 localhost:8080
    python3 probe_model.py -H 192.168.1.100         # 指定主机
    python3 probe_model.py -H 10.0.0.1 -p 8000      # 指定主机和端口
    python3 probe_model.py -o probe_result.json      # 保存 JSON
    python3 probe_model.py --md report.md            # 保存 Markdown 报告
    python3 probe_model.py --md-only                 # 只输出 Markdown 到 stdout
    python3 probe_model.py --json-only               # 只输出 JSON 到 stdout
    python3 probe_model.py --skip-context-probe      # 跳过 context 边界探测（耗时较长）
    python3 probe_model.py -o result.json --md report.md  # 同时保存 JSON 和 Markdown
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


# ── 工具函数 ──────────────────────────────────────────

def api_get(base: str, path: str, timeout: int = 10) -> tuple[int, dict | str | None]:
    """GET 请求，返回 (status_code, body)"""
    try:
        req = urllib.request.Request(f"{base}{path}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode()
        except Exception:
            body = ""
        return e.code, body
    except Exception as e:
        return 0, str(e)


def api_post(base: str, path: str, payload: dict, timeout: int = 120) -> tuple[int, dict | str | None]:
    """POST 请求，返回 (status_code, body)"""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode()
        except Exception:
            body = ""
        return e.code, body
    except Exception as e:
        return 0, str(e)


def api_post_stream(base: str, path: str, payload: dict, timeout: int = 120) -> dict:
    """POST 流式请求，返回 {ttft, tokens, total_time, content, error}"""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    result = {"ttft": 0, "tokens": 0, "total_time": 0, "content": "", "error": None}
    t0 = time.monotonic()
    first_token_time = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buf = ""
            while True:
                chunk = resp.read(1)
                if not chunk:
                    break
                buf += chunk.decode()
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.monotonic()
                            result["tokens"] += 1
                            result["content"] += content
                    except json.JSONDecodeError:
                        pass
    except urllib.error.HTTPError as e:
        result["error"] = f"HTTP {e.code}"
    except Exception as e:
        result["error"] = str(e)

    t1 = time.monotonic()
    result["total_time"] = round(t1 - t0, 3)
    if first_token_time is not None:
        result["ttft"] = round(first_token_time - t0, 3)
    return result


def build_prompt_of_length(n_tokens_approx: int) -> str:
    """构造大约 n_tokens_approx 个 token 的 prompt（英文约 1 word ≈ 1.3 token）"""
    # 用重复的简单句子填充，每个词约 1-1.3 token
    word = "hello "
    # 粗略估算: 1 个 'hello ' ≈ 2 tokens (hello + space)
    # 但不同 tokenizer 不同，这里用保守估计
    repeat = max(1, n_tokens_approx // 2)
    return word * repeat


# ── 探测模块 ──────────────────────────────────────────

def probe_health(base: str) -> dict:
    """健康检查"""
    status, body = api_get(base, "/health")
    return {
        "status_code": status,
        "body": body,
        "ok": status == 200,
    }


def probe_server_props(base: str) -> dict:
    """探测服务端属性（llama-server /props 端点）"""
    info = {}

    # /props — llama-server 特有
    status, body = api_get(base, "/props")
    if status == 200 and isinstance(body, dict):
        info["props"] = body
        info["available"] = True
    else:
        info["props"] = None
        info["available"] = False

    # /slots — llama-server slot 状态（需要 --slots 启动参数）
    status, body = api_get(base, "/slots")
    if status == 200:
        info["slots"] = body
        if isinstance(body, list):
            info["total_slots"] = len(body)
            info["idle_slots"] = sum(1 for s in body if s.get("state") == 0)
    else:
        info["slots"] = None
        info["slots_note"] = "未启用或不可用（需要 --slots 启动参数）"

    # /metrics — Prometheus 指标
    status, body = api_get(base, "/metrics")
    if status == 200 and isinstance(body, str):
        info["metrics_available"] = True
        # 提取关键指标
        metrics = {}
        for line in body.splitlines():
            if line.startswith("#"):
                continue
            for key in [
                "llama_kv_cache_tokens",
                "llama_kv_cache_used_cells",
                "llama_requests_processing",
                "llama_requests_deferred",
                "llama_n_decode_total",
                "llama_n_prompt_tokens_total",
                "llama_prompt_tokens_seconds",
                "llama_tokens_predicted_seconds",
                "llama_n_tokens_predicted_total",
            ]:
                if line.startswith(key):
                    parts = line.split()
                    if len(parts) >= 2:
                        metrics[key] = parts[-1]
        info["prometheus_metrics"] = metrics
    else:
        info["metrics_available"] = False

    return info


def probe_model_info(base: str) -> dict:
    """获取模型信息"""
    info = {}

    # /v1/models
    status, body = api_get(base, "/v1/models")
    if status == 200 and isinstance(body, dict):
        models = body.get("data", [])
        info["models"] = models
        if models:
            info["model_id"] = models[0].get("id", "unknown")
    else:
        info["models"] = []
        info["model_id"] = "unknown"

    return info


def probe_tokenizer(base: str) -> dict:
    """探测 tokenizer（llama-server /tokenize 端点）"""
    info = {}

    # 测试 tokenize
    test_text = "Hello, world! 你好世界"
    status, body = api_post(base, "/tokenize", {"content": test_text})
    if status == 200 and isinstance(body, dict):
        tokens = body.get("tokens", [])
        info["tokenize_available"] = True
        info["test_input"] = test_text
        info["token_count"] = len(tokens)
        info["tokens_sample"] = tokens[:20]

        # 用标准文本测量 token/char 比率
        en_text = "The quick brown fox jumps over the lazy dog. " * 10
        status2, body2 = api_post(base, "/tokenize", {"content": en_text})
        if status2 == 200 and isinstance(body2, dict):
            en_tokens = len(body2.get("tokens", []))
            info["en_chars_per_token"] = round(len(en_text) / en_tokens, 2) if en_tokens else 0

        zh_text = "大型语言模型是一种深度学习模型，通过海量文本数据训练而成。" * 10
        status3, body3 = api_post(base, "/tokenize", {"content": zh_text})
        if status3 == 200 and isinstance(body3, dict):
            zh_tokens = len(body3.get("tokens", []))
            info["zh_chars_per_token"] = round(len(zh_text) / zh_tokens, 2) if zh_tokens else 0
    else:
        info["tokenize_available"] = False

    # 测试 detokenize
    if info.get("tokenize_available") and info.get("tokens_sample"):
        status, body = api_post(base, "/detokenize", {"tokens": info["tokens_sample"]})
        info["detokenize_available"] = status == 200

    return info


def _tokenize_count(base: str, text: str) -> int | None:
    """用 /tokenize 端点计算 token 数，失败返回 None"""
    status, body = api_post(base, "/tokenize", {"content": text})
    if status == 200 and isinstance(body, dict):
        return len(body.get("tokens", []))
    return None


def probe_context_window(base: str, model_id: str) -> dict:
    """探测实际可用的 context 窗口大小（二分法）"""
    info = {"method": "binary_search"}

    # 先用小请求确认服务正常
    status, body = api_post(base, "/v1/chat/completions", {
        "model": model_id, "max_tokens": 1,
        "messages": [{"role": "user", "content": "Hi"}],
    })
    if status != 200:
        info["error"] = f"基础请求失败: HTTP {status}"
        return info

    # 二分法探测 context 上限
    # 保守上限 32K，避免超大 prompt 导致远程服务 OOM 崩溃
    # 如果需要探测更大 context，可通过 --max-context 参数调整
    lo, hi = 512, 32768
    last_ok = lo

    # 先快速确定大致范围（指数增长，步幅 ×2）
    print("      阶段 1: 指数探测范围 (上限 32K) ...", flush=True)
    probe_size = 1024
    while probe_size <= hi:
        prompt = build_prompt_of_length(probe_size)
        actual_tokens = _tokenize_count(base, prompt)
        if actual_tokens:
            print(f"        尝试 ~{actual_tokens} tokens ...", end=" ", flush=True)
        else:
            print(f"        尝试 ~{probe_size} tokens (估算) ...", end=" ", flush=True)

        status, body = api_post(base, "/v1/chat/completions", {
            "model": model_id, "max_tokens": 1,
            "messages": [{"role": "user", "content": prompt}],
        }, timeout=180)

        if status == 200:
            print("OK", flush=True)
            last_ok = probe_size
            probe_size *= 2
        else:
            print("FAIL", flush=True)
            hi = probe_size
            break
    else:
        # 如果 32K 都能过
        info["context_approx_tokens"] = f">={hi}"
        info["note"] = "达到安全探测上限 (32K)，实际 context 可能更大"
        return info

    lo = last_ok

    # 二分精确定位
    print("      阶段 2: 二分精确定位 ...", flush=True)
    iterations = 0
    while hi - lo > 512 and iterations < 10:
        mid = (lo + hi) // 2
        prompt = build_prompt_of_length(mid)
        actual_tokens = _tokenize_count(base, prompt)
        label = f"~{actual_tokens}" if actual_tokens else f"~{mid}"
        print(f"        尝试 {label} tokens ...", end=" ", flush=True)

        status, body = api_post(base, "/v1/chat/completions", {
            "model": model_id, "max_tokens": 1,
            "messages": [{"role": "user", "content": prompt}],
        }, timeout=180)

        if status == 200:
            print("OK", flush=True)
            lo = mid
        else:
            print("FAIL", flush=True)
            hi = mid
        iterations += 1

    # 用 tokenize 端点获取精确 token 数
    final_prompt = build_prompt_of_length(lo)
    exact_tokens = _tokenize_count(base, final_prompt)
    info["context_max_ok_tokens"] = exact_tokens or lo
    info["context_first_fail_tokens"] = hi
    info["note"] = f"实际可用 context 约 {exact_tokens or lo} tokens"

    return info


def probe_capabilities(base: str, model_id: str) -> dict:
    """探测模型支持的功能"""
    info = {}
    base_msg = [{"role": "user", "content": "Say OK"}]

    # 1. 基本 chat completion (非流式)
    status, body = api_post(base, "/v1/chat/completions", {
        "model": model_id, "max_tokens": 8,
        "messages": base_msg, "stream": False,
    })
    info["chat_completion"] = status == 200

    # 2. 流式
    result = api_post_stream(base, "/v1/chat/completions", {
        "model": model_id, "max_tokens": 8,
        "messages": base_msg, "stream": True,
    })
    info["streaming"] = result["error"] is None and result["tokens"] > 0

    # 3. system prompt
    status, body = api_post(base, "/v1/chat/completions", {
        "model": model_id, "max_tokens": 16,
        "messages": [
            {"role": "system", "content": "You are a calculator. Only output numbers."},
            {"role": "user", "content": "What is 2+3?"},
        ],
    })
    info["system_prompt"] = status == 200

    # 4. 多轮对话
    status, body = api_post(base, "/v1/chat/completions", {
        "model": model_id, "max_tokens": 16,
        "messages": [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice!"},
            {"role": "user", "content": "What's my name?"},
        ],
    })
    info["multi_turn"] = status == 200
    if status == 200 and isinstance(body, dict):
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        info["multi_turn_coherent"] = "alice" in content.lower()

    # 5. temperature 控制
    for temp in [0.0, 0.5, 2.0]:
        status, body = api_post(base, "/v1/chat/completions", {
            "model": model_id, "max_tokens": 8,
            "messages": base_msg, "temperature": temp,
        })
        info[f"temperature_{temp}"] = status == 200

    # 6. top_p
    status, body = api_post(base, "/v1/chat/completions", {
        "model": model_id, "max_tokens": 8,
        "messages": base_msg, "top_p": 0.1,
    })
    info["top_p"] = status == 200

    # 7. frequency_penalty / presence_penalty
    status, body = api_post(base, "/v1/chat/completions", {
        "model": model_id, "max_tokens": 8,
        "messages": base_msg, "frequency_penalty": 0.5, "presence_penalty": 0.5,
    })
    info["frequency_presence_penalty"] = status == 200

    # 8. stop tokens
    status, body = api_post(base, "/v1/chat/completions", {
        "model": model_id, "max_tokens": 64,
        "messages": [{"role": "user", "content": "Count from 1 to 20, one number per line."}],
        "stop": ["5"],
    })
    info["stop_tokens"] = status == 200
    if status == 200 and isinstance(body, dict):
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        info["stop_tokens_effective"] = "6" not in content

    # 9. logprobs
    status, body = api_post(base, "/v1/chat/completions", {
        "model": model_id, "max_tokens": 8,
        "messages": base_msg, "logprobs": True, "top_logprobs": 5,
    })
    info["logprobs"] = status == 200

    # 10. n > 1 (多个 completion)
    status, body = api_post(base, "/v1/chat/completions", {
        "model": model_id, "max_tokens": 8,
        "messages": base_msg, "n": 2,
    })
    info["n_completions"] = status == 200

    # 11. seed (可复现)
    results = []
    for _ in range(2):
        status, body = api_post(base, "/v1/chat/completions", {
            "model": model_id, "max_tokens": 16,
            "messages": [{"role": "user", "content": "Write exactly: test123"}],
            "seed": 42, "temperature": 0.0,
        })
        if status == 200 and isinstance(body, dict):
            results.append(body.get("choices", [{}])[0].get("message", {}).get("content", ""))
    info["seed_reproducible"] = len(results) == 2 and results[0] == results[1]

    # 12. /completion 原生端点
    status, body = api_post(base, "/completion", {
        "prompt": "Hello, my name is", "n_predict": 16,
    })
    info["native_completion"] = status == 200

    # 13. /embedding 端点
    status, body = api_post(base, "/embedding", {
        "content": "test embedding",
    })
    info["embedding"] = status == 200

    return info


def probe_baseline_performance(base: str, model_id: str) -> dict:
    """单请求基线性能，不同 prompt 长度"""
    info = {"tests": []}

    prompt_configs = [
        {"label": "短 prompt (~50 tok)", "tokens": 50, "max_tokens": 64},
        {"label": "中 prompt (~500 tok)", "tokens": 500, "max_tokens": 128},
        {"label": "长 prompt (~2000 tok)", "tokens": 2000, "max_tokens": 128},
    ]

    for cfg in prompt_configs:
        print(f"      {cfg['label']} ...", end=" ", flush=True)
        prompt = build_prompt_of_length(cfg["tokens"])

        # 实际 token 数
        actual_input = _tokenize_count(base, prompt)

        # 流式请求测量 TTFT 和 token 速率
        t0 = time.monotonic()
        result = api_post_stream(base, "/v1/chat/completions", {
            "model": model_id,
            "max_tokens": cfg["max_tokens"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0.7,
        }, timeout=180)

        if result["error"]:
            print(f"FAIL ({result['error']})", flush=True)
            info["tests"].append({
                "label": cfg["label"],
                "input_tokens": actual_input or cfg["tokens"],
                "error": result["error"],
            })
            continue

        tps = result["tokens"] / (result["total_time"] - result["ttft"]) if (result["total_time"] - result["ttft"]) > 0 else 0
        prefill_tps = (actual_input or cfg["tokens"]) / result["ttft"] if result["ttft"] > 0 else 0

        print(f"TTFT={result['ttft']:.3f}s, {result['tokens']} tok, {tps:.1f} tok/s", flush=True)

        info["tests"].append({
            "label": cfg["label"],
            "input_tokens": actual_input or cfg["tokens"],
            "output_tokens": result["tokens"],
            "ttft_s": result["ttft"],
            "total_time_s": result["total_time"],
            "decode_tok_per_s": round(tps, 2),
            "prefill_tok_per_s": round(prefill_tps, 2),
        })

    return info


def probe_concurrency(base: str, model_id: str) -> dict:
    """探测并发处理能力"""
    info = {"tests": []}

    for n_concurrent in [1, 2, 4, 8]:
        print(f"      并发 {n_concurrent} 请求 ...", end=" ", flush=True)

        def do_request(_):
            t0 = time.monotonic()
            status, body = api_post(base, "/v1/chat/completions", {
                "model": model_id, "max_tokens": 32,
                "messages": [{"role": "user", "content": "Say hello briefly."}],
            }, timeout=120)
            elapsed = time.monotonic() - t0
            return {"status": status, "time": round(elapsed, 3)}

        t_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
            futures = [pool.submit(do_request, i) for i in range(n_concurrent)]
            results = [f.result() for f in as_completed(futures)]
        wall_time = time.monotonic() - t_start

        successes = sum(1 for r in results if r["status"] == 200)
        times = [r["time"] for r in results if r["status"] == 200]
        avg_time = sum(times) / len(times) if times else 0

        print(f"{successes}/{n_concurrent} OK, wall={wall_time:.2f}s, avg_latency={avg_time:.2f}s", flush=True)

        info["tests"].append({
            "concurrency": n_concurrent,
            "success": successes,
            "total": n_concurrent,
            "wall_time_s": round(wall_time, 3),
            "avg_latency_s": round(avg_time, 3),
            "individual_times": times,
        })

        # 如果全部失败就停止
        if successes == 0:
            info["max_concurrency_ok"] = n_concurrent - 1 if n_concurrent > 1 else 0
            break
    else:
        info["max_concurrency_ok"] = f">={n_concurrent}"

    return info


# ── 输出格式化 ──────────────────────────────────────────

def print_section(title: str, content: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    print(content)


def format_capabilities(caps: dict) -> str:
    lines = []
    labels = {
        "chat_completion": "Chat Completion",
        "streaming": "流式输出",
        "system_prompt": "System Prompt",
        "multi_turn": "多轮对话",
        "multi_turn_coherent": "  └─ 上下文连贯",
        "temperature_0.0": "Temperature 0.0",
        "temperature_0.5": "Temperature 0.5",
        "temperature_2.0": "Temperature 2.0",
        "top_p": "Top-P 采样",
        "frequency_presence_penalty": "Frequency/Presence Penalty",
        "stop_tokens": "Stop Tokens",
        "stop_tokens_effective": "  └─ 实际生效",
        "logprobs": "Log Probabilities",
        "n_completions": "多路生成 (n>1)",
        "seed_reproducible": "Seed 可复现",
        "native_completion": "/completion 原生端点",
        "embedding": "Embedding 端点",
    }
    for key, label in labels.items():
        if key in caps:
            v = caps[key]
            mark = "✓" if v else "✗"
            lines.append(f"    {mark}  {label}")
    return "\n".join(lines)


def format_baseline(baseline: dict) -> str:
    lines = []
    for t in baseline.get("tests", []):
        if "error" in t:
            lines.append(f"    {t['label']}: FAIL ({t['error']})")
        else:
            lines.append(
                f"    {t['label']}:"
                f"  TTFT={t['ttft_s']:.3f}s"
                f"  decode={t['decode_tok_per_s']:.1f} tok/s"
                f"  prefill={t['prefill_tok_per_s']:.0f} tok/s"
                f"  ({t['output_tokens']} tok / {t['total_time_s']:.2f}s)"
            )
    return "\n".join(lines)


def format_concurrency(conc: dict) -> str:
    lines = []
    for t in conc.get("tests", []):
        lines.append(
            f"    并发 {t['concurrency']:>2}:"
            f"  {t['success']}/{t['total']} OK"
            f"  wall={t['wall_time_s']:.2f}s"
            f"  avg_latency={t['avg_latency_s']:.2f}s"
        )
    if "max_concurrency_ok" in conc:
        lines.append(f"    最大稳定并发: {conc['max_concurrency_ok']}")
    return "\n".join(lines)


def print_report(report: dict):
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          LLM 模型能力 & 配置探测报告                    ║")
    print(f"║  目标: {report['target']:<49}║")
    print(f"║  时间: {report['collected_at'][:19]:<49}║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 健康状态
    h = report.get("health", {})
    print_section("健康状态", f"    状态: {'OK' if h.get('ok') else 'FAIL'} ({h.get('body', '')})")

    # 模型信息
    m = report.get("model_info", {})
    print_section("模型信息", f"    Model ID: {m.get('model_id', 'N/A')}")

    # 服务端配置
    sp = report.get("server_props", {})
    if sp.get("available") and sp.get("props"):
        props = sp["props"]
        lines = []
        for k, v in props.items():
            lines.append(f"    {k}: {v}")
        print_section("服务端配置 (/props)", "\n".join(lines))

    if sp.get("total_slots") is not None:
        print_section("Slot 状态", f"    总 slots: {sp['total_slots']}, 空闲: {sp.get('idle_slots', '?')}")

    if sp.get("prometheus_metrics"):
        lines = [f"    {k}: {v}" for k, v in sp["prometheus_metrics"].items()]
        print_section("Prometheus 指标", "\n".join(lines))

    # Tokenizer
    tk = report.get("tokenizer", {})
    if tk.get("tokenize_available"):
        lines = [
            f"    Tokenize 端点: 可用",
            f"    测试: \"{tk['test_input']}\" → {tk['token_count']} tokens",
            f"    英文 chars/token: {tk.get('en_chars_per_token', 'N/A')}",
            f"    中文 chars/token: {tk.get('zh_chars_per_token', 'N/A')}",
        ]
        print_section("Tokenizer", "\n".join(lines))

    # 功能支持
    caps = report.get("capabilities", {})
    if caps:
        print_section("功能支持", format_capabilities(caps))

    # Context 窗口
    ctx = report.get("context_window", {})
    if ctx and "error" not in ctx:
        lines = [f"    最大可用 context: ~{ctx.get('context_max_ok_tokens', '?')} tokens"]
        if ctx.get("note"):
            lines.append(f"    {ctx['note']}")
        print_section("Context 窗口", "\n".join(lines))
    elif ctx and not ctx.get("skipped"):
        print_section("Context 窗口", f"    {ctx.get('error', '探测失败')}")

    # 基线性能
    baseline = report.get("baseline_performance", {})
    if baseline.get("tests"):
        print_section("基线性能 (单请求)", format_baseline(baseline))

    # 并发测试
    conc = report.get("concurrency", {})
    if conc.get("tests"):
        print_section("并发处理能力", format_concurrency(conc))

    print(f"\n{'═' * 60}")
    print("  探测完成")
    print(f"{'═' * 60}")


# ── Markdown 渲染 ──────────────────────────────────────

CAPABILITY_LABELS = {
    "chat_completion": "Chat Completion",
    "streaming": "流式输出 (Streaming)",
    "system_prompt": "System Prompt",
    "multi_turn": "多轮对话",
    "multi_turn_coherent": "多轮上下文连贯",
    "temperature_0.0": "Temperature 0.0 (Greedy)",
    "temperature_0.5": "Temperature 0.5",
    "temperature_2.0": "Temperature 2.0 (High Creativity)",
    "top_p": "Top-P 采样",
    "frequency_presence_penalty": "Frequency / Presence Penalty",
    "stop_tokens": "Stop Tokens",
    "stop_tokens_effective": "Stop Tokens 实际生效",
    "logprobs": "Log Probabilities",
    "n_completions": "多路生成 (n > 1)",
    "seed_reproducible": "Seed 可复现输出",
    "native_completion": "/completion 原生端点",
    "embedding": "Embedding 端点",
}


def render_markdown(report: dict) -> str:
    """将探测结果渲染为 Markdown"""
    lines: list[str] = []

    def h1(text: str):
        lines.append(f"# {text}\n")

    def h2(text: str):
        lines.append(f"## {text}\n")

    def h3(text: str):
        lines.append(f"### {text}\n")

    def p(text: str):
        lines.append(f"{text}\n")

    def table(headers: list[str], rows: list[list[str]]):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")

    # ── 标题 ──

    model_id = report.get("model_info", {}).get("model_id", "Unknown")
    h1(f"LLM 模型探测报告 — {model_id}")

    table(
        ["项目", "值"],
        [
            ["目标地址", f"`{report.get('target', '')}`"],
            ["探测时间 (UTC)", f"`{report.get('collected_at', '')[:19]}`"],
            ["探测器版本", report.get("probe_version", "")],
        ],
    )

    # ── 健康状态 ──

    h2("1. 健康状态")
    health = report.get("health", {})
    status_badge = "**OK**" if health.get("ok") else "**FAIL**"
    p(f"状态: {status_badge} (HTTP {health.get('status_code', '?')})")

    # ── 模型信息 ──

    h2("2. 模型信息")
    mi = report.get("model_info", {})
    p(f"Model ID: `{mi.get('model_id', 'N/A')}`")
    models = mi.get("models", [])
    if models:
        rows = []
        for m in models:
            rows.append([
                f"`{m.get('id', '')}`",
                m.get("object", ""),
                str(m.get("owned_by", "")),
            ])
        table(["ID", "Object", "Owned By"], rows)

    # ── 服务端配置 ──

    h2("3. 服务端配置")
    sp = report.get("server_props", {})

    if sp.get("available") and sp.get("props"):
        h3("3.1 Server Properties (`/props`)")
        rows = [[f"`{k}`", str(v)] for k, v in sp["props"].items()]
        table(["参数", "值"], rows)

    if sp.get("total_slots") is not None:
        h3("3.2 Slot 状态 (`/slots`)")
        table(
            ["总 Slots", "空闲 Slots"],
            [[str(sp["total_slots"]), str(sp.get("idle_slots", "?"))]],
        )
        # 逐 slot 明细
        slots = sp.get("slots")
        if isinstance(slots, list) and slots:
            slot_rows = []
            for s in slots:
                state = "idle" if s.get("state") == 0 else "processing"
                slot_rows.append([
                    str(s.get("id", "")),
                    state,
                    str(s.get("n_ctx", "")),
                    str(s.get("n_predict", "")),
                    str(s.get("model", "")),
                ])
            table(["Slot", "状态", "n_ctx", "n_predict", "Model"], slot_rows)
    elif sp.get("slots_note"):
        h3("3.2 Slot 状态")
        p(f"_{sp['slots_note']}_")

    if sp.get("prometheus_metrics"):
        h3("3.3 Prometheus 指标")
        rows = [[f"`{k}`", str(v)] for k, v in sp["prometheus_metrics"].items()]
        table(["指标", "值"], rows)
    elif not sp.get("metrics_available"):
        h3("3.3 Prometheus 指标")
        p("_`/metrics` 端点不可用_")

    # ── Tokenizer ──

    h2("4. Tokenizer")
    tk = report.get("tokenizer", {})
    if tk.get("tokenize_available"):
        table(
            ["项目", "值"],
            [
                ["/tokenize 端点", "可用"],
                ["/detokenize 端点", "可用" if tk.get("detokenize_available") else "不可用"],
                ["测试文本", f'`{tk.get("test_input", "")}`'],
                ["Token 数", str(tk.get("token_count", ""))],
                ["英文 chars/token", str(tk.get("en_chars_per_token", "N/A"))],
                ["中文 chars/token", str(tk.get("zh_chars_per_token", "N/A"))],
            ],
        )
    else:
        p("_/tokenize 端点不可用，跳过 tokenizer 探测_")

    # ── 功能支持 ──

    h2("5. 功能支持")
    caps = report.get("capabilities", {})
    if caps:
        rows = []
        for key, label in CAPABILITY_LABELS.items():
            if key in caps:
                v = caps[key]
                rows.append([label, "pass" if v else "fail"])
        table(["功能", "状态"], rows)
    else:
        p("_未探测_")

    # ── Context 窗口 ──

    h2("6. Context 窗口")
    ctx = report.get("context_window", {})
    if ctx.get("skipped"):
        p("_已跳过 (`--skip-context-probe`)_")
    elif "error" in ctx:
        p(f"探测失败: {ctx['error']}")
    elif ctx:
        table(
            ["项目", "值"],
            [
                ["最大成功 token 数", f"~{ctx.get('context_max_ok_tokens', '?')}"],
                ["首次失败 token 数", str(ctx.get("context_first_fail_tokens", ""))],
                ["探测方法", ctx.get("method", "")],
                ["备注", ctx.get("note", "")],
            ],
        )

    # ── 基线性能 ──

    h2("7. 基线性能 (单请求)")
    baseline = report.get("baseline_performance", {})
    tests = baseline.get("tests", [])
    if tests:
        rows = []
        for t in tests:
            if "error" in t:
                rows.append([
                    t["label"],
                    str(t.get("input_tokens", "")),
                    "FAIL", "", "", "", t["error"],
                ])
            else:
                rows.append([
                    t["label"],
                    str(t.get("input_tokens", "")),
                    str(t.get("output_tokens", "")),
                    f"{t['ttft_s']:.3f}",
                    f"{t['decode_tok_per_s']:.1f}",
                    f"{t['prefill_tok_per_s']:.0f}",
                    f"{t['total_time_s']:.2f}",
                ])
        table(
            ["Prompt", "Input Tok", "Output Tok", "TTFT (s)", "Decode (tok/s)", "Prefill (tok/s)", "Total (s)"],
            rows,
        )
    else:
        p("_未探测_")

    # ── 并发测试 ──

    h2("8. 并发处理能力")
    conc = report.get("concurrency", {})
    ctests = conc.get("tests", [])
    if ctests:
        rows = []
        for t in ctests:
            rows.append([
                str(t["concurrency"]),
                f"{t['success']}/{t['total']}",
                f"{t['wall_time_s']:.2f}",
                f"{t['avg_latency_s']:.2f}",
            ])
        table(["并发数", "成功率", "Wall Time (s)", "Avg Latency (s)"], rows)
        if "max_concurrency_ok" in conc:
            p(f"最大稳定并发: **{conc['max_concurrency_ok']}**")
    else:
        p("_未探测_")

    # ── 尾部 ──

    lines.append("---\n")
    p(f"_生成工具: probe_model.py v{report.get('probe_version', '?')}_")

    return "\n".join(lines)


# ── 主流程 ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM 模型能力 & 服务配置探测")
    parser.add_argument("-H", "--host", default="localhost", help="目标主机 (默认: localhost)")
    parser.add_argument("-p", "--port", type=int, default=8080, help="目标端口 (默认: 8080)")
    parser.add_argument("-o", "--output", help="保存 JSON 到指定文件")
    parser.add_argument("--md", "--markdown", dest="markdown", help="保存 Markdown 报告到指定文件")
    parser.add_argument("--json-only", action="store_true", help="只输出 JSON 到 stdout")
    parser.add_argument("--md-only", action="store_true", help="只输出 Markdown 到 stdout")
    parser.add_argument("--skip-context-probe", action="store_true", help="跳过 context 窗口探测")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    quiet = args.json_only or args.md_only

    report = {
        "target": base,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "probe_version": "1.0.0",
    }

    def log(msg: str):
        if not quiet:
            print(msg, flush=True)

    # 1. 健康检查
    log("[1/7] 健康检查 ...")
    report["health"] = probe_health(base)
    if not report["health"]["ok"]:
        log(f"  ✗ 服务不可达: {report['health']['body']}")
        log("  终止探测。")
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        sys.exit(1)
    log("      ✓ OK")

    # 2. 服务端配置
    log("[2/7] 服务端配置 ...")
    report["server_props"] = probe_server_props(base)

    # 3. 模型信息
    log("[3/7] 模型信息 ...")
    report["model_info"] = probe_model_info(base)
    model_id = report["model_info"].get("model_id", "unknown")
    log(f"      模型: {model_id}")

    # 4. Tokenizer
    log("[4/7] Tokenizer 探测 ...")
    report["tokenizer"] = probe_tokenizer(base)

    # 5. 功能支持
    log("[5/7] 功能支持探测 ...")
    report["capabilities"] = probe_capabilities(base, model_id)

    # 6. Context 窗口
    if args.skip_context_probe:
        log("[6/7] Context 窗口探测 ... 跳过")
        report["context_window"] = {"skipped": True}
    else:
        log("[6/7] Context 窗口探测 (可能耗时较长) ...")
        report["context_window"] = probe_context_window(base, model_id)

    # 7. 基线性能
    log("[7/7] 基线性能测试 ...")
    report["baseline_performance"] = probe_baseline_performance(base, model_id)

    # 附加: 并发测试
    log("[bonus] 并发处理能力 ...")
    report["concurrency"] = probe_concurrency(base, model_id)

    # 输出
    md_text = render_markdown(report)

    if args.json_only:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.md_only:
        print(md_text)
    else:
        print_report(report)

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        if not quiet:
            print(f"\nJSON 已保存到: {args.output}")

    if args.markdown:
        Path(args.markdown).write_text(md_text)
        if not quiet:
            print(f"Markdown 已保存到: {args.markdown}")


if __name__ == "__main__":
    main()
