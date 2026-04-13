#!/usr/bin/env python3
"""
LLM Full Benchmark — 一键全自动压测
用法: python3 run_full_benchmark.py --host <IP> [--port 8080] [--dry-run] [--native-only] [--max-concurrency N]
"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="run_full_benchmark",
        description="LLM 全自动 Benchmark — 8 组对比测试",
    )
    parser.add_argument("--host", required=True, help="目标机器 IP")
    parser.add_argument("--port", type=int, default=8080, help="llama-server 端口 (默认 8080)")
    parser.add_argument("--max-concurrency", type=int, default=None, help="覆盖自动检测的并发上限")
    parser.add_argument("--dry-run", action="store_true", help="只生成计划，不执行")
    parser.add_argument("--native-only", action="store_true", help="跳过 evalscope 组（仅跑 native 引擎）")
    parser.add_argument("--output-dir", default=None, help="输出目录 (默认 ./results/benchmark-{timestamp})")
    args = parser.parse_args()

    from src.llm_stress_test.benchmark import run_benchmark
    sys.exit(run_benchmark(
        host=args.host,
        port=args.port,
        max_concurrency=args.max_concurrency,
        dry_run=args.dry_run,
        native_only=args.native_only,
        output_dir=args.output_dir,
    ))


if __name__ == "__main__":
    main()
