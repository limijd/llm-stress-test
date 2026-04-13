#!/usr/bin/env python3
"""
llama-server 连通性测试脚本
测试目标：DeepSeek-V3.2-Q4_K-M.gguf via llama-server
零依赖，只用标准库
"""

import json
import sys
import time
import urllib.request
import urllib.error

# ── 配置 ──────────────────────────────────────────────
HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
BASE_URL = f"http://{HOST}:{PORT}"
TIMEOUT = 120  # DeepSeek-V3.2 较大，首 token 可能较慢


def test_health():
    """检查 /health 端点"""
    print(f"[1/3] 检查健康状态 {BASE_URL}/health ...")
    try:
        req = urllib.request.Request(f"{BASE_URL}/health")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            status = body.get("status", "unknown")
            print(f"      状态: {status} (HTTP {resp.status})")
            return status == "ok"
    except urllib.error.URLError as e:
        print(f"      ✗ 连接失败: {e}")
        return False
    except Exception as e:
        print(f"      ✗ 异常: {e}")
        return False


def test_models():
    """检查 /v1/models 端点，获取模型信息"""
    print(f"[2/3] 查询模型列表 {BASE_URL}/v1/models ...")
    try:
        req = urllib.request.Request(f"{BASE_URL}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            models = body.get("data", [])
            for m in models:
                print(f"      模型: {m.get('id', 'N/A')}")
            return models[0]["id"] if models else None
    except urllib.error.URLError as e:
        print(f"      ✗ 连接失败: {e}")
        return None
    except Exception as e:
        print(f"      ✗ 异常: {e}")
        return None


def test_chat(model_id):
    """发送一条简单的 chat completion 请求"""
    print(f"[3/3] 测试对话生成 ...")
    payload = json.dumps({
        "model": model_id or "default",
        "messages": [
            {"role": "user", "content": "请用一句话介绍你自己。"}
        ],
        "max_tokens": 128,
        "temperature": 0.7,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            elapsed = time.time() - t0
            body = json.loads(resp.read())

            # 解析响应
            choice = body["choices"][0]
            content = choice["message"]["content"]
            usage = body.get("usage", {})

            print(f"      响应 (耗时 {elapsed:.2f}s):")
            print(f"      {content[:200]}")
            print(f"      Token 用量: prompt={usage.get('prompt_tokens', '?')}, "
                  f"completion={usage.get('completion_tokens', '?')}")
            return True
    except urllib.error.HTTPError as e:
        print(f"      ✗ HTTP {e.code}: {e.read().decode()[:200]}")
        return False
    except urllib.error.URLError as e:
        print(f"      ✗ 连接失败: {e}")
        return False
    except Exception as e:
        print(f"      ✗ 异常: {e}")
        return False


def main():
    print(f"═══ llama-server 连通性测试 ═══")
    print(f"目标: {BASE_URL}")
    print()

    # 1. 健康检查
    if not test_health():
        print("\n✗ 服务不可达或未就绪，终止测试")
        sys.exit(1)

    # 2. 模型列表
    model_id = test_models()

    # 3. 对话测试
    ok = test_chat(model_id)

    print()
    if ok:
        print("✓ 所有测试通过 — llama-server 工作正常")
    else:
        print("✗ 对话生成失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
