#!/usr/bin/env python3
"""
LLM 主机信息采集脚本
采集 GPU、CPU、内存、存储、网络、内核调优等信息，用于支撑 LLM 压力测试报告。
零依赖，只用 Python 标准库。目标环境：Ubuntu 24.04 + NVIDIA GPU

用法:
    python3 collect_host_info.py                          # 输出到终端
    python3 collect_host_info.py -o host_info.json        # 保存 JSON
    python3 collect_host_info.py --md host_info.md        # 保存 Markdown 报告
    python3 collect_host_info.py --md-only                # 只输出 Markdown 到 stdout
    python3 collect_host_info.py --json-only              # 只输出 JSON 到 stdout
    python3 collect_host_info.py -o info.json --md info.md  # 同时保存 JSON 和 Markdown
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ── 工具函数 ──────────────────────────────────────────

def run(cmd: str, timeout: int = 30) -> str:
    """执行 shell 命令，返回 stdout；失败返回空字符串"""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.stdout.strip()
    except Exception:
        return ""


def read_file(path: str) -> str:
    """读取文件内容，失败返回空字符串"""
    try:
        return Path(path).read_text().strip()
    except Exception:
        return ""


def parse_size_kb(text: str) -> int:
    """从 'xxx kB' 格式解析为 KB 整数"""
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else 0


def fmt_bytes(n: int, unit: str = "KB") -> str:
    """格式化字节数为人类可读"""
    multiplier = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    b = n * multiplier.get(unit, 1)
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024:.1f} KB"


# ── 采集模块 ──────────────────────────────────────────

def collect_system() -> dict:
    """基本系统信息"""
    info = {
        "hostname": run("hostname"),
        "os": read_file("/etc/os-release"),
        "kernel": run("uname -r"),
        "arch": run("uname -m"),
        "uptime": run("uptime -p"),
        "uptime_since": run("uptime -s"),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "date_local": datetime.now().isoformat(),
        "timezone": run("timedatectl show --property=Timezone --value") or time.tzname[0],
    }
    # 解析 os-release
    os_info = {}
    for line in info["os"].splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os_info[k] = v.strip('"')
    info["os_pretty"] = os_info.get("PRETTY_NAME", "Unknown")
    info["os_version"] = os_info.get("VERSION_ID", "Unknown")
    return info


def collect_cpu() -> dict:
    """CPU 详细信息"""
    info = {}
    lscpu = run("lscpu")
    if lscpu:
        fields = {}
        for line in lscpu.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fields[k.strip()] = v.strip()
        info["model"] = fields.get("Model name", "Unknown")
        info["architecture"] = fields.get("Architecture", "Unknown")
        info["sockets"] = int(fields.get("Socket(s)", 0))
        info["cores_per_socket"] = int(fields.get("Core(s) per socket", 0))
        info["threads_per_core"] = int(fields.get("Thread(s) per core", 0))
        info["total_cores"] = info["sockets"] * info["cores_per_socket"]
        info["total_threads"] = info["total_cores"] * info["threads_per_core"]
        info["max_mhz"] = fields.get("CPU max MHz", "")
        info["min_mhz"] = fields.get("CPU min MHz", "")
        info["numa_nodes"] = int(fields.get("NUMA node(s)", 0))
        info["l1d_cache"] = fields.get("L1d cache", "")
        info["l1i_cache"] = fields.get("L1i cache", "")
        info["l2_cache"] = fields.get("L2 cache", "")
        info["l3_cache"] = fields.get("L3 cache", "")
        info["flags"] = fields.get("Flags", "")
        info["virtualization"] = fields.get("Virtualization", "None")
        info["byte_order"] = fields.get("Byte Order", "")

    # NUMA 拓扑
    numa_raw = run("numactl --hardware 2>/dev/null || lscpu | grep 'NUMA node'")
    info["numa_topology"] = numa_raw

    # CPU 当前频率（逐核）
    freq = run("cat /proc/cpuinfo | grep 'cpu MHz' | head -16")
    if freq:
        freqs = [float(m.group(1)) for m in re.finditer(r":\s*([\d.]+)", freq)]
        info["current_freq_mhz"] = {
            "min": round(min(freqs), 1) if freqs else 0,
            "max": round(max(freqs), 1) if freqs else 0,
            "avg": round(sum(freqs) / len(freqs), 1) if freqs else 0,
        }

    # CPU governor
    gov = run("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null")
    info["governor"] = gov or "unknown"

    return info


def collect_memory() -> dict:
    """内存信息"""
    info = {}
    meminfo = read_file("/proc/meminfo")
    if meminfo:
        fields = {}
        for line in meminfo.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fields[k.strip()] = v.strip()
        total = parse_size_kb(fields.get("MemTotal", "0"))
        avail = parse_size_kb(fields.get("MemAvailable", "0"))
        swap_total = parse_size_kb(fields.get("SwapTotal", "0"))
        swap_free = parse_size_kb(fields.get("SwapFree", "0"))
        huge_total = int(fields.get("HugePages_Total", "0").split()[0]) if "HugePages_Total" in fields else 0
        huge_free = int(fields.get("HugePages_Free", "0").split()[0]) if "HugePages_Free" in fields else 0
        huge_size = parse_size_kb(fields.get("Hugepagesize", "0"))

        info["total"] = fmt_bytes(total)
        info["total_kb"] = total
        info["available"] = fmt_bytes(avail)
        info["available_kb"] = avail
        info["used_pct"] = round((1 - avail / total) * 100, 1) if total else 0
        info["swap_total"] = fmt_bytes(swap_total)
        info["swap_used"] = fmt_bytes(swap_total - swap_free)
        info["hugepages_total"] = huge_total
        info["hugepages_free"] = huge_free
        info["hugepage_size"] = fmt_bytes(huge_size)

    # DIMM 信息（需要 root）
    dimm = run("sudo dmidecode -t memory 2>/dev/null | grep -E 'Size|Speed|Type|Manufacturer' | head -32")
    info["dimm_info"] = dimm or "(需要 root 权限执行 dmidecode)"

    # 内存带宽（如果有 lshw）
    mem_bw = run("sudo lshw -class memory -short 2>/dev/null | head -20")
    info["memory_hardware"] = mem_bw or "(需要 root 权限执行 lshw)"

    return info


def collect_gpu() -> dict:
    """NVIDIA GPU 详细信息"""
    info = {"available": False}

    # 检测 nvidia-smi
    smi = run("which nvidia-smi")
    if not smi:
        info["error"] = "nvidia-smi 未找到"
        return info

    info["available"] = True

    # 驱动 & CUDA 版本
    info["driver_version"] = run("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
    cuda_ver = run("nvidia-smi | grep 'CUDA Version' | head -1")
    m = re.search(r"CUDA Version:\s*([\d.]+)", cuda_ver)
    info["cuda_version"] = m.group(1) if m else "unknown"

    # nvcc 编译器版本
    nvcc_ver = run("nvcc --version 2>/dev/null | grep 'release'")
    info["nvcc_version"] = nvcc_ver

    # 逐 GPU 信息
    gpu_query = run(
        "nvidia-smi --query-gpu="
        "index,name,uuid,pci.bus_id,"
        "memory.total,memory.used,memory.free,"
        "temperature.gpu,power.draw,power.limit,power.max_limit,"
        "clocks.current.graphics,clocks.max.graphics,"
        "clocks.current.memory,clocks.max.memory,"
        "utilization.gpu,utilization.memory,"
        "compute_mode,pstate,ecc.errors.corrected.aggregate.total,"
        "mig.mode.current"
        " --format=csv,noheader,nounits"
    )

    gpus = []
    if gpu_query:
        for line in gpu_query.splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) >= 18:
                gpu = {
                    "index": cols[0],
                    "name": cols[1],
                    "uuid": cols[2],
                    "pci_bus_id": cols[3],
                    "memory_total_mib": cols[4],
                    "memory_used_mib": cols[5],
                    "memory_free_mib": cols[6],
                    "temperature_c": cols[7],
                    "power_draw_w": cols[8],
                    "power_limit_w": cols[9],
                    "power_max_limit_w": cols[10],
                    "clock_graphics_mhz": cols[11],
                    "clock_graphics_max_mhz": cols[12],
                    "clock_memory_mhz": cols[13],
                    "clock_memory_max_mhz": cols[14],
                    "utilization_gpu_pct": cols[15],
                    "utilization_memory_pct": cols[16],
                    "compute_mode": cols[17],
                    "pstate": cols[18] if len(cols) > 18 else "",
                    "ecc_errors": cols[19] if len(cols) > 19 else "",
                    "mig_mode": cols[20] if len(cols) > 20 else "",
                }
                gpus.append(gpu)
    info["count"] = len(gpus)
    info["gpus"] = gpus

    # GPU 拓扑（NVLink / PCIe 互联）
    topo = run("nvidia-smi topo -m 2>/dev/null")
    info["topology_matrix"] = topo

    # NVLink 状态
    nvlink = run("nvidia-smi nvlink --status 2>/dev/null | head -60")
    info["nvlink_status"] = nvlink

    # GPU 持久化模式
    persist = run("nvidia-smi --query-gpu=persistence_mode --format=csv,noheader")
    info["persistence_mode"] = persist.splitlines() if persist else []

    # CUDA 设备可见性
    info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "(未设置，所有 GPU 可见)")

    # nvidia-smi 全量输出（原始快照）
    info["nvidia_smi_full"] = run("nvidia-smi")

    return info


def collect_storage() -> dict:
    """存储信息"""
    info = {}

    # 磁盘列表
    lsblk = run("lsblk -d -o NAME,SIZE,TYPE,MODEL,ROTA,TRAN,MOUNTPOINT 2>/dev/null")
    info["block_devices"] = lsblk

    # 文件系统使用情况
    df = run("df -hT | grep -vE '^(tmpfs|devtmpfs|overlay|Filesystem)'")
    info["filesystem_usage"] = df

    # NVMe 设备详情
    nvme = run("sudo nvme list 2>/dev/null || ls /dev/nvme* 2>/dev/null | head -10")
    info["nvme_devices"] = nvme

    # I/O 调度器
    schedulers = run("cat /sys/block/*/queue/scheduler 2>/dev/null | head -10")
    info["io_schedulers"] = schedulers

    # 查找可能的模型文件
    gguf_files = run(
        "find / -name '*.gguf' -o -name '*.safetensors' -o -name '*.bin' 2>/dev/null "
        "| head -20 | while read f; do ls -lh \"$f\" 2>/dev/null; done",
        timeout=15,
    )
    info["model_files"] = gguf_files or "(未搜索到或无权限)"

    return info


def collect_network() -> dict:
    """网络信息"""
    info = {}

    # 网卡列表
    info["interfaces"] = run("ip -br addr 2>/dev/null || ifconfig -a 2>/dev/null | grep -E '^\\w|inet'")

    # 网卡速率
    info["link_speeds"] = run(
        "for dev in $(ls /sys/class/net/ 2>/dev/null); do "
        "  speed=$(cat /sys/class/net/$dev/speed 2>/dev/null); "
        "  [ -n \"$speed\" ] && echo \"$dev: ${speed} Mbps\"; "
        "done"
    )

    # 监听端口（找 LLM 服务）
    info["listening_ports"] = run("ss -tlnp 2>/dev/null | grep -E '(8080|8000|8443|3000|5000|11434)'")

    # 防火墙状态
    info["firewall"] = run("sudo ufw status 2>/dev/null || sudo iptables -L -n 2>/dev/null | head -20")

    return info


def collect_pcie() -> dict:
    """PCIe 拓扑，对多 GPU 通信很重要"""
    info = {}

    # GPU 的 PCIe 链路信息
    pcie_gpu = run(
        "for gpu_pci in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null); do "
        "  bdf=$(echo $gpu_pci | sed 's/00000000://' | tr '[:upper:]' '[:lower:]'); "
        "  echo \"=== $gpu_pci ===\"; "
        "  sudo lspci -vvs $bdf 2>/dev/null | grep -E '(LnkCap|LnkSta|NUMA|MaxPayload|MaxReadReq)'; "
        "done"
    )
    info["gpu_pcie_links"] = pcie_gpu or "(需要 root 权限)"

    # PCIe 拓扑树
    info["pcie_tree"] = run("sudo lspci -tv 2>/dev/null | head -50") or "(需要 root 权限)"

    # IOMMU 状态
    info["iommu"] = run("dmesg 2>/dev/null | grep -i iommu | head -5") or run(
        "cat /proc/cmdline 2>/dev/null | tr ' ' '\\n' | grep iommu"
    )

    return info


def collect_kernel_tuning() -> dict:
    """内核调优参数，影响 LLM 服务性能"""
    params = {
        "vm.swappiness": run("sysctl -n vm.swappiness 2>/dev/null"),
        "vm.overcommit_memory": run("sysctl -n vm.overcommit_memory 2>/dev/null"),
        "vm.dirty_ratio": run("sysctl -n vm.dirty_ratio 2>/dev/null"),
        "vm.dirty_background_ratio": run("sysctl -n vm.dirty_background_ratio 2>/dev/null"),
        "vm.nr_hugepages": run("sysctl -n vm.nr_hugepages 2>/dev/null"),
        "kernel.shmmax": run("sysctl -n kernel.shmmax 2>/dev/null"),
        "net.core.somaxconn": run("sysctl -n net.core.somaxconn 2>/dev/null"),
        "net.core.rmem_max": run("sysctl -n net.core.rmem_max 2>/dev/null"),
        "net.core.wmem_max": run("sysctl -n net.core.wmem_max 2>/dev/null"),
        "net.ipv4.tcp_max_syn_backlog": run("sysctl -n net.ipv4.tcp_max_syn_backlog 2>/dev/null"),
    }
    info = {"sysctl": params}

    # Transparent Huge Pages
    thp = read_file("/sys/kernel/mm/transparent_hugepage/enabled")
    info["transparent_hugepages"] = thp

    # CPU 频率策略
    info["cpu_governor"] = run("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null")

    # 电源管理
    info["power_profile"] = run("powerprofilesctl get 2>/dev/null") or run(
        "cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null"
    )

    # cgroup 限制（可能在容器中）
    cg_mem = read_file("/sys/fs/cgroup/memory.max")
    cg_cpu = read_file("/sys/fs/cgroup/cpu.max")
    info["cgroup_memory_limit"] = cg_mem or "(无限制或不在 cgroup 中)"
    info["cgroup_cpu_limit"] = cg_cpu or "(无限制或不在 cgroup 中)"

    return info


def collect_llm_server() -> dict:
    """检测正在运行的 LLM 服务"""
    info = {}

    # 查找 llama-server 进程
    ps = run("ps aux | grep -E '(llama-server|llama.cpp|vllm|text-generation|ollama)' | grep -v grep")
    info["llm_processes"] = ps

    # 提取 llama-server 启动参数
    llama_cmd = run(
        "ps -eo args | grep 'llama-server' | grep -v grep | head -1"
    )
    info["llama_server_cmdline"] = llama_cmd

    if llama_cmd:
        # 解析关键参数
        params = {}
        # 模型路径
        m = re.search(r"-m\s+(\S+)", llama_cmd)
        params["model_path"] = m.group(1) if m else ""
        # context size
        m = re.search(r"-c\s+(\d+)", llama_cmd)
        params["context_size"] = int(m.group(1)) if m else "default"
        # GPU layers
        m = re.search(r"-ngl\s+(\d+)", llama_cmd)
        params["gpu_layers"] = int(m.group(1)) if m else "default (all)"
        # batch size
        m = re.search(r"-b\s+(\d+)", llama_cmd)
        params["batch_size"] = int(m.group(1)) if m else "default"
        # threads
        m = re.search(r"-t\s+(\d+)", llama_cmd)
        params["threads"] = int(m.group(1)) if m else "default"
        # parallel slots
        m = re.search(r"--parallel\s+(\d+)", llama_cmd)
        params["parallel_slots"] = int(m.group(1)) if m else "default"
        # host/port
        m = re.search(r"--host\s+(\S+)", llama_cmd)
        params["host"] = m.group(1) if m else "127.0.0.1"
        m = re.search(r"--port\s+(\d+)", llama_cmd)
        params["port"] = int(m.group(1)) if m else 8080
        # flash attention
        params["flash_attention"] = "--flash-attn" in llama_cmd or "-fa" in llama_cmd
        # mmap / mlock
        params["use_mmap"] = "--no-mmap" not in llama_cmd
        params["use_mlock"] = "--mlock" in llama_cmd
        # tensor split
        m = re.search(r"--tensor-split\s+(\S+)", llama_cmd)
        params["tensor_split"] = m.group(1) if m else ""
        # cont batching
        params["cont_batching"] = "-cb" in llama_cmd or "--cont-batching" in llama_cmd

        info["parsed_params"] = params

        # 模型文件大小
        model_path = params.get("model_path", "")
        if model_path:
            size = run(f"ls -lh '{model_path}' 2>/dev/null")
            info["model_file_info"] = size

    # llama-server 版本
    info["llama_server_version"] = run("llama-server --version 2>/dev/null")

    # 检查 API 是否响应
    for port in [8080, 8000]:
        health = run(f"curl -s -m 5 http://localhost:{port}/health 2>/dev/null")
        if health:
            info[f"health_port_{port}"] = health
            break

    return info


def collect_container() -> dict:
    """容器 / 虚拟化检测"""
    info = {}
    info["in_docker"] = os.path.exists("/.dockerenv")
    info["in_container"] = bool(run("cat /proc/1/cgroup 2>/dev/null | grep -E '(docker|lxc|kubepods)'"))
    info["systemd_detect_virt"] = run("systemd-detect-virt 2>/dev/null") or "none"

    # Docker 版本（如果有）
    info["docker_version"] = run("docker --version 2>/dev/null")

    # NVIDIA Container Runtime
    info["nvidia_container_runtime"] = run("nvidia-container-runtime --version 2>/dev/null")

    return info


def collect_software() -> dict:
    """相关软件版本"""
    info = {}
    info["python"] = run("python3 --version 2>/dev/null")
    info["gcc"] = run("gcc --version 2>/dev/null | head -1")
    info["cmake"] = run("cmake --version 2>/dev/null | head -1")
    info["cuda_toolkit"] = run("dpkg -l 2>/dev/null | grep cuda-toolkit | head -3") or run(
        "rpm -qa 2>/dev/null | grep cuda-toolkit | head -3"
    )
    info["cudnn"] = run("dpkg -l 2>/dev/null | grep libcudnn | head -3")
    info["nccl"] = run("dpkg -l 2>/dev/null | grep libnccl | head -3")
    return info


# ── 输出格式化 ──────────────────────────────────────────

def print_section(title: str, data: dict, indent: int = 0):
    """递归打印段落"""
    prefix = "  " * indent
    if indent == 0:
        print(f"\n{'═' * 60}")
        print(f"  {title}")
        print(f"{'═' * 60}")
    else:
        print(f"{prefix}── {title} ──")

    for k, v in data.items():
        if isinstance(v, dict):
            print_section(k, v, indent + 1)
        elif isinstance(v, list):
            print(f"{prefix}  {k}:")
            for item in v:
                if isinstance(item, dict):
                    for kk, vv in item.items():
                        print(f"{prefix}    {kk}: {vv}")
                    print()
                else:
                    print(f"{prefix}    {item}")
        elif isinstance(v, str) and "\n" in v:
            print(f"{prefix}  {k}:")
            for line in v.splitlines():
                print(f"{prefix}    {line}")
        else:
            print(f"{prefix}  {k}: {v}")


def print_report(report: dict):
    """打印完整报告"""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          LLM 主机信息采集报告                           ║")
    print(f"║  采集时间: {report['collected_at'][:19]}                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    section_titles = {
        "system": "系统概况",
        "cpu": "CPU 信息",
        "memory": "内存信息",
        "gpu": "GPU 信息 (NVIDIA)",
        "pcie": "PCIe 拓扑",
        "storage": "存储信息",
        "network": "网络信息",
        "kernel_tuning": "内核调优参数",
        "llm_server": "LLM 服务检测",
        "container": "容器 / 虚拟化",
        "software": "软件版本",
    }

    for key, title in section_titles.items():
        if key in report:
            print_section(title, report[key])

    print(f"\n{'═' * 60}")
    print("  采集完成")
    print(f"{'═' * 60}")


# ── Markdown 渲染 ──────────────────────────────────────

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """生成 Markdown 表格"""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.append("")
    return "\n".join(lines)


def _md_code_block(text: str, lang: str = "") -> str:
    """生成 Markdown 代码块"""
    if not text or text.startswith("("):
        return f"_{text or 'N/A'}_\n"
    return f"```{lang}\n{text}\n```\n"


def render_markdown(report: dict) -> str:
    """将主机信息渲染为 Markdown"""
    lines: list[str] = []

    def h1(t: str): lines.append(f"# {t}\n")
    def h2(t: str): lines.append(f"## {t}\n")
    def h3(t: str): lines.append(f"### {t}\n")
    def p(t: str): lines.append(f"{t}\n")

    hostname = report.get("system", {}).get("hostname", "Unknown")
    h1(f"LLM 主机信息采集报告 — {hostname}")

    p(f"采集时间 (UTC): `{report.get('collected_at', '')[:19]}`")
    p(f"采集器版本: {report.get('collector_version', '')}\n")

    # ── 1. 系统概况 ──

    h2("1. 系统概况")
    sys_info = report.get("system", {})
    lines.append(_md_table(
        ["项目", "值"],
        [
            ["主机名", f"`{sys_info.get('hostname', '')}`"],
            ["操作系统", sys_info.get("os_pretty", "")],
            ["OS 版本", sys_info.get("os_version", "")],
            ["内核", f"`{sys_info.get('kernel', '')}`"],
            ["架构", sys_info.get("arch", "")],
            ["运行时间", sys_info.get("uptime", "")],
            ["启动时间", sys_info.get("uptime_since", "")],
            ["时区", sys_info.get("timezone", "")],
        ],
    ))

    # ── 2. CPU ──

    h2("2. CPU 信息")
    cpu = report.get("cpu", {})
    if cpu:
        lines.append(_md_table(
            ["项目", "值"],
            [
                ["型号", cpu.get("model", "")],
                ["架构", cpu.get("architecture", "")],
                ["Sockets", str(cpu.get("sockets", ""))],
                ["物理核心", str(cpu.get("total_cores", ""))],
                ["逻辑线程", str(cpu.get("total_threads", ""))],
                ["每核线程", str(cpu.get("threads_per_core", ""))],
                ["最大频率 (MHz)", cpu.get("max_mhz", "")],
                ["最小频率 (MHz)", cpu.get("min_mhz", "")],
                ["L1d Cache", cpu.get("l1d_cache", "")],
                ["L1i Cache", cpu.get("l1i_cache", "")],
                ["L2 Cache", cpu.get("l2_cache", "")],
                ["L3 Cache", cpu.get("l3_cache", "")],
                ["NUMA Nodes", str(cpu.get("numa_nodes", ""))],
                ["Governor", cpu.get("governor", "")],
                ["虚拟化", cpu.get("virtualization", "")],
            ],
        ))
        freq = cpu.get("current_freq_mhz")
        if freq:
            p(f"当前频率: min={freq['min']} MHz, max={freq['max']} MHz, avg={freq['avg']} MHz")

        numa_topo = cpu.get("numa_topology", "")
        if numa_topo:
            h3("2.1 NUMA 拓扑")
            lines.append(_md_code_block(numa_topo))

    # ── 3. 内存 ──

    h2("3. 内存信息")
    mem = report.get("memory", {})
    if mem:
        lines.append(_md_table(
            ["项目", "值"],
            [
                ["总内存", mem.get("total", "")],
                ["可用内存", mem.get("available", "")],
                ["使用率", f"{mem.get('used_pct', '')}%"],
                ["Swap 总量", mem.get("swap_total", "")],
                ["Swap 已用", mem.get("swap_used", "")],
                ["HugePages 总数", str(mem.get("hugepages_total", ""))],
                ["HugePages 空闲", str(mem.get("hugepages_free", ""))],
                ["HugePage 大小", mem.get("hugepage_size", "")],
            ],
        ))
        dimm = mem.get("dimm_info", "")
        if dimm and not dimm.startswith("("):
            h3("3.1 DIMM 硬件")
            lines.append(_md_code_block(dimm))

    # ── 4. GPU ──

    h2("4. GPU 信息 (NVIDIA)")
    gpu = report.get("gpu", {})
    if not gpu.get("available"):
        p(f"_{gpu.get('error', 'NVIDIA GPU 不可用')}_")
    else:
        lines.append(_md_table(
            ["项目", "值"],
            [
                ["GPU 数量", str(gpu.get("count", ""))],
                ["驱动版本", gpu.get("driver_version", "")],
                ["CUDA 版本", gpu.get("cuda_version", "")],
                ["NVCC 版本", gpu.get("nvcc_version", "") or "N/A"],
                ["CUDA_VISIBLE_DEVICES", f"`{gpu.get('cuda_visible_devices', '')}`"],
            ],
        ))

        gpus = gpu.get("gpus", [])
        if gpus:
            h3("4.1 逐 GPU 详情")
            rows = []
            for g in gpus:
                rows.append([
                    g.get("index", ""),
                    g.get("name", ""),
                    f"{g.get('memory_total_mib', '')} MiB",
                    f"{g.get('memory_used_mib', '')} MiB",
                    f"{g.get('temperature_c', '')} C",
                    f"{g.get('power_draw_w', '')} / {g.get('power_limit_w', '')} W",
                    f"{g.get('clock_graphics_mhz', '')} MHz",
                    f"{g.get('utilization_gpu_pct', '')}%",
                    g.get("pstate", ""),
                    g.get("compute_mode", ""),
                ])
            lines.append(_md_table(
                ["#", "名称", "显存总量", "显存已用", "温度", "功耗/限制", "GPU 时钟", "利用率", "PState", "Compute Mode"],
                rows,
            ))

        # 持久化模式
        persist = gpu.get("persistence_mode", [])
        if persist:
            p(f"Persistence Mode: `{'  '.join(persist)}`")

        # NVLink / 拓扑
        topo = gpu.get("topology_matrix", "")
        if topo:
            h3("4.2 GPU 拓扑矩阵 (NVLink/PCIe)")
            lines.append(_md_code_block(topo))

        nvlink = gpu.get("nvlink_status", "")
        if nvlink:
            h3("4.3 NVLink 状态")
            lines.append(_md_code_block(nvlink))

        # nvidia-smi 全量
        smi_full = gpu.get("nvidia_smi_full", "")
        if smi_full:
            h3("4.4 nvidia-smi 完整输出")
            lines.append(_md_code_block(smi_full))

    # ── 5. PCIe ──

    h2("5. PCIe 拓扑")
    pcie = report.get("pcie", {})
    if pcie:
        gpu_links = pcie.get("gpu_pcie_links", "")
        if gpu_links:
            h3("5.1 GPU PCIe 链路")
            lines.append(_md_code_block(gpu_links))

        pcie_tree = pcie.get("pcie_tree", "")
        if pcie_tree:
            h3("5.2 PCIe 设备树")
            lines.append(_md_code_block(pcie_tree))

        iommu = pcie.get("iommu", "")
        if iommu:
            p(f"IOMMU: `{iommu}`")

    # ── 6. 存储 ──

    h2("6. 存储信息")
    storage = report.get("storage", {})
    if storage:
        blk = storage.get("block_devices", "")
        if blk:
            h3("6.1 块设备")
            lines.append(_md_code_block(blk))

        fs = storage.get("filesystem_usage", "")
        if fs:
            h3("6.2 文件系统使用")
            lines.append(_md_code_block(fs))

        nvme = storage.get("nvme_devices", "")
        if nvme:
            h3("6.3 NVMe 设备")
            lines.append(_md_code_block(nvme))

        sched = storage.get("io_schedulers", "")
        if sched:
            p(f"I/O 调度器: `{sched}`")

        model_files = storage.get("model_files", "")
        if model_files and not model_files.startswith("("):
            h3("6.4 模型文件")
            lines.append(_md_code_block(model_files))

    # ── 7. 网络 ──

    h2("7. 网络信息")
    net = report.get("network", {})
    if net:
        ifaces = net.get("interfaces", "")
        if ifaces:
            h3("7.1 网络接口")
            lines.append(_md_code_block(ifaces))

        speeds = net.get("link_speeds", "")
        if speeds:
            h3("7.2 链路速率")
            lines.append(_md_code_block(speeds))

        ports = net.get("listening_ports", "")
        if ports:
            h3("7.3 LLM 相关监听端口")
            lines.append(_md_code_block(ports))

    # ── 8. 内核调优 ──

    h2("8. 内核调优参数")
    kt = report.get("kernel_tuning", {})
    if kt:
        sysctl = kt.get("sysctl", {})
        if sysctl:
            rows = [[f"`{k}`", str(v)] for k, v in sysctl.items() if v]
            if rows:
                lines.append(_md_table(["参数", "值"], rows))

        thp = kt.get("transparent_hugepages", "")
        if thp:
            p(f"Transparent HugePages: `{thp}`")

        gov = kt.get("cpu_governor", "")
        if gov:
            p(f"CPU Governor: `{gov}`")

        power = kt.get("power_profile", "")
        if power:
            p(f"Power Profile: `{power}`")

        cg_mem = kt.get("cgroup_memory_limit", "")
        cg_cpu = kt.get("cgroup_cpu_limit", "")
        if cg_mem or cg_cpu:
            p(f"cgroup Memory: `{cg_mem}` | cgroup CPU: `{cg_cpu}`")

    # ── 9. LLM 服务检测 ──

    h2("9. LLM 服务检测")
    llm = report.get("llm_server", {})
    if llm:
        ver = llm.get("llama_server_version", "")
        if ver:
            p(f"llama-server 版本: `{ver}`")

        cmdline = llm.get("llama_server_cmdline", "")
        if cmdline:
            h3("9.1 启动命令")
            lines.append(_md_code_block(cmdline, "bash"))

        params = llm.get("parsed_params", {})
        if params:
            h3("9.2 解析后参数")
            rows = [[f"`{k}`", str(v)] for k, v in params.items()]
            lines.append(_md_table(["参数", "值"], rows))

        model_file = llm.get("model_file_info", "")
        if model_file:
            p(f"模型文件: `{model_file}`")

        procs = llm.get("llm_processes", "")
        if procs:
            h3("9.3 LLM 进程")
            lines.append(_md_code_block(procs))

        # 健康检查结果
        for port in [8080, 8000]:
            hkey = f"health_port_{port}"
            if hkey in llm:
                p(f"Health (port {port}): `{llm[hkey]}`")

    # ── 10. 容器 / 虚拟化 ──

    h2("10. 容器 / 虚拟化")
    ctr = report.get("container", {})
    if ctr:
        lines.append(_md_table(
            ["项目", "值"],
            [
                ["Docker 容器内", str(ctr.get("in_docker", False))],
                ["容器环境", str(ctr.get("in_container", False))],
                ["虚拟化检测", ctr.get("systemd_detect_virt", "none")],
                ["Docker 版本", ctr.get("docker_version", "") or "N/A"],
                ["NVIDIA Container Runtime", ctr.get("nvidia_container_runtime", "") or "N/A"],
            ],
        ))

    # ── 11. 软件版本 ──

    h2("11. 软件版本")
    sw = report.get("software", {})
    if sw:
        rows = []
        labels = {
            "python": "Python",
            "gcc": "GCC",
            "cmake": "CMake",
            "cuda_toolkit": "CUDA Toolkit",
            "cudnn": "cuDNN",
            "nccl": "NCCL",
        }
        for k, label in labels.items():
            v = sw.get(k, "")
            if v:
                rows.append([label, f"`{v}`"])
            else:
                rows.append([label, "N/A"])
        lines.append(_md_table(["软件", "版本"], rows))

    # ── 尾部 ──

    lines.append("---\n")
    p(f"_生成工具: collect_host_info.py v{report.get('collector_version', '?')}_")

    return "\n".join(lines)


# ── 主流程 ──────────────────────────────────────────

def collect_all() -> dict:
    """采集所有信息"""
    report = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "collector_version": "1.0.0",
    }

    sections = [
        ("system", collect_system),
        ("cpu", collect_cpu),
        ("memory", collect_memory),
        ("gpu", collect_gpu),
        ("pcie", collect_pcie),
        ("storage", collect_storage),
        ("network", collect_network),
        ("kernel_tuning", collect_kernel_tuning),
        ("llm_server", collect_llm_server),
        ("container", collect_container),
        ("software", collect_software),
    ]

    for name, collector in sections:
        try:
            report[name] = collector()
        except Exception as e:
            report[name] = {"error": str(e)}

    return report


def main():
    parser = argparse.ArgumentParser(description="LLM 主机信息采集")
    parser.add_argument("-o", "--output", help="保存 JSON 到指定文件")
    parser.add_argument("--md", "--markdown", dest="markdown", help="保存 Markdown 报告到指定文件")
    parser.add_argument("--json-only", action="store_true", help="只输出 JSON 到 stdout")
    parser.add_argument("--md-only", action="store_true", help="只输出 Markdown 到 stdout")
    args = parser.parse_args()

    report = collect_all()
    md_text = render_markdown(report)

    if args.json_only:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.md_only:
        print(md_text)
    else:
        print_report(report)

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        if not args.json_only and not args.md_only:
            print(f"\nJSON 已保存到: {args.output}")

    if args.markdown:
        Path(args.markdown).write_text(md_text)
        if not args.json_only and not args.md_only:
            print(f"Markdown 已保存到: {args.markdown}")


if __name__ == "__main__":
    main()
