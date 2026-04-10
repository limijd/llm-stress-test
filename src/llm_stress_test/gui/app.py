"""Tkinter GUI 配置编辑器 — 仅用于创建/编辑 YAML 配置文件，不执行测试"""
from __future__ import annotations

import json
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Any

import yaml

from ..config import validate_config, ConfigError


class ConfigEditorApp:
    """主窗口：滚动表单 + 菜单栏"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LLM 压力测试 — 配置编辑器")
        self.root.geometry("750x850")
        self._current_path: Path | None = None

        self._build_menu()
        self._build_form()
        self._build_buttons()

    # ------------------------------------------------------------------
    # 菜单栏
    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开配置", accelerator="Ctrl+O", command=self._on_open)
        file_menu.add_command(label="保存配置", accelerator="Ctrl+S", command=self._on_save)
        file_menu.add_command(label="另存为...", accelerator="Ctrl+Shift+S", command=self._on_save_as)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        self.root.config(menu=menubar)

        self.root.bind("<Control-o>", lambda _: self._on_open())
        self.root.bind("<Control-s>", lambda _: self._on_save())
        self.root.bind("<Control-S>", lambda _: self._on_save_as())

    # ------------------------------------------------------------------
    # 滚动表单主体
    # ------------------------------------------------------------------
    def _build_form(self) -> None:
        # 外层容器：Canvas + Scrollbar 实现竖向滚动
        container = tk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 实际表单放在 canvas 内的 frame 里
        self._form_frame = tk.Frame(canvas)
        form_window = canvas.create_window((0, 0), window=self._form_frame, anchor="nw")

        def _on_frame_configure(event: tk.Event) -> None:  # noqa: ARG001
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfig(form_window, width=event.width)

        self._form_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        # 鼠标滚轮
        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # 各分区
        self._build_section_target()
        self._build_section_engine()
        self._build_section_criteria()
        self._build_section_degradation()
        self._build_section_output()

    # ------------------------------------------------------------------
    # 辅助：创建分区标题
    # ------------------------------------------------------------------
    def _section_label(self, text: str) -> None:
        lbl = tk.Label(
            self._form_frame,
            text=text,
            font=("", 11, "bold"),
            anchor="w",
            fg="#2c5282",
        )
        lbl.pack(fill=tk.X, padx=8, pady=(14, 2))
        ttk.Separator(self._form_frame, orient="horizontal").pack(fill=tk.X, padx=8, pady=(0, 6))

    # 辅助：单行 label + entry
    def _labeled_entry(self, label: str, default: str = "") -> tk.StringVar:
        row = tk.Frame(self._form_frame)
        row.pack(fill=tk.X, padx=16, pady=2)
        tk.Label(row, text=label, width=20, anchor="w").pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        tk.Entry(row, textvariable=var, width=48).pack(side=tk.LEFT, fill=tk.X, expand=True)
        return var

    # ------------------------------------------------------------------
    # 测试目标
    # ------------------------------------------------------------------
    def _build_section_target(self) -> None:
        self._section_label("测试目标")
        self.v_name = self._labeled_entry("name")
        self.v_api_url = self._labeled_entry("api_url")
        self.v_api_key = self._labeled_entry("api_key")
        self.v_model = self._labeled_entry("model")

    # ------------------------------------------------------------------
    # 引擎与测试参数
    # ------------------------------------------------------------------
    def _build_section_engine(self) -> None:
        self._section_label("引擎与测试参数")

        # engine 单选
        row = tk.Frame(self._form_frame)
        row.pack(fill=tk.X, padx=16, pady=2)
        tk.Label(row, text="engine", width=20, anchor="w").pack(side=tk.LEFT)
        self.v_engine = tk.StringVar(value="evalscope")
        for val in ("evalscope", "native"):
            tk.Radiobutton(row, text=val, variable=self.v_engine, value=val).pack(side=tk.LEFT, padx=4)

        self.v_concurrency = self._labeled_entry("concurrency（逗号分隔）", "1,5,10,20,50")
        self.v_rpl = self._labeled_entry("requests_per_level（逗号分隔）", "10,50,100,200,500")
        self.v_dataset = self._labeled_entry("dataset", "longalpaca")

        # stream 复选框
        row2 = tk.Frame(self._form_frame)
        row2.pack(fill=tk.X, padx=16, pady=2)
        tk.Label(row2, text="stream", width=20, anchor="w").pack(side=tk.LEFT)
        self.v_stream = tk.BooleanVar(value=True)
        tk.Checkbutton(row2, variable=self.v_stream).pack(side=tk.LEFT)

        self.v_extra_args = self._labeled_entry("extra_args（JSON）", "{}")

    # ------------------------------------------------------------------
    # 通过条件
    # ------------------------------------------------------------------
    def _build_section_criteria(self) -> None:
        self._section_label("通过条件")

        hint = tk.Label(
            self._form_frame,
            text="每行一条，格式：metric operator threshold  （如: success_rate >= 1.0）",
            anchor="w",
            fg="#555",
            font=("", 9),
        )
        hint.pack(fill=tk.X, padx=16)

        self.w_criteria = tk.Text(self._form_frame, height=5, width=60, font=("Courier", 10))
        self.w_criteria.pack(fill=tk.X, padx=16, pady=(2, 0))
        # 默认示例
        default_criteria = "success_rate >= 1.0\ngen_toks_per_sec >= 500\navg_ttft <= 10.0"
        self.w_criteria.insert("1.0", default_criteria)

    # ------------------------------------------------------------------
    # 降级策略
    # ------------------------------------------------------------------
    def _build_section_degradation(self) -> None:
        self._section_label("降级策略")

        row = tk.Frame(self._form_frame)
        row.pack(fill=tk.X, padx=16, pady=2)
        tk.Label(row, text="enabled", width=20, anchor="w").pack(side=tk.LEFT)
        self.v_deg_enabled = tk.BooleanVar(value=True)
        tk.Checkbutton(row, variable=self.v_deg_enabled).pack(side=tk.LEFT)

        self.v_deg_start = self._labeled_entry("start_concurrency", "50")
        self.v_deg_step = self._labeled_entry("step", "10")
        self.v_deg_min = self._labeled_entry("min_concurrency", "10")

    # ------------------------------------------------------------------
    # 输出设置
    # ------------------------------------------------------------------
    def _build_section_output(self) -> None:
        self._section_label("输出设置")

        self.v_out_dir = self._labeled_entry("dir", "./results")

        # 格式复选框
        row = tk.Frame(self._form_frame)
        row.pack(fill=tk.X, padx=16, pady=2)
        tk.Label(row, text="formats", width=20, anchor="w").pack(side=tk.LEFT)
        self.v_fmt_json = tk.BooleanVar(value=True)
        self.v_fmt_csv = tk.BooleanVar(value=True)
        self.v_fmt_html = tk.BooleanVar(value=True)
        tk.Checkbutton(row, text="JSON", variable=self.v_fmt_json).pack(side=tk.LEFT)
        tk.Checkbutton(row, text="CSV", variable=self.v_fmt_csv).pack(side=tk.LEFT)
        tk.Checkbutton(row, text="HTML", variable=self.v_fmt_html).pack(side=tk.LEFT)

        row2 = tk.Frame(self._form_frame)
        row2.pack(fill=tk.X, padx=16, pady=2)
        tk.Label(row2, text="charts", width=20, anchor="w").pack(side=tk.LEFT)
        self.v_charts = tk.BooleanVar(value=True)
        tk.Checkbutton(row2, variable=self.v_charts).pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # 底部操作按钮
    # ------------------------------------------------------------------
    def _build_buttons(self) -> None:
        bar = tk.Frame(self.root)
        bar.pack(fill=tk.X, padx=8, pady=8)

        tk.Button(bar, text="打开配置", width=12, command=self._on_open).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="保存配置", width=12, command=self._on_save).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="另存为...", width=12, command=self._on_save_as).pack(side=tk.LEFT, padx=4)

    # ------------------------------------------------------------------
    # 核心：表单 → dict
    # ------------------------------------------------------------------
    def _to_config(self) -> dict:
        # 解析逗号分隔的整数列表
        def _parse_int_list(raw: str) -> list[int]:
            return [int(x.strip()) for x in raw.split(",") if x.strip()]

        # 解析 extra_args JSON
        extra_raw = self.v_extra_args.get().strip()
        try:
            extra_args: Any = json.loads(extra_raw) if extra_raw else {}
        except json.JSONDecodeError as exc:
            raise ConfigError(f"extra_args 不是合法 JSON: {exc}") from exc

        # 解析通过条件
        criteria_raw = self.w_criteria.get("1.0", tk.END).strip()
        pass_criteria: list[dict] = []
        for line in criteria_raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ConfigError(f"通过条件格式错误（应为 'metric operator threshold'）: {line!r}")
            metric, operator, threshold_str = parts
            try:
                threshold: float | int = float(threshold_str)
                # 若是整数值则保存为 int
                if threshold == int(threshold):
                    threshold = int(threshold)
            except ValueError as exc:
                raise ConfigError(f"threshold 必须是数字: {threshold_str!r}") from exc
            pass_criteria.append({"metric": metric, "operator": operator, "threshold": threshold})

        # 格式列表
        formats: list[str] = []
        if self.v_fmt_json.get():
            formats.append("json")
        if self.v_fmt_csv.get():
            formats.append("csv")
        if self.v_fmt_html.get():
            formats.append("html")

        cfg: dict = {
            "target": {
                "name": self.v_name.get().strip(),
                "api_url": self.v_api_url.get().strip(),
                "api_key": self.v_api_key.get().strip(),
                "model": self.v_model.get().strip(),
            },
            "engine": self.v_engine.get(),
            "request": {
                "stream": self.v_stream.get(),
                "extra_args": extra_args,
            },
            "test": {
                "concurrency": _parse_int_list(self.v_concurrency.get()),
                "requests_per_level": _parse_int_list(self.v_rpl.get()),
                "dataset": self.v_dataset.get().strip(),
            },
            "pass_criteria": pass_criteria,
            "degradation": {
                "enabled": self.v_deg_enabled.get(),
                "start_concurrency": int(self.v_deg_start.get().strip()),
                "step": int(self.v_deg_step.get().strip()),
                "min_concurrency": int(self.v_deg_min.get().strip()),
            },
            "output": {
                "dir": self.v_out_dir.get().strip(),
                "formats": formats,
                "charts": self.v_charts.get(),
            },
        }
        return cfg

    # ------------------------------------------------------------------
    # 核心：dict → 表单
    # ------------------------------------------------------------------
    def _from_config(self, cfg: dict) -> None:
        target = cfg.get("target", {})
        self.v_name.set(target.get("name", ""))
        self.v_api_url.set(target.get("api_url", ""))
        self.v_api_key.set(target.get("api_key", ""))
        self.v_model.set(target.get("model", ""))

        self.v_engine.set(cfg.get("engine", "evalscope"))

        request = cfg.get("request", {})
        self.v_stream.set(bool(request.get("stream", True)))
        extra_args = request.get("extra_args", {})
        self.v_extra_args.set(json.dumps(extra_args, ensure_ascii=False) if extra_args else "{}")

        test = cfg.get("test", {})
        concurrency = test.get("concurrency", [])
        rpl = test.get("requests_per_level", [])
        self.v_concurrency.set(",".join(str(x) for x in concurrency))
        self.v_rpl.set(",".join(str(x) for x in rpl))
        self.v_dataset.set(test.get("dataset", ""))

        # 通过条件
        self.w_criteria.delete("1.0", tk.END)
        for criterion in cfg.get("pass_criteria", []):
            line = f"{criterion.get('metric','')} {criterion.get('operator','')} {criterion.get('threshold','')}"
            self.w_criteria.insert(tk.END, line + "\n")

        degradation = cfg.get("degradation", {})
        self.v_deg_enabled.set(bool(degradation.get("enabled", True)))
        self.v_deg_start.set(str(degradation.get("start_concurrency", 50)))
        self.v_deg_step.set(str(degradation.get("step", 10)))
        self.v_deg_min.set(str(degradation.get("min_concurrency", 10)))

        output = cfg.get("output", {})
        self.v_out_dir.set(output.get("dir", "./results"))
        formats = output.get("formats", [])
        self.v_fmt_json.set("json" in formats)
        self.v_fmt_csv.set("csv" in formats)
        self.v_fmt_html.set("html" in formats)
        self.v_charts.set(bool(output.get("charts", True)))

    # ------------------------------------------------------------------
    # 文件操作
    # ------------------------------------------------------------------
    def _on_open(self) -> None:
        path = filedialog.askopenfilename(
            title="打开配置文件",
            filetypes=[("YAML 文件", "*.yaml *.yml"), ("所有文件", "*.*")],
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                raise ConfigError("配置文件顶层必须是字典")
            self._from_config(cfg)
            self._current_path = Path(path)
            self.root.title(f"LLM 压力测试 — 配置编辑器  [{self._current_path.name}]")
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def _on_save(self) -> None:
        if self._current_path is None:
            self._on_save_as()
            return
        self._write_to(self._current_path)

    def _on_save_as(self) -> None:
        initial = str(self._current_path) if self._current_path else "config.yaml"
        path = filedialog.asksaveasfilename(
            title="另存为",
            initialfile=initial,
            defaultextension=".yaml",
            filetypes=[("YAML 文件", "*.yaml *.yml"), ("所有文件", "*.*")],
        )
        if path:
            self._write_to(Path(path))

    def _write_to(self, path: Path) -> None:
        # 收集表单值
        try:
            cfg = self._to_config()
        except (ConfigError, ValueError) as exc:
            messagebox.showerror("表单错误", str(exc))
            return

        # 验证
        try:
            validate_config(cfg)
        except ConfigError as exc:
            messagebox.showerror("配置校验失败", str(exc))
            return

        # 写文件
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
            self._current_path = path
            self.root.title(f"LLM 压力测试 — 配置编辑器  [{path.name}]")
            messagebox.showinfo("已保存", f"配置已保存到:\n{path}")
        except OSError as exc:
            messagebox.showerror("写入失败", str(exc))


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------
def main() -> None:
    # 解析 --config <path>
    config_path: str | None = None
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            break

    root = tk.Tk()
    app = ConfigEditorApp(root)

    if config_path:
        app._load_file(config_path)

    root.mainloop()


if __name__ == "__main__":
    main()
