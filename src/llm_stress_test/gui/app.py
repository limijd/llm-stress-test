"""本地 Web UI 配置编辑器 — 基于 Python 标准库 http.server，零外部 GUI 依赖"""
from __future__ import annotations

import json
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs

from .. import _yaml as yaml
from ..config import validate_config, ConfigError

_HOST = "127.0.0.1"
_DEFAULT_PORT = 9753

# ======================================================================
# HTML 页面（内嵌，不依赖任何外部文件）
# ======================================================================
_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>LLM 压力测试 — 配置编辑器</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #f5f7fa; color: #2d3748; line-height: 1.5; }
.container { max-width: 720px; margin: 24px auto; padding: 0 16px; }
h1 { text-align: center; padding: 16px 0 8px; color: #2c5282; font-size: 22px; }
.subtitle { text-align: center; color: #718096; font-size: 13px; margin-bottom: 20px; }
.card { background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.1);
        padding: 20px; margin-bottom: 16px; }
.card h2 { font-size: 15px; color: #2c5282; border-bottom: 1px solid #e2e8f0;
            padding-bottom: 6px; margin-bottom: 12px; }
.field { display: flex; align-items: center; margin-bottom: 8px; }
.field label { width: 180px; font-size: 13px; color: #4a5568; flex-shrink: 0; }
.field input[type=text], .field select { flex: 1; padding: 6px 10px; border: 1px solid #cbd5e0;
  border-radius: 4px; font-size: 13px; font-family: inherit; }
.field input:focus, .field select:focus, textarea:focus {
  outline: none; border-color: #4A90D9; box-shadow: 0 0 0 2px rgba(74,144,217,.2); }
textarea { width: 100%; padding: 8px 10px; border: 1px solid #cbd5e0; border-radius: 4px;
           font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 12px; resize: vertical; }
.hint { font-size: 11px; color: #a0aec0; margin: 2px 0 8px 180px; }
.checkbox-row { display: flex; align-items: center; margin-bottom: 8px; }
.checkbox-row label:first-child { width: 180px; font-size: 13px; color: #4a5568; }
.checkbox-row label { font-size: 13px; margin-right: 14px; cursor: pointer; }
.checkbox-row input[type=checkbox] { margin-right: 4px; }
.actions { display: flex; gap: 10px; justify-content: center; padding: 16px 0; }
.btn { padding: 8px 24px; border: none; border-radius: 6px; font-size: 14px;
       cursor: pointer; transition: background .15s; }
.btn-primary { background: #4A90D9; color: #fff; }
.btn-primary:hover { background: #3a7bc8; }
.btn-secondary { background: #e2e8f0; color: #2d3748; }
.btn-secondary:hover { background: #cbd5e0; }
.toast { position: fixed; top: 16px; right: 16px; padding: 12px 20px; border-radius: 6px;
         color: #fff; font-size: 13px; z-index: 999; opacity: 0; transition: opacity .3s; }
.toast.show { opacity: 1; }
.toast.success { background: #48bb78; }
.toast.error { background: #f56565; }
</style>
</head>
<body>
<div class="container">
  <h1>LLM 压力测试 — 配置编辑器</h1>
  <p class="subtitle">编辑完成后保存为 YAML，用 <code>llm-stress-test run --config &lt;file&gt;</code> 执行测试</p>

  <div class="card">
    <h2>测试目标</h2>
    <div class="field"><label>name</label><input type="text" id="name" value="DeepSeek-V3.2-Exp"></div>
    <div class="field"><label>api_url</label><input type="text" id="api_url" value="https://llmapi.paratera.com/v1/chat/completions"></div>
    <div class="field"><label>api_key</label><input type="text" id="api_key" value="${LLM_API_KEY}"></div>
    <div class="hint">支持 ${ENV_VAR} 环境变量引用，避免明文写入</div>
    <div class="field"><label>model</label><input type="text" id="model" value="DeepSeek-V3.2-Exp"></div>
  </div>

  <div class="card">
    <h2>引擎与测试参数</h2>
    <div class="field"><label>engine</label>
      <select id="engine"><option value="evalscope">evalscope</option><option value="native">native</option></select>
    </div>
    <div class="field"><label>concurrency</label><input type="text" id="concurrency" value="1,5,10,20,50"></div>
    <div class="field"><label>requests_per_level</label><input type="text" id="rpl" value="10,50,100,200,500"></div>
    <div class="hint">与 concurrency 按索引一一对应，长度必须一致</div>
    <div class="field"><label>dataset</label><input type="text" id="dataset" value="longalpaca"></div>
    <div class="hint">内置: openqa | longalpaca，或自定义 JSONL 文件路径</div>
    <div class="checkbox-row"><label>stream</label><label><input type="checkbox" id="stream" checked> 启用流式响应</label></div>
    <div class="field"><label>extra_args (JSON)</label><input type="text" id="extra_args" value='{"chat_template_kwargs":{"thinking":true}}'></div>
  </div>

  <div class="card">
    <h2>通过条件</h2>
    <textarea id="criteria" rows="4">success_rate >= 1.0
gen_toks_per_sec >= 500
avg_ttft <= 10.0</textarea>
    <div class="hint" style="margin-left:0">每行一条，格式: metric operator threshold</div>
  </div>

  <div class="card">
    <h2>降级策略</h2>
    <div class="checkbox-row"><label>enabled</label><label><input type="checkbox" id="deg_enabled" checked> 启用自动降级</label></div>
    <div class="field"><label>step</label><input type="text" id="deg_step" value="10"></div>
    <div class="field"><label>min_concurrency</label><input type="text" id="deg_min" value="10"></div>
  </div>

  <div class="card">
    <h2>输出设置</h2>
    <div class="field"><label>dir</label><input type="text" id="out_dir" value="./results"></div>
    <div class="checkbox-row"><label>formats</label>
      <label><input type="checkbox" id="fmt_json" checked> JSON</label>
      <label><input type="checkbox" id="fmt_csv" checked> CSV</label>
      <label><input type="checkbox" id="fmt_html" checked> HTML</label>
    </div>
    <div class="checkbox-row"><label>charts</label><label><input type="checkbox" id="charts" checked> 生成图表</label></div>
  </div>

  <div class="actions">
    <label class="btn btn-secondary" style="position:relative;overflow:hidden">
      打开配置
      <input type="file" id="file_open" accept=".yaml,.yml" style="position:absolute;inset:0;opacity:0;cursor:pointer">
    </label>
    <button class="btn btn-primary" onclick="onSave()">保存配置</button>
    <button class="btn btn-secondary" onclick="onValidate()">校验</button>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
function toast(msg, type) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast show ' + type;
  setTimeout(() => el.className = 'toast', 3000);
}

function collectConfig() {
  function intList(s) { return s.split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x)); }
  function parseCriteria(text) {
    return text.trim().split('\n').filter(l => l.trim()).map(l => {
      const p = l.trim().split(/\s+/);
      return {metric: p[0], operator: p[1], threshold: parseFloat(p[2])};
    });
  }
  let extra = {};
  try { extra = JSON.parse(document.getElementById('extra_args').value || '{}'); } catch(e) {}
  const fmts = [];
  if (document.getElementById('fmt_json').checked) fmts.push('json');
  if (document.getElementById('fmt_csv').checked) fmts.push('csv');
  if (document.getElementById('fmt_html').checked) fmts.push('html');
  const conc = intList(document.getElementById('concurrency').value);
  return {
    target: { name: document.getElementById('name').value,
              api_url: document.getElementById('api_url').value,
              api_key: document.getElementById('api_key').value,
              model: document.getElementById('model').value },
    engine: document.getElementById('engine').value,
    request: { stream: document.getElementById('stream').checked, extra_args: extra },
    test: { concurrency: conc,
            requests_per_level: intList(document.getElementById('rpl').value),
            dataset: document.getElementById('dataset').value },
    pass_criteria: parseCriteria(document.getElementById('criteria').value),
    degradation: { enabled: document.getElementById('deg_enabled').checked,
                   start_concurrency: conc.length ? conc[conc.length-1] : 50,
                   step: parseInt(document.getElementById('deg_step').value) || 10,
                   min_concurrency: parseInt(document.getElementById('deg_min').value) || 10 },
    output: { dir: document.getElementById('out_dir').value, formats: fmts,
              charts: document.getElementById('charts').checked }
  };
}

function fillForm(cfg) {
  const t = cfg.target || {};
  document.getElementById('name').value = t.name || '';
  document.getElementById('api_url').value = t.api_url || '';
  document.getElementById('api_key').value = t.api_key || '';
  document.getElementById('model').value = t.model || '';
  document.getElementById('engine').value = cfg.engine || 'evalscope';
  const test = cfg.test || {};
  document.getElementById('concurrency').value = (test.concurrency||[]).join(',');
  document.getElementById('rpl').value = (test.requests_per_level||[]).join(',');
  document.getElementById('dataset').value = test.dataset || '';
  const req = cfg.request || {};
  document.getElementById('stream').checked = req.stream !== false;
  document.getElementById('extra_args').value = JSON.stringify(req.extra_args||{});
  const criteria = (cfg.pass_criteria||[]).map(c => c.metric+' '+c.operator+' '+c.threshold).join('\n');
  document.getElementById('criteria').value = criteria;
  const deg = cfg.degradation || {};
  document.getElementById('deg_enabled').checked = deg.enabled !== false;
  document.getElementById('deg_step').value = deg.step || 10;
  document.getElementById('deg_min').value = deg.min_concurrency || 10;
  const out = cfg.output || {};
  document.getElementById('out_dir').value = out.dir || './results';
  const fmts = out.formats || [];
  document.getElementById('fmt_json').checked = fmts.includes('json');
  document.getElementById('fmt_csv').checked = fmts.includes('csv');
  document.getElementById('fmt_html').checked = fmts.includes('html');
  document.getElementById('charts').checked = out.charts !== false;
}

// 打开本地文件
document.getElementById('file_open').addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(ev) {
    // 发给服务端解析 YAML（浏览器原生不能解析 YAML）
    fetch('/api/parse-yaml', {method:'POST', body: ev.target.result,
          headers:{'Content-Type':'text/plain'}})
      .then(r => r.json()).then(data => {
        if (data.error) { toast(data.error, 'error'); return; }
        fillForm(data.config);
        toast('已加载: ' + file.name, 'success');
      });
  };
  reader.readAsText(file);
  e.target.value = '';
});

function onValidate() {
  const cfg = collectConfig();
  fetch('/api/validate', {method:'POST', body: JSON.stringify(cfg),
        headers:{'Content-Type':'application/json'}})
    .then(r => r.json()).then(data => {
      if (data.valid) toast('配置有效 ✓', 'success');
      else toast('校验失败: ' + data.error, 'error');
    });
}

function onSave() {
  const cfg = collectConfig();
  fetch('/api/validate', {method:'POST', body: JSON.stringify(cfg),
        headers:{'Content-Type':'application/json'}})
    .then(r => r.json()).then(data => {
      if (!data.valid) { toast('校验失败: ' + data.error, 'error'); return; }
      // 校验通过，触发下载 YAML
      fetch('/api/to-yaml', {method:'POST', body: JSON.stringify(cfg),
            headers:{'Content-Type':'application/json'}})
        .then(r => r.blob()).then(blob => {
          const a = document.createElement('a');
          a.href = URL.createObjectURL(blob);
          a.download = (document.getElementById('name').value || 'config').replace(/[^a-zA-Z0-9_-]/g,'_') + '.yaml';
          a.click();
          toast('配置已下载', 'success');
        });
    });
}
</script>
</body>
</html>"""


# ======================================================================
# HTTP Handler
# ======================================================================
class _Handler(BaseHTTPRequestHandler):
    """处理 GUI 的 HTTP 请求"""

    def log_message(self, format, *args):
        # 静默普通请求日志
        pass

    def do_GET(self):
        # 所有 GET 都返回主页面
        self._respond(200, "text/html", _HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if self.path == "/api/parse-yaml":
            self._handle_parse_yaml(body)
        elif self.path == "/api/validate":
            self._handle_validate(body)
        elif self.path == "/api/to-yaml":
            self._handle_to_yaml(body)
        else:
            self._respond(404, "application/json", json.dumps({"error": "not found"}).encode())

    def _handle_parse_yaml(self, body: bytes):
        """解析 YAML 文本，返回 JSON"""
        try:
            cfg = yaml.safe_load(body.decode("utf-8"))
            if not isinstance(cfg, dict):
                raise ValueError("顶层必须是字典")
            self._respond_json({"config": cfg})
        except Exception as e:
            self._respond_json({"error": str(e)})

    def _handle_validate(self, body: bytes):
        """校验配置"""
        try:
            cfg = json.loads(body)
            validate_config(cfg)
            self._respond_json({"valid": True})
        except ConfigError as e:
            self._respond_json({"valid": False, "error": str(e)})
        except Exception as e:
            self._respond_json({"valid": False, "error": str(e)})

    def _handle_to_yaml(self, body: bytes):
        """将 JSON 配置转为 YAML 并返回"""
        try:
            cfg = json.loads(body)
            yaml_text = yaml.dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False)
            self._respond(200, "application/x-yaml", yaml_text.encode("utf-8"))
        except Exception as e:
            self._respond_json({"error": str(e)})

    def _respond(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_json(self, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._respond(200, "application/json", body)


# ======================================================================
# 入口
# ======================================================================
def main() -> None:
    # 解析参数
    config_path: str | None = None
    port = _DEFAULT_PORT
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        else:
            i += 1

    # 如果指定了 --config，把初始配置注入到 HTML 中
    global _HTML_PAGE
    if config_path:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                inject_script = f"<script>window.addEventListener('load',()=>fillForm({json.dumps(cfg, ensure_ascii=False)}));</script>"
                _HTML_PAGE = _HTML_PAGE.replace("</body>", inject_script + "</body>")
                print(f"已加载配置: {config_path}")
        except Exception as e:
            print(f"警告: 无法加载配置 {config_path}: {e}")

    url = f"http://{_HOST}:{port}"
    server = HTTPServer((_HOST, port), _Handler)
    print(f"配置编辑器已启动: {url}")
    print("按 Ctrl+C 停止")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.server_close()


if __name__ == "__main__":
    main()
