# Trading Control Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the static dashboard into a local control panel: a served page whose buttons actually run the forward tick, paper runs, and tests, with a live log pane, a status-light strip, and a tabbed grid layout.

**Architecture:** A new stdlib-only HTTP server (`scripts/control_panel.py`) reuses `make_dashboard.render_all()` for the page body and injects a control bar + log pane + polling JS. One subprocess job at a time, output buffered in memory, polled via `GET /status`. The status strip and tab layout are added inside `make_dashboard.py` so the static file gets them too.

**Tech Stack:** Python stdlib only (`http.server`, `subprocess`, `threading`, `json`), vanilla JS/CSS injected as strings. No new dependencies.

## Global Constraints

- Stdlib only — no new entries in `requirements.txt` / `pyproject.toml`.
- Server binds `127.0.0.1` only, never `0.0.0.0`.
- Subprocesses always `shell=False`; `engine` validated against a whitelist, `symbols` against `[A-Za-z0-9,]+`.
- One job at a time (`ponytail:` global single-job lock; per-job queue only if ever needed).
- All commands run with `cwd=<repo root>` and `PYTHONPATH=src`, using `sys.executable` (launch the panel with `.venv/bin/python`).
- All displayed numbers stay at 2 decimals (existing `_pct`/`_money`/`_num` helpers — do not add new format strings with other precisions).
- Tests run with: `/Users/adijain/robinhood agentic trading/.venv/bin/python -m pytest` (adjust to `.venv/bin/python -m pytest` from repo root).
- Existing tests must keep passing (208 at time of writing).

## File Structure

- `scripts/control_panel.py` — **create.** Job runner class, HTTP handler, panel-chrome injection, `main()`. The only new runtime file.
- `scripts/make_dashboard.py` — **modify.** Add `_status_strip()` (status lights) and `_tabs()` (tab layout) used by `render_all()`; add CSS for both; add a runbook row for the panel.
- `tests/test_control_panel.py` — **create.** Job runner, endpoints, injection tests.
- `tests/test_dashboard.py` — **modify.** Assertions for status strip and tabs.

---

### Task 1: Job runner — one subprocess, output captured in memory

**Files:**
- Create: `scripts/control_panel.py`
- Test: `tests/test_control_panel.py`

**Interfaces:**
- Produces: `class Job` with `start(name: str, cmd: list[str]) -> bool`, `.running: bool` (property), `.lines: list[str]` (accumulated output, first line is the command, last line `[exit N]`), `.name: str`. Module globals `ROOT: Path` (repo root), `PY: str` (`sys.executable`), `JOB: Job` (singleton). Task 2 builds the HTTP layer on exactly these names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_control_panel.py`:

```python
import importlib.util
import sys
import time
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "control_panel.py"
    spec = importlib.util.spec_from_file_location("control_panel", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _wait_done(job, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not job.running and job.lines and job.lines[-1].startswith("[exit"):
            return
        time.sleep(0.05)
    raise AssertionError(f"job never finished: {job.lines}")


def test_job_captures_output_and_exit_code():
    job = _module().Job()
    assert job.start("demo", [sys.executable, "-c", "print('hi')"]) is True
    _wait_done(job)
    assert "hi" in job.lines
    assert job.lines[-1] == "[exit 0]"
    assert job.name == "demo"


def test_job_refuses_second_start_while_running():
    job = _module().Job()
    assert job.start("sleep", [sys.executable, "-c", "import time; time.sleep(5)"])
    assert job.start("again", [sys.executable, "-c", "print('no')"]) is False
    job.proc.kill()


def test_job_records_nonzero_exit():
    job = _module().Job()
    job.start("boom", [sys.executable, "-c", "raise SystemExit(3)"])
    _wait_done(job)
    assert job.lines[-1] == "[exit 3]"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_control_panel.py -v`
Expected: FAIL — `FileNotFoundError` (no `scripts/control_panel.py`).

- [ ] **Step 3: Write the implementation**

Create `scripts/control_panel.py`:

```python
"""Serve the trading dashboard as a live control panel.

    .venv/bin/python scripts/control_panel.py            # http://127.0.0.1:8321
    .venv/bin/python scripts/control_panel.py --port N

Buttons on the page run the forward tick, paper runs, and tests as local
subprocesses; output streams into a log pane. Localhost only, stdlib only.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


class Job:
    """One subprocess at a time; output accumulates in memory.

    ponytail: single global job + in-memory log; add a queue/persistence
    only if parallel runs are ever actually wanted.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.lines: list[str] = []
        self.proc: subprocess.Popen[str] | None = None
        self.name = ""

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, name: str, cmd: list[str]) -> bool:
        with self._lock:
            if self.running:
                return False
            self.name, self.lines = name, [f"$ {' '.join(cmd)}"]
            self.proc = subprocess.Popen(
                cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env={**os.environ, "PYTHONPATH": "src"},
            )
            threading.Thread(target=self._pump, daemon=True).start()
            return True

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.append(line.rstrip("\n"))
        self.lines.append(f"[exit {self.proc.wait()}]")


JOB = Job()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_control_panel.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/control_panel.py tests/test_control_panel.py
git commit -m "feat: control panel job runner — one subprocess, buffered output"
```

---

### Task 2: HTTP endpoints — /status and /action/<name>

**Files:**
- Modify: `scripts/control_panel.py`
- Test: `tests/test_control_panel.py`

**Interfaces:**
- Consumes: `Job`, `JOB`, `PY`, `ROOT` from Task 1.
- Produces: `class Handler(BaseHTTPRequestHandler)`; `_cmd(name: str, q: dict[str, list[str]]) -> list[str] | None` (None = bad params); `ENGINES: list[str]`. Endpoints: `GET /status?since=N` → `{"running": bool, "name": str, "lines": [str], "next": int}`; `POST /action/{tick|test|run}` → `{"started": true}` 200, 409 if busy, 400 bad params, 404 unknown. Task 3 adds `GET /` to `do_GET`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_control_panel.py`:

```python
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer


def _serve(mod):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mod.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _post(url):
    return urllib.request.urlopen(urllib.request.Request(url, method="POST"))


def test_status_endpoint_reports_idle():
    mod = _module()
    mod.JOB = mod.Job()
    srv, base = _serve(mod)
    try:
        with urllib.request.urlopen(base + "/status") as r:
            assert json.load(r) == {"running": False, "name": "", "lines": [], "next": 0}
    finally:
        srv.shutdown()


def test_action_runs_command_and_status_streams_output():
    mod = _module()
    mod.JOB = mod.Job()
    mod._cmd = lambda name, q: [sys.executable, "-c", "print('ran-' + '%s')" % name]
    srv, base = _serve(mod)
    try:
        with _post(base + "/action/tick") as r:
            assert json.load(r) == {"started": True}
        _wait_done(mod.JOB)
        with urllib.request.urlopen(base + "/status?since=0") as r:
            s = json.load(r)
        assert "ran-tick" in s["lines"] and s["running"] is False
        assert s["next"] == len(s["lines"])
    finally:
        srv.shutdown()


def test_action_conflicts_and_validation():
    mod = _module()
    mod.JOB = mod.Job()
    srv, base = _serve(mod)
    try:
        for url, code in [
            (base + "/action/run?engine=bogus&symbols=all", 400),
            (base + "/action/run?engine=momentum&symbols=NV;rm", 400),
            (base + "/action/nope", 404),
        ]:
            try:
                _post(url)
                raise AssertionError(f"expected {code} for {url}")
            except urllib.error.HTTPError as e:
                assert e.code == code
        mod.JOB.start("busy", [sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            _post(base + "/action/test")
            raise AssertionError("expected 409")
        except urllib.error.HTTPError as e:
            assert e.code == 409
        mod.JOB.proc.kill()
    finally:
        srv.shutdown()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_control_panel.py -v`
Expected: new tests FAIL with `AttributeError: module 'control_panel' has no attribute 'Handler'`; Task 1 tests still PASS.

- [ ] **Step 3: Write the implementation**

In `scripts/control_panel.py`, add to the imports:

```python
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
```

Append after `JOB = Job()`:

```python
ENGINES = ["mean_reversion", "momentum", "linreg", "pairs", "agent"]


def _cmd(name: str, q: dict[str, list[str]]) -> list[str] | None:
    """Build the subprocess argv for an action; None means bad params."""
    if name == "tick":
        return [PY, "-m", "rhagent.forward"]
    if name == "test":
        return [PY, "-m", "pytest", "-q"]
    # run
    engine = q.get("engine", ["mean_reversion"])[0]
    symbols = q.get("symbols", ["all"])[0]
    if engine not in ENGINES or not re.fullmatch(r"[A-Za-z0-9,]+", symbols):
        return None
    return [PY, "-m", "rhagent.papertrade", "--engine", engine, "--symbols", symbols]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # keep the terminal quiet
        pass

    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/status":
            since = int(parse_qs(url.query).get("since", ["0"])[0])
            self._json({"running": JOB.running, "name": JOB.name,
                        "lines": JOB.lines[since:], "next": len(JOB.lines)})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        name = url.path.removeprefix("/action/")
        if name not in ("tick", "test", "run"):
            return self._json({"error": "unknown action"}, 404)
        cmd = _cmd(name, parse_qs(url.query))
        if cmd is None:
            return self._json({"error": "bad params"}, 400)
        if not JOB.start(name, cmd):
            return self._json({"error": "a job is already running"}, 409)
        self._json({"started": True})
```

Note: `do_POST` must call the module-level `_cmd` (as written) so tests can monkeypatch it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_control_panel.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/control_panel.py tests/test_control_panel.py
git commit -m "feat: control panel HTTP endpoints — /status polling, /action/{tick,test,run}"
```

---

### Task 3: Panel page — dashboard body + control bar, log pane, polling JS

**Files:**
- Modify: `scripts/control_panel.py`
- Test: `tests/test_control_panel.py`

**Interfaces:**
- Consumes: `Handler.do_GET`, `Handler._json` from Task 2; `make_dashboard.render_all(base_dir: Path) -> str` (exists; returns full HTML containing `<h1>Trading Dashboard</h1>` and `</body>`).
- Produces: `BASE_DIR: Path` module global (default `ROOT / "journal" / "papertrade"`, tests override it); `panel_page() -> str`; `main(argv: list[str] | None = None) -> int`; `GET /` serves the panel page.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_control_panel.py` (the `_write_run` helper is intentionally duplicated from `tests/test_dashboard.py`):

```python
import pandas as pd


def _write_run(run_dir: Path, *, engine="mean_reversion", net=(0.01,)):
    run_dir.mkdir(parents=True)
    run_dir.joinpath("run.json").write_text(json.dumps({
        "run_id": run_dir.name, "engine": engine, "symbols": ["A"],
        "start": "2026-07-01", "end": "2026-07-02",
        "cost_bps": 1.0, "notional": 10_000.0,
    }))
    run_dir.joinpath("trades.jsonl").write_text("")
    idx = pd.date_range("2026-07-01", periods=len(net), freq="D")
    pd.DataFrame({"date": idx, "net": list(net)}).to_csv(
        run_dir / "returns.csv", index=False)


def test_panel_page_injects_controls_into_dashboard(tmp_path):
    mod = _module()
    paper = tmp_path / "journal" / "papertrade"
    _write_run(paper / "2026-07-12T00-00-00Z-aaaaaaaa")
    _write_run(tmp_path / "journal" / "forward" / "mean_reversion")
    mod.BASE_DIR = paper
    html = mod.panel_page()
    assert "Trading Dashboard" in html          # dashboard body is there
    assert 'data-action="tick"' in html         # control bar
    assert 'data-action="test"' in html
    assert 'id="runform"' in html               # engine/symbols form
    assert 'id="log"' in html                   # log pane
    assert "/status?since=" in html             # polling JS wired
    for engine in mod.ENGINES:
        assert f"<option>{engine}</option>" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_control_panel.py::test_panel_page_injects_controls_into_dashboard -v`
Expected: FAIL with `AttributeError: ... no attribute 'BASE_DIR'`

- [ ] **Step 3: Write the implementation**

In `scripts/control_panel.py`, add to the imports (top of file, after `PY = sys.executable`):

```python
import argparse
import webbrowser

sys.path.insert(0, str(ROOT / "scripts"))
import make_dashboard  # noqa: E402

BASE_DIR = ROOT / "journal" / "papertrade"
```

Append after the `Handler` class:

```python
_PANEL_CSS = """<style>
.ctrl{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:10px;align-items:center;
background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 14px;margin:14px 0}
.ctrl button,.ctrl select,.ctrl input{font:inherit;color:var(--fg);background:var(--panel2);
border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer}
.ctrl button:hover{border-color:var(--accent)}
.ctrl form{display:flex;gap:6px;align-items:center;margin:0}
#log{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px;
max-height:260px;overflow:auto;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
white-space:pre-wrap;margin:0 0 14px}
</style>"""

_PANEL_HTML = _PANEL_CSS + """
<div class="ctrl">
  <button data-action="tick">&#9654; forward tick</button>
  <form id="runform">
    <select name="engine"><option>mean_reversion</option><option>momentum</option>
    <option>linreg</option><option>pairs</option><option>agent</option></select>
    <input name="symbols" value="all" size="10">
    <button type="submit">&#9654; paper run</button>
  </form>
  <button data-action="test">&#9654; tests</button>
  <button type="button" onclick="location.reload()">&#8635; refresh page</button>
</div>
<pre id="log" hidden></pre>"""

_PANEL_JS = """<script>
let next = 0;
const log = document.getElementById('log');
async function poll(){
  const r = await (await fetch('/status?since=' + next)).json();
  if (r.lines.length){
    log.hidden = false;
    log.textContent += r.lines.join('\\n') + '\\n';
    log.scrollTop = log.scrollHeight;
  }
  next = r.next;
  if (r.running) setTimeout(poll, 1000);
}
async function act(name, params){
  const qs = params ? '?' + new URLSearchParams(params) : '';
  const r = await fetch('/action/' + name + qs, {method: 'POST'});
  if (r.ok) poll();
  else alert((await r.json()).error || 'request failed');
}
document.querySelectorAll('button[data-action]').forEach(b =>
  b.addEventListener('click', () => act(b.dataset.action)));
document.getElementById('runform').addEventListener('submit', e => {
  e.preventDefault();
  const f = e.target;
  act('run', {engine: f.engine.value, symbols: f.symbols.value});
});
</script>"""


def panel_page() -> str:
    html = make_dashboard.render_all(BASE_DIR)
    return (
        html.replace("<h1>Trading Dashboard</h1>",
                     "<h1>Trading Dashboard</h1>" + _PANEL_HTML)
        .replace("</body>", _PANEL_JS + "</body>")
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="control_panel")
    p.add_argument("--port", type=int, default=8321)
    p.add_argument("--base-dir", default=str(BASE_DIR))
    p.add_argument("--no-open", action="store_true", help="don't open a browser")
    args = p.parse_args(argv)
    global BASE_DIR
    BASE_DIR = Path(args.base_dir)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"control panel at {url}  (ctrl-c to stop)")
    if not args.no_open:
        webbrowser.open(url)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

And extend `Handler.do_GET` — replace the final `else` branch:

```python
        elif url.path == "/":
            body = panel_page().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json({"error": "not found"}, 404)
```

Finally, add the panel to the runbook in `scripts/make_dashboard.py` — insert as the FIRST entry of `_RUNBOOK`:

```python
    ("live control panel", ".venv/bin/python scripts/control_panel.py"),
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS (208 existing + 7 new).

- [ ] **Step 5: Smoke-test by hand**

Run: `.venv/bin/python scripts/control_panel.py --no-open --port 8321 &` then `curl -s http://127.0.0.1:8321/ | grep -c runform` (expect `1`) and `curl -s -X POST http://127.0.0.1:8321/action/tick` (expect `{"started": true}`); `kill %1` when done.

- [ ] **Step 6: Commit**

```bash
git add scripts/control_panel.py scripts/make_dashboard.py tests/test_control_panel.py
git commit -m "feat: control panel page — live buttons, log pane, polling JS"
```

---

### Task 4: Status strip — three system lights above the fold

**Files:**
- Modify: `scripts/make_dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: existing helpers in `make_dashboard.py`: `_days_old(value: str) -> int | None`, `_status_class(days_old) -> str` ("ok"/"warn"/"bad"), `_status_label(days_old) -> str`, `_run_dirs(base_dir) -> list[Path]`, `load_run(run_dir) -> (meta, trades, net)`.
- Produces: `_status_strip(base_dir: Path, forward_dir: Path) -> str` — a `<div class='strip'>` of three `.status` pills (price cache / forward tick / research run), rendered at the very top of `render_all()`. The price cache is read from `base_dir.parents[1] / "data"` (repo root `data/`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py`:

```python
def test_status_strip_shows_three_lights(tmp_path):
    mod = _dashboard_module()
    paper = tmp_path / "journal" / "papertrade"
    _write_run(paper / "2026-07-12T00-00-00Z-aaaaaaaa")
    _write_run(tmp_path / "journal" / "forward" / "mean_reversion")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "NVDA.csv").write_text("date,close\n2026-07-01,1.00\n")

    assert mod.main(["--base-dir", str(paper)]) == 0
    html = (tmp_path / "journal" / "dashboard.html").read_text()
    assert "class='strip'" in html
    assert "price cache" in html
    assert "forward tick" in html
    assert "research run" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py::test_status_strip_shows_three_lights -v`
Expected: FAIL on `assert "class='strip'" in html`

- [ ] **Step 3: Write the implementation**

In `scripts/make_dashboard.py`, add after `_status_label`:

```python
def _latest_end_days(base: Path) -> int | None:
    ends = [str(load_run(d)[0].get("end", "")) for d in _run_dirs(base)]
    return _days_old(max(ends)) if ends else None


def _status_strip(base_dir: Path, forward_dir: Path) -> str:
    cache_dir = base_dir.parents[1] / "data"
    cache_days = None
    csvs = list(cache_dir.glob("*.csv"))
    if csvs:
        newest = date.fromtimestamp(max(c.stat().st_mtime for c in csvs))
        cache_days = (date.today() - newest).days
    items = [
        ("price cache", cache_days),
        ("forward tick", _latest_end_days(forward_dir)),
        ("research run", _latest_end_days(base_dir)),
    ]
    return "<div class='strip'>" + "".join(
        f"<span class='status {_status_class(d)}'>{escape(label)}: {_status_label(d)}</span>"
        for label, d in items
    ) + "</div>"
```

In `render_all()`, prepend the strip — change:

```python
    index = (
        _overview_cards(base_dir, forward_dir, comparison)
```

to:

```python
    index = (
        _status_strip(base_dir, forward_dir)
        + _overview_cards(base_dir, forward_dir, comparison)
```

In `_CSS`, add one line (next to the `.status` rules):

```css
.strip{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 4px}
.strip .status{font-size:12px;padding:4px 12px}
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_dashboard.py tests/test_dashboard.py
git commit -m "feat: status strip — cache/forward/research freshness lights"
```

---

### Task 5: Tab layout — Overview / Runs / Bake-off instead of one long scroll

**Files:**
- Modify: `scripts/make_dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: the section-building expressions currently concatenated in `render_all()`.
- Produces: `_tabs(tabs: list[tuple[str, str]]) -> str` — nav buttons + panes + toggle JS. `render_all()` splits its body into three panes: **Overview** (status strip, overview cards, runbook, forward table, research pulse), **Runs** (comparison table + all run details — same pane so `#run-...` anchor links keep working), **Bake-off** (bake-off table + code graph). The single-run `render()` page is unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py`:

```python
def test_dashboard_is_tabbed(tmp_path):
    mod = _dashboard_module()
    paper = tmp_path / "journal" / "papertrade"
    _write_run(paper / "2026-07-12T00-00-00Z-aaaaaaaa")
    _write_run(tmp_path / "journal" / "forward" / "mean_reversion")

    assert mod.main(["--base-dir", str(paper)]) == 0
    html = (tmp_path / "journal" / "dashboard.html").read_text()
    assert html.count("class='tabbtn") == 3
    for pane in ["tab-overview", "tab-runs", "tab-bakeoff"]:
        assert f"id='{pane}'" in html
    runs_pane = html.split("id='tab-runs'")[1].split("id='tab-bakeoff'")[0]
    assert "rundetail" in runs_pane          # details live with the run table
    assert "href='#run-" in runs_pane        # anchors stay within the pane
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py::test_dashboard_is_tabbed -v`
Expected: FAIL on `html.count("class='tabbtn") == 3`

- [ ] **Step 3: Write the implementation**

In `scripts/make_dashboard.py`, add before `_page`:

```python
def _tabs(tabs: list[tuple[str, str, str]]) -> str:
    """tabs = [(slug, label, body_html)]; first tab starts active."""
    nav = "".join(
        f"<button class='tabbtn{' active' if i == 0 else ''}' "
        f"data-tab='tab-{slug}'>{escape(label)}</button>"
        for i, (slug, label, _) in enumerate(tabs)
    )
    panes = "".join(
        f"<div class='pane{' active' if i == 0 else ''}' id='tab-{slug}'>{body}</div>"
        for i, (slug, _, body) in enumerate(tabs)
    )
    js = (
        "<script>"
        "document.querySelectorAll('.tabbtn').forEach(b=>b.onclick=()=>{"
        "document.querySelectorAll('.tabbtn,.pane').forEach(e=>e.classList.remove('active'));"
        "b.classList.add('active');"
        "document.getElementById(b.dataset.tab).classList.add('active');});"
        "if(location.hash.startsWith('#run-')){"
        "document.querySelectorAll('.tabbtn')[1].click();"
        "document.getElementById(location.hash.slice(1))?.scrollIntoView();}"
        "</script>"
    )
    return f"<nav class='tabs'>{nav}</nav>{panes}{js}"
```

Rewrite `render_all()`'s body assembly — replace everything from `index = (` through the `return _page(...)` with:

```python
    overview = (
        _status_strip(base_dir, forward_dir)
        + _overview_cards(base_dir, forward_dir, comparison)
        + "<h2>Runbook · every command from here</h2>"
        f"<div class='tblscroll'>{_runbook()}</div>"
        "<h2>Now · forward track record</h2>"
        f"<div class='tblscroll'>{_forward_table(forward_dir)}</div>"
        "<h2>Research pulse</h2>"
        + _latest_summary(latest, "latest paper-trade run")
        + (_latest_summary(best_dir, "best paper-trade run") if best_dir != latest else "")
    )
    runs_pane = (
        f"<h2>All paper-trade runs · {len(runs)} total</h2>"
        f"<div class='tblscroll'>{_compare_table(comparison, '', link=True)}</div>"
        "<h2>Run details</h2>"
        + _run_details(runs, latest)
    )
    bakeoff = (
        "<h2>Bake-off · robust Sharpe (fold + bootstrap + deflated)</h2>"
        "<p class='sub'>A variant beats baseline only if its 95% CI lower bound "
        "clears the baseline Sharpe.</p>"
        f"<div class='tblscroll'>{_bakeoff_table(base_dir)}</div>"
        "<h2>Code graph</h2>"
        f"{_code_health(graph_dir)}"
    )
    body = _tabs([
        ("overview", "Overview", overview),
        ("runs", "Runs", runs_pane),
        ("bakeoff", "Bake-off", bakeoff),
    ])
    return _page(
        f"Trading dashboard — {len(runs)} research runs", body,
        f"Generated from {escape(str(base_dir.parent))} · rhagent trading harness",
    )
```

In `_CSS`, add:

```css
.tabs{display:flex;gap:6px;margin:18px 0 0;border-bottom:1px solid var(--line)}
.tabbtn{font:inherit;background:none;border:none;border-bottom:2px solid transparent;
color:var(--muted);padding:8px 14px;cursor:pointer}
.tabbtn:hover{color:var(--fg)}
.tabbtn.active{color:var(--fg);border-bottom-color:var(--accent)}
.pane{display:none;padding-top:8px}
.pane.active{display:block}
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS. If `test_default_dashboard_writes_only_one_dashboard_html` breaks on a string assertion (section headings now live inside panes — content is unchanged, only nesting), fix the assertion, not the feature.

- [ ] **Step 5: Regenerate and eyeball**

Run: `.venv/bin/python scripts/make_dashboard.py --open`
Expected: three tabs; clicking a run id in Runs jumps to its expanded detail; Overview fits roughly one screen.

- [ ] **Step 6: Commit**

```bash
git add scripts/make_dashboard.py tests/test_dashboard.py
git commit -m "feat: tabbed layout — Overview / Runs / Bake-off"
```

---

## Self-Review (done at plan time)

- **Coverage:** working buttons + log pane (Tasks 1–3), status strip (Task 4), grid/tab layout (Task 5) — all three control-panel pillars from the spec have tasks. Auto-reload after jobs was deliberately dropped (would wipe the log the user is reading); a manual refresh button replaces it.
- **Placeholders:** none — every step has full code or an exact command.
- **Type consistency:** `Job.start(name, cmd) -> bool`, `_cmd(name, q) -> list[str] | None`, `panel_page() -> str`, `_status_strip(base_dir, forward_dir) -> str`, `_tabs(list[tuple[str, str, str]]) -> str` are used identically everywhere they appear.
