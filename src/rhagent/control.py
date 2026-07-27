"""Localhost-only control panel: see forward records, run an offline backtest,
trigger a paper tick, flip the kill switch, edit the safe config knobs.

Server-rendered HTML, plain <form> POSTs, no JS, no framework. Binds
127.0.0.1 ONLY -- hardcoded, not read from any argument or env var, so this
can never be exposed off the local machine.

Cannot place a real order: it never imports or references the live broker
class, never reads the live-trading env var, and the only tick it can
trigger (rhagent.paper_run) always constructs the mock broker itself (see
paper_run.py).

Usage:
    PYTHONPATH=src .venv/bin/python -m rhagent.control
"""

from __future__ import annotations

import http.server
import json
import re
import sys
import threading
from html import escape
from pathlib import Path
from urllib.parse import parse_qs

import pandas as pd

from . import backtest
from .config import load as load_config
from .strategies import build as build_strategy

HOST = "127.0.0.1"  # loopback only -- never made configurable, see module docstring
PORT = 8765

_BUSY_LOCK = threading.Lock()  # ponytail: one global lock, not a job queue --
# fine for a single local operator clicking one button at a time; upgrade to
# per-action locks if that stops being true.


def _tail_jsonl(path: Path, n: int) -> list[dict]:
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines[-n:]]


def _forward_records(forward_dir: Path) -> list[dict]:
    from .evaluate import aggregate, load_run

    out = []
    for run_json in sorted(forward_dir.glob("*/run.json")):
        eval_dir = run_json.parent
        meta, trades, net = load_run(eval_dir)
        a = aggregate(trades, net)
        pos_files = sorted(eval_dir.glob("pos_*.csv"))
        positions = {}
        for pos_csv in pos_files:
            sym = pos_csv.stem.removeprefix("pos_")
            df = pd.read_csv(pos_csv)
            if len(df) and float(df.iloc[-1]["pos"]) != 0.0:
                positions[sym] = float(df.iloc[-1]["pos"])
        out.append({"name": eval_dir.name, "meta": meta, "metrics": a, "positions": positions,
                   "tracks_positions": bool(pos_files)})
    return out


def _run_backtest(strategy: str, symbols: list[str], start: str, end: str,
                  cost_bps: float, cache_dir: str = "data") -> dict:
    """Cache-only: never fetches over the network. Missing symbols are
    reported, not silently fetched -- reuses backtest.py/strategies as-is."""
    cache_dir = Path(cache_dir)
    strat = build_strategy(strategy, {})
    legs, missing = {}, []
    for sym in symbols:
        path = cache_dir / f"{sym}.csv"
        if not path.exists():
            missing.append(sym)
            continue
        bars = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        bars = bars.loc[(bars.index >= start) & (bars.index <= end)]
        if len(bars) < 2:
            missing.append(sym)
            continue
        pos = strat.positions(bars)
        legs[sym] = backtest.net_returns(bars, pos, cost_bps)
    if not legs:
        return {"ok": False, "missing": missing, "error": "no cached symbol had data in range"}
    df = pd.concat(legs, axis=1)
    combined = df.mean(axis=1)
    res = backtest.result_from_returns(combined)
    return {
        "ok": True, "missing": missing, "n_symbols": len(legs),
        "total_return": res.total_return, "sharpe": res.sharpe,
        "max_drawdown": res.max_drawdown, "hit_rate": res.hit_rate, "n_days": res.n_days,
    }


def _fmt(v, dp=4):
    return "n/a" if v is None else f"{v:.{dp}f}"


_CONFIG_KNOBS = [
    ("cost_bps", r"(cost_bps:\s*)([0-9.]+)"),
    ("per_trade_max_usd", r"(per_trade_max_usd:\s*)([0-9.]+)"),
    ("allow_short", r"(allow_short:\s*)(true|false)"),
]


def _patch_config_yaml(config_path: Path, updates: dict[str, str]) -> None:
    """Line-level regex substitution -- NOT a yaml load/dump round-trip, which
    would silently strip every explanatory comment in config.yaml. Each knob
    name appears exactly once in the file today, so a single count=1
    substitution per key is unambiguous."""
    text = config_path.read_text()
    for name, pattern in _CONFIG_KNOBS:
        if name not in updates:
            continue
        text, n = re.subn(pattern, lambda m: m.group(1) + updates[name], text, count=1)
        if n == 0:
            raise ValueError(f"config knob {name!r} not found in {config_path}")
    config_path.write_text(text)


class ControlApp:
    """Holds the (overridable) filesystem paths so tests can point every
    action at a tmp_path instead of the real journal/data/config.yaml."""

    def __init__(self, *, journal_dir: Path = Path("journal"), data_dir: Path = Path("data"),
                config_path: Path = Path("config.yaml"), halt_file: Path = Path("HALT")) -> None:
        self.journal_dir = Path(journal_dir)
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        self.halt_file = Path(halt_file)
        self.last_action: str | None = None

    # -- read-only state for the GET page --------------------------------
    def page(self) -> str:
        cfg = load_config()
        forward = _forward_records(self.journal_dir / "forward")
        decisions = _tail_jsonl(self.journal_dir / "forward" / "agent" / "decisions.jsonl", 20)
        halted = self.halt_file.exists()

        rows = []
        for r in forward:
            m, meta = r["metrics"], r["meta"]
            if not r["tracks_positions"]:
                pos = "not tracked per-symbol (non-agent engine)"
            else:
                pos = ", ".join(f"{s}={q:g}" for s, q in r["positions"].items()) or "flat"
            rows.append(
                f"<tr><td>{escape(r['name'])}</td><td>{escape(meta.get('engine',''))}</td>"
                f"<td>{escape(str(meta.get('start',''))[:10])} → {escape(str(meta.get('end',''))[:10])}</td>"
                f"<td>{m['n_trades']} trades / {m['n_return_days']} days</td>"
                f"<td>{_fmt(m['total_return'], 4)}</td><td>{_fmt(m['sharpe'], 2)}</td>"
                f"<td>{_fmt(m['max_drawdown'], 4)}</td><td>{escape(pos)}</td></tr>"
            )
        dec_rows = "".join(
            f"<li><code>{escape(d.get('date',''))} {escape(d.get('symbol',''))} "
            f"target={d.get('target')} status={escape(d.get('status','ok'))}</code></li>"
            for d in reversed(decisions)
        ) or "<li>no decisions logged</li>"

        universe = ", ".join(cfg.strategy.universe) if cfg.strategy else ""
        return f"""<!doctype html><html><head><meta charset="utf-8">
<title>rhagent control panel</title>
<style>body{{font-family:monospace;max-width:1000px;margin:20px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left;font-size:13px}}
fieldset{{margin:16px 0}} .warn{{color:#b00}} .ok{{color:#080}}</style></head><body>
<h2>rhagent control panel <small>(loopback only, no auth)</small></h2>
{f'<p class="warn"><b>HALT is SET</b> -- paper_run will do nothing until cleared.</p>' if halted else '<p class="ok">HALT clear -- armed.</p>'}
{f'<p>last action: {escape(self.last_action)}</p>' if self.last_action else ''}

<h3>Forward records</h3>
<table><tr><th>record</th><th>engine</th><th>window</th><th>sample</th>
<th>total_return</th><th>sharpe</th><th>max_dd</th><th>positions</th></tr>
{''.join(rows) or '<tr><td colspan=8>none yet</td></tr>'}</table>

<h3>Agent's recent decisions (journal/forward/agent/decisions.jsonl)</h3>
<ul>{dec_rows}</ul>

<fieldset><legend>Kill switch</legend>
<form method="post" action="/halt"><input type="hidden" name="action" value="{'clear' if halted else 'set'}">
<button type="submit">{'Clear HALT' if halted else 'Set HALT'}</button></form></fieldset>

<fieldset><legend>Run a backtest (cached data/*.csv only, no network)</legend>
<form method="post" action="/backtest">
strategy <input name="strategy" value="mean_reversion" size=14>
symbols (comma-separated) <input name="symbols" value="{escape(universe[:60])}" size=40><br>
start <input type="date" name="start" value="2026-01-01">
end <input type="date" name="end" value="2026-07-01">
cost_bps <input type="number" step="0.1" name="cost_bps" value="7.0" size=6>
<button type="submit">Run backtest</button></form></fieldset>

<fieldset><legend>Trigger a paper tick (rhagent.paper_run, always MockBroker)</legend>
<form method="post" action="/paper-tick"><button type="submit">Run paper tick now</button></form></fieldset>

<fieldset><legend>Edit safe config knobs (config.yaml, next run picks these up)</legend>
<form method="post" action="/config">
cost_bps <input name="cost_bps" value="{cfg.strategy.cost_bps if cfg.strategy else ''}" size=6>
per_trade_max_usd <input name="per_trade_max_usd" value="{cfg.limits.per_trade_max_usd}" size=8>
allow_short <select name="allow_short"><option {'selected' if cfg.agent.allow_short else ''} value="true">true</option>
<option {'selected' if not cfg.agent.allow_short else ''} value="false">false</option></select>
<button type="submit">Save</button></form></fieldset>
</body></html>"""

    # -- state-changing actions (POST only) -------------------------------
    def do_halt(self, action: str) -> str:
        if action == "set":
            self.halt_file.touch()
            return "HALT file created"
        self.halt_file.unlink(missing_ok=True)
        return "HALT file removed"

    def do_backtest(self, params: dict) -> str:
        symbols = [s.strip().upper() for s in params.get("symbols", [""])[0].split(",") if s.strip()]
        result = _run_backtest(
            params.get("strategy", ["mean_reversion"])[0], symbols,
            params.get("start", [""])[0], params.get("end", [""])[0],
            float(params.get("cost_bps", ["7.0"])[0]), self.data_dir,
        )
        return f"backtest {result}"

    def do_paper_tick(self) -> str:
        with _BUSY_LOCK:
            from . import paper_run
            result = paper_run.run(halt_file=self.halt_file)
        return f"paper tick: {result}"

    def do_config(self, params: dict) -> str:
        updates = {k: v[0] for k, v in params.items() if v}
        _patch_config_yaml(self.config_path, updates)
        return f"config.yaml updated: {updates}"


class ControlHandler(http.server.BaseHTTPRequestHandler):
    app: ControlApp = ControlApp()

    def log_message(self, fmt, *args):  # quieter default logging
        pass

    def _send(self, status: int, body: str, location: str | None = None) -> None:
        self.send_response(status)
        if location:
            self.send_header("Location", location)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        body_bytes = body.encode("utf-8")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        if body_bytes:
            self.wfile.write(body_bytes)

    _POST_ONLY_PATHS = {"/halt", "/backtest", "/paper-tick", "/config"}

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, self.app.page())
            return
        if path in self._POST_ONLY_PATHS:
            self._send(405, "GET not allowed on a state-changing route; use POST")
            return
        self._send(404, "not found")

    # Every state-changing route only exists under do_POST -- a GET to any
    # of these paths falls through to the 405 branch below, never a 3xx/200.
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else ""
        params = parse_qs(body)
        path = self.path.split("?")[0]
        try:
            if path == "/halt":
                self.app.last_action = self.app.do_halt(params.get("action", ["set"])[0])
            elif path == "/backtest":
                self.app.last_action = self.app.do_backtest(params)
            elif path == "/paper-tick":
                self.app.last_action = self.app.do_paper_tick()
            elif path == "/config":
                self.app.last_action = self.app.do_config(params)
            else:
                self._send(404, "not found")
                return
        except Exception as e:  # surface the error instead of a bare 500
            self.app.last_action = f"ERROR: {e}"
        # PRG pattern: redirect back to GET / so a refresh never re-submits.
        self._send(303, "", location="/")


def main() -> int:
    try:
        server = http.server.ThreadingHTTPServer((HOST, PORT), ControlHandler)
    except OSError as e:
        # Overwhelmingly this is "address already in use" from a panel left
        # running in another terminal. The raw socket traceback buries that.
        print(f"[control] cannot bind {HOST}:{PORT} -- {e}\n"
              f"  in use? find it:  lsof -nP -iTCP:{PORT} -sTCP:LISTEN\n"
              f"  then:             kill <PID>", file=sys.stderr)
        return 1
    print(f"[control] serving on http://{HOST}:{PORT} (loopback only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
