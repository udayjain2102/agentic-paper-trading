"""Render the trading system into a single self-contained HTML dashboard.

Forward records, the locked-config scorecard, every research run, the robust
bake-off, the equity curve, and the ledger / decision-quality buckets — one
static page with a small vanilla-JS layer (sort, filter, per-run drawer).

    python scripts/make_dashboard.py                 # writes journal/dashboard.html
    python scripts/make_dashboard.py --open          # also open in a browser

Reads the ledgers under journal/papertrade/ and journal/forward/; reuses
rhagent.evaluate so the numbers match the CLI report exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime, timezone
from html import escape
from pathlib import Path

# src-layout: make the rhagent package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rhagent.config import load as load_config  # noqa: E402
from rhagent.evaluate import _bucket_labels, aggregate, load_run, spy_benchmark  # noqa: E402
from rhagent.evaluate_robust import robust_table  # noqa: E402
from rhagent.features import flatten_trades  # noqa: E402
from rhagent.memory import read_memory  # noqa: E402

HALT_FILE = Path("HALT")

# GitHub renders this badge live, so the static page shows CI state with no JS.
_ACTIONS_URL = ("https://github.com/udayjain2102/papertrading-model-"
                "/actions/workflows/daily-paper-run.yml")

_RUNBOOK = [
    ("daily forward tick", "PYTHONPATH=src .venv/bin/python -m rhagent.forward"),
    ("new research run", "PYTHONPATH=src .venv/bin/python -m rhagent.papertrade --engine mean_reversion --symbols all"),
    ("unattended daily loop", "scripts/paper_cron.sh"),
    ("rebuild this page", ".venv/bin/python scripts/make_dashboard.py --open"),
    ("run the tests", ".venv/bin/python -m pytest"),
]


def _run_dirs(base_dir: Path) -> list[Path]:
    return sorted(p.parent for p in base_dir.glob("*/run.json"))


def _run_row(run_dir: Path) -> dict:
    """Every per-run number the page shows, for one archived run."""
    meta, trades, net = load_run(run_dir)
    a = aggregate(trades, net)
    pnl_s = trades["pnl_abs"].astype(float) if len(trades) else pd.Series(dtype=float)
    symbols = meta.get("symbols", [])
    rid = str(meta["run_id"])
    notional = float(meta.get("notional", 10_000.0))
    return {
        "id": rid, "sid": rid[:10] + "·" + rid.split("-")[-1],
        "engine": meta.get("engine", ""), "overlay": meta.get("overlay", "") or "",
        "notional": notional, "balance": notional * (1.0 + a["total_return"]),
        "start": str(meta.get("start", ""))[:10], "end": str(meta.get("end", ""))[:10],
        "n": a["n_trades"],
        "won": int((trades["outcome"] == "win").sum()) if len(trades) else 0,
        "lost": int((trades["outcome"] == "loss").sum()) if len(trades) else 0,
        "pnl": notional * a["total_return"], "ret": a["total_return"],
        "gw": float(pnl_s[pnl_s > 0].sum()), "gl": float(-pnl_s[pnl_s < 0].sum()),
        "pf": a["profit_factor"], "wr": a["win_rate"], "sharpe": a["sharpe"],
        "dd": a["max_drawdown"], "avgWin": a["avg_win"], "avgLoss": a["avg_loss"],
        "avgHold": a["avg_holding_bars"],
        "uni": ", ".join(symbols) if len(symbols) <= 5 else f"universe ({len(symbols)})",
    }


def _forward_leg(eval_dir: Path) -> dict:
    """One forward track (agent / baseline / honest-fill baseline).

    fill_mode is a newer field than cost_bps, so older run.json files fall back
    to "close" — the fill every record used before fill_mode was written.
    """
    if not (eval_dir / "run.json").exists():
        return {"symbols": [], "days": 0, "ret": 0.0, "pnl": 0.0, "notional": 10_000.0,
                "costBps": 0.0, "fillMode": "close",
                "spy": {"return": 0.0, "start": None, "end": None, "n_days": 0}}
    meta, trades, net = load_run(eval_dir)
    a = aggregate(trades, net)
    notional = float(meta.get("notional", 10_000.0))
    return {
        "symbols": meta.get("symbols", []), "days": len(net), "ret": a["total_return"],
        "pnl": notional * a["total_return"], "notional": notional,
        "costBps": float(meta.get("cost_bps", 0.0)),
        "fillMode": meta.get("fill_mode", "close"), "spy": spy_benchmark(net.index),
    }


def _locked_run(runs: list[Path], cfg) -> Path:
    """The most recent run matching config.yaml `strategy:`, else the latest run."""
    if cfg.strategy is not None:
        candidates = [(str(m["run_id"]), d) for d, m in ((d, load_run(d)[0]) for d in runs)
                      if m.get("engine") == cfg.strategy.name
                      and m.get("overlay", "none") == cfg.strategy.overlay]
        if candidates:
            return max(candidates, key=lambda c: c[0])[1]
    return runs[-1]


def _cross_run_buckets(base_dir: Path) -> tuple[list[dict], list[dict]]:
    """Top loss and win buckets across every bucketing dimension (vol, gap,
    holding, symbol, side, dow, near_high, ...) over every archived run."""
    frames = [f for f in (load_run(d)[1] for d in _run_dirs(base_dir)) if len(f)]
    if not frames:
        return [], []
    trades = flatten_trades(pd.concat(frames, ignore_index=True))
    n_losses = int((trades["outcome"] == "loss").sum())
    n_wins = int((trades["outcome"] == "win").sum())
    rows = []
    for dim, labels in _bucket_labels(trades).items():
        for bucket, idx in trades.groupby(labels).groups.items():
            sub = trades.loc[idx]
            loss_n = int((sub["outcome"] == "loss").sum())
            win_n = int((sub["outcome"] == "win").sum())
            rows.append({"dim": dim, "bucket": str(bucket), "lossN": loss_n, "winN": win_n,
                         "totalN": int(len(sub)),
                         "wr": float((sub["outcome"] == "win").mean()) if len(sub) else 0.0,
                         "lossShare": loss_n / n_losses if n_losses else 0.0,
                         "winShare": win_n / n_wins if n_wins else 0.0})
    top = lambda key: [{**r, "share": r[key]} for r in sorted(rows, key=lambda r: -r[key])[:5]]  # noqa: E731
    return top("lossShare"), top("winShare")


def _bakeoff_data(base_dir: Path, engine: str) -> list[dict]:
    df = robust_table(base_dir)
    if len(df) == 0 or not engine:
        return []
    df = df[df["engine"] == engine]
    order = {"none": 0, "conviction": 1, "bucket": 2, "winprob": 3}
    rows = []
    for overlay, grp in df.groupby(df["overlay"].fillna("none").replace("", "none")):
        best = grp.loc[grp["deflated"].idxmax()]
        rows.append({
            "overlay": overlay, "point": float(best["point_sharpe"]),
            "deflated": float(best["deflated"]),
            "fold": f"{best['fold_mean']:.2f}±{best['fold_std']:.2f}",
            "ci": f"[{best['ci_lo']:.2f}, {best['ci_hi']:.2f}]",
            "beats": bool(best["beats_baseline"]),
        })
    return sorted(rows, key=lambda r: order.get(r["overlay"], 99))


def _win_trades(run_dir: Path) -> list[dict]:
    _, trades, _ = load_run(run_dir)
    return [{
        "id": f"{i:04d}", "sym": t.symbol,
        "entry": str(t.entry_ts)[:10], "exit": str(t.exit_ts)[:10],
        "bars": int(t.holding_bars), "pnlPct": float(t.pnl_pct),
        "pnl": float(t.pnl_abs), "oc": t.outcome,
    } for i, t in enumerate(trades.itertuples(), start=1)]


def _build_data(base_dir: Path) -> dict:
    runs = _run_dirs(base_dir)
    if not runs:
        raise SystemExit(f"no runs found under {base_dir} — run rhagent.papertrade first")
    cfg = load_config()
    locked_dir = _locked_run(runs, cfg)
    _, _, net = load_run(locked_dir)
    equity = (1.0 + net.astype(float)).cumprod()
    forward_dir = base_dir.parent / "forward"
    loss_buckets, win_buckets = _cross_run_buckets(base_dir)
    g = cfg.limits
    memory = read_memory()

    return {
        "updated": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
        "lockedEngine": cfg.strategy.name if cfg.strategy else "",
        "lockedOverlay": cfg.strategy.overlay if cfg.strategy else "",
        "guardrails": {
            "per_trade_max_usd": g.per_trade_max_usd,
            "total_deployed_max_usd": g.total_deployed_max_usd,
            "max_new_positions_per_run": g.max_new_positions_per_run,
            "max_orders_per_run": g.max_orders_per_run,
            "daily_loss_limit_usd": g.daily_loss_limit_usd, "live": not cfg.dry_run,
            "halt": HALT_FILE.exists(), "model": cfg.agent.model,
        },
        "forward": {
            "agent": _forward_leg(forward_dir / "agent"),
            "baseline": _forward_leg(forward_dir / "mean_reversion"),
            "real": _forward_leg(forward_dir / "mean_reversion_real"),
        },
        "bakeoff": _bakeoff_data(base_dir, cfg.strategy.name if cfg.strategy else ""),
        "curveDaily": [float(v) for v in equity.tolist()],
        "curveDates": [str(d)[:10] for d in equity.index],
        "runs": [_run_row(d) for d in runs],
        "winScore": {**_run_row(locked_dir), "spy": spy_benchmark(net.index)},
        "winTrades": _win_trades(locked_dir),
        "buckets": loss_buckets, "winBuckets": win_buckets,
        "runbook": [list(x) for x in _RUNBOOK],
        "reflections": memory.split("\n## ")[1:][-3:] if memory else [],
        "actionsUrl": _ACTIONS_URL,
    }


_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0c10;--panel:#12161c;--panel2:#171d25;--line:#232a34;--line2:#2e3742;--fg:#e8edf4;--muted:#828d9b;--up:#05c46b;--down:#ff5c5c;--accent:#4db8ff;--warn:#ffb020;--purple:#b388ff}
*{box-sizing:border-box}
html,body{margin:0}
body{background:radial-gradient(1200px 600px at 78% -8%,rgba(77,184,255,.06),transparent 60%),var(--bg);color:var(--fg);font-family:'IBM Plex Sans',system-ui,sans-serif;font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}
::selection{background:rgba(5,196,107,.25)}
@keyframes drawerIn{from{transform:translateX(24px);opacity:0}to{transform:none;opacity:1}}
.m{font-family:'IBM Plex Mono',monospace}
.mu{color:var(--muted)}
section{margin-top:30px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px 22px}
.card h3,.card h2{margin:0 0 4px;font-size:15px;font-weight:600}
.sub{font-size:12px;color:var(--muted);margin-bottom:16px}
.grid{display:grid;gap:11px}
.two{display:grid;gap:20px}
.col{display:flex;flex-direction:column;gap:14px}
.hdr{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;flex-wrap:wrap}
.tile{background:var(--bg);border:1px solid var(--line);border-radius:11px;padding:13px 15px}
.tile .v{font-size:20px;font-weight:700;font-family:'IBM Plex Mono',monospace;letter-spacing:-.01em}
.tile .k{font-size:11px;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:.04em}
.chipbar{display:flex;gap:5px;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:3px;flex-wrap:wrap}
.chip{border:none;cursor:pointer;font:600 12px 'IBM Plex Sans',sans-serif;padding:6px 13px;border-radius:7px;background:transparent;color:var(--muted);display:inline-flex;align-items:center;gap:6px}
.chip[aria-pressed=true]{background:var(--panel2);color:var(--fg)}
.tag{padding:2px 9px;border-radius:6px;font:700 11px 'IBM Plex Mono',monospace;white-space:nowrap}
.pill{display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border-radius:999px;font:600 11px 'IBM Plex Mono',monospace;border:1px solid var(--line);background:var(--panel2);color:var(--muted)}
.dot{width:7px;height:7px;border-radius:50%;flex:none;background:currentColor}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{padding:8px 10px;font-weight:600;border-bottom:1px solid var(--line);color:var(--muted);text-align:right;white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid var(--line);font-family:'IBM Plex Mono',monospace;text-align:right}
th.l,td.l{text-align:left}
th.c,td.c{text-align:center}
.bar{height:8px;background:var(--bg);border:1px solid var(--line);border-radius:4px;overflow:hidden}
.bar>i{display:block;height:100%;border-radius:4px}
.scroll{overflow-x:auto}
.scroll::-webkit-scrollbar{height:9px;width:9px}
.scroll::-webkit-scrollbar-thumb{background:#2b333f;border-radius:6px}
.row:hover{background:var(--panel2)}
.row{cursor:pointer}
.note{padding:11px 13px;border-radius:10px;font-size:11.5px;color:var(--muted);text-wrap:pretty}
.btn{border:1px solid var(--line);background:var(--panel2);color:var(--fg);cursor:pointer;font:600 12.5px 'IBM Plex Sans',sans-serif;padding:8px 16px;border-radius:8px}
.sortable{cursor:pointer}
</style>
</head>
<body>
<header class="hdr" style="position:sticky;top:0;z-index:20;align-items:center;padding:14px 26px;background:rgba(10,12,16,.82);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)">
  <div style="line-height:1.15">
    <div style="font-weight:700;font-size:15px">RHAGENT<span class="mu" style="font-weight:500"> · Trading Control Room</span></div>
    <div class="m mu" style="font-size:11px">autonomous US-equities agent · guardrail-enforced</div>
  </div>
  <div id="headerpills" class="hdr" style="align-items:center;gap:8px"></div>
</header>

<main style="max-width:1240px;margin:0 auto;padding:26px 26px 90px">
  <section style="margin-top:0">
    <h2 class="mu" style="margin:0 0 12px;font-size:12px;letter-spacing:.14em;text-transform:uppercase">The verdict · agent vs baseline · forward paper track</h2>
    <div id="verdict"></div>
  </section>

  <section class="card">
    <h3>Equity curve · locked strategy candidate</h3><div class="sub" id="chartsub"></div>
    <div id="chart"></div>
    <div class="mu" style="margin-top:6px;font-size:11.5px">green dot = peak, red dot = trough of the max-drawdown window (shaded); dashed line = break-even.</div>
  </section>

  <section class="two" style="grid-template-columns:minmax(0,0.9fr) minmax(0,1.1fr)">
    <div class="card">
      <h3>Guardrails · armed</h3><div class="sub">Hard caps enforced in code — the model cannot talk its way past them.</div>
      <div id="guardrails" class="grid" style="grid-template-columns:repeat(auto-fit,minmax(140px,1fr))"></div>
    </div>
    <div class="card">
      <h3>Overlay bake-off · robust Sharpe</h3><div class="sub">A variant beats baseline only if its 95% CI lower bound clears the baseline Sharpe.</div>
      <div class="scroll"><table><thead><tr><th class="l">overlay</th><th>point</th><th>deflated</th><th>fold ±sd</th><th>95% CI</th><th class="c">vs base</th></tr></thead><tbody id="bakeoff"></tbody></table></div>
    </div>
  </section>

  <section class="card">
    <div class="hdr">
      <div><h3 id="runcount"></h3><div class="sub">Every paper-trade run in the archive. Click a column to sort, a row to open.</div></div>
      <div id="enginechips" class="chipbar"></div>
    </div>
    <div class="scroll" style="border:1px solid var(--line);border-radius:12px"><table style="min-width:760px"><thead id="runcols"></thead><tbody id="runrows"></tbody></table></div>
  </section>

  <section class="card">
    <div class="hdr" style="align-items:baseline;gap:10px;margin-bottom:6px">
      <h3>Locked-config scorecard</h3><span id="winid" class="m mu" style="font-size:11px"></span>
      <span class="tag" style="background:rgba(5,196,107,.14);color:var(--up)">FORWARD CANDIDATE</span>
      <span class="tag" style="background:rgba(255,176,32,.14);color:var(--warn)">IN-SAMPLE</span>
      <div style="flex:1"></div>
    </div>
    <div class="sub">Measured over the same window the strategy was selected on — selection-biased, not out-of-sample evidence.</div>
    <div id="scoretiles" class="grid" style="grid-template-columns:repeat(auto-fit,minmax(130px,1fr))"></div>
    <div id="scorespy" class="mu" style="margin-top:12px;font-size:11.5px"></div>
  </section>

  <section class="two" style="grid-template-columns:minmax(0,0.85fr) minmax(0,1.15fr)">
    <div class="col" style="gap:20px">
      <div class="card">
        <h3>Where losses concentrate</h3><div class="sub">Top buckets by share of all losses, across every archived run.</div>
        <div id="buckets" class="col"></div>
        <div class="note" style="margin-top:16px;background:rgba(255,176,32,.07);border:1px solid rgba(255,176,32,.22)">⚠ Measured across every archived paper-trade run, not just the locked config.</div>
      </div>
      <div class="card">
        <h3>Where we win</h3><div class="sub">Top buckets by share of all wins, across every archived run.</div>
        <div id="winbuckets" class="col"></div>
      </div>
    </div>
    <div class="card">
      <div class="hdr" style="margin-bottom:14px">
        <div><h3>Trade ledger</h3><div id="ledgercount" class="mu" style="font-size:12px"></div></div>
        <div id="tradechips" class="chipbar"></div>
      </div>
      <div class="scroll"><table style="min-width:520px"><thead><tr><th class="l">#</th><th class="l">sym</th><th class="l">entry → exit</th><th>bars</th><th>pnl %</th><th>pnl $</th><th style="min-width:90px">weight</th></tr></thead><tbody id="ledger"></tbody></table></div>
      <div id="ledgermore" style="margin-top:12px;text-align:center"></div>
    </div>
  </section>

  <section class="card">
    <div class="hdr" style="align-items:center">
      <div><h3>Runbook</h3><div class="mu" style="font-size:12px">Every command that drives this system. Click to copy.</div></div>
      <div class="hdr" style="align-items:center;gap:10px"><span id="trigger-status" class="m mu" style="font-size:11px"></span><button id="trigger-btn" class="btn">Run new research run</button></div>
    </div>
    <div id="runbook" class="col" style="gap:8px;margin-top:16px"></div>
  </section>

  <section id="agentnotes" class="card" style="display:none">
    <h3>Agent's own lessons (self-written)</h3><div class="sub">Reflections the agent journaled after past runs.</div>
    <div id="reflections" class="col" style="gap:10px;font-size:12.5px"></div>
  </section>

  <footer class="hdr m mu" style="margin-top:40px;padding-top:20px;border-top:1px solid var(--line);font-size:11.5px">
    <span>rhagent trading harness · journal/papertrade + journal/forward</span>
    <span>numbers reproduced from rhagent.evaluate · not investment advice</span>
  </footer>
</main>
<div id="drawerwrap"></div>

<script>
const DATA = __DATA_JSON__;
const ST = { engine: 'all', runSort: 'id', runDir: -1, tradeFilter: 'all', copied: -1, selectedRun: null, ledgerAll: false };
const LEDGER_PREVIEW = 10;
const $ = id => document.getElementById(id);

const money = (x, dp = 2) => (x < 0 ? '-' : '') + '$' + Math.abs(x).toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
const signed = (x, dp = 2) => (x >= 0 ? '+' : '') + money(x, dp);
const pct = (x, dp = 2) => (x >= 0 ? '+' : '') + (x * 100).toFixed(dp) + '%';
const pctAbs = (x, dp = 1) => (x * 100).toFixed(dp) + '%';
const num = x => x >= 999 ? '∞' : x.toFixed(2);
const ud = x => x >= 0 ? 'var(--up)' : 'var(--down)';
const engineColor = e => ({ mean_reversion: 'var(--accent)', momentum: 'var(--warn)', linreg: 'var(--purple)', agent: 'var(--up)' }[e] || 'var(--muted)');
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// one tile, one chip row, one bucket bar — reused by every panel.
const tile = (v, k, color) => `<div class="tile"><div class="v" style="color:${color || 'var(--fg)'}">${esc(v)}</div><div class="k">${esc(k)}</div></div>`;
const chips = (attr, items, active) => items.map(([id, label, dot]) =>
  `<button class="chip" ${attr}="${esc(id)}" aria-pressed="${id === active}">${dot ? `<span class="dot" style="color:${dot}"></span>` : ''}${esc(label)}</button>`).join('');
const bucketRow = (b, nKey, color) => `<div>
  <div class="hdr" style="align-items:baseline;gap:8px;margin-bottom:5px"><span style="font-size:12.5px"><span class="m mu" style="font-size:11px">${esc(b.dim)}</span> · ${esc(b.bucket)}</span><span class="m" style="font-size:12.5px;font-weight:700;color:${color}">${pctAbs(b.share, 1)}</span></div>
  <div class="bar"><i style="width:${(b.share * 100).toFixed(0)}%;background:${color}"></i></div>
  <div class="m mu" style="font-size:10.5px">${b[nKey].toLocaleString()} of ${b.totalN.toLocaleString()} trades · ${pctAbs(b.wr, 0)} win rate</div></div>`;

function renderHeaderPills() {
  const g = DATA.guardrails, U = 'var(--up)', W = 'var(--warn)', D = 'var(--down)';
  $('headerpills').innerHTML = `
    <span class="pill" style="color:${g.live ? U : W};border-color:currentColor"><span class="dot"></span>${g.live ? 'LIVE · TRADING' : 'PAPER · DRY-RUN'}</span>
    <span class="pill" style="color:${g.halt ? D : U};border-color:currentColor">HALT · ${g.halt ? 'SET' : 'CLEAR'}</span>
    <a href="${DATA.actionsUrl}"><img src="${DATA.actionsUrl}/badge.svg?branch=main" alt="daily paper-run status"></a>
    <span class="pill">upd ${esc(DATA.updated)}</span>`;
}

function renderVerdict() {
  const { agent, baseline: base, real } = DATA.forward;
  const days = agent.days || base.days || 0;
  let badge = 'TOO EARLY TO CALL', note = `Forward track has ${days} day(s) logged. Verdict needs weeks of OOS data.`;
  if (days >= 5) {
    badge = agent.pnl > base.pnl ? 'AGENT LEADS' : base.pnl > agent.pnl ? 'BASELINE LEADS' : 'TIED';
    note = `Forward P&L over the tracked window: ${badge.toLowerCase()}.`;
  }
  const fill = leg => `${leg.fillMode === 'next_open' ? 'next-open' : 'same-close'} fill @ ${leg.costBps}bp`;
  const sub = leg => `${leg.days} day(s) · bal ${money(leg.notional + leg.pnl, 0)} · net ${pct(leg.ret)} · ${(leg.symbols || []).join(', ') || '—'} · ${fill(leg)}`;
  const spyLine = leg => leg.spy && leg.spy.start
    ? `SPY buy&amp;hold ${leg.spy.start}→${leg.spy.end}: <b style="color:var(--fg)">${pct(leg.spy.return)}</b>`
    : 'SPY buy&amp;hold: not enough tracked days yet';
  const side = (leg, title, color, align) => `<div style="padding:22px 26px;display:flex;flex-direction:column;gap:6px;text-align:${align}">
      <div style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:${color};font-weight:600">${title}</div>
      <div class="m" style="font-size:34px;font-weight:700;letter-spacing:-.02em">${money(leg.pnl)}</div>
      <div class="m mu" style="font-size:12px">${esc(sub(leg))}</div>
      <div class="m mu" style="font-size:11px">${spyLine(leg)}</div></div>`;
  const banner = (color, bg, html) => `<div class="note" style="margin-top:8px;font-size:13px;color:var(--fg);background:${bg};border:1px solid ${color}">${html}</div>`;
  $('verdict').innerHTML = `
    <div style="display:grid;grid-template-columns:1fr auto 1fr;align-items:stretch;background:var(--panel);border:1px solid var(--line);border-radius:16px;overflow:hidden">
      ${side(agent, 'LLM Agent', 'var(--accent)', 'left')}
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;padding:22px 30px;background:var(--panel2);border-left:1px solid var(--line);border-right:1px solid var(--line);min-width:220px">
        <div style="padding:6px 16px;border-radius:999px;background:rgba(255,176,32,.12);border:1px solid rgba(255,176,32,.35);color:var(--warn);font-weight:700;font-size:13px">${badge}</div>
        <div class="mu" style="font-size:12px;text-align:center;max-width:210px;text-wrap:pretty">${esc(note)}</div>
      </div>
      ${side(base, 'Mean-Reversion Baseline', 'var(--purple)', 'right')}
    </div>
    ${banner('rgba(5,196,107,.22)', 'rgba(5,196,107,.06)', `<b style="color:var(--up)">Research winner locked:</b> ${esc(DATA.lockedEngine)}, gated by the <b>${esc(DATA.lockedOverlay || 'no')}</b> overlay. This is the config the forward track is now paper-trading.`)}
    ${real.days ? banner('rgba(77,184,255,.22)', 'rgba(77,184,255,.06)', `<b style="color:var(--accent)">Honest-fill record (mean_reversion_real):</b> ${money(real.pnl)} over ${esc(sub(real))}. The baseline above is ${fill(base)} — flattering by comparison, kept only because its track record predates this fill.`) : ''}
    <div class="mu" style="margin-top:8px;font-size:11px">SPY benchmark is buy-and-hold over each leg's own tracked window; the strategy is not always fully invested, so this isn't an apples-to-apples exposure comparison.</div>`;
}

function renderChart() {
  const ys = DATA.curveDaily, dates = DATA.curveDates, n = ys.length;
  $('chartsub').textContent = `${DATA.lockedEngine}${DATA.lockedOverlay ? ' + ' + DATA.lockedOverlay : ''} · ${dates[0] || ''} → ${dates[n - 1] || ''} · ${money(DATA.winScore.notional, 0)} notional`;
  if (!n) { $('chart').innerHTML = '<p class="mu">no equity series</p>'; return; }
  const W = 960, H = 320, padL = 54, padR = 18, padT = 20, padB = 30;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  let lo = Math.min(...ys), hi = Math.max(...ys);
  const m = (hi - lo) * 0.14 || 0.01; lo -= m; hi += m;
  const span = hi - lo || 1;
  const px = i => padL + plotW * (i / Math.max(n - 1, 1));
  const py = v => padT + plotH * (1 - (v - lo) / span);
  const txt = (x, y, s, anchor, fill, size) => `<text x="${x}" y="${y}" text-anchor="${anchor}" fill="${fill}" font-size="${size}" font-family="'IBM Plex Mono',monospace">${esc(s)}</text>`;
  let axes = '';
  for (let k = 0; k < 5; k++) {
    const v = lo + span * k / 4, y = py(v);
    axes += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#232a34" opacity="0.55"/>` + txt(padL - 9, y + 3.5, v.toFixed(2) + '×', 'end', '#828d9b', 11);
  }
  axes += [0, Math.floor(n / 2), n - 1].map((i, k) => txt(px(i), H - 9, dates[i], ['start', 'middle', 'end'][k === 0 ? 0 : k === 2 ? 2 : 1], '#828d9b', 11)).join('');
  const line = ys.map((v, i) => px(i).toFixed(1) + ',' + py(v).toFixed(1)).join(' ');
  const stroke = ys[n - 1] >= 1 ? '#05c46b' : '#ff5c5c';
  let rm = -Infinity, ddMin = 0, ddI = 0, peakI = 0, pk = -Infinity;
  ys.forEach((v, i) => { rm = Math.max(rm, v); if (v / rm - 1 < ddMin) { ddMin = v / rm - 1; ddI = i; } });
  for (let i = 0; i <= ddI; i++) if (ys[i] > pk) { pk = ys[i]; peakI = i; }
  const dot = (i, c) => `<circle cx="${px(i)}" cy="${py(ys[i])}" r="3.5" fill="${c}" stroke="#0a0c10" stroke-width="1.5"/>`;
  $('chart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block">
    ${ddMin < 0 ? `<rect x="${px(peakI)}" y="${padT}" width="${Math.max(px(ddI) - px(peakI), 1)}" height="${plotH}" fill="#ff5c5c" opacity="0.07"/>` : ''}
    ${axes}
    <polygon points="${px(0).toFixed(1)},${py(lo).toFixed(1)} ${line} ${px(n - 1).toFixed(1)},${py(lo).toFixed(1)}" fill="${stroke}" opacity="0.1"/>
    ${(1 >= lo && 1 <= hi) ? `<line x1="${padL}" y1="${py(1)}" x2="${W - padR}" y2="${py(1)}" stroke="#4db8ff" stroke-dasharray="4 4" opacity="0.5"/>` : ''}
    <polyline points="${line}" fill="none" stroke="${stroke}" stroke-width="2" stroke-linejoin="round"/>
    ${dot(peakI, '#05c46b')}${ddMin < 0 ? dot(ddI, '#ff5c5c') : ''}
  </svg>`;
}

function renderGuardrails() {
  const g = DATA.guardrails;
  $('guardrails').innerHTML = [
    ['per-trade max', money(g.per_trade_max_usd, 0)], ['total deployed max', money(g.total_deployed_max_usd, 0)],
    ['new positions / run', g.max_new_positions_per_run], ['orders / run', g.max_orders_per_run],
    ['daily realized-loss kill', money(g.daily_loss_limit_usd, 0)], ['model', g.model.split('/').pop()],
  ].map(([k, v]) => tile(v, k, 'var(--up)')).join('');
}

function renderBakeoff() {
  const rows = DATA.bakeoff;
  if (!rows.length) { $('bakeoff').innerHTML = '<tr><td colspan="6" class="l mu">no bake-off data</td></tr>'; return; }
  const maxPoint = Math.max(...rows.map(b => b.point));
  $('bakeoff').innerHTML = rows.map(b => `<tr style="background:${b.beats ? 'rgba(5,196,107,.07)' : 'transparent'}">
    <td class="l" style="color:${b.beats ? 'var(--up)' : 'var(--fg)'};font-weight:${b.beats ? 700 : 500}">${esc(b.overlay)}</td>
    <td><div style="display:flex;align-items:center;justify-content:flex-end;gap:8px"><span class="bar" style="width:44px;height:5px"><i style="width:${(b.point / maxPoint * 100).toFixed(0)}%;background:${b.beats ? 'var(--up)' : 'var(--muted)'}"></i></span>${b.point.toFixed(2)}</div></td>
    <td class="mu">${b.deflated.toFixed(2)}</td><td class="mu">${esc(b.fold)}</td><td class="mu">${esc(b.ci)}</td>
    <td class="c">${b.beats ? '✓ beats' : '—'}</td></tr>`).join('');
}

const RUN_COLS = [['id', 'run'], ['engine', 'engine'], ['overlay', 'overlay'], ['n', 'trades'], ['wr', 'win %'], ['pf', 'PF'], ['pnl', 'P&L'], ['ret', 'return']];

function renderRuns() {
  $('runcount').textContent = `Research runs · ${DATA.runs.length} total`;
  $('enginechips').innerHTML = chips('data-engine',
    [['all', 'all', 'var(--muted)'], ...new Set(DATA.runs.map(r => r.engine))].map(e => Array.isArray(e) ? e : [e, e, engineColor(e)]), ST.engine);
  $('runcols').innerHTML = '<tr>' + RUN_COLS.map(([k, label], i) =>
    `<th class="sortable ${i < 3 ? 'l' : ''}" data-sort="${k}">${esc(label)}<span style="color:var(--accent)">${ST.runSort === k ? (ST.runDir < 0 ? ' ↓' : ' ↑') : ''}</span></th>`).join('') + '</tr>';

  const rows = DATA.runs.filter(r => ST.engine === 'all' || r.engine === ST.engine).slice().sort((a, b) => {
    const av = a[ST.runSort], bv = b[ST.runSort];
    return typeof av === 'string' ? ST.runDir * av.localeCompare(bv) : ST.runDir * (av - bv);
  });
  const maxAbsRet = Math.max(...rows.map(r => Math.abs(r.ret)), 0.01);
  $('runrows').innerHTML = rows.map(r => {
    const win = r.id === DATA.winScore.id;
    return `<tr class="row" data-open="${esc(r.id)}" style="background:${win ? 'rgba(5,196,107,.06)' : 'transparent'};border-left:3px solid ${win ? 'var(--up)' : 'transparent'}">
      <td class="l"><span style="color:var(--accent)">${esc(r.sid)}</span>${win ? '<span style="color:var(--up);font-weight:700;font-size:10px;margin-left:6px">◆ LOCKED</span>' : ''}</td>
      <td class="l"><span class="dot" style="display:inline-block;color:${engineColor(r.engine)}"></span> ${esc(r.engine)}</td>
      <td class="l mu">${esc(r.overlay || '—')}</td><td>${r.n}</td><td>${pctAbs(r.wr, 1)}</td><td>${num(r.pf)}</td>
      <td style="color:${ud(r.pnl)};font-weight:600">${signed(r.pnl, 0)}</td>
      <td><div style="display:flex;align-items:center;justify-content:flex-end;gap:8px"><span class="bar" style="width:52px;height:5px;flex:none"><i style="width:${(Math.abs(r.ret) / maxAbsRet * 100).toFixed(0)}%;background:${ud(r.ret)}"></i></span><span style="color:${ud(r.ret)};min-width:56px">${pct(r.ret, 1)}</span></div></td></tr>`;
  }).join('');
}

const SCORE_TILES = S => [
  ['balance', money(S.balance), ud(S.balance - S.notional)], ['net P&L', signed(S.pnl), ud(S.pnl)],
  ['total return', pct(S.ret), ud(S.ret)], ['trades', String(S.n)], ['win rate', pctAbs(S.wr, 1)],
  ['profit factor', num(S.pf), ud(S.pf - 1)], ['avg win', money(S.avgWin), 'var(--up)'],
  ['avg loss', money(S.avgLoss), 'var(--down)'], ['sharpe', S.sharpe.toFixed(2)],
  ['max drawdown', pctAbs(S.dd, 2), 'var(--down)'], ['avg holding', S.avgHold.toFixed(1) + ' bars'],
  ['gross win', money(S.gw, 0), 'var(--up)'], ['gross loss', money(-S.gl, 0), 'var(--down)'],
];

function renderScorecard() {
  const S = DATA.winScore, spy = S.spy;
  $('winid').textContent = S.id;
  $('scoretiles').innerHTML = SCORE_TILES(S).map(([k, v, c]) => tile(v, k, c)).join('');
  $('scorespy').textContent = spy && spy.start
    ? `SPY buy-and-hold over the same window (${spy.start} → ${spy.end}): ${pct(spy.return)} — vs strategy total return ${pct(S.ret)}. The strategy is not always fully invested, so this isn't apples-to-apples on exposure.`
    : 'SPY buy-and-hold benchmark unavailable for this window (price cache does not cover it).';
}

function renderBuckets() {
  $('buckets').innerHTML = DATA.buckets.map(b => bucketRow(b, 'lossN', 'var(--down)')).join('');
  $('winbuckets').innerHTML = DATA.winBuckets.map(b => bucketRow(b, 'winN', 'var(--up)')).join('');
}

function renderLedger() {
  const S = DATA.winScore;
  $('ledgercount').textContent = `${S.won} wins · ${S.lost} losses · ${S.n} trades`;
  $('tradechips').innerHTML = chips('data-tradefilter', [['all', 'all'], ['win', 'wins'], ['loss', 'losses']], ST.tradeFilter);
  const all = DATA.winTrades.filter(t => ST.tradeFilter === 'all' || t.oc === ST.tradeFilter);
  const maxAbsPnl = Math.max(...DATA.winTrades.map(t => Math.abs(t.pnl)), 1);
  $('ledger').innerHTML = (ST.ledgerAll ? all : all.slice(0, LEDGER_PREVIEW)).map(t => {
    const w = (Math.abs(t.pnl) / maxAbsPnl * 100).toFixed(0), c = ud(t.pnl);
    const half = (side, on) => `<span style="flex:1;display:flex;justify-content:flex-${side}"><span style="height:8px;width:${on ? w : 0}%;background:${c};border-radius:2px"></span></span>`;
    return `<tr style="border-left:3px solid ${c}">
      <td class="l mu">#${esc(t.id)}</td><td class="l" style="font-weight:600">${esc(t.sym)}</td>
      <td class="l" style="font-size:11px">${esc(t.entry)} → ${esc(t.exit)}</td><td>${t.bars}</td>
      <td style="color:${c}">${pct(t.pnlPct)}</td><td style="color:${c};font-weight:600">${signed(t.pnl)}</td>
      <td><div style="display:flex;align-items:center;height:12px">${half('end', t.pnl < 0)}<span style="width:1px;height:12px;background:var(--line2)"></span>${half('start', t.pnl >= 0)}</div></td></tr>`;
  }).join('');
  $('ledgermore').innerHTML = all.length > LEDGER_PREVIEW
    ? `<button class="btn" data-ledgertoggle="1">${ST.ledgerAll ? 'show fewer' : `show all ${all.length} trades`}</button>` : '';
}

function renderRunbook() {
  $('runbook').innerHTML = DATA.runbook.map(([label, cmd], i) => `
    <button class="btn" data-copy="${i}" style="text-align:left;display:flex;align-items:center;gap:14px;background:var(--bg)">
      <span class="mu" style="font-size:12px;min-width:150px;flex:none">${esc(label)}</span>
      <span class="m" style="font-size:12px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(cmd)}</span>
      <span class="m" style="font-size:11px;color:${ST.copied === i ? 'var(--up)' : 'var(--muted)'};flex:none">${ST.copied === i ? 'COPIED' : 'COPY'}</span>
    </button>`).join('');
}

function renderAgentNotes() {
  const has = DATA.reflections && DATA.reflections.length;
  $('agentnotes').style.display = has ? '' : 'none';
  if (has) $('reflections').innerHTML = DATA.reflections.map(r => `<p style="margin:0">## ${esc(r)}</p>`).join('');
}

function renderDrawer() {
  const wrap = $('drawerwrap');
  const r = ST.selectedRun && DATA.runs.find(x => x.id === ST.selectedRun);
  if (!r) { wrap.innerHTML = ''; return; }
  const win = r.id === DATA.winScore.id;
  wrap.innerHTML = `
    <div data-close style="position:fixed;inset:0;z-index:40;background:rgba(4,6,9,.6);backdrop-filter:blur(2px)"></div>
    <aside class="scroll" style="position:fixed;top:0;right:0;z-index:41;height:100vh;width:min(540px,94vw);background:var(--panel);border-left:1px solid var(--line2);box-shadow:-24px 0 60px rgba(0,0,0,.5);overflow-y:auto;animation:drawerIn .22s cubic-bezier(.2,.8,.2,1) both">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;padding:20px 24px;border-bottom:1px solid var(--line)">
        <div>
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
            <span style="font-weight:600;font-size:14px"><span class="dot" style="display:inline-block;color:${engineColor(r.engine)}"></span> ${esc(r.engine)}</span>
            <span class="pill" style="border-radius:6px">${esc(r.overlay || 'no overlay')}</span>
            ${win ? '<span class="tag" style="background:rgba(5,196,107,.14);color:var(--up)">◆ LOCKED CONFIG</span>' : ''}
          </div>
          <div class="m mu" style="font-size:11.5px;margin-top:6px">${esc(r.id)}</div>
        </div>
        <button class="btn" data-close style="padding:4px 10px">✕</button>
      </div>
      <div style="padding:22px 24px">
        <div style="display:flex;align-items:baseline;gap:12px">
          <div class="m" style="font-size:38px;font-weight:700;color:${ud(r.ret)}">${pct(r.ret)}</div>
          <div class="m" style="font-size:14px;font-weight:600;color:${ud(r.pnl)}">${signed(r.pnl)}</div>
        </div>
        <div class="mu" style="font-size:12px">total return on ${money(r.notional, 0)} notional · ${esc(r.uni)} · ${r.start} → ${r.end}</div>
        <div class="grid" style="grid-template-columns:repeat(3,1fr);margin-top:18px">${SCORE_TILES(r).map(([k, v, c]) => tile(v, k, c)).join('')}</div>
        <div style="margin-top:16px">
          <div class="m mu" style="font-size:11px;margin-bottom:6px">win / loss split · ${r.won}W / ${r.lost}L</div>
          <div style="display:flex;height:9px;border-radius:5px;overflow:hidden;background:var(--panel2)">
            <div style="width:${(r.won / (r.n || 1) * 100).toFixed(1)}%;background:var(--up)"></div>
            <div style="width:${(r.lost / (r.n || 1) * 100).toFixed(1)}%;background:var(--down)"></div>
          </div>
        </div>
        <div class="note" style="margin-top:16px;background:rgba(77,184,255,.06);border:1px solid rgba(77,184,255,.22)">${win
          ? 'This is the config the forward paper track is now trading — its per-trade ledger is in the Trade ledger panel.'
          : `Per-trade ledger for this run: rhagent.evaluate --run ${esc(r.id)}`}</div>
      </div>
    </aside>`;
}

async function triggerResearchRun() {
  const status = $('trigger-status'), btn = $('trigger-btn');
  btn.disabled = true;
  status.textContent = 'triggering…';
  status.style.color = 'var(--muted)';
  try {
    const r = await fetch('/api/trigger-run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    const body = r.ok ? {} : await r.json().catch(() => ({}));
    status.textContent = r.ok ? 'triggered — running on GitHub Actions' : 'failed: ' + (body.error || r.status);
    status.style.color = r.ok ? 'var(--up)' : 'var(--down)';
  } catch (e) {
    status.textContent = 'failed: ' + e.message;
    status.style.color = 'var(--down)';
  } finally {
    btn.disabled = false;
  }
}

document.addEventListener('click', e => {
  const t = e.target.closest('[data-engine],[data-sort],[data-open],[data-tradefilter],[data-ledgertoggle],[data-copy],[data-close],#trigger-btn');
  if (!t) return;
  const d = t.dataset;
  if (d.engine) { ST.engine = d.engine; renderRuns(); }
  else if (d.sort) {
    if (ST.runSort === d.sort) ST.runDir = -ST.runDir;
    else { ST.runSort = d.sort; ST.runDir = ['id', 'engine', 'overlay'].includes(d.sort) ? 1 : -1; }
    renderRuns();
  } else if (d.open) { ST.selectedRun = d.open; renderDrawer(); }
  else if (d.tradefilter) { ST.tradeFilter = d.tradefilter; ST.ledgerAll = false; renderLedger(); }
  else if (d.ledgertoggle) { ST.ledgerAll = !ST.ledgerAll; renderLedger(); }
  else if (t.id === 'trigger-btn') { triggerResearchRun(); }
  else if (d.copy != null) {
    const i = Number(d.copy);
    if (navigator.clipboard) navigator.clipboard.writeText(DATA.runbook[i][1]).catch(() => {});
    ST.copied = i; renderRunbook();
    setTimeout(() => { ST.copied = -1; renderRunbook(); }, 1400);
  } else if (d.close != null) { ST.selectedRun = null; renderDrawer(); }
});
document.addEventListener('keydown', e => { if (e.key === 'Escape' && ST.selectedRun) { ST.selectedRun = null; renderDrawer(); } });

[renderHeaderPills, renderVerdict, renderChart, renderGuardrails, renderBakeoff, renderRuns,
 renderScorecard, renderBuckets, renderLedger, renderRunbook, renderAgentNotes, renderDrawer].forEach(f => f());
</script>
</body>
</html>
"""


def render_control_room(base_dir: Path) -> str:
    data = _build_data(base_dir)
    title = f"Trading Control Room — {len(data['runs'])} research runs"
    return (_TEMPLATE.replace("__TITLE__", escape(title))
            .replace("__DATA_JSON__", json.dumps(data)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="make_dashboard")
    p.add_argument("--base-dir", default="journal/papertrade")
    p.add_argument("--out", help="output HTML path (default: journal/dashboard.html)")
    p.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = p.parse_args(argv)

    base_dir = Path(args.base_dir)
    out = Path(args.out) if args.out else base_dir.parent / "dashboard.html"
    out.write_text(render_control_room(base_dir), encoding="utf-8")
    print(f"wrote {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
