"""Render the trading system into a single self-contained HTML dashboard.

Everything happening in one page: forward records, the latest paper-trade
scorecard, all research runs, the robust bake-off, the equity curve, and the
latest run's ledger/failure buckets. The unified view ("control room") is a
static HTML page with a small vanilla-JS layer for chart mode toggles, run
sorting/filtering, and a per-run detail drawer — no build step, no framework.

    python scripts/make_dashboard.py                 # unified control room
    python scripts/make_dashboard.py --run <run_id>  # a specific run
    python scripts/make_dashboard.py --open          # also open in a browser

Reads the ledgers written under journal/papertrade/ and journal/forward/; it
reuses rhagent.evaluate so the numbers match the CLI report exactly.
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
from rhagent.evaluate import (  # noqa: E402
    _bucket_labels,
    aggregate,
    compare_runs,
    failure_buckets,
    load_run,
)
from rhagent.evaluate_robust import robust_table  # noqa: E402
from rhagent.features import flatten_trades  # noqa: E402
from rhagent.learn import lessons_from_runs  # noqa: E402
from rhagent.memory import read_memory  # noqa: E402

HALT_FILE = Path("HALT")

# GitHub renders this badge live (in progress / passing / failing), so the
# static dashboard shows current CI state without any JS or re-render.
_ACTIONS_URL = ("https://github.com/udayjain2102/papertrading-model-"
                "/actions/workflows/daily-paper-run.yml")

_RUNBOOK = [
    ("daily forward tick", "PYTHONPATH=src .venv/bin/python -m rhagent.forward"),
    ("new research run", "PYTHONPATH=src .venv/bin/python -m rhagent.papertrade --engine mean_reversion --symbols all"),
    ("unattended daily loop", "scripts/paper_cron.sh"),
    ("rebuild this page", ".venv/bin/python scripts/make_dashboard.py --open"),
    ("run the tests", ".venv/bin/python -m pytest"),
]


# ── formatting helpers ──────────────────────────────────────────────────────

def _pct(x: float) -> str:
    return f"{x:.2%}"


def _money(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def _num(x: float) -> str:
    return "∞" if x == float("inf") else f"{x:.2f}"


def _return_pnl(total_return: float, notional: float) -> float:
    return float(notional) * float(total_return)


def _run_dirs(base_dir: Path) -> list[Path]:
    return sorted(p.parent for p in base_dir.glob("*/run.json"))


def _run_anchor(run_id: str) -> str:
    safe = "".join(c if c.isalnum() else "-" for c in str(run_id)).strip("-")
    return f"run-{safe}"


# ── SVG equity curve (used by the single-run detail page) ──────────────────

def _equity_svg(net: pd.Series, width: int = 900, height: int = 300) -> str:
    if len(net) == 0:
        return "<p class='muted'>no return series</p>"
    equity = (1.0 + net.astype(float)).cumprod()
    ys = equity.tolist()
    dates = [str(d)[:10] for d in equity.index]
    n = len(ys)
    lo, hi = min(ys), max(ys)
    # pad the value range so the line never hugs the frame
    margin = (hi - lo) * 0.08 or 0.01
    lo, hi = lo - margin, hi + margin
    span = hi - lo
    padL, padR, padT, padB = 56, 16, 28, 30
    plot_h = height - padT - padB

    def px(i: int) -> float:
        return padL + (width - padL - padR) * (i / max(n - 1, 1))

    def py(v: float) -> float:
        return padT + plot_h * (1 - (v - lo) / span)

    grid, ylabels = [], []
    for k in range(5):
        v = lo + span * k / 4
        y = py(v)
        grid.append(f"<line x1='{padL}' y1='{y:.1f}' x2='{width - padR}' y2='{y:.1f}' class='grid'/>")
        ylabels.append(f"<text x='{padL - 8}' y='{y + 4:.1f}' class='ytick'>{v:.2f}×</text>")

    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(ys))
    area = f"{px(0):.1f},{py(lo):.1f} " + pts + f" {px(n - 1):.1f},{py(lo):.1f}"
    final = ys[-1]
    stroke = "var(--up)" if final >= 1.0 else "var(--down)"

    base_y = py(1.0) if lo <= 1.0 <= hi else None
    baseline = (
        f"<line x1='{padL}' y1='{base_y:.1f}' x2='{width - padR}' y2='{base_y:.1f}' "
        f"class='baseline'/><text x='{padL - 8}' y='{base_y + 4:.1f}' class='ytick base'>1.00×</text>"
        if base_y is not None else ""
    )

    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    dd_shade = dd_marker = ""
    if len(dd) and dd.min() < 0:
        trough = int(dd.values.argmin())
        peak = int(equity.values[: trough + 1].argmax())
        x0, x1 = px(peak), px(trough)
        dd_shade = (
            f"<rect x='{x0:.1f}' y='{padT}' width='{max(x1 - x0, 1):.1f}' height='{plot_h}' "
            f"class='ddband'/>"
        )
        dd_marker = (
            f"<circle cx='{px(trough):.1f}' cy='{py(ys[trough]):.1f}' r='3.5' class='ddpt'/>"
            f"<text x='{px(trough):.1f}' y='{py(ys[trough]) + 18:.1f}' class='ddlabel'>"
            f"max DD {_pct(dd.min())}</text>"
        )

    peak_i = int(equity.values.argmax())
    peak_dot = f"<circle cx='{px(peak_i):.1f}' cy='{py(ys[peak_i]):.1f}' r='3.5' class='peakpt'/>"

    xlabels = (
        f"<text x='{padL}' y='{height - 8}' class='xtick' text-anchor='start'>{dates[0]}</text>"
        f"<text x='{width - padR}' y='{height - 8}' class='xtick' text-anchor='end'>{dates[-1]}</text>"
    )

    return f"""<svg viewBox="0 0 {width} {height}" class="equity" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Equity curve, final {equity.iloc[-1]:.2f} times starting capital, {_pct(final - 1)}">
  {''.join(grid)}
  {dd_shade}
  <polygon points="{area}" fill="{stroke}" opacity="0.12"/>
  {baseline}
  <polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="2" stroke-linejoin="round"/>
  {peak_dot}{dd_marker}
  {''.join(ylabels)}{xlabels}
  <text x="{padL}" y="16" class="svglabel">equity {equity.iloc[-1]:.2f}× · {_pct(final - 1)}</text>
</svg>"""


# ── HTML fragments (single-run detail page) ─────────────────────────────────

def _scorecard(a: dict, trades: pd.DataFrame, notional: float) -> str:
    def tile(label: str, value: str, cls: str = "", sub: str = "") -> str:
        subhtml = f"<div class='tile-s'>{sub}</div>" if sub else ""
        return (
            f"<div class='tile'><div class='tile-v {cls}'>{value}</div>"
            f"<div class='tile-l'>{label}</div>{subhtml}</div>"
        )

    pnl = trades["pnl_abs"].astype(float) if len(trades) else pd.Series(dtype=float)
    winnings = float(pnl[pnl > 0].sum())
    loss = float(-pnl[pnl < 0].sum())
    net_pl = _return_pnl(a["total_return"], notional)
    balance = notional + net_pl
    bal_cls = "up" if balance >= notional else "down"

    ret_cls = "up" if a["total_return"] >= 0 else "down"
    pf_cls = "up" if a["profit_factor"] >= 1 else "down"
    return "<div class='tiles'>" + "".join([
        tile("current balance", _money(balance), bal_cls,
             sub=f"start {_money(notional)} · net {'+' if net_pl >= 0 else ''}{_money(net_pl)}"),
        tile("gross trade wins", _money(winnings), "up"),
        tile("gross trade losses", _money(-loss), "down"),
        tile("total return", _pct(a["total_return"]), ret_cls),
        tile("trades", str(a["n_trades"])),
        tile("win rate", _pct(a["win_rate"])),
        tile("profit factor", _num(a["profit_factor"]), pf_cls),
        tile("avg win", _money(a["avg_win"]), "up"),
        tile("avg loss", _money(a["avg_loss"]), "down"),
        tile("sharpe", _num(a["sharpe"])),
        tile("max drawdown", _pct(a["max_drawdown"]), "down"),
        tile("avg holding", f"{a['avg_holding_bars']:.2f} bars"),
    ]) + "</div>"


def _trades_table(trades: pd.DataFrame) -> str:
    if len(trades) == 0:
        return "<p class='muted'>no trades</p>"
    cols = [
        ("trade_id", "trade id"), ("symbol", "sym"), ("side", "side"),
        ("entry_ts", "entry"), ("entry_price", "in"),
        ("exit_ts", "exit"), ("exit_price", "out"),
        ("holding_bars", "bars"), ("pnl_pct", "pnl %"),
        ("pnl_abs", "pnl $"), ("outcome", ""), ("entry_reason", "reason"),
        ("exit_reason", "exit reason"),
    ]
    head = "".join(f"<th>{escape(h)}</th>" for _, h in cols)
    rows = []
    for _, t in trades.iterrows():
        oc = t["outcome"]
        seq = str(t["trade_id"]).split("#")[-1]
        cells = [
            f"<td class='mono' title='{escape(str(t['trade_id']))}'>#{seq}</td>",
            f"<td>{escape(str(t['symbol']))}</td>",
            f"<td>{escape(str(t['side']))}</td>",
            f"<td class='mono'>{escape(str(t['entry_ts'])[:10])}</td>",
            f"<td class='num'>{t['entry_price']:.2f}</td>",
            f"<td class='mono'>{escape(str(t['exit_ts'])[:10])}</td>",
            f"<td class='num'>{t['exit_price']:.2f}</td>",
            f"<td class='num'>{int(t['holding_bars'])}</td>",
            f"<td class='num {oc}'>{_pct(t['pnl_pct'])}</td>",
            f"<td class='num {oc}'>{_money(t['pnl_abs'])}</td>",
            f"<td><span class='pill {oc}'>{escape(oc)}</span></td>",
            f"<td class='reason'>{escape(str(t['entry_reason']))}</td>",
            f"<td class='reason'>{escape(str(t['exit_reason']))}</td>",
        ]
        rows.append(f"<tr class='row-{oc}'>{''.join(cells)}</tr>")
    return f"<table class='grid'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _buckets_table(b: pd.DataFrame) -> str:
    if len(b) == 0 or (b["loss_share"] == 0).all():
        return "<p class='muted'>no losing trades — nothing to bucket</p>"
    b = b[b["loss_share"] > 0]
    rows = []
    for _, r in b.iterrows():
        share = r["loss_share"]
        bar = (
            f"<div class='barwrap'><div class='bar' style='width:{share*100:.1f}%'></div>"
            f"<span>{_pct(share)}</span></div>"
        )
        rows.append(
            f"<tr><td>{escape(str(r['dimension']))}</td>"
            f"<td class='mono'>{escape(str(r['bucket']))}</td>"
            f"<td class='num'>{int(r['n_trades'])}</td>"
            f"<td class='num'>{_pct(r['win_rate'])}</td>"
            f"<td class='sharecell'>{bar}</td></tr>"
        )
    return (
        "<table class='grid'><thead><tr><th>dimension</th><th>bucket</th>"
        "<th>trades</th><th>win rate</th><th>share of total loss</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _compare_table(df: pd.DataFrame, current: str, link: bool = False) -> str:
    if len(df) == 0:
        return "<p class='muted'>no runs</p>"
    best_id = df.loc[df["total_return"].idxmax(), "run_id"] if len(df) else None
    rows = []
    for _, r in df.iterrows():
        rid = str(r["run_id"])
        cls = "cur" if rid == current else ""
        ret_cls = "up" if r["total_return"] >= 0 else "down"
        badges = ""
        if rid == best_id:
            badges += "<span class='tag best'>best</span>"
        if rid == current:
            badges += "<span class='tag cur'>viewing</span>"
        idcell = f"<a class='mono' href='#{_run_anchor(rid)}'>{escape(rid)}</a>" if link else f"{escape(rid)}"
        pnl = float(r["net_pnl"])
        pnl_cls = "up" if pnl >= 0 else "down"
        rows.append(
            f"<tr class='{cls}'><td class='mono'>{idcell}{badges}</td>"
            f"<td>{escape(str(r['engine']))}</td>"
            f"<td class='num'>{int(r['n_trades'])}</td>"
            f"<td class='num up'>{int(r['won'])}</td>"
            f"<td class='num down'>{int(r['lost'])}</td>"
            f"<td class='num'>{_pct(r['win_rate'])}</td>"
            f"<td class='num'>{_num(r['profit_factor'])}</td>"
            f"<td class='num {pnl_cls}'>{'+' if pnl >= 0 else ''}{_money(pnl)}</td>"
            f"<td class='num {ret_cls}'>{_pct(r['total_return'])}</td>"
            f"<td class='num'>{_num(r['sharpe'])}</td>"
            f"<td class='num down'>{_pct(r['max_drawdown'])}</td></tr>"
        )
    return (
        "<table class='grid'><thead><tr><th>run id</th><th>engine</th><th>trades</th>"
        "<th>won</th><th>lost</th>"
        "<th>win rate</th><th>profit factor</th><th>return p&amp;l</th><th>total return</th><th>sharpe</th>"
        f"<th>max dd</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


_CSS = """
:root{--bg:#0f1216;--panel:#171b21;--panel2:#1c2128;--line:#2a313b;--fg:#e6edf3;
--muted:#8b949e;--up:#3fb950;--down:#f85149;--accent:#58a6ff}
@media(prefers-color-scheme:light){:root{--bg:#f6f8fa;--panel:#fff;--panel2:#f0f3f6;
--line:#d0d7de;--fg:#1f2328;--muted:#636c76;--up:#1a7f37;--down:#cf222e;--accent:#0969da}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
margin:36px 0 12px;font-weight:600}
h3{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
margin:22px 0 10px;font-weight:600}
.runcard{border-top:2px solid var(--line);margin-top:44px;padding-top:4px;scroll-margin-top:16px}
a.mono{color:var(--accent);text-decoration:none}
a.mono:hover{text-decoration:underline}
.backlink{font-size:12px;font-weight:400;margin-left:14px;color:var(--accent);text-decoration:none}
.backlink:hover{text-decoration:underline}
.sub{color:var(--muted);margin:0 0 8px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.chip{background:var(--panel2);border:1px solid var(--line);border-radius:999px;
padding:3px 12px;font-size:12px}
.chip b{color:var(--fg)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile-v{font-size:22px;font-weight:700;letter-spacing:-.01em}
.tile-l{color:var(--muted);font-size:12px;margin-top:2px;text-transform:uppercase;letter-spacing:.04em}
.tile-s{color:var(--muted);font-size:11px;margin-top:6px;font-variant-numeric:tabular-nums}
.up{color:var(--up)}.down{color:var(--down)}
.tblscroll{overflow-x:auto;border:1px solid var(--line);border-radius:12px}
table.grid{border-collapse:collapse;width:100%;font-size:13px}
table.grid th{position:sticky;top:0;background:var(--panel2);text-align:left;
padding:9px 12px;color:var(--muted);font-weight:600;white-space:nowrap;border-bottom:1px solid var(--line);z-index:1}
table.grid td{padding:8px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
table.grid tbody tr:nth-child(even){background:color-mix(in srgb,var(--fg) 3%,transparent)}
table.grid tbody tr:hover{background:var(--panel2)}
tr.row-win td:first-child{box-shadow:inset 3px 0 var(--up)}
tr.row-loss td:first-child{box-shadow:inset 3px 0 var(--down)}
tr.row-flat td:first-child{box-shadow:inset 3px 0 var(--muted)}
.num{text-align:right;font-variant-numeric:tabular-nums}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.reason{color:var(--muted);max-width:260px;overflow:hidden;text-overflow:ellipsis}
.pill{padding:1px 9px;border-radius:999px;font-size:11px;font-weight:600;text-transform:uppercase}
.pill.win{background:color-mix(in srgb,var(--up) 18%,transparent);color:var(--up)}
.pill.loss{background:color-mix(in srgb,var(--down) 18%,transparent);color:var(--down)}
.pill.flat{background:var(--panel2);color:var(--muted)}
.muted{color:var(--muted)}
.equity{width:100%;height:auto;background:var(--panel);border:1px solid var(--line);border-radius:12px}
.grid{stroke:var(--line);stroke-width:1;opacity:.5}
.baseline{stroke:var(--accent);stroke-dasharray:4 4;opacity:.7}
.ddband{fill:var(--down);opacity:.08}
.ddpt{fill:var(--down)}.peakpt{fill:var(--up)}
.ddlabel{fill:var(--down);font-size:11px;text-anchor:middle}
.ytick{fill:var(--muted);font-size:10.5px;text-anchor:end;font-variant-numeric:tabular-nums}
.ytick.base{fill:var(--accent)}
.xtick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.svglabel{fill:var(--muted);font-size:12px}
.tag{margin-left:8px;padding:1px 7px;border-radius:999px;font-size:10px;font-weight:700;
text-transform:uppercase;letter-spacing:.04em;vertical-align:middle}
.tag.best{background:color-mix(in srgb,var(--up) 20%,transparent);color:var(--up)}
.tag.cur{background:color-mix(in srgb,var(--accent) 20%,transparent);color:var(--accent)}
.sharecell{min-width:220px}
.barwrap{display:flex;align-items:center;gap:8px}
.barwrap .bar{height:10px;background:var(--down);border-radius:3px;min-width:2px}
.barwrap span{color:var(--muted);font-size:12px}
tr.cur{background:color-mix(in srgb,var(--accent) 14%,transparent)}
tr.cur td:first-child{box-shadow:inset 3px 0 var(--accent)}
footer{margin-top:48px;color:var(--muted);font-size:12px;text-align:center}
"""


def _run_section(run_dir: Path, anchored: bool = False) -> str:
    """One run's full detail: header, scorecard, equity, ledger, buckets."""
    meta, trades, net = load_run(run_dir)
    a = aggregate(trades, net)
    buckets = failure_buckets(trades)
    rid = meta["run_id"]

    chips = "".join(
        f"<span class='chip'>{escape(k)} <b>{escape(str(v))}</b></span>"
        for k, v in [
            ("engine", meta["engine"]),
            ("symbols", ", ".join(meta["symbols"])),
            ("period", f"{str(meta['start'])[:10]} → {str(meta['end'])[:10]}"),
            ("cost", f"{meta['cost_bps']} bps"),
            ("notional", _money(meta["notional"])),
        ]
    )
    aid = f" id='{_run_anchor(rid)}'" if anchored else ""
    back = "<a class='backlink' href='#allruns'>← all runs</a>" if anchored else ""
    return f"""
  <section class="runcard"{aid}>
  <h2 class="runhead">{escape(str(meta['engine']))} · <span class="mono">{escape(rid)}</span>{back}</h2>
  <div class="meta">{chips}</div>

  <h3>Scorecard</h3>
  {_scorecard(a, trades, meta["notional"])}

  <h3>Equity curve</h3>
  {_equity_svg(net)}

  <h3>Trade ledger · {len(trades)} trades</h3>
  <div class="tblscroll">{_trades_table(trades)}</div>

  <h3>Failure buckets · where losses concentrate</h3>
  <div class="tblscroll">{_buckets_table(buckets)}</div>
  </section>"""


def _page(title: str, body: str, footer: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title>
<style>{_CSS}</style></head><body><div class="wrap">
  <h1>Trading Dashboard</h1>
  <p class="sub"><a href="{_ACTIONS_URL}">
    <img src="{_ACTIONS_URL}/badge.svg?branch=main" alt="daily paper-run status"></a>
    — live status of the daily paper-run (click for logs)</p>
  {body}
  <footer>{footer}</footer>
</div></body></html>"""


def render(run_dir: Path, base_dir: Path) -> str:
    meta = load_run(run_dir)[0]
    comparison = compare_runs(base_dir)
    body = (
        _run_section(run_dir)
        + f"\n  <h2>Run comparison · {len(comparison)} runs</h2>"
        + f"\n  <div class='tblscroll'>{_compare_table(comparison, meta['run_id'])}</div>"
    )
    return _page(
        f"Paper-trade dashboard — {meta['run_id']}", body,
        f"Generated from {escape(str(run_dir))} · rhagent paper-trade harness",
    )


# ── Trading Control Room: unified dashboard ─────────────────────────────────
#
# Data is computed server-side into one JSON blob (schema below) and rendered
# client-side by a small vanilla-JS layer — chart mode toggle, run table sort/
# filter, trade filter, and a per-run detail drawer. No React, no build step.

def _forward_leg(eval_dir: Path) -> dict:
    if not (eval_dir / "run.json").exists():
        return {
            "engine": "", "symbols": [], "start": "", "end": "", "days": 0,
            "ret": 0.0, "pnl": 0.0, "notional": 10_000.0, "sharpe": 0.0, "dd": 0.0,
        }
    meta, trades, net = load_run(eval_dir)
    a = aggregate(trades, net)
    notional = float(meta.get("notional", 10_000.0))
    return {
        "engine": meta.get("engine", ""), "symbols": meta.get("symbols", []),
        "start": str(meta.get("start", ""))[:10], "end": str(meta.get("end", ""))[:10],
        "days": len(net), "ret": a["total_return"],
        "pnl": _return_pnl(a["total_return"], notional),
        "notional": notional, "sharpe": a["sharpe"], "dd": a["max_drawdown"],
    }


def _locked_run(base_dir: Path, runs: list[Path], cfg) -> Path:
    """The run matching the locked strategy config (config.yaml `strategy:`),
    most recent first. Falls back to the latest run if config has no match."""
    if cfg.strategy is not None:
        candidates = []
        for run_dir in runs:
            meta = load_run(run_dir)[0]
            if meta.get("engine") == cfg.strategy.name and meta.get("overlay", "none") == cfg.strategy.overlay:
                candidates.append((str(meta["run_id"]), run_dir))
        if candidates:
            return max(candidates, key=lambda c: c[0])[1]
    return runs[-1]


def _runs_data(base_dir: Path) -> list[dict]:
    rows = []
    for run_dir in _run_dirs(base_dir):
        meta, trades, net = load_run(run_dir)
        a = aggregate(trades, net)
        pnl_series = trades["pnl_abs"].astype(float) if len(trades) else pd.Series(dtype=float)
        won = int((trades["outcome"] == "win").sum()) if len(trades) else 0
        lost = int((trades["outcome"] == "loss").sum()) if len(trades) else 0
        symbols = meta.get("symbols", [])
        uni = ", ".join(symbols) if len(symbols) <= 5 else f"universe ({len(symbols)})"
        rid = str(meta["run_id"])
        notional = float(meta.get("notional", 10_000.0))
        rows.append({
            "id": rid, "engine": meta.get("engine", ""), "overlay": meta.get("overlay", "") or "",
            "notional": notional, "start": str(meta.get("start", ""))[:10], "end": str(meta.get("end", ""))[:10],
            "n": a["n_trades"], "won": won, "lost": lost,
            "pnl": _return_pnl(a["total_return"], notional),
            "gw": float(pnl_series[pnl_series > 0].sum()), "gl": float(-pnl_series[pnl_series < 0].sum()),
            "pf": a["profit_factor"], "ret": a["total_return"], "wr": a["win_rate"],
            "uni": uni, "nsym": len(symbols), "sid": rid[:10] + "·" + rid.split("-")[-1],
        })
    return rows


def _win_score(run_dir: Path) -> dict:
    meta, trades, net = load_run(run_dir)
    a = aggregate(trades, net)
    pnl_series = trades["pnl_abs"].astype(float) if len(trades) else pd.Series(dtype=float)
    won = int((trades["outcome"] == "win").sum()) if len(trades) else 0
    lost = int((trades["outcome"] == "loss").sum()) if len(trades) else 0
    gw = float(pnl_series[pnl_series > 0].sum())
    gl = float(-pnl_series[pnl_series < 0].sum())
    notional = float(meta.get("notional", 10_000.0))
    pnl = _return_pnl(a["total_return"], notional)
    return {
        "notional": notional, "ret": a["total_return"], "pnl": pnl, "balance": notional + pnl,
        "n": a["n_trades"], "won": won, "lost": lost, "wr": a["win_rate"], "pf": a["profit_factor"],
        "avgWin": a["avg_win"], "avgLoss": a["avg_loss"], "sharpe": a["sharpe"], "dd": a["max_drawdown"],
        "avgHold": a["avg_holding_bars"], "gw": gw, "gl": gl,
    }


def _win_trades(run_dir: Path) -> list[dict]:
    _, trades, _ = load_run(run_dir)
    rows = []
    for i, t in enumerate(trades.itertuples(), start=1):
        rows.append({
            "id": f"{i:04d}", "sym": t.symbol, "side": t.side,
            "entry": str(t.entry_ts)[:10], "in": float(t.entry_price),
            "exit": str(t.exit_ts)[:10], "out": float(t.exit_price),
            "bars": int(t.holding_bars), "pnlPct": float(t.pnl_pct),
            "pnl": float(t.pnl_abs), "oc": t.outcome,
        })
    return rows


def _cross_run_buckets(base_dir: Path) -> tuple[list[dict], dict, list[dict], dict]:
    """Top loss + win buckets ranked across every bucketing dimension
    (vol, gap, holding, symbol, side, dow, near_high, ...), over every
    archived run."""
    frames = [load_run(d)[1] for d in _run_dirs(base_dir)]
    frames = [f for f in frames if len(f)]
    if not frames:
        return [], {}, [], {}
    trades = flatten_trades(pd.concat(frames, ignore_index=True))
    n_total = len(trades)
    n_losses = int((trades["outcome"] == "loss").sum())
    n_wins = int((trades["outcome"] == "win").sum())
    src = f"every paper-trade run archived so far ({len(frames)} runs)"
    rows = []
    for dim, labels in _bucket_labels(trades).items():
        for bucket, idx in trades.groupby(labels).groups.items():
            sub = trades.loc[idx]
            loss_n = int((sub["outcome"] == "loss").sum())
            win_n = int((sub["outcome"] == "win").sum())
            wr = float((sub["outcome"] == "win").mean()) if len(sub) else 0.0
            rows.append({"dim": dim, "bucket": str(bucket), "lossN": loss_n, "winN": win_n,
                         "totalN": int(len(sub)), "wr": wr,
                         "loss_share": loss_n / n_losses if n_losses else 0.0,
                         "win_share": win_n / n_wins if n_wins else 0.0})
    loss_rows = [{**r, "share": r["loss_share"]} for r in sorted(rows, key=lambda r: -r["loss_share"])[:5]]
    win_rows = [{**r, "share": r["win_share"]} for r in sorted(rows, key=lambda r: -r["win_share"])[:5]]
    loss_meta = {"n": n_total, "losses": n_losses, "runs": len(frames), "source": src,
                 "caveat": "Measured across every archived paper-trade run, not just the locked config."}
    win_meta = {"n": n_total, "wins": n_wins, "source": src,
                "note": "Top buckets across all dimensions, ranked by share of total wins."}
    return loss_rows, loss_meta, win_rows, win_meta


def _bakeoff_data(base_dir: Path, engine: str) -> list[dict]:
    df = robust_table(base_dir)
    if len(df) == 0 or not engine:
        return []
    df = df[df["engine"] == engine]
    if len(df) == 0:
        return []
    order = {"none": 0, "conviction": 1, "bucket": 2, "winprob": 3}
    rows = []
    for overlay, grp in df.groupby(df["overlay"].fillna("none").replace("", "none")):
        best = grp.loc[grp["deflated"].idxmax()]
        rows.append({
            "overlay": overlay, "point": float(best["point_sharpe"]), "deflated": float(best["deflated"]),
            "fold": f"{best['fold_mean']:.2f}±{best['fold_std']:.2f}",
            "ci": f"[{best['ci_lo']:.2f}, {best['ci_hi']:.2f}]",
            "beats": bool(best["beats_baseline"]),
        })
    rows.sort(key=lambda r: order.get(r["overlay"], 99))
    return rows


def _equity_curve(run_dir: Path) -> tuple[list[float], list[str]]:
    _, _, net = load_run(run_dir)
    equity = (1.0 + net.astype(float)).cumprod()
    return [float(v) for v in equity.tolist()], [str(d)[:10] for d in equity.index]


def _agent_reflections() -> list[str]:
    memory = read_memory()
    if not memory:
        return []
    return memory.split("\n## ")[1:][-3:]


def _build_control_room_data(base_dir: Path) -> dict:
    runs = _run_dirs(base_dir)
    if not runs:
        raise SystemExit(f"no runs found under {base_dir} — run rhagent.papertrade first")
    cfg = load_config()
    locked_dir = _locked_run(base_dir, runs, cfg)

    forward_dir = base_dir.parent / "forward"
    curve_vals, curve_dates = _equity_curve(locked_dir)
    loss_buckets, loss_meta, win_buckets, win_meta = _cross_run_buckets(base_dir)
    g = cfg.limits

    return {
        "updated": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
        "winRunId": str(load_run(locked_dir)[0]["run_id"]),
        "lockedEngine": cfg.strategy.name if cfg.strategy else "",
        "lockedOverlay": cfg.strategy.overlay if cfg.strategy else "",
        "guardrails": {
            "per_trade_max_usd": g.per_trade_max_usd, "total_deployed_max_usd": g.total_deployed_max_usd,
            "max_new_positions_per_run": g.max_new_positions_per_run, "max_orders_per_run": g.max_orders_per_run,
            "daily_loss_limit_usd": g.daily_loss_limit_usd, "live": not cfg.dry_run,
            "halt": HALT_FILE.exists(), "model": cfg.agent.model,
        },
        "forward": {"agent": _forward_leg(forward_dir / "agent"), "baseline": _forward_leg(forward_dir / "mean_reversion")},
        "bakeoff": _bakeoff_data(base_dir, cfg.strategy.name if cfg.strategy else ""),
        "curveDaily": curve_vals, "curveDates": curve_dates,
        "runs": _runs_data(base_dir),
        "winScore": _win_score(locked_dir),
        "winTrades": _win_trades(locked_dir),
        "buckets": loss_buckets, "bucketsMeta": loss_meta,
        "winBuckets": win_buckets, "winBucketsMeta": win_meta,
        "runbook": [list(x) for x in _RUNBOOK],
        "lessons": lessons_from_runs(base_dir) or "",
        "reflections": _agent_reflections(),
        "actionsUrl": _ACTIONS_URL,
    }


_CONTROL_ROOM_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  *{box-sizing:border-box}
  html,body{margin:0}
  body{background:#0a0c10}
  ::selection{background:rgba(5,196,107,.25)}
  a{color:#4db8ff;text-decoration:none}
  a:hover{color:#7fcfff}
  @keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
  @keyframes drawerIn{from{transform:translateX(24px);opacity:0}to{transform:none;opacity:1}}
  @keyframes backdropIn{from{opacity:0}to{opacity:1}}
  .cr-scroll::-webkit-scrollbar{height:9px;width:9px}
  .cr-scroll::-webkit-scrollbar-thumb{background:#2b333f;border-radius:6px}
  .cr-scroll::-webkit-scrollbar-track{background:transparent}
  .cr-row:hover{background:var(--panel2)}
  .cr-btn{border:none;cursor:pointer;font-family:'IBM Plex Sans',sans-serif}
</style>
</head>
<body>
<div id="cr-root" style="--bg:#0a0c10;--panel:#12161c;--panel2:#171d25;--line:#232a34;--line2:#2e3742;--fg:#e8edf4;--muted:#828d9b;--up:#05c46b;--down:#ff5c5c;--accent:#4db8ff;--warn:#ffb020;--purple:#b388ff;min-height:100vh;background:radial-gradient(1200px 600px at 78% -8%,rgba(77,184,255,.06),transparent 60%),var(--bg);color:var(--fg);font-family:'IBM Plex Sans',system-ui,sans-serif;font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased">

  <header style="position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:18px;flex-wrap:wrap;padding:14px 26px;background:rgba(10,12,16,.82);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)">
    <div style="display:flex;align-items:center;gap:11px">
      <div style="width:26px;height:26px;border-radius:7px;background:linear-gradient(135deg,var(--up),#00a35a);display:flex;align-items:center;justify-content:center;box-shadow:0 0 0 1px rgba(5,196,107,.35),0 4px 14px rgba(5,196,107,.25)">
        <div style="width:9px;height:9px;background:#04120b;transform:rotate(45deg);border-radius:1px"></div>
      </div>
      <div style="line-height:1.15">
        <div style="font-weight:700;letter-spacing:-.01em;font-size:15px">RHAGENT<span style="color:var(--muted);font-weight:500"> · Trading Control Room</span></div>
        <div style="font-size:11px;color:var(--muted);font-family:'IBM Plex Mono',monospace">autonomous US-equities agent · guardrail-enforced</div>
      </div>
    </div>
    <div style="flex:1"></div>
    <div id="cr-headerpills" style="display:flex;align-items:center;gap:8px;flex-wrap:wrap"></div>
  </header>

  <main style="max-width:1240px;margin:0 auto;padding:26px 26px 90px">

    <section style="animation:fadeUp .4s ease both">
      <div style="display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:12px">
        <h2 style="margin:0;font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600">The verdict · agent vs baseline</h2>
        <span style="font-size:11px;color:var(--muted);font-family:'IBM Plex Mono',monospace">forward paper track</span>
      </div>
      <div id="cr-verdict"></div>
    </section>

    <section style="margin-top:30px">
      <div id="cr-kpis" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px"></div>
    </section>

    <section style="margin-top:30px;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px 22px">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:14px;flex-wrap:wrap;margin-bottom:6px">
        <div>
          <h3 style="margin:0;font-size:15px;font-weight:600">Equity curve · locked strategy candidate</h3>
          <div id="cr-chartsub" style="font-size:12px;color:var(--muted);margin-top:3px"></div>
        </div>
        <div id="cr-chartmodes" style="display:flex;gap:5px;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:3px"></div>
      </div>
      <div style="display:flex;gap:18px;flex-wrap:wrap;margin-bottom:8px">
        <span style="display:inline-flex;align-items:center;gap:6px;font-size:11.5px;color:var(--muted);font-family:'IBM Plex Mono',monospace"><span style="width:16px;height:3px;background:var(--up);border-radius:2px"></span>equity</span>
        <span style="display:inline-flex;align-items:center;gap:6px;font-size:11.5px;color:var(--muted);font-family:'IBM Plex Mono',monospace"><span style="width:16px;height:0;border-top:2px dashed var(--accent)"></span>break-even</span>
        <span style="display:inline-flex;align-items:center;gap:6px;font-size:11.5px;color:var(--muted);font-family:'IBM Plex Mono',monospace"><span style="width:12px;height:12px;border-radius:50%;background:rgba(255,92,92,.12);border:1px solid var(--down)"></span>max drawdown window</span>
      </div>
      <div id="cr-chart"></div>
    </section>

    <section style="margin-top:30px;display:grid;grid-template-columns:minmax(0,0.9fr) minmax(0,1.1fr);gap:20px">
      <div style="background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px 22px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
          <h3 style="margin:0;font-size:15px;font-weight:600">Guardrails · armed</h3>
          <span style="display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:700;color:var(--up);font-family:'IBM Plex Mono',monospace"><span style="width:7px;height:7px;border-radius:50%;background:var(--up);box-shadow:0 0 0 3px rgba(5,196,107,.18)"></span>0 BREACHES</span>
        </div>
        <div style="font-size:12px;color:var(--muted);margin-bottom:16px">Hard caps enforced in code — the model cannot talk its way past them.</div>
        <div id="cr-guardrails" style="display:flex;flex-direction:column;gap:14px"></div>
        <div id="cr-guardrail-chips" style="display:flex;gap:7px;flex-wrap:wrap;margin-top:18px;padding-top:16px;border-top:1px solid var(--line)"></div>
      </div>

      <div style="background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px 22px">
        <h3 style="margin:0 0 4px;font-size:15px;font-weight:600">Overlay bake-off · robust Sharpe</h3>
        <div style="font-size:12px;color:var(--muted);margin-bottom:16px">A variant beats baseline only if its 95% CI lower bound clears the baseline Sharpe.</div>
        <div class="cr-scroll" style="overflow-x:auto">
          <table style="width:100%;border-collapse:collapse;font-size:12.5px">
            <thead>
              <tr style="color:var(--muted);text-align:left">
                <th style="padding:8px 10px;font-weight:600;border-bottom:1px solid var(--line)">overlay</th>
                <th style="padding:8px 10px;font-weight:600;border-bottom:1px solid var(--line);text-align:right">point</th>
                <th style="padding:8px 10px;font-weight:600;border-bottom:1px solid var(--line);text-align:right">deflated</th>
                <th style="padding:8px 10px;font-weight:600;border-bottom:1px solid var(--line);text-align:right">fold ±sd</th>
                <th style="padding:8px 10px;font-weight:600;border-bottom:1px solid var(--line);text-align:right">95% CI</th>
                <th style="padding:8px 10px;font-weight:600;border-bottom:1px solid var(--line);text-align:center">vs base</th>
              </tr>
            </thead>
            <tbody id="cr-bakeoff"></tbody>
          </table>
        </div>
      </div>
    </section>

    <section style="margin-top:30px;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px 22px">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:14px;flex-wrap:wrap;margin-bottom:14px">
        <div>
          <h3 id="cr-runcount" style="margin:0;font-size:15px;font-weight:600"></h3>
          <div style="font-size:12px;color:var(--muted);margin-top:3px">Every paper-trade run in the archive. Click a column to sort, a row to open.</div>
        </div>
        <div id="cr-enginechips" style="display:flex;gap:5px;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:3px;flex-wrap:wrap"></div>
      </div>
      <div class="cr-scroll" style="overflow-x:auto;border:1px solid var(--line);border-radius:12px">
        <div style="min-width:760px">
          <div id="cr-runcols" style="display:grid;grid-template-columns:minmax(150px,1.5fr) 1fr 0.9fr 0.65fr 0.7fr 0.6fr 1fr 1.25fr;background:var(--panel2);border-bottom:1px solid var(--line)"></div>
          <div id="cr-runrows"></div>
        </div>
      </div>
    </section>

    <section style="margin-top:30px">
      <h3 style="margin:0 0 14px;font-size:15px;font-weight:600">Engine leaderboard <span style="color:var(--muted);font-weight:400;font-size:13px">· best P&amp;L per strategy family</span></h3>
      <div id="cr-leaderboard" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px"></div>
    </section>

    <section style="margin-top:30px;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px 22px">
      <div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:16px">
        <h3 style="margin:0;font-size:15px;font-weight:600">Locked-config scorecard</h3>
        <span id="cr-winid" style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted)"></span>
        <span style="padding:2px 9px;border-radius:6px;background:rgba(5,196,107,.14);color:var(--up);font-size:11px;font-weight:700;font-family:'IBM Plex Mono',monospace">FORWARD CANDIDATE</span>
      </div>
      <div id="cr-scoretiles" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:11px"></div>
    </section>

    <section style="margin-top:30px;display:grid;grid-template-columns:minmax(0,0.85fr) minmax(0,1.15fr);gap:20px;align-items:stretch">
      <div style="display:flex;flex-direction:column;gap:20px;height:100%">
        <div style="background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px 22px">
          <h3 style="margin:0 0 4px;font-size:15px;font-weight:600">Where losses concentrate</h3>
          <div style="font-size:12px;color:var(--muted);margin-bottom:16px">Very short exits and very long holds both underperform.</div>
          <div id="cr-buckets" style="display:flex;flex-direction:column;gap:14px"></div>
          <div id="cr-bucketscaveat" style="margin-top:16px;padding:11px 13px;background:rgba(255,176,32,.07);border:1px solid rgba(255,176,32,.22);border-radius:10px;font-size:11.5px;color:var(--muted);text-wrap:pretty"></div>
        </div>
        <div style="background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px 22px">
          <h3 style="margin:0 0 4px;font-size:15px;font-weight:600">Where we win</h3>
          <div style="font-size:12px;color:var(--muted);margin-bottom:16px">Top buckets across all dimensions.</div>
          <div id="cr-winbuckets" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:14px"></div>
          <div id="cr-winbucketsnote" style="padding:11px 13px;background:rgba(5,196,107,.06);border:1px solid rgba(5,196,107,.2);border-radius:10px;font-size:11.5px;color:var(--muted);text-wrap:pretty"></div>
        </div>
      </div>

      <div style="background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px 22px">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:14px">
          <div>
            <h3 style="margin:0;font-size:15px;font-weight:600">Trade ledger</h3>
            <div id="cr-ledgercount" style="font-size:12px;color:var(--muted);margin-top:3px"></div>
          </div>
          <div id="cr-tradechips" style="display:flex;gap:5px;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:3px"></div>
        </div>
        <div class="cr-scroll" style="overflow-x:auto">
          <table style="width:100%;border-collapse:collapse;font-size:12px;min-width:520px">
            <thead>
              <tr style="color:var(--muted);text-align:right">
                <th style="padding:8px 9px;font-weight:600;border-bottom:1px solid var(--line);text-align:left">#</th>
                <th style="padding:8px 9px;font-weight:600;border-bottom:1px solid var(--line);text-align:left">sym</th>
                <th style="padding:8px 9px;font-weight:600;border-bottom:1px solid var(--line);text-align:left">entry → exit</th>
                <th style="padding:8px 9px;font-weight:600;border-bottom:1px solid var(--line)">bars</th>
                <th style="padding:8px 9px;font-weight:600;border-bottom:1px solid var(--line)">pnl %</th>
                <th style="padding:8px 9px;font-weight:600;border-bottom:1px solid var(--line)">pnl $</th>
                <th style="padding:8px 9px;font-weight:600;border-bottom:1px solid var(--line);text-align:right;min-width:90px">weight</th>
              </tr>
            </thead>
            <tbody id="cr-ledger"></tbody>
          </table>
        </div>
      </div>
    </section>

    <section style="margin-top:30px;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px 22px">
      <h3 style="margin:0 0 4px;font-size:15px;font-weight:600">Runbook</h3>
      <div style="font-size:12px;color:var(--muted);margin-bottom:16px">Every command that drives this system. Click to copy.</div>
      <div id="cr-runbook" style="display:flex;flex-direction:column;gap:8px"></div>
    </section>

    <section id="cr-agentnotes" style="margin-top:30px;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px 22px;display:none">
      <h3 style="margin:0 0 4px;font-size:15px;font-weight:600">Agent's own lessons (self-written)</h3>
      <div style="font-size:12px;color:var(--muted);margin-bottom:12px">Reflections the agent journaled after past runs.</div>
      <div id="cr-reflections" style="display:flex;flex-direction:column;gap:10px;font-size:12.5px;color:var(--fg)"></div>
    </section>

    <footer style="margin-top:40px;padding-top:20px;border-top:1px solid var(--line);display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;color:var(--muted);font-size:11.5px;font-family:'IBM Plex Mono',monospace">
      <span>rhagent trading harness · journal/papertrade + journal/forward</span>
      <span>numbers reproduced from rhagent.evaluate · not investment advice</span>
    </footer>
  </main>

  <div id="cr-drawerwrap"></div>
</div>

<script>
const DATA = __DATA_JSON__;
const ACTIONS_URL = DATA.actionsUrl;

const ST = { chartMode: 'cum', hoverIdx: null, engine: 'all', runSort: 'pnl', runDir: -1, tradeFilter: 'all', copied: -1, selectedRun: null };

function money(x, dp = 2) { const s = x < 0 ? '-' : ''; return s + '$' + Math.abs(x).toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp }); }
function pct(x, dp = 2) { return (x >= 0 ? '+' : '') + (x * 100).toFixed(dp) + '%'; }
function pctAbs(x, dp = 1) { return (x * 100).toFixed(dp) + '%'; }
function num(x) { return x >= 999 ? '∞' : x.toFixed(2); }
function engineColor(e) { return { mean_reversion: 'var(--accent)', momentum: 'var(--warn)', linreg: 'var(--purple)', agent: 'var(--up)' }[e] || 'var(--muted)'; }
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

function renderHeaderPills() {
  const g = DATA.guardrails;
  const el = document.getElementById('cr-headerpills');
  el.innerHTML = `
    <span style="display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border-radius:999px;background:${g.live ? 'rgba(5,196,107,.1)' : 'rgba(255,176,32,.12)'};border:1px solid ${g.live ? 'rgba(5,196,107,.28)' : 'rgba(255,176,32,.3)'};color:${g.live ? 'var(--up)' : 'var(--warn)'};font-size:11px;font-weight:600;font-family:'IBM Plex Mono',monospace"><span style="width:6px;height:6px;border-radius:50%;background:currentColor"></span>${g.live ? 'LIVE · TRADING' : 'PAPER · DRY-RUN'}</span>
    <span style="display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border-radius:999px;background:${g.halt ? 'rgba(255,92,92,.12)' : 'rgba(5,196,107,.1)'};border:1px solid ${g.halt ? 'var(--down)' : 'rgba(5,196,107,.28)'};color:${g.halt ? 'var(--down)' : 'var(--up)'};font-size:11px;font-weight:600;font-family:'IBM Plex Mono',monospace">HALT · ${g.halt ? 'SET' : 'CLEAR'}</span>
    <a href="${ACTIONS_URL}" style="display:inline-flex"><img src="${ACTIONS_URL}/badge.svg?branch=main" alt="daily paper-run status"></a>
    <span style="padding:5px 11px;border-radius:999px;background:var(--panel2);border:1px solid var(--line);color:var(--muted);font-size:11px;font-family:'IBM Plex Mono',monospace">upd ${esc(DATA.updated)}</span>`;
}

function renderVerdict() {
  const f = DATA.forward, agent = f.agent, base = f.baseline;
  const days = agent.days || base.days || 0;
  let badge = 'TOO EARLY TO CALL', note = `Forward track has ${days} day(s) logged. Verdict needs weeks of OOS data.`;
  if (days >= 5) {
    if (agent.pnl > base.pnl) { badge = 'AGENT LEADS'; note = 'Agent forward P&L ahead of the mean-reversion baseline over the tracked window.'; }
    else if (base.pnl > agent.pnl) { badge = 'BASELINE LEADS'; note = 'Baseline forward P&L ahead of the agent over the tracked window.'; }
    else { badge = 'TIED'; note = 'Agent and baseline forward P&L are tied so far.'; }
  }
  const sub = leg => `${leg.days} day${leg.days === 1 ? '' : 's'} · ${(leg.symbols || []).join(', ') || '—'} · ${leg.ret >= 0 ? 'up' : 'down'}`;
  document.getElementById('cr-verdict').innerHTML = `
    <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:0;align-items:stretch;background:var(--panel);border:1px solid var(--line);border-radius:16px;overflow:hidden">
      <div style="padding:22px 26px;display:flex;flex-direction:column;gap:6px">
        <div style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);font-weight:600">LLM Agent</div>
        <div style="font-size:34px;font-weight:700;font-family:'IBM Plex Mono',monospace;letter-spacing:-.02em">${money(agent.pnl)}</div>
        <div style="font-size:12px;color:var(--muted);font-family:'IBM Plex Mono',monospace">${sub(agent)}</div>
      </div>
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;padding:22px 30px;background:var(--panel2);border-left:1px solid var(--line);border-right:1px solid var(--line);min-width:220px">
        <div style="padding:6px 16px;border-radius:999px;background:rgba(255,176,32,.12);border:1px solid rgba(255,176,32,.35);color:var(--warn);font-weight:700;font-size:13px;letter-spacing:.03em">${badge}</div>
        <div style="font-size:12px;color:var(--muted);text-align:center;max-width:210px;text-wrap:pretty">${esc(note)}</div>
      </div>
      <div style="padding:22px 26px;display:flex;flex-direction:column;gap:6px;text-align:right">
        <div style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--purple);font-weight:600">Mean-Reversion Baseline</div>
        <div style="font-size:34px;font-weight:700;font-family:'IBM Plex Mono',monospace;letter-spacing:-.02em">${money(base.pnl)}</div>
        <div style="font-size:12px;color:var(--muted);font-family:'IBM Plex Mono',monospace">${sub(base)}</div>
      </div>
    </div>
    <div style="margin-top:10px;display:flex;align-items:center;gap:10px;padding:12px 16px;background:rgba(5,196,107,.06);border:1px solid rgba(5,196,107,.22);border-radius:12px">
      <span style="width:8px;height:8px;border-radius:50%;background:var(--up);flex:none;box-shadow:0 0 0 4px rgba(5,196,107,.15)"></span>
      <div style="font-size:13px;color:var(--fg)"><b style="color:var(--up)">Research winner locked:</b> ${esc(DATA.lockedEngine)}, gated by the <b>${esc(DATA.lockedOverlay || 'no')}</b> overlay. This is the config the forward track is now paper-trading.</div>
    </div>`;
}

function renderKpis() {
  const S = DATA.winScore, f = DATA.forward.agent;
  const kpis = [
    { label: 'Forward balance', value: money(f.notional + f.pnl), color: 'var(--fg)', bar: 'var(--accent)', sub: `${f.days} day(s) tracked · net ${pct(f.ret)}` },
    { label: 'Locked-config return', value: pct(S.ret), color: 'var(--up)', bar: 'var(--up)', sub: `${esc(DATA.lockedEngine)} + ${esc(DATA.lockedOverlay)}` },
    { label: 'Win rate', value: pctAbs(S.wr, 1), color: 'var(--up)', bar: 'var(--up)', sub: `${S.won}W / ${S.lost}L` },
    { label: 'Profit factor', value: S.pf.toFixed(2), color: 'var(--up)', bar: 'var(--up)', sub: `${money(S.gw, 0)} / ${money(S.gl, 0)}` },
    { label: 'Max drawdown', value: pctAbs(S.dd, 2), color: 'var(--down)', bar: 'var(--down)', sub: `${S.avgHold.toFixed(1)} bar avg hold` },
    { label: 'System health', value: DATA.guardrails.halt ? 'HALTED' : 'ARMED', color: DATA.guardrails.halt ? 'var(--down)' : 'var(--up)', bar: DATA.guardrails.halt ? 'var(--down)' : 'var(--up)', sub: '5 guardrails · 0 breaches' },
  ];
  document.getElementById('cr-kpis').innerHTML = kpis.map(k => `
    <div style="background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px;position:relative;overflow:hidden">
      <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:${k.bar}"></div>
      <div style="font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600">${esc(k.label)}</div>
      <div style="font-size:27px;font-weight:700;font-family:'IBM Plex Mono',monospace;letter-spacing:-.02em;margin-top:6px;color:${k.color}">${esc(k.value)}</div>
      <div style="font-size:11.5px;color:var(--muted);margin-top:5px;font-family:'IBM Plex Mono',monospace">${esc(k.sub)}</div>
    </div>`).join('');
}

function buildChartSvg() {
  const ys = DATA.curveDaily, dates = DATA.curveDates, n = ys.length;
  if (!n) return '<p style="color:var(--muted)">no equity series</p>';
  const W = 960, H = 320, padL = 54, padR = 18, padT = 20, padB = 30;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const mode = ST.chartMode;
  const vals = mode === 'cum' ? ys : ys.map(v => (v - 1) * 100);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const m = (hi - lo) * 0.14 || 0.01; lo -= m; hi += m; const span = hi - lo || 1;
  const px = i => padL + plotW * (i / Math.max(n - 1, 1));
  const py = v => padT + plotH * (1 - (v - lo) / span);
  let grid = '', yl = '';
  for (let k = 0; k < 5; k++) {
    const v = lo + span * k / 4, y = py(v);
    grid += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#232a34" stroke-width="1" opacity="0.55"/>`;
    yl += `<text x="${padL - 9}" y="${y + 3.5}" text-anchor="end" fill="#828d9b" font-size="11" font-family="'IBM Plex Mono',monospace">${mode === 'cum' ? v.toFixed(2) + '×' : v.toFixed(0) + '%'}</text>`;
  }
  const line = vals.map((v, i) => px(i).toFixed(1) + ',' + py(v).toFixed(1)).join(' ');
  const area = px(0).toFixed(1) + ',' + py(lo).toFixed(1) + ' ' + line + ' ' + px(n - 1).toFixed(1) + ',' + py(lo).toFixed(1);
  const stroke = ys[n - 1] >= 1 ? '#05c46b' : '#ff5c5c';
  const baseVal = mode === 'cum' ? 1 : 0;
  const baseEl = (baseVal >= lo && baseVal <= hi) ? `<line x1="${padL}" y1="${py(baseVal)}" x2="${W - padR}" y2="${py(baseVal)}" stroke="#4db8ff" stroke-dasharray="4 4" opacity="0.5"/>` : '';
  let rm = -Infinity, ddMin = 0, ddI = 0, peakI = 0;
  ys.forEach((v, i) => { rm = Math.max(rm, v); const x = v / rm - 1; if (x < ddMin) { ddMin = x; ddI = i; } });
  let pk = -Infinity; for (let i = 0; i <= ddI; i++) if (ys[i] > pk) { pk = ys[i]; peakI = i; }
  const ddBand = ddMin < 0 ? `<rect x="${px(peakI)}" y="${padT}" width="${Math.max(px(ddI) - px(peakI), 1)}" height="${plotH}" fill="#ff5c5c" opacity="0.07"/>` : '';
  const peakDot = `<circle cx="${px(peakI)}" cy="${py(vals[peakI])}" r="3.5" fill="#05c46b" stroke="#0a0c10" stroke-width="1.5"/>`;
  const troughDot = ddMin < 0 ? `<circle cx="${px(ddI)}" cy="${py(vals[ddI])}" r="3.5" fill="#ff5c5c" stroke="#0a0c10" stroke-width="1.5"/>` : '';
  const xl = [0, Math.floor(n / 2), n - 1].map((i, k) => `<text x="${px(i)}" y="${H - 9}" text-anchor="${k === 0 ? 'start' : k === 2 ? 'end' : 'middle'}" fill="#828d9b" font-size="11" font-family="'IBM Plex Mono',monospace">${esc(dates[i])}</text>`).join('');
  let hov = '';
  if (ST.hoverIdx != null) {
    const hIdx = ST.hoverIdx, hx = px(hIdx), hy = py(vals[hIdx]);
    const label = mode === 'cum' ? ys[hIdx].toFixed(3) + '×  (' + pct(ys[hIdx] - 1) + ')' : pct(ys[hIdx] - 1);
    const tw = 172, tx = Math.min(Math.max(hx - tw / 2, padL), W - padR - tw), ty = Math.max(hy - 50, padT + 2);
    hov = `<g pointer-events="none"><line x1="${hx}" y1="${padT}" x2="${hx}" y2="${padT + plotH}" stroke="#828d9b" stroke-width="1" opacity="0.5"/>
      <circle cx="${hx}" cy="${hy}" r="4.5" fill="${stroke}" stroke="#0a0c10" stroke-width="2"/>
      <g transform="translate(${tx},${ty})"><rect width="${tw}" height="42" rx="7" fill="#05070a" stroke="#2e3742"/>
      <text x="11" y="18" fill="#e8edf4" font-size="12.5" font-family="'IBM Plex Mono',monospace" font-weight="600">${esc(label)}</text>
      <text x="11" y="33" fill="#828d9b" font-size="11" font-family="'IBM Plex Mono',monospace">${esc(dates[hIdx])}</text></g></g>`;
  }
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block" preserveAspectRatio="xMidYMid meet" id="cr-chart-svg">
    ${grid}${ddBand}<polygon points="${area}" fill="${stroke}" opacity="0.1"/>${baseEl}
    <polyline points="${line}" fill="none" stroke="${stroke}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
    ${peakDot}${troughDot}${yl}${xl}${hov}
    <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="transparent" id="cr-chart-overlay"/>
  </svg>`;
}

function renderChart() {
  const cm = ST.chartMode;
  document.getElementById('cr-chartmodes').innerHTML = ['cum', 'ret'].map(id => `
    <button class="cr-btn" data-chartmode="${id}" style="padding:6px 14px;border-radius:7px;font-size:12px;font-weight:600;background:${cm === id ? 'var(--accent)' : 'transparent'};color:${cm === id ? '#04121f' : 'var(--muted)'}">${id === 'cum' ? 'Cumulative ×' : 'Return %'}</button>`).join('');
  const dates = DATA.curveDates;
  document.getElementById('cr-chartsub').textContent = `${esc(DATA.lockedEngine)}${DATA.lockedOverlay ? ' + ' + DATA.lockedOverlay : ''} · ${dates[0] || ''} → ${dates[dates.length - 1] || ''} · ${money(DATA.winScore.notional, 0)} notional`;
  document.getElementById('cr-chart').innerHTML = buildChartSvg();
  const overlay = document.getElementById('cr-chart-overlay');
  if (overlay) {
    overlay.addEventListener('mousemove', e => {
      const svg = overlay.ownerSVGElement, r = svg.getBoundingClientRect();
      const xr = (e.clientX - r.left) / r.width * 960;
      let i = Math.round((xr - 54) / (960 - 54 - 18) * (DATA.curveDaily.length - 1));
      i = Math.max(0, Math.min(DATA.curveDaily.length - 1, i));
      if (i !== ST.hoverIdx) { ST.hoverIdx = i; renderChart(); }
    });
    overlay.addEventListener('mouseleave', () => { if (ST.hoverIdx != null) { ST.hoverIdx = null; renderChart(); } });
  }
}

function renderGuardrails() {
  const g = DATA.guardrails;
  const rows = [
    { label: 'Per-trade max', cap: money(g.per_trade_max_usd, 0), note: 'rejected above ceiling' },
    { label: 'Total deployed max', cap: money(g.total_deployed_max_usd, 0), note: 'hard cap on live exposure' },
    { label: 'Max new positions / run', cap: String(g.max_new_positions_per_run), note: 'per cron tick' },
    { label: 'Max orders / run', cap: String(g.max_orders_per_run), note: 'rate limit' },
    { label: 'Daily realized-loss kill', cap: money(g.daily_loss_limit_usd, 0), note: 'halts the day when breached' },
  ];
  document.getElementById('cr-guardrails').innerHTML = rows.map(r => `
    <div>
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px">
        <span style="font-size:12.5px;color:var(--fg)">${esc(r.label)}</span>
        <span style="font-size:13px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:var(--fg)">${esc(r.cap)}</span>
      </div>
      <div style="height:6px;background:var(--bg);border-radius:4px;overflow:hidden;border:1px solid var(--line)">
        <div style="height:100%;width:100%;background:linear-gradient(90deg,var(--up),#3ad98a);border-radius:4px"></div>
      </div>
      <div style="font-size:10.5px;color:var(--muted);margin-top:4px;font-family:'IBM Plex Mono',monospace">${esc(r.note)}</div>
    </div>`).join('');
  document.getElementById('cr-guardrail-chips').innerHTML = ['US-equities only', 'daily-loss kill switch', 'HALT file', g.model.split('/').pop()]
    .map(c => `<span style="padding:4px 10px;border-radius:6px;background:var(--panel2);border:1px solid var(--line);font-size:11px;color:var(--muted);font-family:'IBM Plex Mono',monospace">${esc(c)}</span>`).join('');
}

function renderBakeoff() {
  const rows = DATA.bakeoff;
  if (!rows.length) { document.getElementById('cr-bakeoff').innerHTML = `<tr><td colspan="6" style="padding:12px;color:var(--muted)">no bake-off data</td></tr>`; return; }
  const maxPoint = Math.max(...rows.map(b => b.point));
  document.getElementById('cr-bakeoff').innerHTML = rows.map(b => `
    <tr style="background:${b.beats ? 'rgba(5,196,107,.07)' : 'transparent'}">
      <td style="padding:9px 10px;border-bottom:1px solid var(--line);font-family:'IBM Plex Mono',monospace;color:${b.beats ? 'var(--up)' : 'var(--fg)'};font-weight:${b.beats ? 700 : 500}">${esc(b.overlay)}</td>
      <td style="padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;font-family:'IBM Plex Mono',monospace">
        <div style="display:flex;align-items:center;justify-content:flex-end;gap:8px"><span style="width:44px;height:5px;background:var(--bg);border-radius:3px;overflow:hidden;border:1px solid var(--line)"><span style="display:block;height:100%;width:${(b.point / maxPoint * 100).toFixed(0)}%;background:${b.beats ? 'var(--up)' : 'var(--muted)'}"></span></span>${b.point.toFixed(2)}</div>
      </td>
      <td style="padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;font-family:'IBM Plex Mono',monospace;color:var(--muted)">${b.deflated.toFixed(2)}</td>
      <td style="padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;font-family:'IBM Plex Mono',monospace;color:var(--muted)">${esc(b.fold)}</td>
      <td style="padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;font-family:'IBM Plex Mono',monospace;color:var(--muted)">${esc(b.ci)}</td>
      <td style="padding:9px 10px;border-bottom:1px solid var(--line);text-align:center">${b.beats ? '✓ beats' : '—'}</td>
    </tr>`).join('');
}

const RUN_COLS = [
  { key: 'sid', label: 'run', align: 'left' }, { key: 'engine', label: 'engine', align: 'left' },
  { key: 'overlay', label: 'overlay', align: 'left' }, { key: 'n', label: 'trades', align: 'right' },
  { key: 'wr', label: 'win %', align: 'right' }, { key: 'pf', label: 'PF', align: 'right' },
  { key: 'pnl', label: 'P&L', align: 'right' }, { key: 'ret', label: 'return', align: 'right' },
];

function renderRuns() {
  document.getElementById('cr-runcount').textContent = `Research runs · ${DATA.runs.length} total`;
  document.getElementById('cr-enginechips').innerHTML = ['all', ...new Set(DATA.runs.map(r => r.engine))].map(e => `
    <button class="cr-btn" data-engine="${esc(e)}" style="padding:6px 13px;border-radius:7px;font-size:12px;font-weight:600;background:${ST.engine === e ? 'var(--panel2)' : 'transparent'};color:${ST.engine === e ? 'var(--fg)' : 'var(--muted)'};display:inline-flex;align-items:center;gap:6px"><span style="width:7px;height:7px;border-radius:50%;background:${e === 'all' ? 'var(--muted)' : engineColor(e)}"></span>${esc(e)}</button>`).join('');
  document.getElementById('cr-runcols').innerHTML = RUN_COLS.map(c => `
    <div class="cr-btn" data-sort="${c.key}" style="padding:10px 12px;font-weight:600;color:var(--muted);font-size:12.5px;white-space:nowrap;cursor:pointer;text-align:${c.align}">${esc(c.label)}<span style="color:var(--accent)">${ST.runSort === c.key ? (ST.runDir < 0 ? ' ↓' : ' ↑') : ''}</span></div>`).join('');

  let rows = DATA.runs.filter(r => ST.engine === 'all' || r.engine === ST.engine);
  rows = rows.slice().sort((a, b) => {
    const av = a[ST.runSort], bv = b[ST.runSort];
    return typeof av === 'string' ? ST.runDir * av.localeCompare(bv) : ST.runDir * (av - bv);
  });
  const maxAbsRet = Math.max(...rows.map(r => Math.abs(r.ret)), 0.01);
  document.getElementById('cr-runrows').innerHTML = rows.map(r => {
    const win = r.id === DATA.winRunId;
    const w = Math.abs(r.ret) / maxAbsRet * 100;
    return `<div class="cr-row cr-btn" data-open="${esc(r.id)}" style="display:grid;grid-template-columns:minmax(150px,1.5fr) 1fr 0.9fr 0.65fr 0.7fr 0.6fr 1fr 1.25fr;align-items:center;background:${win ? 'rgba(5,196,107,.06)' : 'transparent'};border-bottom:1px solid var(--line);border-left:3px solid ${win ? 'var(--up)' : 'transparent'};font-size:12.5px">
      <div style="padding:9px 12px;font-family:'IBM Plex Mono',monospace;white-space:nowrap"><span style="color:var(--accent);border-bottom:1px dashed rgba(77,184,255,.45);padding-bottom:1px">${esc(r.sid)}</span>${win ? '<span style="color:var(--up);font-weight:700;font-size:10px;margin-left:6px">◆ LOCKED</span>' : ''}</div>
      <div style="padding:9px 12px;white-space:nowrap;display:flex;align-items:center;gap:6px"><span style="width:7px;height:7px;border-radius:50%;background:${engineColor(r.engine)};flex:none"></span>${esc(r.engine)}</div>
      <div style="padding:9px 12px;font-family:'IBM Plex Mono',monospace;color:var(--muted)">${esc(r.overlay || '—')}</div>
      <div style="padding:9px 12px;text-align:right;font-family:'IBM Plex Mono',monospace">${r.n}</div>
      <div style="padding:9px 12px;text-align:right;font-family:'IBM Plex Mono',monospace">${pctAbs(r.wr, 1)}</div>
      <div style="padding:9px 12px;text-align:right;font-family:'IBM Plex Mono',monospace">${num(r.pf)}</div>
      <div style="padding:9px 12px;text-align:right;font-family:'IBM Plex Mono',monospace;color:${r.pnl >= 0 ? 'var(--up)' : 'var(--down)'};font-weight:600">${(r.pnl >= 0 ? '+' : '') + money(r.pnl, 0)}</div>
      <div style="padding:9px 12px;font-family:'IBM Plex Mono',monospace;display:flex;align-items:center;justify-content:flex-end;gap:8px"><span style="width:52px;height:5px;background:var(--bg);border-radius:3px;overflow:hidden;border:1px solid var(--line);flex:none"><span style="display:block;height:100%;width:${w.toFixed(0)}%;background:${r.ret >= 0 ? 'var(--up)' : 'var(--down)'}"></span></span><span style="color:${r.ret >= 0 ? 'var(--up)' : 'var(--down)'};min-width:56px;text-align:right">${pct(r.ret, 1)}</span></div>
    </div>`;
  }).join('');
}

function renderLeaderboard() {
  const byEngine = {};
  DATA.runs.forEach(r => { if (!byEngine[r.engine] || r.pnl > byEngine[r.engine].pnl) byEngine[r.engine] = r; });
  const arr = Object.values(byEngine).sort((a, b) => b.pnl - a.pnl);
  const maxLb = Math.max(...arr.map(r => r.pnl), 1);
  document.getElementById('cr-leaderboard').innerHTML = arr.map(r => `
    <div style="background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px"><span style="width:9px;height:9px;border-radius:50%;background:${engineColor(r.engine)}"></span><span style="font-weight:600;font-size:14px">${esc(r.engine)}</span></div>
      <div style="font-size:26px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:${r.pnl >= 0 ? 'var(--up)' : 'var(--down)'};letter-spacing:-.02em">${(r.pnl >= 0 ? '+' : '') + money(r.pnl, 0)}</div>
      <div style="height:7px;background:var(--bg);border:1px solid var(--line);border-radius:4px;overflow:hidden;margin:10px 0 8px"><div style="height:100%;width:${(r.pnl / maxLb * 100).toFixed(0)}%;background:${engineColor(r.engine)};border-radius:4px"></div></div>
      <div style="font-size:11.5px;color:var(--muted);font-family:'IBM Plex Mono',monospace">${r.n} trades · ${pctAbs(r.wr, 1)} win · ${esc(r.overlay || 'no overlay')}</div>
    </div>`).join('');
}

function renderScorecard() {
  const S = DATA.winScore;
  document.getElementById('cr-winid').textContent = DATA.winRunId;
  const tiles = [
    { label: 'balance', value: money(S.balance), color: 'var(--up)' },
    { label: 'net P&L', value: '+' + money(S.pnl), color: 'var(--up)' },
    { label: 'total return', value: pct(S.ret), color: 'var(--up)' },
    { label: 'trades', value: String(S.n), color: 'var(--fg)' },
    { label: 'win rate', value: pctAbs(S.wr, 1), color: 'var(--fg)' },
    { label: 'profit factor', value: S.pf.toFixed(2), color: 'var(--up)' },
    { label: 'avg win', value: money(S.avgWin), color: 'var(--up)' },
    { label: 'avg loss', value: money(S.avgLoss), color: 'var(--down)' },
    { label: 'sharpe', value: S.sharpe.toFixed(2), color: 'var(--fg)' },
    { label: 'max drawdown', value: pctAbs(S.dd, 2), color: 'var(--down)' },
    { label: 'avg holding', value: S.avgHold.toFixed(1) + ' bars', color: 'var(--fg)' },
    { label: 'gross win', value: money(S.gw, 0), color: 'var(--fg)' },
  ];
  document.getElementById('cr-scoretiles').innerHTML = tiles.map(t => `
    <div style="background:var(--bg);border:1px solid var(--line);border-radius:11px;padding:13px 15px">
      <div style="font-size:20px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:${t.color};letter-spacing:-.01em">${esc(t.value)}</div>
      <div style="font-size:11px;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:.04em">${esc(t.label)}</div>
    </div>`).join('');
}

function renderBuckets() {
  const bm = DATA.bucketsMeta || {};
  document.getElementById('cr-buckets').innerHTML = DATA.buckets.map(b => `
    <div>
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px">
        <span style="font-size:12.5px"><span style="color:var(--muted);font-family:'IBM Plex Mono',monospace;font-size:11px">${esc(b.dim)}</span> · ${esc(b.bucket)}</span>
        <span style="font-size:12.5px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:var(--down)">${pctAbs(b.share, 1)}</span>
      </div>
      <div style="height:8px;background:var(--bg);border:1px solid var(--line);border-radius:4px;overflow:hidden"><div style="height:100%;width:${(b.share * 100).toFixed(0)}%;background:linear-gradient(90deg,var(--down),#ff8a8a);border-radius:4px"></div></div>
      <div style="font-size:10.5px;color:var(--muted);margin-top:4px;font-family:'IBM Plex Mono',monospace">${b.lossN.toLocaleString()} of ${b.totalN.toLocaleString()} trades · ${pctAbs(b.wr, 0)} win rate</div>
    </div>`).join('');
  document.getElementById('cr-bucketscaveat').innerHTML = bm.n ? `⚠ ${esc(bm.caveat || '')}` : '';

  const wm = DATA.winBucketsMeta || {};
  document.getElementById('cr-winbuckets').innerHTML = DATA.winBuckets.map(w => `
    <div>
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px">
        <span style="font-size:12.5px"><span style="color:var(--muted);font-family:'IBM Plex Mono',monospace;font-size:11px">${esc(w.dim)}</span> · ${esc(w.bucket)}</span>
        <span style="font-size:12.5px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:var(--up)">${pctAbs(w.share, 1)}</span>
      </div>
      <div style="height:8px;background:var(--bg);border:1px solid var(--line);border-radius:4px;overflow:hidden"><div style="height:100%;width:${(w.share * 100).toFixed(0)}%;background:linear-gradient(90deg,var(--up),#3ad98a);border-radius:4px"></div></div>
      <div style="font-size:10.5px;color:var(--muted);margin-top:4px;font-family:'IBM Plex Mono',monospace">${w.winN.toLocaleString()} of ${w.totalN.toLocaleString()} trades · ${pctAbs(w.wr, 0)} win rate</div>
    </div>`).join('');
  document.getElementById('cr-winbucketsnote').textContent = wm.note || '';
}

function renderLedger() {
  const S = DATA.winScore;
  document.getElementById('cr-ledgercount').textContent = `${S.won} wins · ${S.lost} losses · ${S.n} trades`;
  document.getElementById('cr-tradechips').innerHTML = [['all', 'all'], ['win', 'wins'], ['loss', 'losses']].map(([id, label]) => `
    <button class="cr-btn" data-tradefilter="${id}" style="padding:6px 12px;border-radius:7px;font-size:12px;font-weight:600;background:${ST.tradeFilter === id ? 'var(--panel2)' : 'transparent'};color:${ST.tradeFilter === id ? 'var(--fg)' : 'var(--muted)'}">${esc(label)}</button>`).join('');
  const trades = DATA.winTrades.filter(t => ST.tradeFilter === 'all' || t.oc === ST.tradeFilter);
  const maxAbsPnl = Math.max(...DATA.winTrades.map(t => Math.abs(t.pnl)), 1);
  document.getElementById('cr-ledger').innerHTML = trades.map(t => {
    const w = Math.abs(t.pnl) / maxAbsPnl * 100;
    const color = t.oc === 'win' ? 'var(--up)' : 'var(--down)';
    return `<tr style="border-left:3px solid ${color}">
      <td style="padding:8px 9px;border-bottom:1px solid var(--line);font-family:'IBM Plex Mono',monospace;color:var(--muted)">#${esc(t.id)}</td>
      <td style="padding:8px 9px;border-bottom:1px solid var(--line);font-family:'IBM Plex Mono',monospace;font-weight:600">${esc(t.sym)}</td>
      <td style="padding:8px 9px;border-bottom:1px solid var(--line);font-family:'IBM Plex Mono',monospace;font-size:11px">${esc(t.entry)} → ${esc(t.exit)}</td>
      <td style="padding:8px 9px;border-bottom:1px solid var(--line);text-align:right;font-family:'IBM Plex Mono',monospace">${t.bars}</td>
      <td style="padding:8px 9px;border-bottom:1px solid var(--line);text-align:right;font-family:'IBM Plex Mono',monospace;color:${color}">${pct(t.pnlPct)}</td>
      <td style="padding:8px 9px;border-bottom:1px solid var(--line);text-align:right;font-family:'IBM Plex Mono',monospace;color:${color};font-weight:600">${(t.pnl >= 0 ? '+' : '') + money(t.pnl)}</td>
      <td style="padding:8px 9px;border-bottom:1px solid var(--line)">
        <div style="display:flex;align-items:center;gap:0;height:12px"><span style="flex:1;display:flex;justify-content:flex-end"><span style="height:8px;width:${t.pnl < 0 ? w.toFixed(0) : 0}%;background:var(--down);border-radius:2px 0 0 2px"></span></span><span style="width:1px;height:12px;background:var(--line2)"></span><span style="flex:1;display:flex;justify-content:flex-start"><span style="height:8px;width:${t.pnl >= 0 ? w.toFixed(0) : 0}%;background:var(--up);border-radius:0 2px 2px 0"></span></span></div>
      </td>
    </tr>`;
  }).join('');
}

function renderRunbook() {
  document.getElementById('cr-runbook').innerHTML = DATA.runbook.map(([label, cmd], i) => `
    <button class="cr-btn" data-copy="${i}" style="text-align:left;display:flex;align-items:center;gap:14px;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:11px 14px">
      <span style="font-size:12px;color:var(--muted);min-width:150px;flex:none">${esc(label)}</span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--fg);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(cmd)}</span>
      <span style="font-size:11px;font-weight:600;color:${ST.copied === i ? 'var(--up)' : 'var(--muted)'};flex:none;font-family:'IBM Plex Mono',monospace">${ST.copied === i ? 'COPIED' : 'COPY'}</span>
    </button>`).join('');
}

function renderAgentNotes() {
  const sec = document.getElementById('cr-agentnotes');
  if (!DATA.reflections || !DATA.reflections.length) { sec.style.display = 'none'; return; }
  sec.style.display = '';
  document.getElementById('cr-reflections').innerHTML = DATA.reflections.map(r => `<p style="margin:0">## ${esc(r)}</p>`).join('');
}

function buildDetail(runId) {
  const r = DATA.runs.find(x => x.id === runId); if (!r) return null;
  const win = r.id === DATA.winRunId;
  const balance = r.notional * (1 + r.ret);
  const avgWin = r.won ? r.gw / r.won : 0, avgLoss = r.lost ? -r.gl / r.lost : 0;
  const tiles = [
    { label: 'balance', value: money(balance), color: balance >= r.notional ? 'var(--up)' : 'var(--down)' },
    { label: 'net P&L', value: (r.pnl >= 0 ? '+' : '') + money(r.pnl), color: r.pnl >= 0 ? 'var(--up)' : 'var(--down)' },
    { label: 'return', value: pct(r.ret), color: r.ret >= 0 ? 'var(--up)' : 'var(--down)' },
    { label: 'trades', value: String(r.n), color: 'var(--fg)' },
    { label: 'win rate', value: pctAbs(r.wr, 1), color: 'var(--fg)' },
    { label: 'profit factor', value: num(r.pf), color: r.pf >= 1 ? 'var(--up)' : 'var(--down)' },
    { label: 'avg win', value: money(avgWin), color: 'var(--up)' },
    { label: 'avg loss', value: money(avgLoss), color: 'var(--down)' },
    { label: 'gross win', value: money(r.gw, 0), color: 'var(--fg)' },
  ];
  return {
    idFull: r.id, engine: r.engine, dot: engineColor(r.engine), overlay: r.overlay || 'no overlay',
    winTag: win ? '◆ LOCKED CONFIG' : '', ret: pct(r.ret), retColor: r.ret >= 0 ? 'var(--up)' : 'var(--down)',
    pnl: (r.pnl >= 0 ? '+' : '') + money(r.pnl), notional: money(r.notional, 0),
    symbols: r.uni, period: r.start + ' → ' + r.end, tiles, won: r.won, lost: r.lost,
    winW: (r.won / r.n * 100).toFixed(1) + '%', lossW: (r.lost / r.n * 100).toFixed(1) + '%',
    note: win ? 'This is the config the forward paper track is now trading.' : `Per-trade ledger for this run is available from the CLI: rhagent.evaluate --run ${r.id}.`,
  };
}

function renderDrawer() {
  const wrap = document.getElementById('cr-drawerwrap');
  if (!ST.selectedRun) { wrap.innerHTML = ''; return; }
  const d = buildDetail(ST.selectedRun);
  if (!d) { wrap.innerHTML = ''; return; }
  wrap.innerHTML = `
    <div data-close-drawer style="position:fixed;inset:0;z-index:40;background:rgba(4,6,9,.6);backdrop-filter:blur(2px);animation:backdropIn .18s ease both"></div>
    <aside style="position:fixed;top:0;right:0;z-index:41;height:100vh;width:min(540px,94vw);background:var(--panel);border-left:1px solid var(--line2);box-shadow:-24px 0 60px rgba(0,0,0,.5);overflow-y:auto;animation:drawerIn .22s cubic-bezier(.2,.8,.2,1) both" class="cr-scroll">
      <div style="position:sticky;top:0;z-index:2;display:flex;align-items:flex-start;justify-content:space-between;gap:12px;padding:20px 24px;background:rgba(18,22,28,.92);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)">
        <div>
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
            <span style="display:inline-flex;align-items:center;gap:6px;font-weight:600;font-size:14px"><span style="width:8px;height:8px;border-radius:50%;background:${d.dot}"></span>${esc(d.engine)}</span>
            <span style="padding:2px 8px;border-radius:6px;background:var(--panel2);border:1px solid var(--line);font-size:11px;color:var(--muted);font-family:'IBM Plex Mono',monospace">${esc(d.overlay)}</span>
            ${d.winTag ? `<span style="padding:2px 8px;border-radius:6px;background:rgba(5,196,107,.14);color:var(--up);font-size:10px;font-weight:700;font-family:'IBM Plex Mono',monospace">${d.winTag}</span>` : ''}
          </div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--muted);margin-top:6px">${esc(d.idFull)}</div>
        </div>
        <button data-close-drawer style="flex:none;width:30px;height:30px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--muted);cursor:pointer;font-size:15px;line-height:1">✕</button>
      </div>
      <div style="padding:22px 24px">
        <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px">
          <div style="font-size:38px;font-weight:700;font-family:'IBM Plex Mono',monospace;letter-spacing:-.02em;color:${d.retColor}">${d.ret}</div>
          <div style="font-size:14px;color:${d.retColor};font-family:'IBM Plex Mono',monospace;font-weight:600">${d.pnl}</div>
        </div>
        <div style="font-size:12px;color:var(--muted)">total return on ${d.notional} notional</div>
        <div style="display:flex;gap:7px;flex-wrap:wrap;margin:18px 0 20px">
          <span style="padding:5px 11px;border-radius:7px;background:var(--bg);border:1px solid var(--line);font-size:11.5px;color:var(--muted);font-family:'IBM Plex Mono',monospace">${esc(d.symbols)}</span>
          <span style="padding:5px 11px;border-radius:7px;background:var(--bg);border:1px solid var(--line);font-size:11.5px;color:var(--muted);font-family:'IBM Plex Mono',monospace">${esc(d.period)}</span>
        </div>
        <h4 style="margin:0 0 12px;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:600">Scorecard</h4>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">
          ${d.tiles.map(t => `<div style="background:var(--bg);border:1px solid var(--line);border-radius:11px;padding:12px 14px">
            <div style="font-size:18px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:${t.color};letter-spacing:-.01em">${esc(t.value)}</div>
            <div style="font-size:10.5px;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:.04em">${esc(t.label)}</div></div>`).join('')}
        </div>
        <div style="margin-top:16px;display:flex;align-items:center;gap:10px;padding:12px 14px;background:var(--bg);border:1px solid var(--line);border-radius:11px">
          <div style="flex:1">
            <div style="font-size:11px;color:var(--muted);margin-bottom:6px;font-family:'IBM Plex Mono',monospace">win / loss split · ${d.won}W / ${d.lost}L</div>
            <div style="display:flex;height:9px;border-radius:5px;overflow:hidden;background:var(--panel2)">
              <div style="height:100%;width:${d.winW};background:var(--up)"></div>
              <div style="height:100%;width:${d.lossW};background:var(--down)"></div>
            </div>
          </div>
        </div>
        <div style="margin-top:16px;padding:12px 14px;background:rgba(77,184,255,.06);border:1px solid rgba(77,184,255,.22);border-radius:11px;font-size:12px;color:var(--muted);text-wrap:pretty">${esc(d.note)}</div>
      </div>
    </aside>`;
}

function renderAll() {
  renderHeaderPills(); renderVerdict(); renderKpis(); renderChart(); renderGuardrails();
  renderBakeoff(); renderRuns(); renderLeaderboard(); renderScorecard(); renderBuckets();
  renderLedger(); renderRunbook(); renderAgentNotes(); renderDrawer();
}

document.addEventListener('click', e => {
  const t = e.target.closest('[data-chartmode],[data-engine],[data-sort],[data-open],[data-tradefilter],[data-copy],[data-close-drawer]');
  if (!t) return;
  if (t.dataset.chartmode) { ST.chartMode = t.dataset.chartmode; renderChart(); }
  else if (t.dataset.engine) { ST.engine = t.dataset.engine; renderRuns(); }
  else if (t.dataset.sort) {
    if (ST.runSort === t.dataset.sort) ST.runDir = -ST.runDir;
    else { ST.runSort = t.dataset.sort; ST.runDir = ['sid', 'engine', 'overlay'].includes(t.dataset.sort) ? 1 : -1; }
    renderRuns();
  } else if (t.dataset.open) { ST.selectedRun = t.dataset.open; renderDrawer(); }
  else if (t.dataset.tradefilter) { ST.tradeFilter = t.dataset.tradefilter; renderLedger(); }
  else if (t.dataset.copy != null) {
    const i = Number(t.dataset.copy);
    const cmd = DATA.runbook[i][1];
    if (navigator.clipboard) navigator.clipboard.writeText(cmd).catch(() => {});
    ST.copied = i; renderRunbook();
    setTimeout(() => { ST.copied = -1; renderRunbook(); }, 1400);
  } else if (t.hasAttribute('data-close-drawer')) { ST.selectedRun = null; renderDrawer(); }
});
document.addEventListener('keydown', e => { if (e.key === 'Escape' && ST.selectedRun) { ST.selectedRun = null; renderDrawer(); } });

renderAll();
</script>
</body>
</html>
"""


def render_control_room(base_dir: Path) -> str:
    data = _build_control_room_data(base_dir)
    title = f"Trading Control Room — {len(data['runs'])} research runs"
    html = _CONTROL_ROOM_TEMPLATE.replace("__TITLE__", escape(title))
    return html.replace("__DATA_JSON__", json.dumps(data))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="make_dashboard")
    p.add_argument("--run", help="render only this run_id in detail (default: all runs)")
    p.add_argument("--base-dir", default="journal/papertrade")
    p.add_argument("--out", help="output HTML path (default: journal/dashboard.html)")
    p.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = p.parse_args(argv)

    base_dir = Path(args.base_dir)
    if args.run:
        run_dir = base_dir / args.run
        if not (run_dir / "run.json").exists():
            raise SystemExit(f"no run.json in {run_dir}")
        html = render(run_dir, base_dir)
        label = run_dir.name
    else:
        html = render_control_room(base_dir)
        label = "all runs"

    out = Path(args.out) if args.out else base_dir.parent / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({label})")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
