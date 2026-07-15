"""Render the trading system into a single self-contained HTML dashboard.

Everything happening in one page: forward records, the latest paper-trade
scorecard, all research runs, the robust bake-off, the equity curve, and the
latest run's ledger/failure buckets.

    python scripts/make_dashboard.py                 # unified dashboard
    python scripts/make_dashboard.py --run <run_id>  # a specific run
    python scripts/make_dashboard.py --open          # also open in a browser

Reads the ledgers written under journal/papertrade/ and journal/forward/; it
reuses rhagent.evaluate so the numbers match the CLI report exactly.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from datetime import date
from html import escape
from pathlib import Path

# src-layout: make the rhagent package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rhagent.evaluate import (  # noqa: E402
    aggregate,
    compare_runs,
    failure_buckets,
    load_run,
)


def _latest_run(base_dir: Path) -> Path:
    runs = sorted(p.parent for p in base_dir.glob("*/run.json"))
    if not runs:
        raise SystemExit(f"no runs found under {base_dir} — run rhagent.papertrade first")
    return runs[-1]


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


def _safe_compare(base_dir: Path) -> pd.DataFrame:
    if not base_dir.exists():
        return pd.DataFrame()
    return compare_runs(base_dir)


def _days_old(value: str) -> int | None:
    try:
        end = pd.to_datetime(value).date()
    except (TypeError, ValueError):
        return None
    return (date.today() - end).days


def _status_class(days_old: int | None) -> str:
    if days_old is None:
        return "warn"
    if days_old <= 3:
        return "ok"
    if days_old <= 10:
        return "warn"
    return "bad"


def _status_label(days_old: int | None) -> str:
    if days_old is None:
        return "unknown"
    if days_old == 0:
        return "current"
    if days_old == 1:
        return "1 day old"
    return f"{days_old} days old"


# ── SVG equity curve ────────────────────────────────────────────────────────

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

    # horizontal gridlines + y-axis equity labels
    grid, ylabels = [], []
    for k in range(5):
        v = lo + span * k / 4
        y = py(v)
        grid.append(f"<line x1='{padL}' y1='{y:.1f}' x2='{width - padR}' y2='{y:.1f}' class='grid'/>")
        ylabels.append(f"<text x='{padL - 8}' y='{y + 4:.1f}' class='ytick'>{v:.3f}×</text>")

    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(ys))
    area = f"{px(0):.1f},{py(lo):.1f} " + pts + f" {px(n - 1):.1f},{py(lo):.1f}"
    final = ys[-1]
    stroke = "var(--up)" if final >= 1.0 else "var(--down)"

    base_y = py(1.0) if lo <= 1.0 <= hi else None
    baseline = (
        f"<line x1='{padL}' y1='{base_y:.1f}' x2='{width - padR}' y2='{base_y:.1f}' "
        f"class='baseline'/><text x='{padL - 8}' y='{base_y + 4:.1f}' class='ytick base'>1.000×</text>"
        if base_y is not None else ""
    )

    # max-drawdown window: shade from the running-max peak to the deepest trough
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

    # peak marker
    peak_i = int(equity.values.argmax())
    peak_dot = (
        f"<circle cx='{px(peak_i):.1f}' cy='{py(ys[peak_i]):.1f}' r='3.5' class='peakpt'/>"
    )

    xlabels = (
        f"<text x='{padL}' y='{height - 8}' class='xtick' text-anchor='start'>{dates[0]}</text>"
        f"<text x='{width - padR}' y='{height - 8}' class='xtick' text-anchor='end'>{dates[-1]}</text>"
    )

    return f"""<svg viewBox="0 0 {width} {height}" class="equity" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Equity curve, final {equity.iloc[-1]:.3f} times starting capital, {_pct(final - 1)}">
  {''.join(grid)}
  {dd_shade}
  <polygon points="{area}" fill="{stroke}" opacity="0.12"/>
  {baseline}
  <polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="2" stroke-linejoin="round"/>
  {peak_dot}{dd_marker}
  {''.join(ylabels)}{xlabels}
  <text x="{padL}" y="16" class="svglabel">equity {equity.iloc[-1]:.3f}× · {_pct(final - 1)}</text>
</svg>"""


# ── HTML fragments ──────────────────────────────────────────────────────────

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
        tile("avg holding", f"{a['avg_holding_bars']:.1f} bars"),
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
        idcell = (
            f"<a class='mono' href='#run-{escape(rid)}'>{escape(rid)}</a>"
            if link else f"{escape(rid)}"
        )
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


def _forward_table(forward_dir: Path) -> str:
    df = _safe_compare(forward_dir)
    if len(df) == 0:
        return "<p class='muted'>no forward records yet</p>"
    rows = []
    for run_dir in _run_dirs(forward_dir):
        meta, _, net = load_run(run_dir)
        a = aggregate(pd.DataFrame(), net)
        days_old = _days_old(str(meta.get("end", "")))
        ret_cls = "up" if a["total_return"] >= 0 else "down"
        pnl = _return_pnl(a["total_return"], float(meta.get("notional", 10_000.0)))
        pnl_cls = "up" if pnl >= 0 else "down"
        rows.append(
            f"<tr><td class='mono'>{escape(str(meta['run_id']))}</td>"
            f"<td>{escape(str(meta['engine']))}</td>"
            f"<td><span class='status {_status_class(days_old)}'>{_status_label(days_old)}</span></td>"
            f"<td class='num'>{len(net)}</td>"
            f"<td class='mono'>{escape(str(meta.get('start', ''))[:10])}</td>"
            f"<td class='mono'>{escape(str(meta.get('end', ''))[:10])}</td>"
            f"<td class='num {pnl_cls}'>{'+' if pnl >= 0 else ''}{_money(pnl)}</td>"
            f"<td class='num {ret_cls}'>{_pct(a['total_return'])}</td>"
            f"<td class='num'>{_num(a['sharpe'])}</td>"
            f"<td class='num down'>{_pct(a['max_drawdown'])}</td></tr>"
        )
    return (
        "<table class='grid'><thead><tr><th>record</th><th>engine</th><th>freshness</th><th>days</th>"
        "<th>start</th><th>end</th><th>return p&amp;l</th><th>total return</th>"
        f"<th>sharpe</th><th>max dd</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _latest_summary(run_dir: Path, label: str) -> str:
    meta, trades, net = load_run(run_dir)
    a = aggregate(trades, net)
    pnl = _return_pnl(a["total_return"], float(meta.get("notional", 10_000.0)))
    pnl_cls = "up" if pnl >= 0 else "down"
    return (
        f"<div class='summary'><div><div class='eyebrow'>{escape(label)}</div>"
        f"<strong>{escape(str(meta['engine']))}</strong> "
        f"<span class='mono'>{escape(str(meta['run_id']))}</span></div>"
        f"<div class='summary-metric {pnl_cls}'>{'+' if pnl >= 0 else ''}{_money(pnl)}</div>"
        f"<div class='summary-mini'>{_pct(a['total_return'])} return · "
        f"{_num(a['sharpe'])} Sharpe · {_pct(a['max_drawdown'])} max DD</div></div>"
    )


def _code_health(graph_dir: Path) -> str:
    report = graph_dir / "GRAPH_REPORT.md"
    graph = graph_dir / "graph.html"
    if not report.exists() and not graph.exists():
        return "<p class='muted'>no code graph generated yet</p>"
    links = []
    if graph.exists():
        links.append(f"<a class='mono' href='{escape(graph.resolve().as_uri())}'>graph.html</a>")
    if report.exists():
        links.append(f"<a class='mono' href='{escape(report.resolve().as_uri())}'>GRAPH_REPORT.md</a>")
    return (
        "<div class='notice'><b>Code topology available.</b> "
        "Use it for dependency spelunking, but keep this trading dashboard as the operational source of truth. "
        + " · ".join(links) + "</div>"
    )


def _overview_cards(base_dir: Path, forward_dir: Path, comparison: pd.DataFrame) -> str:
    forward_runs = _run_dirs(forward_dir)
    latest_forward_end = ""
    for run_dir in forward_runs:
        meta, _, _ = load_run(run_dir)
        end = str(meta.get("end", ""))
        if end > latest_forward_end:
            latest_forward_end = end
    days_old = _days_old(latest_forward_end)
    best = comparison.loc[comparison["total_return"].idxmax()] if len(comparison) else None
    best_text = "no research runs"
    best_cls = ""
    if best is not None:
        best_text = f"{best['engine']} {_pct(float(best['total_return']))}"
        best_cls = "up" if float(best["total_return"]) >= 0 else "down"
    cards = [
        ("Forward records", str(len(forward_runs)), "", "live paper tracks"),
        ("Latest forward", _status_label(days_old), _status_class(days_old), str(latest_forward_end)[:10] or "none"),
        ("Research runs", str(len(_run_dirs(base_dir))), "", "paper-trade archive"),
        ("Best return", best_text, best_cls, "research only"),
    ]
    return "<div class='overview'>" + "".join(
        f"<div class='overview-card'><div class='tile-l'>{escape(label)}</div>"
        f"<div class='overview-v {cls}'>{escape(value)}</div>"
        f"<div class='tile-s'>{escape(sub)}</div></div>"
        for label, value, cls, sub in cards
    ) + "</div>"


def _bakeoff_table(base_dir) -> str:
    from rhagent.evaluate_robust import robust_table
    df = robust_table(base_dir)
    if len(df) == 0:
        return "<p class='muted'>no runs</p>"
    rows = []
    for _, r in df.iterrows():
        win = "beats" if r["beats_baseline"] else ""
        cls = "up" if r["beats_baseline"] else ""
        rows.append(
            f"<tr><td class='mono'>{escape(str(r['run_id']))}</td>"
            f"<td>{escape(str(r['engine']))}</td>"
            f"<td>{escape(str(r['overlay']))}</td>"
            f"<td class='num'>{r['point_sharpe']:.2f}</td>"
            f"<td class='num'>{r['fold_mean']:.2f}±{r['fold_std']:.2f}</td>"
            f"<td class='num'>[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]</td>"
            f"<td class='num'>{r['deflated']:.2f}</td>"
            f"<td class='num {cls}'>{win}</td></tr>"
        )
    return (
        "<table class='grid'><thead><tr><th>run id</th><th>engine</th><th>overlay</th>"
        "<th>point sharpe</th><th>fold mean±sd</th><th>95% CI</th><th>deflated</th>"
        f"<th>vs baseline</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
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
h2.runhead{font-size:19px;text-transform:none;letter-spacing:-.01em;color:var(--fg);margin:20px 0 10px}
a.mono{color:var(--accent);text-decoration:none}
a.mono:hover{text-decoration:underline}
.backlink{font-size:12px;font-weight:400;margin-left:14px;color:var(--accent);text-decoration:none}
.backlink:hover{text-decoration:underline}
.sub{color:var(--muted);margin:0 0 8px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.chip{background:var(--panel2);border:1px solid var(--line);border-radius:999px;
padding:3px 12px;font-size:12px}
.chip b{color:var(--fg)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}
.summary{display:grid;grid-template-columns:1fr auto;gap:4px 18px;align-items:center;
background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:12px 0}
.summary strong{font-size:17px}
.summary-metric{font-size:22px;font-weight:700;grid-row:1 / span 2}
.summary-mini{color:var(--muted);font-size:12px}
.eyebrow{text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-size:11px;margin-bottom:2px}
.notice{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;color:var(--muted)}
.notice b{color:var(--fg)}
.overview{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0 6px}
.overview-card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.overview-v{font-size:20px;font-weight:700;margin-top:4px}
.status{display:inline-block;border-radius:999px;padding:1px 8px;font-size:11px;font-weight:700;text-transform:uppercase}
.status.ok{background:color-mix(in srgb,var(--up) 18%,transparent);color:var(--up)}
.status.warn{background:color-mix(in srgb,var(--accent) 18%,transparent);color:var(--accent)}
.status.bad{background:color-mix(in srgb,var(--down) 18%,transparent);color:var(--down)}
.ok{color:var(--up)}.warn{color:var(--accent)}.bad{color:var(--down)}
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
    aid = f" id='run-{escape(rid)}'" if anchored else ""
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


def render_all(base_dir: Path) -> str:
    """Unified control room: forward records, research index, latest detail."""
    runs = sorted(_run_dirs(base_dir), reverse=True)
    if not runs:
        raise SystemExit(f"no runs found under {base_dir} — run rhagent.papertrade first")
    comparison = compare_runs(base_dir)
    latest = runs[0]
    best = comparison.loc[comparison["total_return"].idxmax(), "run_id"]
    best_dir = base_dir / str(best)
    forward_dir = base_dir.parent / "forward"
    graph_dir = base_dir.parents[0].parent / "graphify-out"
    index = (
        _overview_cards(base_dir, forward_dir, comparison)
        + "<h2>Now · forward track record</h2>"
        f"<div class='tblscroll'>{_forward_table(forward_dir)}</div>"
        "<h2>Research pulse</h2>"
        + _latest_summary(latest, "latest paper-trade run")
        + (_latest_summary(best_dir, "best paper-trade run") if best_dir != latest else "")
        + f"<h2>All paper-trade runs · {len(runs)} total</h2>"
        + f"<div class='tblscroll'>{_compare_table(comparison, '', link=False)}</div>"
    )
    index += (
        "<h2>Bake-off · robust Sharpe (fold + bootstrap + deflated)</h2>"
        "<p class='sub'>A variant beats baseline only if its 95% CI lower bound "
        "clears the baseline Sharpe.</p>"
        f"<div class='tblscroll'>{_bakeoff_table(base_dir)}</div>"
        "<h2>Code graph</h2>"
        f"{_code_health(graph_dir)}"
        "<h2>Latest run detail</h2>"
    )
    return _page(
        f"Trading dashboard — {len(runs)} research runs", index + _run_section(latest),
        f"Generated from {escape(str(base_dir.parent))} · rhagent trading harness",
    )


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
        html = render_all(base_dir)
        label = "all runs"

    out = Path(args.out) if args.out else base_dir.parent / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    if not args.run and args.out is None:
        _write_redirect_stubs(out)
    print(f"wrote {out}  ({label})")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


def _write_redirect_stubs(target: Path) -> None:
    """Keep legacy dashboard paths from becoming competing sources of truth."""
    for legacy in [target.parent / "papertrade" / "dashboard.html",
                   target.parent / "forward" / "dashboard.html"]:
        rel = legacy.parent.relative_to(target.parent)
        href = ("../" * len(rel.parts)) + target.name
        legacy.write_text(
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta http-equiv='refresh' content='0; url={escape(href)}'>"
            "<title>Trading Dashboard</title></head><body>"
            f"<p>Moved to <a href='{escape(href)}'>the unified trading dashboard</a>.</p>"
            "</body></html>",
            encoding="utf-8",
        )


if __name__ == "__main__":
    raise SystemExit(main())
