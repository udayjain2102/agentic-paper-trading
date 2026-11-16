"""Render the trading system into a single self-contained HTML dashboard.

Forward records, the locked-config scorecard, every research run, the robust
bake-off, the equity curve, and the ledger / decision-quality buckets — one
static page with a small vanilla-JS layer (sort, filter, per-run drawer).

    python scripts/make_dashboard.py                 # writes journal/dashboard.html
    python scripts/make_dashboard.py --open          # also open in a browser

Reads the ledgers under journal/papertrade/ and journal/forward/; reuses
rhagent.evaluate so the numbers match the CLI report exactly.

PAYLOAD CONTRACT -- keys added to the dict _build_data returns (every older key
and its shape is unchanged). Floats are raw (the page formats them); dates are
"YYYY-MM-DD" strings; a section that cannot be built yields the empty value
below AND an entry in `health` (level "warn", id "build:<key>"); it never raises.

  asOf         str | null          date of the newest bar in data/SPY.csv.
  records      list[Record], always 5, in order: mean_reversion_real,
               mean_reversion, agent, agent_ctx, agent_open. Record =
               {id: str, label: str, kind: "rule"|"agent", basis: str, note: str,
                missing: bool,   # true: returns.csv absent, empty or unreadable
                days: int|null, start: str|null, end: str|null,
                totalReturn: float|null, sharpe: float|null,
                sharpeNeeds: int|null,   # min return-days; set only when sharpe is null on a non-missing record
                maxDrawdown: float|null,   # <= 0
                spyReturn: float|null,     # SPY buy-and-hold over [start, end]
                lastNet: float|null,       # last recorded day's return
                behind: int|null,          # SPY bars after `end`, minus 1, floor 0; null if no SPY data
                curve: {dates: [str], equity: [float]},   # equity[0] = 1 + first net
                decisions: {ok: int, failed: int} | null}
               decisions: agent kinds only; final status per (date, symbol)
               (agent_open: per date); a row with no status (legacy) counts as neither.
               null for rule kinds. A missing record has every nullable field null,
               curve {dates: [], equity: []}, decisions null.
  today        {date: str|null,    # = asOf
                open: {date, status, scanDate, nFloor, screens: [..],
                       screenReason, candidates, positions: [{symbol, target, reason}],
                       reason, fail} | null,   # null until agent_open has decided a day;
                       # scalars absent from the log line are null, lists are []
                agents: [{id, date: str, long: int, short: int, flat: int, failed: int,
                          changes: [{symbol, from: float, to: float, reason: str|null}]}],
                       # ids agent, agent_ctx in that order; [] on failure; a record
                       # with no pos files is omitted. long/short/flat count only
                       # status-ok rows on `date`; failed rows hold the prior position
                       # and are counted in `failed`. changes (max 15, biggest move
                       # first) compare ok rows to the previous decided date (absent = 0).
                rule: {date: str, held: [str], entered: [str], exited: [str]} | null,
                       # mean_reversion_real's target positions, newest bar vs the
                       # one before; held = long on the newest bar.
                paper: {date: str, orders: int, accepted: int, rejected: int,
                        rejectReasons: {reason: count}, halted: bool,
                        haltReason: str|null,
                        account: {date, cashUsd, realizedPnlTodayUsd,
                                  positions: [str]} | null} | null}
                       # newest paper tick in journal/paper_orders.jsonl.
  memory       {date: str|null, bullets: [str], legacy: bool, archiveEntries: int}
               # empty value: {date: null, bullets: [], legacy: false, archiveEntries: 0}
  experiments  list[{created, experiment, variant, sharpe, ciLo, ciHi, annReturn,
               maxDrawdown, start, end, note}], newest first; [] if none.
  health       list[{id: str, level: "ok"|"warn"|"bad", title: str, detail: str}]
  healthLevel  "ok"|"warn"|"bad"   the worst level in `health` ("ok" if it is empty).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import webbrowser
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

# src-layout: make the rhagent package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rhagent.config import load as load_config  # noqa: E402
from rhagent.evaluate import (  # noqa: E402
    MIN_RETURN_DAYS_FOR_SHARPE,
    MIN_TRADES_FOR_RATE_STATS,
    _bucket_labels,
    aggregate,
    load_run,
    spy_benchmark,
)
from rhagent.evaluate_robust import robust_table  # noqa: E402
from rhagent.data import get_bars  # noqa: E402
from rhagent.features import flatten_trades  # noqa: E402
from rhagent.forward import _agent_outage, _latest_decisions, _positions  # noqa: E402
from rhagent.memory import archive_path, read_memory  # noqa: E402
from rhagent.open_universe import _rows, _stuck  # noqa: E402

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
        "n": a["n_trades"], "nDays": a["n_return_days"],
        "won": int((trades["outcome"] == "win").sum()) if len(trades) else 0,
        "lost": int((trades["outcome"] == "loss").sum()) if len(trades) else 0,
        "pnl": notional * a["total_return"], "ret": a["total_return"],
        "gw": float(pnl_s[pnl_s > 0].sum()), "gl": float(-pnl_s[pnl_s < 0].sum()),
        "pf": a["profit_factor"], "wr": a["win_rate"], "sharpe": a["sharpe"],
        "dd": a["max_drawdown"], "avgWin": a["avg_win"], "avgLoss": a["avg_loss"],
        "avgHold": a["avg_holding_bars"],
        "uni": ", ".join(symbols) if len(symbols) <= 5 else f"universe ({len(symbols)})",
    }


def _rejected_order_count(journal_path: Path = Path("journal/paper_orders.jsonl")) -> int:
    """Real count of order_rejected events ever journaled by executor.py.
    Absent file means the paper-run journal has never been written -- that is
    genuinely zero rejections, not a fabricated figure."""
    if not journal_path.exists():
        return 0
    n = 0
    for line in journal_path.read_text().splitlines():
        if not line.strip():
            continue
        if json.loads(line).get("event") == "order_rejected":
            n += 1
    return n


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


_HOLDING_NAMES = {"short": "under 5 bars", "long": "5+ bars"}  # not the trade side


def _locked_buckets(locked_dir: Path) -> tuple[list[dict], list[dict]]:
    """Where the locked strategy's loss dollars and win dollars concentrate.

    One run only (the strategy actually being paper-traded), in dollars, and
    ranked by lift = a bucket's share of the dollars / its share of the trades.
    Ranking by raw share just lists the biggest bucket first, in both panels.
    """
    trades = flatten_trades(load_run(locked_dir)[1])
    if not len(trades):
        return [], []
    pnl = trades["pnl_abs"].astype(float)
    total_loss, total_win = float(-pnl[pnl < 0].sum()), float(pnl[pnl > 0].sum())
    rows = []
    for dim, labels in _bucket_labels(trades).items():
        if dim == "symbol":
            continue  # ~12 trades a name: noise, not a setup
        for bucket, idx in trades.groupby(labels).groups.items():
            sub = pnl.loc[idx]
            share = len(sub) / len(trades)
            loss = float(-sub[sub < 0].sum()) / total_loss if total_loss else 0.0
            win = float(sub[sub > 0].sum()) / total_win if total_win else 0.0
            rows.append({
                "dim": dim,
                "bucket": _HOLDING_NAMES.get(str(bucket), str(bucket)) if dim == "holding" else str(bucket),
                "tradeShare": share, "lossShare": loss, "winShare": win,
                "wr": float((trades.loc[idx, "outcome"] == "win").mean()),
            })

    def top(key):
        out = [{**r, "share": r[key], "lift": r[key] / r["tradeShare"]} for r in rows]
        return sorted((r for r in out if r["lift"] > 1), key=lambda r: -r["lift"])[:5]

    return top("lossShare"), top("winShare")


def _bakeoff_data(base_dir: Path, engine: str) -> list[dict]:
    df = robust_table(base_dir)
    if len(df) == 0 or not engine:
        return []
    df = df[df["engine"] == engine]
    order = {"none": 0, "conviction": 1, "bucket": 2, "winprob": 3}
    rows = []
    for overlay, grp in df.groupby(df["overlay"].fillna("none").replace("", "none")):
        # fold/bootstrap stats are None for short records (too few 60-day
        # folds / a too-short series for the bootstrap) -- rank by deflated
        # where it exists, else just take the first row rather than crashing.
        best = (
            grp.loc[grp["deflated"].idxmax()] if grp["deflated"].notna().any() else grp.iloc[0]
        )

        def _f(v):
            # DataFrame column: a None mixed with real floats upcasts to NaN,
            # not Python None -- pd.isna() catches both, `is None` alone would
            # miss the NaN case.
            return None if pd.isna(v) else float(v)

        fold_mean, fold_std = _f(best["fold_mean"]), _f(best["fold_std"])
        ci_lo, ci_hi = _f(best["ci_lo"]), _f(best["ci_hi"])
        n_days = int(best["n_return_days"])
        rows.append({
            "overlay": overlay, "point": float(best["point_sharpe"]), "deflated": _f(best["deflated"]),
            "nDays": n_days,
            "fold": f"{fold_mean:.2f}±{fold_std:.2f}" if fold_mean is not None and fold_std is not None
                    else f"n/a ({n_days}d, too few 60d folds)",
            "ci": f"[{ci_lo:.2f}, {ci_hi:.2f}]" if ci_lo is not None and ci_hi is not None else f"n/a ({n_days}d)",
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


# ---- data layer for the single-page overview (see PAYLOAD CONTRACT above) ----

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments" / "results.jsonl"
_LEVELS = ["ok", "warn", "bad"]
_RECORDS = [  # (id, label, kind, basis, note)
    ("mean_reversion_real", "Rule, honest fill", "rule", "7 bp · next-open fill",
     "The honest rule number: 7 bp costs, filled at the next open."),
    ("mean_reversion", "Rule, 1 bp", "rule", "1 bp · close fill",
     "Flattering: 1 bp cost and a same-close fill; kept only for continuity with older numbers."),
    ("agent", "Agent", "agent", "7 bp · next-open fill",
     "A language model picks long, short or flat for each stock on a fixed list every day."),
    ("agent_ctx", "Agent + market context", "agent", "7 bp · next-open fill",
     "The same agent, also shown a market-context block (SPY moves, breadth, momentum rank)."),
    ("agent_open", "Agent, no stock list", "agent", "7 bp · next-open fill",
     "The agent screens a daily scan of the wider market and picks its own names."),
]


def _safe(health: list, key: str, empty, fn, *args):
    """fn(*args), or `empty` plus a health entry -- a dashboard number must never fail the run."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        health.append({"id": f"build:{key}", "level": "warn",
                       "title": f"{key} could not be built", "detail": f"{type(e).__name__}: {e}"})
        return empty


def _spy_dates(data_dir: Path) -> list:
    path = data_dir / "SPY.csv"
    if not path.exists():
        return []
    return sorted(pd.read_csv(path, parse_dates=["date"])["date"])


def _decision_counts(eval_dir: Path, per_date: bool) -> dict:
    if per_date:  # agent_open decides one day at a time; the last row for a date wins
        latest = {r["date"]: r for r in _rows(eval_dir / "decisions.jsonl")}
    else:
        latest = _latest_decisions(eval_dir)
    status = [r.get("status") for r in latest.values()]
    return {"ok": status.count("ok"), "failed": sum(s not in ("ok", None) for s in status)}


def _empty_record(spec: tuple) -> dict:
    rid, label, kind, basis, note = spec
    return {"id": rid, "label": label, "kind": kind, "basis": basis, "note": note,
            "missing": True, "days": None, "start": None, "end": None, "totalReturn": None,
            "sharpe": None, "sharpeNeeds": None, "maxDrawdown": None, "spyReturn": None,
            "lastNet": None, "behind": None, "curve": {"dates": [], "equity": []},
            "decisions": None}


def _record(spec: tuple, forward_dir: Path, spy_dates: list, data_dir: Path) -> dict:
    out = _empty_record(spec)
    eval_dir = forward_dir / spec[0]
    if not (eval_dir / "returns.csv").exists():
        return out
    net = pd.read_csv(eval_dir / "returns.csv", parse_dates=["date"]).set_index("date")["net"].astype(float)
    if not len(net):
        return out
    a = aggregate(pd.DataFrame(), net)
    spy = spy_benchmark(net.index, data_dir)
    end = net.index[-1]
    out.update({
        "missing": False, "days": a["n_return_days"],
        "start": str(net.index[0])[:10], "end": str(end)[:10],
        "totalReturn": a["total_return"], "sharpe": a["sharpe"],
        "sharpeNeeds": MIN_RETURN_DAYS_FOR_SHARPE if a["sharpe"] is None else None,
        "maxDrawdown": a["max_drawdown"],
        "spyReturn": spy["return"] if spy["n_days"] else None,
        "lastNet": float(net.iloc[-1]),
        "behind": max(0, sum(d > end for d in spy_dates) - 1) if spy_dates else None,
        "curve": {"dates": [str(d)[:10] for d in net.index],
                  "equity": [float(v) for v in (1.0 + net).cumprod()]},
        "decisions": _decision_counts(eval_dir, spec[0] == "agent_open") if spec[2] == "agent" else None,
    })
    return out


def _today_open(forward_dir: Path) -> dict | None:
    rows = _rows(forward_dir / "agent_open" / "decisions.jsonl")
    if not rows:
        return None
    r = rows[-1]
    return {"date": r.get("date"), "status": r.get("status"), "scanDate": r.get("scan_date"),
            "nFloor": r.get("n_floor"), "screens": r.get("screens") or [],
            "screenReason": r.get("screen_reason"), "candidates": r.get("candidates"),
            "positions": [{"symbol": p.get("symbol"), "target": p.get("target"), "reason": p.get("reason")}
                          for p in r.get("positions") or []],
            "reason": r.get("reason"), "fail": r.get("fail")}


def _today_agent(forward_dir: Path, rid: str) -> dict | None:
    eval_dir = forward_dir / rid
    frames = []
    for f in sorted(eval_dir.glob("pos_*.csv")):
        df = pd.read_csv(f)
        if "status" not in df:
            df["status"] = "ok"  # rows seeded before the status column existed
        df["symbol"] = f.stem[len("pos_"):]
        frames.append(df)
    if not frames:
        return None
    pos = pd.concat(frames)
    dates = sorted(pos["date"].unique())
    date = dates[-1]
    prev = pos[pos["date"] == dates[-2]].set_index("symbol")["pos"] if len(dates) > 1 else pd.Series(dtype=float)
    cur = pos[pos["date"] == date]
    ok = cur[cur["status"] == "ok"]
    reasons = {s: r.get("reason") for (d, s), r in _latest_decisions(eval_dir).items() if d == date}
    changes = [{"symbol": r.symbol, "from": float(prev.get(r.symbol, 0.0)), "to": float(r.pos),
                "reason": reasons.get(r.symbol)}
               for r in ok.itertuples() if float(r.pos) != float(prev.get(r.symbol, 0.0))]
    changes.sort(key=lambda c: (-abs(c["to"] - c["from"]), c["symbol"]))
    return {"id": rid, "date": str(date), "long": int((ok["pos"] > 0).sum()),
            "short": int((ok["pos"] < 0).sum()), "flat": int((ok["pos"] == 0).sum()),
            "failed": int(len(cur) - len(ok)), "changes": changes[:15]}


def _today_agents(forward_dir: Path) -> list:
    return [a for a in (_today_agent(forward_dir, rid) for rid in ("agent", "agent_ctx")) if a]


def _today_rule(cfg, data_dir: Path) -> dict | None:
    """Same call rhagent.paper_run makes for the rule's target positions."""
    bars = get_bars(list(cfg.strategy.universe), "", "", fetch=lambda *a: {}, cache_dir=data_dir)
    pos, _ = _positions(cfg, cfg.strategy.name, bars, Path())
    pos = {s: p for s, p in pos.items() if len(p)}
    if not pos:
        return None
    last = {s: (float(p.iloc[-1]), float(p.iloc[-2]) if len(p) > 1 else 0.0) for s, p in pos.items()}
    return {"date": str(max(p.index[-1] for p in pos.values()))[:10],
            "held": sorted(s for s, (now, _) in last.items() if now > 0),
            "entered": sorted(s for s, (now, was) in last.items() if now > 0 >= was),
            "exited": sorted(s for s, (now, was) in last.items() if was > 0 >= now)}


def _today_paper(journal: Path) -> dict | None:
    """Newest paper tick: the events journaled since the previous paper_run_end /
    paper_run_halted, closed by the newest one (rhagent.paper_run writes both)."""
    rows = _rows(journal / "paper_orders.jsonl")
    ends = [i for i, r in enumerate(rows) if r.get("event") in ("paper_run_end", "paper_run_halted")]
    if not ends:
        return None
    end = rows[ends[-1]]
    tick = rows[(ends[-2] + 1 if len(ends) > 1 else 0):ends[-1]]
    reasons: dict = {}
    for r in tick:
        if r.get("event") == "order_rejected":
            reasons[r.get("reason")] = reasons.get(r.get("reason"), 0) + 1
    halted = end["event"] == "paper_run_halted"
    acct = None
    if (journal / "paper_account.json").exists():
        raw = json.loads((journal / "paper_account.json").read_text())
        acct = {"date": raw.get("date"), "cashUsd": raw.get("cash_usd"),
                "realizedPnlTodayUsd": raw.get("realized_pnl_today_usd"),
                "positions": sorted(raw.get("positions", {}))}
    return {"date": str(end.get("ts", ""))[:10], "orders": int(end.get("n_orders", 0)),
            "accepted": int(end.get("n_accepted", 0)), "rejected": sum(reasons.values()),
            "rejectReasons": reasons, "halted": halted,
            "haltReason": end.get("reason") if halted else None, "account": acct}


def _memory(journal: Path) -> dict:
    mem_path = journal / "agent_memory.md"
    arch = archive_path(mem_path)
    lines = read_memory(mem_path).strip().splitlines()
    return {"date": lines[0][3:].strip() if lines else None,
            "bullets": [re.sub(r"^[-*•]\s+", "", x.strip()) for x in lines[1:] if x.strip()],
            "legacy": not arch.exists(),
            "archiveEntries": ("\n" + arch.read_text()).count("\n## ") if arch.exists() else 0}


def _experiments(path: Path) -> list[dict]:
    rows = _rows(path)
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["created"], i), reverse=True)
    return [{"created": r["created"], "experiment": r["experiment"], "variant": r["variant"],
             "sharpe": r.get("sharpe"), "ciLo": r.get("sharpe_ci_lo"), "ciHi": r.get("sharpe_ci_hi"),
             "annReturn": r.get("ann_return"), "maxDrawdown": r.get("max_drawdown"),
             "start": r.get("start"), "end": r.get("end"), "note": r.get("note", "")}
            for r in (rows[i] for i in order)]


def _item(id_: str, level: str, title: str, detail: str = "") -> dict:
    return {"id": id_, "level": level, "title": title, "detail": detail}


def _chk_records(records: list) -> list:
    out = []
    for r in records:
        rid, b = r["id"], r["behind"]
        if r["missing"]:
            lvl = "warn" if rid == "agent_open" else "bad"
            why = "starts on the next scheduled run" if rid == "agent_open" else "no returns recorded"
            out.append(_item(f"record:{rid}", lvl, f"{r['label']}: no record", why))
        elif b is None:
            out.append(_item(f"record:{rid}", "warn", f"{r['label']}: freshness unknown",
                             "no SPY bars to compare against"))
        else:
            lvl = "ok" if b == 0 else "warn" if b <= 3 else "bad"
            out.append(_item(f"record:{rid}", lvl,
                             f"{r['label']}: " + ("up to date" if b == 0 else f"{b} bar(s) behind"),
                             f"last recorded day {r['end']}"))
    return out


def _chk_outage(forward_dir: Path) -> list:
    out = []
    for rid in ("agent", "agent_ctx"):
        if (forward_dir / rid).exists():
            down = _agent_outage(forward_dir / rid)
            out.append(_item(f"outage:{rid}", "bad" if down else "ok",
                             f"{rid}: " + ("model outage" if down else "deciding"),
                             "every decision on the newest bar failed" if down else ""))
    return out


def _chk_open(forward_dir: Path, today_open: dict | None, data_dir: Path, now: datetime) -> list:
    if today_open is None:
        return []
    failed = today_open["status"] == "failed"
    out = [_item("open:decision", "bad" if failed else "ok",
                 "agent_open: " + (f"newest decision failed ({today_open['date']})" if failed
                                   else "newest decision ok"),
                 str(today_open["fail"] or "") if failed else "")]
    stuck = _stuck(forward_dir / "agent_open", data_dir / "open", now.replace(tzinfo=None))
    if stuck:
        out.append(_item("open:stuck", "warn", "agent_open: days unrealized for over 5 days",
                         "; ".join(f"{d} (no bar after it for: {', '.join(m) or 'calendar'})"
                                   for d, m in stuck.items())))
    return out


def _chk_decisions(forward_dir: Path, today) -> list:
    """Failed-decision share over the last 30 days, per agent record.

    Cumulative failed counts are meaningless -- retries heal old failures, so
    the totals in `decisions` only ever grow (agent: 845, agent_ctx: 585 on
    real data, nearly all healed). What matters is whether decisions are
    failing NOW: more failures than successes in the window means the record
    is mostly outage, not signal -- the hole record freshness can't see,
    because returns.csv keeps appending flat days through an outage.
    """
    since = (today - timedelta(days=30)).isoformat()
    out = []
    for rid, per_date in (("agent", False), ("agent_ctx", False), ("agent_open", True)):
        log = forward_dir / rid / "decisions.jsonl"
        if not log.exists():
            continue
        if per_date:  # agent_open decides one day at a time; last row per date wins
            latest = {r["date"]: r for r in _rows(log)}
            rows = [r for d, r in latest.items() if d >= since]
        else:
            latest = _latest_decisions(forward_dir / rid)
            rows = [r for (d, _s), r in latest.items() if d >= since]
        ok = sum(r.get("status") == "ok" for r in rows)
        failed = sum(r.get("status") not in (None, "ok") for r in rows)
        total = ok + failed  # legacy rows (no status) count as neither
        if not total:
            continue
        lvl = "bad" if failed > ok else "warn" if failed else "ok"
        out.append(_item(f"decisions:{rid}", lvl,
                         f"{rid}: {failed} failed of {total} decisions (30d)",
                         "failures hold the prior position; healed dates count ok"
                         if failed else ""))
    return out


def _chk_memory(memory: dict, agent: dict) -> list:
    out = []
    if memory["legacy"]:
        out.append(_item("memory:legacy", "warn", "Memory is the old log",
                         "reflection has not rewritten the memory yet"))
    if memory["date"] and agent["end"]:
        age = (pd.Timestamp(agent["end"]) - pd.Timestamp(memory["date"])).days
        out.append(_item("memory:fresh", "warn" if age > 3 else "ok",
                         "Memory " + (f"is {age} days older than the agent record" if age > 3 else "is current"),
                         f"memory {memory['date']}, agent record ends {agent['end']}"))
    return out


def _chk_data(as_of: str | None, now: datetime) -> list:
    if as_of is None:
        return [_item("data:spy", "warn", "No SPY price data", "data/SPY.csv is missing or empty")]
    age = (pd.Timestamp(now.replace(tzinfo=None)).normalize() - pd.Timestamp(as_of)).days
    return [_item("data:spy", "warn" if age > 4 else "ok",
                  "Price data " + (f"is {age} days old" if age > 4 else "is fresh"),
                  f"newest SPY bar {as_of}")]


def _health(records, today, memory, as_of, forward_dir, data_dir, now) -> list:
    agent = next(r for r in records if r["id"] == "agent")
    checks = {"records": lambda: _chk_records(records), "outage": lambda: _chk_outage(forward_dir),
              "open": lambda: _chk_open(forward_dir, today["open"], data_dir, now),
              "decisions": lambda: _chk_decisions(forward_dir, now.date()),
              "memory": lambda: _chk_memory(memory, agent), "data": lambda: _chk_data(as_of, now)}
    out: list = []
    for key, fn in checks.items():
        out += _safe(out, f"health:{key}", [], fn)
    return out


def _new_sections(base_dir: Path, cfg, data_dir: Path, now: datetime) -> dict:
    journal, health = base_dir.parent, []
    forward_dir = journal / "forward"
    spy = _safe(health, "asOf", [], _spy_dates, data_dir)
    as_of = str(spy[-1])[:10] if spy else None
    records = [_safe(health, f"record:{spec[0]}", _empty_record(spec), _record, spec, forward_dir, spy, data_dir)
               for spec in _RECORDS]
    today = {"date": as_of,
             "open": _safe(health, "today.open", None, _today_open, forward_dir),
             "agents": _safe(health, "today.agents", [], _today_agents, forward_dir),
             "rule": _safe(health, "today.rule", None, _today_rule, cfg, data_dir),
             "paper": _safe(health, "today.paper", None, _today_paper, journal)}
    memory = _safe(health, "memory", {"date": None, "bullets": [], "legacy": False, "archiveEntries": 0},
                   _memory, journal)
    experiments = _safe(health, "experiments", [], _experiments, _EXPERIMENTS)
    health += _safe(health, "health", [], _health, records, today, memory, as_of, forward_dir, data_dir, now)
    level = max((h["level"] for h in health), key=_LEVELS.index, default="ok")
    return {"asOf": as_of, "records": records, "today": today, "memory": memory,
            "experiments": experiments, "health": health, "healthLevel": level}


def _build_data(base_dir: Path) -> dict:
    runs = _run_dirs(base_dir)
    if not runs:
        raise SystemExit(f"no runs found under {base_dir} — run rhagent.papertrade first")
    cfg = load_config()
    locked_dir = _locked_run(runs, cfg)
    _, _, net = load_run(locked_dir)
    equity = (1.0 + net.astype(float)).cumprod()
    forward_dir = base_dir.parent / "forward"
    loss_buckets, win_buckets = _locked_buckets(locked_dir)
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
            "daily_loss_limit_usd": g.daily_loss_limit_usd,
            # "live" reflects the LIVE env var only -- it does NOT mean orders
            # are ever actually placed. paper_run.py hardcodes MockBroker
            # unconditionally, so no order path reaches a real broker
            # regardless of this flag (see the "NO LIVE ORDER PATH" badge).
            "live": not cfg.dry_run,
            "halt": HALT_FILE.exists(), "model": cfg.agent.model,
            "rejectedOrders": _rejected_order_count(),
        },
        "thresholds": {
            "minTrades": MIN_TRADES_FOR_RATE_STATS,
            "minReturnDays": MIN_RETURN_DAYS_FOR_SHARPE,
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
        **_new_sections(base_dir, cfg, Path("data"), datetime.now(timezone.utc)),
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
.tile .v{font-size:20px;font-weight:700;font-family:'IBM Plex Mono',monospace;letter-spacing:-.01em;overflow-wrap:anywhere}
.tile .v.sm{font-size:13px;line-height:1.35}
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
  <div id="healthbanner" style="margin-bottom:16px"></div>
  <section style="margin-top:0">
    <h2 class="mu" style="margin:0 0 12px;font-size:12px;letter-spacing:.14em;text-transform:uppercase">The verdict · agent vs baseline · forward paper track</h2>
    <div id="verdict"></div>
  </section>

  <section class="card">
    <div class="hdr" style="align-items:baseline;gap:10px;margin-bottom:6px">
      <div><h3>Forward records</h3><div class="sub">All five live paper tracks on one basis where noted. A record only counts days where every name got a real decision.</div></div>
      <span id="asof" class="m mu" style="font-size:11px"></span>
    </div>
    <div class="scroll"><table style="min-width:980px"><thead><tr>
      <th class="l">record</th><th class="l">basis</th><th>days</th><th class="l">window</th><th>equity</th>
      <th>total return</th><th>sharpe</th><th>max DD</th><th>vs SPY</th><th>behind</th><th>decisions</th>
    </tr></thead><tbody id="records"></tbody></table></div>
  </section>

  <section class="two" style="grid-template-columns:minmax(0,1.2fr) minmax(0,0.8fr)">
    <div class="card">
      <div class="hdr" style="align-items:baseline;gap:10px;margin-bottom:6px">
        <div><h3>Today</h3><div class="sub" id="todaysub"></div></div>
      </div>
      <div id="today" class="col"></div>
    </div>
    <div class="card">
      <div class="hdr" style="align-items:baseline;gap:10px;margin-bottom:6px">
        <div><h3>Agent memory</h3><div class="sub">The lessons block the agent is (or isn't) running with.</div></div>
        <span id="memlegacy"></span>
      </div>
      <div id="memory" class="col" style="font-size:12.5px"></div>
    </div>
  </section>

  <section class="card">
    <h3>Equity curve · locked strategy candidate</h3><div class="sub" id="chartsub"></div>
    <div id="chart"></div>
    <div class="mu" style="margin-top:6px;font-size:11.5px">green dot = peak, red dot = trough of the max-drawdown window (shaded); dashed line = break-even.</div>
  </section>

  <section class="two" style="grid-template-columns:minmax(0,0.9fr) minmax(0,1.1fr)">
    <div class="card">
      <h3>Guardrails · armed</h3><div class="sub">Hard caps enforced in code: the model cannot talk its way past them.</div>
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
    <div class="sub">Measured over the same window the strategy was selected on, which is selection-biased, not out-of-sample evidence.</div>
    <div id="scoretiles" class="grid" style="grid-template-columns:repeat(auto-fit,minmax(130px,1fr))"></div>
    <div id="scorespy" class="mu" style="margin-top:12px;font-size:11.5px"></div>
  </section>

  <section class="two" style="grid-template-columns:minmax(0,0.85fr) minmax(0,1.15fr)">
    <div class="col" style="gap:20px">
      <div class="card">
        <h3>Where losses concentrate</h3><div class="sub">Share of dollars lost, locked strategy only. Ranked by how far a bucket exceeds its share of trades.</div>
        <div id="buckets" class="col"></div>
      </div>
      <div class="card">
        <h3>Where we win</h3><div class="sub">Share of dollars won, locked strategy only. Same ranking.</div>
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
    <div class="hdr" style="align-items:baseline;gap:10px;margin-bottom:6px">
      <div><h3>Experiments</h3><div class="sub">Recorded research experiments, newest first.</div></div>
    </div>
    <div class="scroll"><table style="min-width:860px"><thead><tr>
      <th class="l">created</th><th class="l">experiment</th><th class="l">variant</th><th>sharpe</th><th>95% CI</th>
      <th>ann. return</th><th>max DD</th><th class="l">window</th><th class="l">note</th>
    </tr></thead><tbody id="experiments"></tbody></table></div>
  </section>

  <section class="card">
    <div class="hdr" style="align-items:center">
      <div><h3>Runbook</h3><div class="mu" style="font-size:12px">Every command that drives this system. Click to copy.</div></div>
      <div class="hdr" style="align-items:center;gap:10px"><span id="trigger-status" class="m mu" style="font-size:11px"></span><button id="trigger-btn" class="btn">Run new research run</button></div>
    </div>
    <div id="runbook" class="col" style="gap:8px;margin-top:16px"></div>
  </section>

  <section id="agentnotes" class="card" style="display:none">
    <h3>Agent's own lessons (self-written)</h3><div class="sub">The agent's current lessons, rewritten after each run.</div>
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
// A metric below its sample-size floor (see evaluate.py's MIN_TRADES_FOR_RATE_STATS
// / MIN_RETURN_DAYS_FOR_SHARPE) comes through as null, not a fake 0 -- render that
// honestly instead of crashing on `null.toFixed()` or silently showing "$0.00".
const naOr = (v, fmt, need, have, unit) => v == null ? `n/a - needs ${need}+ ${unit}, have ${have}` : fmt(v);
const engineColor = e => ({ mean_reversion: 'var(--accent)', momentum: 'var(--warn)', linreg: 'var(--purple)', agent: 'var(--up)' }[e] || 'var(--muted)');
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// one tile, one chip row, one bucket bar — reused by every panel.
// `small` shrinks the value type for long unbreakable strings (a model slug),
// which at 20px wraps to 6 lines and drags the whole tile row taller with it.
const tile = (v, k, color, small) => `<div class="tile"><div class="v${small ? ' sm' : ''}" style="color:${color || 'var(--fg)'}">${esc(v)}</div><div class="k">${esc(k)}</div></div>`;
// A 65-name universe printed in full floods every panel it appears in; collapse
// it the way _run_row already does and keep the full list in the tooltip.
const universe = syms => !syms || !syms.length ? 'no symbols'
  : syms.length <= 5 ? syms.join(', ') : `universe (${syms.length})`;
const chips = (attr, items, active) => items.map(([id, label, dot]) =>
  `<button class="chip" ${attr}="${esc(id)}" aria-pressed="${id === active}">${dot ? `<span class="dot" style="color:${dot}"></span>` : ''}${esc(label)}</button>`).join('');
const bucketRow = (b, color) => `<div>
  <div class="hdr" style="align-items:baseline;gap:8px;margin-bottom:5px"><span style="font-size:12.5px"><span class="m mu" style="font-size:11px">${esc(b.dim)}</span> · ${esc(b.bucket)}</span><span class="m" style="font-size:12.5px;font-weight:700;color:${color}">${pctAbs(b.share, 1)}</span></div>
  <div class="bar"><i style="width:${(b.share * 100).toFixed(0)}%;background:${color}"></i></div>
  <div class="m mu" style="font-size:10.5px">${pctAbs(b.tradeShare, 0)} of trades · ${b.lift.toFixed(1)}× its size · ${pctAbs(b.wr, 0)} win rate</div></div>`;

function renderHeaderPills() {
  const g = DATA.guardrails, U = 'var(--up)', W = 'var(--warn)', D = 'var(--down)';
  $('headerpills').innerHTML = `
    <span class="pill" style="color:${W};border-color:currentColor" title="paper_run.py hardcodes MockBroker unconditionally -- no code path here can place a real order, regardless of the LIVE env var"><span class="dot"></span>NO LIVE ORDER PATH</span>
    <span class="pill" style="color:${g.halt ? D : U};border-color:currentColor">HALT · ${g.halt ? 'SET' : 'CLEAR'}</span>
    <a href="${DATA.actionsUrl}"><img src="${DATA.actionsUrl}/badge.svg?branch=main" alt="daily paper-run status"></a>
    <span class="pill">upd ${esc(DATA.updated)}</span>`;
}

function renderVerdict() {
  const { agent, baseline: base, real } = DATA.forward;
  // Both legs need their own tracked days: a leg with 0 days sits at exactly
  // $0.00, which would otherwise "beat" a real leg that is merely down a little.
  const days = Math.min(agent.days, base.days);
  let badge = 'TOO EARLY TO CALL',
    note = agent.days && base.days
      ? `Shortest track has ${days} day(s) logged. Verdict needs weeks of OOS data.`
      : `${!agent.days ? 'Agent' : 'Baseline'} leg has no forward days logged yet, so there is nothing to compare.`;
  if (days >= 5) {
    badge = agent.pnl > base.pnl ? 'AGENT LEADS' : base.pnl > agent.pnl ? 'BASELINE LEADS' : 'TIED';
    note = `Forward P&L over the tracked window: ${badge.toLowerCase()}.`;
  }
  const fill = leg => `${leg.fillMode === 'next_open' ? 'next-open' : 'same-close'} fill @ ${leg.costBps}bp`;
  const sub = leg => `${leg.days} day(s) · bal ${money(leg.notional + leg.pnl, 0)} · net ${pct(leg.ret)} · ${universe(leg.symbols)} · ${fill(leg)}`;
  const spyLine = leg => leg.spy && leg.spy.start
    ? `SPY buy&amp;hold ${leg.spy.start}→${leg.spy.end}: <b style="color:var(--fg)">${pct(leg.spy.return)}</b>`
    : 'SPY buy&amp;hold: not enough tracked days yet';
  const side = (leg, title, color, align) => `<div style="padding:22px 26px;display:flex;flex-direction:column;gap:6px;text-align:${align}">
      <div style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:${color};font-weight:600">${title}</div>
      <div class="m" style="font-size:34px;font-weight:700;letter-spacing:-.02em">${money(leg.pnl)}</div>
      <div class="m mu" style="font-size:12px" title="${esc((leg.symbols || []).join(', '))}">${esc(sub(leg))}</div>
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
    ${real.days ? banner('rgba(77,184,255,.22)', 'rgba(77,184,255,.06)', `<b style="color:var(--accent)">Honest-fill record (mean_reversion_real):</b> ${money(real.pnl)} over ${esc(sub(real))}. The baseline above is ${fill(base)}, flattering by comparison, kept only because its track record predates this fill.`) : ''}
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
    ['daily realized-loss kill', money(g.daily_loss_limit_usd, 0)],
    // real, journaled count -- never a hardcoded number.
    ['order rejections (all-time)', String(g.rejectedOrders)],
    ['model', g.model.split('/').pop(), true],
  ].map(([k, v, small]) => tile(v, k, 'var(--up)', small)).join('');
}

function renderBakeoff() {
  const rows = DATA.bakeoff;
  if (!rows.length) { $('bakeoff').innerHTML = '<tr><td colspan="6" class="l mu">no bake-off data</td></tr>'; return; }
  const maxPoint = Math.max(...rows.map(b => b.point));
  $('bakeoff').innerHTML = rows.map(b => `<tr style="background:${b.beats ? 'rgba(5,196,107,.07)' : 'transparent'}">
    <td class="l" style="color:${b.beats ? 'var(--up)' : 'var(--fg)'};font-weight:${b.beats ? 700 : 500}">${esc(b.overlay)}</td>
    <td><div style="display:flex;align-items:center;justify-content:flex-end;gap:8px"><span class="bar" style="width:44px;height:5px"><i style="width:${(b.point / maxPoint * 100).toFixed(0)}%;background:${b.beats ? 'var(--up)' : 'var(--muted)'}"></i></span>${b.point.toFixed(2)}</div></td>
    <td class="mu">${b.deflated == null ? `n/a (${b.nDays}d)` : b.deflated.toFixed(2)}</td><td class="mu">${esc(b.fold)}</td><td class="mu">${esc(b.ci)}</td>
    <td class="c">${b.beats ? '✓ beats' : 'no'}</td></tr>`).join('');
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
      <td class="l mu">${esc(r.overlay || 'none')}</td><td>${r.n}</td>
      <td title="${r.wr == null ? `needs ${DATA.thresholds.minTrades}+ trades, have ${r.n}` : ''}">${r.wr == null ? 'n/a' : pctAbs(r.wr, 1)}</td>
      <td title="${r.pf == null ? `needs ${DATA.thresholds.minTrades}+ trades, have ${r.n}` : ''}">${r.pf == null ? 'n/a' : num(r.pf)}</td>
      <td style="color:${ud(r.pnl)};font-weight:600">${signed(r.pnl, 0)}</td>
      <td><div style="display:flex;align-items:center;justify-content:flex-end;gap:8px"><span class="bar" style="width:52px;height:5px;flex:none"><i style="width:${(Math.abs(r.ret) / maxAbsRet * 100).toFixed(0)}%;background:${ud(r.ret)}"></i></span><span style="color:${ud(r.ret)};min-width:56px">${pct(r.ret, 1)}</span></div></td></tr>`;
  }).join('');
}

const SCORE_TILES = S => {
  const t = DATA.thresholds, rate = (v, fmt) => naOr(v, fmt, t.minTrades, S.n, 'trades');
  return [
    ['balance', money(S.balance), ud(S.balance - S.notional)], ['net P&L', signed(S.pnl), ud(S.pnl)],
    ['total return', pct(S.ret), ud(S.ret)], ['trades', String(S.n)],
    ['win rate', rate(S.wr, v => pctAbs(v, 1))],
    ['profit factor', rate(S.pf, num), S.pf == null ? 'var(--muted)' : ud(S.pf - 1)],
    ['avg win', rate(S.avgWin, v => money(v)), 'var(--up)'],
    ['avg loss', rate(S.avgLoss, v => money(v)), 'var(--down)'],
    ['sharpe', naOr(S.sharpe, v => v.toFixed(2), t.minReturnDays, S.nDays, 'return-days')],
    ['max drawdown', pctAbs(S.dd, 2), 'var(--down)'],
    ['avg holding', rate(S.avgHold, v => v.toFixed(1) + ' bars')],
    ['gross win', money(S.gw, 0), 'var(--up)'], ['gross loss', money(-S.gl, 0), 'var(--down)'],
  ];
};

function renderScorecard() {
  const S = DATA.winScore, spy = S.spy;
  $('winid').textContent = S.id;
  $('scoretiles').innerHTML = SCORE_TILES(S).map(([k, v, c]) => tile(v, k, c)).join('');
  $('scorespy').textContent = spy && spy.start
    ? `SPY buy-and-hold over the same window (${spy.start} → ${spy.end}): ${pct(spy.return)} vs strategy total return ${pct(S.ret)}. The strategy is not always fully invested, so this isn't apples-to-apples on exposure.`
    : 'SPY buy-and-hold benchmark unavailable for this window (price cache does not cover it).';
}

function renderBuckets() {
  $('buckets').innerHTML = DATA.buckets.map(b => bucketRow(b, 'var(--down)')).join('');
  $('winbuckets').innerHTML = DATA.winBuckets.map(b => bucketRow(b, 'var(--up)')).join('');
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

// ---- overview sections: health, five forward records, today, memory, experiments ----
// Every field the payload contract marks nullable is guarded here -- a missing
// section must render empty, never throw (one throw would blank every panel
// registered after it in the render list below).

function renderHealth() {
  const items = DATA.health || [];
  const el = $('healthbanner');
  if (!items.length) { el.innerHTML = ''; return; }
  const lvl = { ok: 'var(--up)', warn: 'var(--warn)', bad: 'var(--down)' };
  const c = lvl[DATA.healthLevel] || 'var(--line)';
  const bad = items.filter(h => h.level !== 'ok').slice(0, 6);
  const more = items.filter(h => h.level !== 'ok').length - bad.length;
  el.innerHTML = `<div class="note" style="border:1px solid ${c};background:var(--panel)">
    <div class="hdr" style="align-items:center;gap:10px;flex-wrap:wrap">
      <span class="tag" style="background:${c}22;color:${c}">${esc((DATA.healthLevel || 'ok').toUpperCase())}</span>
      <span class="mu" style="font-size:12px">${bad.length
        ? bad.map(h => `<b style="color:${lvl[h.level] || 'var(--fg)'}">${esc(h.title)}</b>${h.detail ? ` — ${esc(h.detail)}` : ''}`).join('<br>')
          + (more > 0 ? `<br><span class="mu">+${more} more</span>` : '')
        : 'all checks passing'}</span>
    </div></div>`;
}

const spark = eq => {
  if (!eq || eq.length < 2) return '<span class="mu">—</span>';
  const w = 96, h = 24;
  const lo = Math.min(...eq), hi = Math.max(...eq), sp = (hi - lo) || 1;
  const pts = eq.map((v, i) => `${(i / (eq.length - 1) * w).toFixed(1)},${(h - 2 - (v - lo) / sp * (h - 4)).toFixed(1)}`).join(' ');
  const c = eq[eq.length - 1] >= eq[0] ? 'var(--up)' : 'var(--down)';
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="display:block"><polyline points="${pts}" fill="none" stroke="${c}" stroke-width="1.5"/></svg>`;
};

function renderRecords() {
  const recs = DATA.records || [];
  $('asof').textContent = DATA.asOf ? `data through ${DATA.asOf}` : '';
  $('records').innerHTML = recs.map(r => {
    if (r.missing) {
      return `<tr><td class="l"><b>${esc(r.label)}</b><div class="mu" style="font-size:10.5px">${esc(r.note)}</div></td>
        <td class="l mu" style="font-size:11px">${esc(r.basis)}</td>
        <td colspan="9" class="l mu">no record yet</td></tr>`;
    }
    const sharpe = r.sharpe == null
      ? `<span class="mu" title="needs ${r.sharpeNeeds} return-days, have ${r.days}">n/a</span>` : r.sharpe.toFixed(2);
    const dec = r.decisions == null ? '—'
      : `${r.decisions.ok} ok · ${r.decisions.failed ? `<b style="color:var(--down)">${r.decisions.failed} failed</b>` : '0 failed'}`;
    const behind = r.behind == null ? '—'
      : r.behind === 0 ? '<span style="color:var(--up)">current</span>'
      : `<span style="color:var(--warn)">${r.behind} behind</span>`;
    return `<tr>
      <td class="l"><b>${esc(r.label)}</b>${r.kind === 'agent' ? ' <span class="tag" style="background:rgba(77,184,255,.12);color:var(--accent);font-size:9.5px">AGENT</span>' : ''}
        <div class="mu" style="font-size:10.5px">${esc(r.note)}</div></td>
      <td class="l mu" style="font-size:11px">${esc(r.basis)}</td>
      <td>${r.days}</td>
      <td class="l mu" style="font-size:11px">${esc(r.start)} → ${esc(r.end)}</td>
      <td>${spark(r.curve && r.curve.equity)}</td>
      <td style="color:${ud(r.totalReturn)};font-weight:600">${r.totalReturn == null ? '—' : pct(r.totalReturn, 1)}</td>
      <td>${sharpe}</td>
      <td style="color:var(--down)">${r.maxDrawdown == null ? '—' : pctAbs(r.maxDrawdown, 1)}</td>
      <td>${r.spyReturn == null ? '—' : `<span style="color:${ud(r.spyReturn)}">${pct(r.spyReturn, 1)}</span>`}</td>
      <td>${behind}</td>
      <td class="mu" style="font-size:11px">${dec}</td></tr>`;
  }).join('') || '<tr><td colspan="11" class="l mu">no records</td></tr>';
}

const todaySub = (title, inner) => `<div>
  <div style="font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600;margin-bottom:6px">${esc(title)}</div>
  ${inner}</div>`;

function renderToday() {
  const t = DATA.today || {};
  $('todaysub').textContent = t.date ? `as of ${t.date}` : '';
  const blocks = [];

  const o = t.open;
  blocks.push(todaySub('agent_open scan', !o ? '<span class="mu">no scan decision recorded yet</span>' : `
    <div class="m" style="font-size:12px">${o.date ? esc(o.date) + ' · ' : ''}status
      <b style="color:${o.status === 'ok' ? 'var(--up)' : 'var(--down)'}">${esc(o.status || '?')}</b>
      ${o.scanDate ? ` · scan ${esc(o.scanDate)}` : ''}${o.nFloor != null ? ` · floor ${o.nFloor}` : ''}</div>
    <div class="mu" style="font-size:11.5px;margin-top:4px">${esc(o.screenReason || '')}${o.screens && o.screens.length ? ' · screens: ' + o.screens.map(esc).join(', ') : ''}${o.candidates != null ? ` · ${o.candidates} candidates` : ''}</div>
    ${o.fail ? `<div style="font-size:11.5px;color:var(--down);margin-top:4px">${esc(o.fail)}</div>` : ''}
    <div style="margin-top:6px">${(o.positions || []).map(p =>
      `<span class="pill" style="border-radius:6px;margin:0 4px 4px 0">${esc(p.symbol)} ${p.target > 0 ? '+' : p.target < 0 ? '−' : '·'}${p.reason ? ' ' + esc(p.reason) : ''}</span>`).join('') || '<span class="mu">no positions</span>'}</div>`));

  const agents = t.agents || [];
  blocks.push(todaySub('agents today', agents.length ? agents.map(a => `
    <div style="margin-bottom:10px">
      <div class="m" style="font-size:12px"><b>${esc(a.id)}</b> · ${esc(a.date)} ·
        <span style="color:var(--up)">${a.long}L</span> <span style="color:var(--down)">${a.short}S</span> ${a.flat} flat
        ${a.failed ? ` · <b style="color:var(--down)">${a.failed} failed</b>` : ''}</div>
      ${(a.changes || []).slice(0, 8).map(c =>
        `<div class="m mu" style="font-size:11px">${esc(c.symbol)} ${c.from} → ${c.to}${c.reason ? ' · ' + esc(c.reason) : ''}</div>`).join('')
        || '<div class="mu" style="font-size:11px">no position changes</div>'}
    </div>`).join('') : '<span class="mu">no agent positions yet</span>'));

  const r = t.rule;
  blocks.push(todaySub('rule · mean_reversion_real', r ? `
    <div class="m" style="font-size:12px">${esc(r.date)}</div>
    ${['held', 'entered', 'exited'].map(k =>
      `<div class="mu" style="font-size:11px;margin-top:4px"><b style="color:var(--fg)">${k}</b>: ${(r[k] || []).join(', ') || '—'}</div>`).join('')}`
    : '<span class="mu">no rule positions</span>'));

  const p = t.paper;
  blocks.push(todaySub('paper account', p ? `
    <div class="m" style="font-size:12px">${p.orders} orders · ${p.accepted} accepted · ${p.rejected} rejected${p.halted ? ' · <b style="color:var(--down)">HALTED</b>' : ''}</div>
    ${p.haltReason ? `<div style="font-size:11.5px;color:var(--down);margin-top:3px">${esc(p.haltReason)}</div>` : ''}
    ${p.rejectReasons && Object.keys(p.rejectReasons).length ? `<div class="mu" style="font-size:11px;margin-top:3px">${Object.entries(p.rejectReasons).map(([k, v]) => `${esc(k)} ×${v}`).join(' · ')}</div>` : ''}
    ${p.account ? `<div class="mu" style="font-size:11px;margin-top:5px">cash ${p.account.cashUsd == null ? '—' : money(p.account.cashUsd, 0)} · today ${p.account.realizedPnlTodayUsd == null ? '—' : signed(p.account.realizedPnlTodayUsd, 0)} · ${p.account.positions.length} positions</div>` : ''}`
    : '<span class="mu">no paper tick recorded</span>'));

  $('today').innerHTML = blocks.join('');
}

function renderMemory() {
  const m = DATA.memory || {};
  $('memlegacy').innerHTML = m.legacy
    ? '<span class="tag" style="background:rgba(255,176,32,.14);color:var(--warn)">LEGACY LOG</span>' : '';
  $('memory').innerHTML = `
    <div class="mu" style="font-size:11px">${m.date ? 'updated ' + esc(m.date) : 'no memory file'}${m.archiveEntries ? ` · ${m.archiveEntries} archived reflections` : ''}</div>
    <div class="col" style="gap:8px;margin-top:10px">${(m.bullets || []).map(b =>
      `<div style="padding-left:10px;border-left:2px solid var(--line2)">${esc(b)}</div>`).join('') || '<span class="mu">empty</span>'}</div>`;
}

function renderExperiments() {
  const rows = DATA.experiments || [];
  $('experiments').innerHTML = rows.length ? rows.map(e => `<tr>
    <td class="l mu" style="font-size:11px">${esc(e.created)}</td>
    <td class="l">${esc(e.experiment)}</td><td class="l mu">${esc(e.variant)}</td>
    <td>${e.sharpe == null ? '—' : e.sharpe.toFixed(2)}</td>
    <td class="mu" style="font-size:11px">${e.ciLo == null ? '—' : `${e.ciLo.toFixed(2)} → ${e.ciHi.toFixed(2)}`}</td>
    <td style="color:${ud(e.annReturn)}">${e.annReturn == null ? '—' : pct(e.annReturn, 1)}</td>
    <td style="color:var(--down)">${e.maxDrawdown == null ? '—' : pctAbs(e.maxDrawdown, 1)}</td>
    <td class="l mu" style="font-size:11px">${esc(e.start)} → ${esc(e.end)}</td>
    <td class="l mu" style="font-size:11px;max-width:300px">${esc(e.note)}</td></tr>`).join('')
    : '<tr><td colspan="9" class="l mu">no experiments recorded</td></tr>';
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
          ? 'This is the config the forward paper track is now trading. Its per-trade ledger is in the Trade ledger panel.'
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
    status.textContent = r.ok ? 'triggered, running on GitHub Actions' : 'failed: ' + (body.error || r.status);
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

[renderHealth, renderHeaderPills, renderVerdict, renderRecords, renderToday, renderMemory,
 renderChart, renderGuardrails, renderBakeoff, renderRuns,
 renderScorecard, renderBuckets, renderLedger, renderExperiments, renderRunbook,
 renderAgentNotes, renderDrawer].forEach(f => f());
</script>
</body>
</html>
"""


def render_control_room(base_dir: Path) -> str:
    data = _build_data(base_dir)
    title = f"Trading Control Room - {len(data['runs'])} research runs"
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
