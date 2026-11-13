"""Fifth forward record, `agent_open`: no stock list.

Every day the agent decides what to hunt for across the whole US market (call 1
chooses screens over fixed scan features) and which names to hold (call 2 picks
from the screened candidates). Only a tradability FLOOR from config.yaml bounds
it. Return series only -- no orders anywhere.

A day that fails (scan, either call, or validation) is recorded as "failed",
yesterday's positions are held, and the date is EXCLUDED from returns.csv, the
same principle as forward.py: a held-over position is the P&L of an outage, not
of a decision. Past dates are never retried or rewritten.

Lookahead: the scan only ever sees bars dated <= its latest date, and a day is
refused unless that date is <= the decision date T (the latest SPY bar in the
main cache). The position decided for T earns from T+1 (net_returns, next_open).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

from .backtest import net_returns
from .data import _read_csv
from .engine import _fail_reason, nvidia_complete

BASE = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/US"
# The numeric scan columns the agent may screen on (the model picks names, never code).
FEATURES = ["close", "ret_1d", "ret_5d", "ret_20d", "ret_60d", "vol_20d",
            "dist_high_250d", "volume_ratio", "z20", "dollar_volume_20d"]
MAX_STALE_DAYS = 5  # scan older than this many days before T = dataset stuck, not "one day behind"


def features(px: pd.DataFrame, cfg) -> tuple[str, pd.DataFrame]:
    """(latest date, one row per company passing the floor) from long bars
    [symbol, date, close, volume, sector]. Pure; every figure uses bars <= latest."""
    px = px[px["sector"].fillna("").astype(str).str.strip() != ""]
    px = px.drop_duplicates(["symbol", "date"])
    c = px.pivot(index="date", columns="symbol", values="close").sort_index()
    v = px.pivot(index="date", columns="symbol", values="volume").reindex_like(c)
    if len(c) < 61:
        raise ValueError(f"scan has {len(c)} bars, need 61")
    last, w, r = c.iloc[-1], c.tail(20), c / c.shift(1) - 1
    df = pd.DataFrame({
        "close": last,
        **{f"ret_{k}d": last / c.iloc[-1 - k] - 1 for k in (1, 5, 20, 60)},
        "vol_20d": r.tail(20).std(),
        "dist_high_250d": last / c.tail(250).max() - 1,
        "volume_ratio": v.iloc[-1] / v.tail(20).mean(),
        "z20": (last - w.mean()) / w.std(),
        "dollar_volume_20d": (c * v).tail(20).median(),
    })[FEATURES]
    df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()
    df = df[(df["close"] >= cfg.min_price) & (df["dollar_volume_20d"] >= cfg.min_dollar_volume)]
    df.insert(0, "sector", px.drop_duplicates("symbol").set_index("symbol")["sector"].reindex(df.index))
    return str(c.index[-1]), df.rename_axis("symbol").reset_index()


def default_scan(cfg) -> tuple[str, pd.DataFrame]:
    import duckdb  # lazy: only the live scan needs it

    px = duckdb.connect().sql(f"""
        select p.symbol, p.report_date as date, p.close::double close,
               p.volume::double volume, f.sector
        from '{BASE}/stock_prices.parquet' p join '{BASE}/stock_profile.parquet' f using (symbol)
        where p.close > 0 and coalesce(f.sector, '') <> ''
          and p.report_date >= (select (max(report_date)::date - 400)::varchar
                                from '{BASE}/stock_prices.parquet')""").df()
    return features(px, cfg)


def _json(raw: str) -> dict:
    """Parse the reply. The model sometimes drops the opening quote of a reason
    value; only that one repair is allowed, so symbols/targets can never be
    altered by it -- anything else that doesn't parse raises."""
    i, j = raw.find("{"), raw.rfind("}")
    if i < 0 or j < i:
        raise ValueError(f"no JSON object in model reply: {raw[:120]!r}")
    s = raw[i:j + 1]
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        try:
            obj = json.loads(re.sub(r'("reason"\s*:\s*)(?=[^"\s])', r'\1"', s))
        except json.JSONDecodeError as e:
            raise ValueError(f"unparseable reply: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    return obj


def _screens(raw: str) -> tuple[list[dict], str]:
    obj = _json(raw)
    screens = obj.get("screens")
    if not isinstance(screens, list) or not 1 <= len(screens) <= 3:
        raise ValueError("screens must be a list of 1..3")
    for s in screens:
        if (not isinstance(s, dict) or s.get("feature") not in FEATURES
                or s.get("order") not in ("lowest", "highest")
                or type(s.get("n")) is not int or not 1 <= s["n"] <= 20):
            raise ValueError(f"bad screen: {s!r}")
    return screens, str(obj.get("reason", ""))


def _picks(raw: str, candidates: set, max_positions: int) -> tuple[list[dict], str]:
    obj = _json(raw)
    pos = obj.get("positions")
    if not isinstance(pos, list) or len(pos) > max_positions:
        raise ValueError(f"positions must be a list of at most {max_positions}")
    seen = set()
    for p in pos:
        if (not isinstance(p, dict) or p.get("symbol") not in candidates
                or p["symbol"] in seen or type(p.get("target")) is not int
                or p["target"] not in (-1, 1)):
            raise ValueError(f"bad position: {p!r}")
        seen.add(p["symbol"])
    return pos, str(obj.get("reason", ""))


def _held_text(held: dict) -> str:
    return ", ".join(f"{s} ({p:+.0f})" for s, p in sorted(held.items())) or "none (all cash)"


def _screen_prompt(df: pd.DataFrame, held: dict, cfg) -> str:
    return (
        "You are a trading agent hunting for stocks across the whole US market.\n"
        f"Market today: {len(df)} companies pass the liquidity floor; "
        f"median ret_5d={df['ret_5d'].median():+.4f} ret_20d={df['ret_20d'].median():+.4f}; "
        f"{(df['z20'] > 0).mean():.0%} trade above their 20-day mean, "
        f"{(df['z20'] < -1).mean():.0%} are more than 1 std below it.\n"
        f"Current holdings: {_held_text(held)}\n"
        f"Available features: {', '.join(FEATURES)}\n"
        "Choose up to 3 screens to find candidates. Respond with ONLY this JSON object, "
        'no markdown, no other text: {"screens": [{"feature": <a feature above>, '
        '"order": "lowest" | "highest", "n": 1-20}], "reason": "<=25 words"}'
    )


def _pick_prompt(cand: pd.DataFrame, held: dict, cfg) -> str:
    lines = [f"{r.symbol} {r.sector} " + " ".join(f"{f}={getattr(r, f):.4g}" for f in FEATURES)
             for r in cand.itertuples()]
    return (
        "You are a trading agent choosing today's portfolio from these candidates "
        "(returns are fractions, 0.05 = 5%):\n" + "\n".join(lines) + "\n"
        f"Current holdings: {_held_text(held)}. Holdings you do not list are closed.\n"
        f"Hold at most {cfg.open_universe.max_positions} names, each equal weight; "
        "target 1 = long, -1 = short. An empty list means all cash. Respond with ONLY "
        'this JSON object, no markdown, no other text: {"positions": [{"symbol": "...", '
        '"target": -1 | 1, "reason": "<=15 words"}], "reason": "<=25 words"}'
    )


def _decide(df: pd.DataFrame, held: dict, complete, cfg) -> dict:
    """Both model calls. Never raises: any failure comes back as out["fail"] and
    the caller records the day failed."""
    out: dict = {"n_floor": len(df)}
    try:
        out["screens"], out["screen_reason"] = _screens(complete(_screen_prompt(df, held, cfg)))
        ou = cfg.open_universe
        picked: list = []
        for s in out["screens"]:
            top = df.nsmallest(s["n"], s["feature"]) if s["order"] == "lowest" \
                else df.nlargest(s["n"], s["feature"])
            picked += [x for x in top["symbol"] if x not in picked]
        picked = picked[:ou.max_candidates]
        picked += [s for s in held if s in set(df["symbol"]) and s not in picked]
        cand = df[df["symbol"].isin(picked)].set_index("symbol").loc[picked].reset_index()
        out["candidates"] = len(cand)
        out["positions"], out["reason"] = _picks(
            complete(_pick_prompt(cand, held, cfg)), set(cand["symbol"]), ou.max_positions)
    except Exception as e:  # noqa: BLE001 -- any failure is a failed day, never a guess
        out["fail"] = _fail_reason(e)
    return out


def _default_complete(cfg):
    """The one place the model callable is built. A new record has no history to
    stay comparable with, so reasoning is switched off (the "detailed thinking
    off" system message alone doesn't stop nemotron-3.5 reasoning past max_tokens)."""
    return nvidia_complete(max_tokens=cfg.agent.max_tokens, thinking=False)


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _append(path: Path, header: list, rows: list) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def _decide_day(eval_dir: Path, T: str, scan, complete, cfg) -> str:
    """Decide date T once and append its decision + positions. Returns the status."""
    pos_path = eval_dir / "positions.csv"
    pos = pd.read_csv(pos_path) if pos_path.exists() else pd.DataFrame(columns=["date", "symbol", "pos"])
    last = pos[pos["date"] == pos["date"].max()] if len(pos) else pos
    held = {r.symbol: float(r.pos) for r in last.itertuples() if r.pos}
    row: dict = {"date": T}
    try:
        scan_date, df = scan(cfg.open_universe)
        row["scan_date"] = scan_date
        lag = (pd.Timestamp(T) - pd.Timestamp(scan_date)).days
        if not 0 <= lag <= MAX_STALE_DAYS:
            raise ValueError(f"scan dated {scan_date} vs decision date {T}: lookahead or stale")
        complete = complete or _default_complete(cfg)
        row.update(_decide(df, held, complete, cfg))
    except Exception as e:  # noqa: BLE001
        row["fail"] = _fail_reason(e)
    row["status"] = "failed" if "fail" in row else "ok"
    if row["status"] == "ok":
        book = {p["symbol"]: float(p["target"]) for p in row["positions"]}
        if not cfg.agent.allow_short:
            book = {s: t for s, t in book.items() if t > 0}
    else:
        book = held  # outage: yesterday's positions carry, the date is excluded from returns
    _append(pos_path, ["date", "symbol", "pos", "status"],
            [(T, s, book.get(s, 0.0), row["status"]) for s in sorted(set(held) | set(book))])
    with (eval_dir / "decisions.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row["status"]


def _pending(eval_dir: Path):
    """(decided-ok dates not yet in returns.csv, symbols whose P&L each needs,
    position grid, all decided dates)."""
    dec = _rows(eval_dir / "decisions.jsonl")
    dates = pd.Index(sorted(r["date"] for r in dec))
    ret_path = eval_dir / "returns.csv"
    done = set(pd.read_csv(ret_path)["date"].astype(str)) if ret_path.exists() else set()
    pending = [r["date"] for r in dec if r["status"] == "ok" and r["date"] not in done]
    pos = pd.read_csv(eval_dir / "positions.csv") if (eval_dir / "positions.csv").exists() \
        else pd.DataFrame(columns=["date", "symbol", "pos"])
    P = pos.pivot(index="date", columns="symbol", values="pos").reindex(dates).ffill().fillna(0.0)
    prev = P.shift(1).fillna(0.0)
    return pending, {d: [s for s in P.columns if P.at[d, s] or prev.at[d, s]] for d in pending}, P, dates


def _realize(eval_dir: Path, cache_dir: Path, fetch, cfg, last_bar: pd.Timestamp) -> int:
    """Append every decided day whose legs all have their T+1 bar AND that the
    calendar (`last_bar`, SPY) has moved past. Returns rows appended."""
    pending, legs, P, dates = _pending(eval_dir)
    need = sorted({s for v in legs.values() for s in v})
    if need:
        try:
            days = max(30, (pd.Timestamp.now() - pd.Timestamp(dates[0])).days + 10)
            fetch(need, cache_dir, days)
        except Exception as e:  # noqa: BLE001 -- the legs just stay unrealized
            print(f"!! price fetch failed (non-fatal, days stay pending): {e}", file=sys.stderr)
    net = {}
    for s in need:
        if not (cache_dir / f"{s}.csv").exists():
            continue
        bars = _read_csv(cache_dir / f"{s}.csv")
        p = P[s].copy()
        p.index = pd.to_datetime(p.index)
        p = p.reindex(p.index.union(bars.index)).ffill().reindex(bars.index).fillna(0.0)
        net[s] = net_returns(bars, p, cfg.strategy.cost_bps, cfg.strategy.fill_mode)
    rows = []
    for d in pending:
        ts = pd.Timestamp(d)
        if ts < last_bar and all(s in net and ts in net[s].index for s in legs[d]):
            rows.append(f"{d},{sum(float(net[s][ts]) for s in legs[d]) / cfg.open_universe.max_positions!r}")
    if rows:
        # Date-sorted file (evaluate builds its equity curve in file order). Lines
        # already recorded are carried over as text, so their values never change.
        ret_path = eval_dir / "returns.csv"
        old = ret_path.read_text().splitlines()[1:] if ret_path.exists() else []
        body = sorted(old + rows, key=lambda x: x.split(",")[0])
        ret_path.write_text("date,net\n" + "\n".join(body) + "\n")
    return len(rows)


def _stuck(eval_dir: Path, cache_dir: Path, today=None) -> dict:
    """{date: [symbols with no bar after it]} for ok days unrealized > 5 days."""
    today = pd.Timestamp(today or pd.Timestamp.now()).normalize()
    pending, legs, _, _ = _pending(eval_dir)
    out = {}
    for d in pending:
        if (today - pd.Timestamp(d)).days > 5:
            out[d] = [s for s in legs[d] if not (cache_dir / f"{s}.csv").exists()
                      or not (_read_csv(cache_dir / f"{s}.csv").index > pd.Timestamp(d)).any()]
    return out


def _default_fetch(symbols, cache_dir, days) -> None:
    from .refresh import fetch_and_update

    fetch_and_update(cache_dir=cache_dir, symbols=symbols, days=days, source="yahoo")


def tick(cfg, eval_dir: Path, *, scan=default_scan, complete=None, fetch=_default_fetch,
         data_dir="data", cache_dir="data/open") -> dict:
    eval_dir.mkdir(parents=True, exist_ok=True)
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    # Decision date = latest settled bar in the shared cache (SPY is the calendar).
    last_bar = _read_csv(data_dir / "SPY.csv").index.max()
    T = str(last_bar.date())
    status = None
    if T not in {r["date"] for r in _rows(eval_dir / "decisions.jsonl")}:  # never decide a date twice
        status = _decide_day(eval_dir, T, scan, complete, cfg)
    appended = _realize(eval_dir, cache_dir, fetch, cfg, last_bar)
    ret = pd.read_csv(eval_dir / "returns.csv") if (eval_dir / "returns.csv").exists() \
        else pd.DataFrame(columns=["date", "net"])
    meta = {"run_id": eval_dir.name, "engine": "agent_open", "symbols": [],
            "cost_bps": cfg.strategy.cost_bps, "fill_mode": cfg.strategy.fill_mode,
            "notional": 10_000.0, "max_positions": cfg.open_universe.max_positions,
            "start": str(ret["date"].min()) if len(ret) else "",
            "end": str(ret["date"].max()) if len(ret) else ""}
    (eval_dir / "run.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    (eval_dir / "trades.jsonl").touch()  # evaluate.load_run expects the file
    if not (eval_dir / "returns.csv").exists():
        _append(eval_dir / "returns.csv", ["date", "net"], [])
    return {"date": T, "status": status, "appended": appended}


def _report(eval_dir: Path, cache_dir: Path = Path("data/open"), today=None) -> None:
    from .evaluate import aggregate, load_run

    meta, trades, net = load_run(eval_dir)
    a = aggregate(trades, net)
    dec = _rows(eval_dir / "decisions.jsonl")
    print(f"forward record: {meta['engine']}  {meta['start'][:10]} -> {meta['end'][:10]}  "
          f"({len(net)} return days, {sum(r['status'] == 'ok' for r in dec)}/{len(dec)} decisions ok)")
    print(f"  total_return   {a['total_return']:+.2%}")
    print(f"  max_drawdown   {a['max_drawdown']:.2%}")
    for d, missing in _stuck(eval_dir, cache_dir, today).items():
        print(f"!! agent_open: decided day {d} still unrealized after >5 days; "
              f"no price bar after it for: {', '.join(missing) or 'none (calendar)'}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    from .config import load

    p = argparse.ArgumentParser(prog="rhagent.open_universe")
    p.add_argument("--out-dir", default="journal/forward")
    p.add_argument("--eval-id", default="agent_open")
    p.add_argument("--data-dir", default="data", help="main cache; its SPY.csv is the calendar")
    p.add_argument("--cache-dir", default="data/open", help="Yahoo bars for held names")
    p.add_argument("--report", action="store_true", help="report only, no tick")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    cfg = load()
    eval_dir = Path(args.out_dir) / args.eval_id
    res = None
    if not args.report:
        res = tick(cfg, eval_dir, data_dir=args.data_dir, cache_dir=args.cache_dir)
        print(f"tick: {res['date']} decision={res['status'] or 'already decided'}, "
              f"appended {res['appended']} return day(s)")
    try:
        _report(eval_dir, Path(args.cache_dir))
    except Exception as e:  # noqa: BLE001 -- report only; the tick is already on disk
        print(f"!! report failed (non-fatal, tick already persisted): {e}", file=sys.stderr)
    # Non-zero only AFTER everything is on disk, so paper_cron's tick_failed alerts.
    return 1 if res and res["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
