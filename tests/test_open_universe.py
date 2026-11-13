"""agent_open record: floor, strict validation, failure rule, returns. No network."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest
import yaml

from rhagent import open_universe as ou
from rhagent.config import OpenUniverseConfig

DATES = ["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06"]
# symbol -> (open, close) per date
BARS = {
    "AAA": [(10, 10), (10, 11), (12, 12), (12, 13.2), (13, 13)],
    "BBB": [(20, 20), (20, 20), (21, 22), (22, 21), (21, 21)],
}
SCREEN = json.dumps({"screens": [{"feature": "ret_5d", "order": "highest", "n": 5}], "reason": "momentum"})
CONFIG = Path(__file__).parent.parent / "config.yaml"


def _pick(*syms, reason="r"):
    return json.dumps({"positions": [{"symbol": s, "target": t, "reason": "x"} for s, t in syms],
                       "reason": reason})


def _cfg(allow_short=True):
    return SimpleNamespace(
        strategy=SimpleNamespace(cost_bps=7.0, fill_mode="next_open"),
        agent=SimpleNamespace(allow_short=allow_short),
        open_universe=OpenUniverseConfig())


def _scan_df():
    rows = [{"symbol": s, "sector": "Tech", **{f: 1.0 + i for f in ou.FEATURES}}
            for i, s in enumerate(["AAA", "BBB", "CCC"])]
    return pd.DataFrame(rows)


class Env:
    """One record in tmp_path; `day(i, ...)` runs a tick as of DATES[i]."""

    def __init__(self, tmp_path, cfg=None):
        self.cfg, self.dir = cfg or _cfg(), tmp_path / "rec"
        self.data, self.cache = tmp_path / "data", tmp_path / "open"
        self.data.mkdir()
        self.calls = 0
        self.skip = set()  # symbols the fake Yahoo has no bars for

    def day(self, i, replies=(), scan_fails=False, complete=None):
        pd.DataFrame({"date": DATES[:i + 1], "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                      "volume": 1.0}).to_csv(self.data / "SPY.csv", index=False)
        queue = list(replies)

        def fake(prompt):
            self.calls += 1
            if complete:
                return complete(prompt)
            return SCREEN if "Available features" in prompt else queue.pop(0)

        def scan(_cfg):
            if scan_fails:
                raise OSError("hf down")
            return DATES[i], _scan_df()

        def fetch(symbols, cache_dir, days):
            cache_dir.mkdir(exist_ok=True)
            for s in [x for x in symbols if x not in self.skip]:
                rows = [{"date": d, "open": o, "high": max(o, c), "low": min(o, c), "close": c,
                         "volume": 1e6} for d, (o, c) in zip(DATES[:i + 1], BARS[s])]
                pd.DataFrame(rows).to_csv(cache_dir / f"{s}.csv", index=False)

        return ou.tick(self.cfg, self.dir, scan=scan, complete=fake, fetch=fetch,
                       data_dir=self.data, cache_dir=self.cache)

    def returns(self):
        return pd.read_csv(self.dir / "returns.csv").set_index("date")["net"]

    def decisions(self):
        return [json.loads(x) for x in (self.dir / "decisions.jsonl").read_text().splitlines()]


# --- floor -----------------------------------------------------------------

def _px(spec):
    days = pd.bdate_range("2026-01-01", periods=70).strftime("%Y-%m-%d")
    return pd.DataFrame([{"symbol": sym, "date": d, "close": price * (1 + 0.001 * i),
                          "volume": vol, "sector": sector}
                         for sym, (price, vol, sector) in spec.items() for i, d in enumerate(days)])


def test_floor_excludes_penny_illiquid_and_fund():
    px = _px({"GOOD": (50, 1e6, "Tech"), "PENNY": (2, 1e9, "Tech"),
              "ILLIQ": (50, 1e3, "Tech"), "FUND": (50, 1e6, ""), "NOSEC": (50, 1e6, None)})
    date, df = ou.features(px, OpenUniverseConfig())
    assert list(df["symbol"]) == ["GOOD"]
    assert date == px["date"].max()
    assert set(ou.FEATURES) <= set(df.columns)


def test_features_ignore_nothing_beyond_latest_date_and_follow_it():
    px = _px({"GOOD": (50, 1e6, "Tech")})
    d0, a = ou.features(px, OpenUniverseConfig())
    # Truncating the last bar moves the whole scan back to the earlier date: every
    # figure is a function of bars dated <= the scan's latest date.
    d1, b = ou.features(px[px["date"] < d0], OpenUniverseConfig())
    assert d1 < d0 and a["close"].iloc[0] != b["close"].iloc[0]


def test_no_symbol_is_hardcoded():
    src = Path(ou.__file__).read_text()
    universe = yaml.safe_load(CONFIG.read_text())["strategy"]["universe"]
    assert not [s for s in universe if f'"{s}"' in src or f"'{s}'" in src]


def test_config_defaults_when_block_absent(tmp_path):
    from rhagent.config import load
    raw = yaml.safe_load(CONFIG.read_text())
    assert raw["open_universe"]["max_positions"] == 10
    raw.pop("open_universe")
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw))
    assert load(p).open_universe == OpenUniverseConfig()


# --- strict validation -----------------------------------------------------

@pytest.mark.parametrize("screens", [
    [{"feature": "nope", "order": "lowest", "n": 3}],
    [{"feature": "z20", "order": "sideways", "n": 3}],
    [{"feature": "z20", "order": "lowest", "n": 0}],
    [{"feature": "z20", "order": "lowest", "n": 21}],
    [{"feature": "z20", "order": "lowest", "n": 3}] * 4,
])
def test_bad_screens_rejected(screens):
    with pytest.raises(ValueError):
        ou._screens(json.dumps({"screens": screens, "reason": "x"}))


def test_good_screens_accepted():
    screens, reason = ou._screens(SCREEN)
    assert screens[0]["feature"] == "ret_5d" and reason == "momentum"


CAND = {f"S{i:02d}" for i in range(12)}


@pytest.mark.parametrize("positions", [
    [{"symbol": "ZZZ", "target": 1, "reason": "x"}],       # not a candidate
    [{"symbol": "S00", "target": 0, "reason": "x"}],
    [{"symbol": "S00", "target": 2, "reason": "x"}],
    [{"symbol": "S00", "target": True, "reason": "x"}],
    [{"symbol": "S00", "target": 1}, {"symbol": "S00", "target": -1}],  # duplicate
    [{"symbol": f"S{i:02d}", "target": 1} for i in range(11)],          # too many
])
def test_bad_picks_rejected(positions):
    with pytest.raises(ValueError):
        ou._picks(json.dumps({"positions": positions, "reason": "x"}), CAND, 10)


def test_empty_picks_valid_and_ten_ok():
    assert ou._picks(json.dumps({"positions": [], "reason": "cash"}), CAND, 10)[0] == []
    ten = [{"symbol": f"S{i:02d}", "target": -1} for i in range(10)]
    assert len(ou._picks(json.dumps({"positions": ten, "reason": "x"}), CAND, 10)[0]) == 10


def test_dropped_reason_quote_tolerated_but_never_symbols_or_targets():
    ok = '{"positions": [{"symbol": "S00", "target": 1, "reason": Cheap vs peers"}], "reason": fine"}'
    assert ou._picks(ok, CAND, 10)[0][0]["symbol"] == "S00"
    for bad in ('{"positions": [{"symbol": S00", "target": 1, "reason": "x"}], "reason": "x"}',
                '{"positions": [{"symbol": "S00", "target": "1", "reason": "x"}], "reason": "x"}',
                '{"positions": [{"symbol": "S00", "target": 1, "reason": Cheap "vs" peers"}], "reason": "x"}'):
        with pytest.raises(ValueError):
            ou._picks(bad, CAND, 10)


# --- day flow --------------------------------------------------------------

def test_failed_screen_holds_positions_and_is_excluded(tmp_path):
    e = Env(tmp_path)
    e.day(0, [_pick(("AAA", 1))])
    bad = json.dumps({"screens": [{"feature": "nope", "order": "lowest", "n": 3}], "reason": "x"})
    r = e.day(1, complete=lambda p: bad)
    assert r["status"] == "failed"
    pos = pd.read_csv(e.dir / "positions.csv")
    d1 = pos[pos["date"] == DATES[1]]
    assert list(d1["symbol"]) == ["AAA"] and list(d1["pos"]) == [1.0] and list(d1["status"]) == ["failed"]
    assert "parse-fail" in e.decisions()[1]["fail"]
    assert DATES[1] not in e.returns().index


def test_failed_pick_and_scan_failure(tmp_path):
    e = Env(tmp_path)
    assert e.day(0, [_pick(("ZZZ", 1))])["status"] == "failed"   # symbol outside the candidate table
    assert e.day(1, scan_fails=True)["status"] == "failed"
    assert e.calls == 2                                          # the scan failure made no model call
    assert len(e.returns()) == 0


def test_omitted_held_symbol_is_closed_and_empty_list_is_all_cash(tmp_path):
    e = Env(tmp_path)
    e.day(0, [_pick(("AAA", 1), ("BBB", 1))])
    assert e.day(1, [_pick(("BBB", 1))])["status"] == "ok"
    assert e.day(2, [_pick()])["status"] == "ok"
    pos = pd.read_csv(e.dir / "positions.csv").set_index(["date", "symbol"])["pos"]
    assert pos[(DATES[1], "AAA")] == 0 and pos[(DATES[1], "BBB")] == 1
    assert pos[(DATES[2], "BBB")] == 0


def test_allow_short_false_drops_shorts_to_flat(tmp_path):
    e = Env(tmp_path, _cfg(allow_short=False))
    e.day(0, [_pick(("AAA", -1), ("BBB", 1))])
    pos = pd.read_csv(e.dir / "positions.csv").set_index("symbol")["pos"]
    assert pos["BBB"] == 1 and pos.get("AAA", 0) == 0


def test_dead_endpoint_makes_exactly_one_call(tmp_path):
    from openai import NotFoundError

    def gone(prompt):
        raise NotFoundError("Function not found", body=None, response=httpx.Response(
            404, request=httpx.Request("POST", "http://x")))

    e = Env(tmp_path)
    assert e.day(0, complete=gone)["status"] == "failed"
    assert e.calls == 1
    assert "404" in e.decisions()[0]["fail"]


@pytest.mark.parametrize("on_call", [1, 2])
def test_truncated_response_fails_day_and_holds_positions(tmp_path, on_call):
    from rhagent.engine import TruncatedResponse

    e = Env(tmp_path)
    e.day(0, [_pick(("AAA", 1))])
    n = {"i": 0}

    def trunc(prompt):
        n["i"] += 1
        if n["i"] == on_call:
            raise TruncatedResponse("hit max_tokens=4000")
        return SCREEN

    assert e.day(1, complete=trunc)["status"] == "failed"
    fail = e.decisions()[1]["fail"]
    assert fail.startswith("truncated:") and "parse-fail" not in fail
    pos = pd.read_csv(e.dir / "positions.csv")
    d1 = pos[pos["date"] == DATES[1]]
    assert list(d1["symbol"]) == ["AAA"] and list(d1["pos"]) == [1.0]
    assert DATES[1] not in e.returns().index


def test_second_same_day_tick_makes_no_calls_and_appends_nothing(tmp_path):
    e = Env(tmp_path)
    e.day(0, [_pick(("AAA", 1))])
    e.day(1, [_pick(("AAA", 1))])
    calls = e.calls
    snap = {f: (e.dir / f).read_text() for f in ("decisions.jsonl", "positions.csv", "returns.csv")}
    r = e.day(1)
    assert e.calls == calls and r["status"] is None and r["appended"] == 0
    assert snap == {f: (e.dir / f).read_text() for f in snap}


# --- returns ---------------------------------------------------------------

def test_returns_hand_computed_next_open_7bp_fixed_slots(tmp_path):
    """AAA long d0-d1 (closed on d2), BBB long d1-d2 (closed on d3); 10 slots, 7 bp/side.
    AAA net: d0 .1-7bp, d1 12/11-1, d2 -7bp, d3 0.  BBB net: d0 0, d1 22/21-1-7bp,
    d2 21/22-1, d3 -7bp (entry/exit days earn open[t+1]->close[t+1])."""
    e = Env(tmp_path)
    e.day(0, [_pick(("AAA", 1))])
    assert len(e.returns()) == 0                       # d0 needs the d1 bar: not recorded yet
    e.day(1, [_pick(("AAA", 1), ("BBB", 1))])
    assert list(e.returns().index) == [DATES[0]]       # d1 still unrealized
    e.day(2, [_pick(("BBB", 1))])
    e.day(3, [_pick()])
    e.day(4, [])
    r = e.returns()
    assert list(r.index) == DATES[:4]
    assert r[DATES[0]] == pytest.approx((0.1 - 0.0007) / 10, abs=1e-12)
    assert r[DATES[1]] == pytest.approx((12 / 11 - 1 + 22 / 21 - 1 - 0.0007) / 10, abs=1e-12)
    assert r[DATES[2]] == pytest.approx((-0.0007 + 21 / 22 - 1) / 10, abs=1e-12)
    assert r[DATES[3]] == pytest.approx(-0.0007 / 10, abs=1e-12)


def test_failed_day_absent_from_returns_and_recorded_days_never_rewritten(tmp_path):
    e = Env(tmp_path)
    e.day(0, [_pick(("AAA", 1))])
    e.day(1, scan_fails=True)
    e.day(2, [_pick(("AAA", 1))])
    e.day(3, [_pick(("AAA", 1))])
    before = (e.dir / "returns.csv").read_text()
    r = e.returns()
    assert DATES[1] not in r.index and DATES[0] in r.index and DATES[2] in r.index
    e.day(4, [_pick(("AAA", 1))])
    assert (e.dir / "returns.csv").read_text().startswith(before)


def test_run_json_loads_with_evaluate(tmp_path):
    from rhagent.evaluate import load_run
    e = Env(tmp_path)
    e.day(0, [_pick(("AAA", 1))])
    e.day(1, [_pick(("AAA", 1))])
    meta, trades, net = load_run(e.dir)
    assert meta["engine"] == "agent_open" and meta["symbols"] == [] and len(trades) == 0
    assert len(net) == 1 and np.isfinite(net.iloc[0])


def test_all_cash_day_waits_for_a_later_calendar_bar(tmp_path):
    e = Env(tmp_path)
    e.day(0, [_pick()])
    assert len(e.returns()) == 0                       # newest date: no later SPY bar yet
    e.day(1, [_pick()])
    assert list(e.returns().index) == [DATES[0]] and e.returns()[DATES[0]] == 0.0


def _stuck_record(tmp_path):
    """BBB has no bars: d0/d1 stay pending while the cash days d2.. realize."""
    e = Env(tmp_path)
    e.skip = {"BBB"}
    e.day(0, [_pick(("BBB", 1))])
    e.day(1, [_pick()])
    e.day(2, [_pick()])
    e.day(3, [_pick()])
    return e


def test_earlier_date_realizing_late_keeps_file_date_sorted_and_old_lines_identical(tmp_path):
    e = _stuck_record(tmp_path)
    assert list(e.returns().index) == [DATES[2]]
    old_line = [x for x in (e.dir / "returns.csv").read_text().splitlines() if x.startswith(DATES[2])]
    e.skip = set()
    e.day(4, [_pick()])
    r = e.returns()
    assert list(r.index) == DATES[:4] == sorted(r.index)
    assert [x for x in (e.dir / "returns.csv").read_text().splitlines() if x.startswith(DATES[2])] == old_line


def test_report_names_days_unrealized_over_five_days(tmp_path, capsys):
    e = _stuck_record(tmp_path)
    ou._report(e.dir, e.cache, today=pd.Timestamp("2026-03-20"))
    err = capsys.readouterr().err
    assert DATES[0] in err and DATES[1] in err and "BBB" in err
    ou._report(e.dir, e.cache, today=pd.Timestamp("2026-03-05"))
    assert capsys.readouterr().err == ""
