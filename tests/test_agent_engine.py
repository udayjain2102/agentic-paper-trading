"""AgentEngine: fake-model unit tests + a papertrade integration run (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from rhagent.engine import AgentEngine, Decision
from rhagent.papertrade import PaperTrader


def _hist(closes):
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes},
        index=pd.date_range("2024-01-01", periods=len(closes), freq="D"),
    )


def test_targets_and_reason_carried():
    hist = _hist([10, 11, 12, 13, 14, 15, 16])
    for tgt in (-1, 0, 1):
        fake = lambda p, t=tgt: json.dumps({"target": t, "reason": f"go {t}"})
        # allow_short=True so the -1 target carries through unclamped (default is
        # now long-only); this test checks target/reason plumbing, not clamping.
        d = AgentEngine(complete=fake, allow_short=True).decide("NVDA", hist, 0.0)
        assert isinstance(d, Decision)
        assert d.target == float(tgt)
        assert f"go {tgt}" in d.reason
        assert d.status == "ok"


def test_allow_short_clamps():
    hist = _hist([10, 11, 12])
    fake = lambda p: json.dumps({"target": -1, "reason": "short"})
    assert AgentEngine(complete=fake, allow_short=False).decide("X", hist, 0.0).target == 0.0
    assert AgentEngine(complete=fake, allow_short=True).decide("X", hist, 0.0).target == -1.0


def test_decide_all_is_one_call_per_symbol_and_isolates_failures():
    """The per-symbol path: one call PER symbol, each prompt mentioning only
    its own symbol, decide()'s failure semantics preserved per symbol."""
    hists = {s: _hist([10, 11, 12, 13, 14, 15, 16]) for s in ("AAA", "BBB", "CCC")}
    syms = list(hists)
    cur = {"AAA": 0.0, "BBB": 1.0, "CCC": -1.0}
    calls = []

    def fake(prompt):
        calls.append(prompt)
        return json.dumps({"target": -1, "reason": "go short"})

    out = AgentEngine(complete=fake).decide_all(syms, hists, cur)
    assert len(calls) == 3                                    # one call PER symbol
    for prompt in calls:
        assert sum(s in prompt for s in syms) == 1            # each mentions only its own symbol
    assert out["AAA"].target == 0.0 and out["AAA"].status == "ok"   # -1 clamped
    assert out["BBB"].target == 0.0 and out["BBB"].status == "ok"
    assert out["CCC"].target == 0.0 and out["CCC"].status == "ok"

    # allow_short leaves the short alone
    assert AgentEngine(complete=fake, allow_short=True).decide_all(
        syms, hists, cur)["AAA"].target == -1.0

    # a failed call fails EVERY symbol, each holding its own current_pos
    def boom(_p):
        raise TimeoutError("model down")

    out = AgentEngine(complete=boom).decide_all(syms, hists, cur)
    assert {s: d.target for s, d in out.items()} == cur
    assert all(d.status == "failed" for d in out.values())


def test_decide_all_one_bad_call_costs_one_symbol_not_the_bar():
    """THE regression this file exists to prevent. forward.py drops a whole bar
    from the forward record if any symbol on it isn't status=="ok", so the cost
    of one failed model call is the thing that decides whether the record moves
    at all. Under the old 13-symbol batching one failure took 13 symbols -- and
    therefore the day -- down with it, and the record advanced one day in a
    week. One failed call must now cost exactly one symbol, and the retry must
    not amplify into extra calls."""
    syms = [f"S{i}" for i in range(20)]
    hists = {s: _hist([10, 11, 12]) for s in syms}
    cur = {s: (0.0 if i % 2 == 0 else 1.0) for i, s in enumerate(syms)}
    bad = syms[7]
    calls = []

    def fake(prompt):
        calls.append(prompt)
        if bad in prompt:
            raise TimeoutError("model down")
        return json.dumps({"target": 1, "reason": "long"})

    out = AgentEngine(complete=fake).decide_all(syms, hists, cur)

    failed = [s for s in syms if out[s].status == "failed"]
    assert failed == [bad]
    assert out[bad].target == cur[bad]
    assert all(out[s].status == "ok" for s in syms if s != bad)
    assert len(calls) == 20  # one call per symbol, no split-retry amplification


def test_parse_fail_holds_current_pos():
    hist = _hist([10, 11, 12])
    d = AgentEngine(complete=lambda p: "not json at all").decide("X", hist, 1.0)
    assert d.target == 1.0
    assert "parse-fail" in d.reason
    assert "ValueError" in d.reason  # names the real exception, not just "parse-fail"
    assert "not json at all" in d.reason  # raw reply kept, not a bare crash message
    # A malformed reply must not be silently recorded as a real trading
    # decision: status distinguishes it so agent performance metrics can
    # exclude it instead of counting a forced hold as a genuine flat call.
    assert d.status == "failed"


def test_default_complete_uses_configured_max_tokens(monkeypatch):
    """AgentEngine() with no explicit max_tokens must reach the API with
    cfg.agent.max_tokens -- not a hardcoded default that silently ignores
    config.yaml. Asserts against the live config value, not a literal, so
    retuning the budget doesn't break the test that guards the wiring."""
    import openai

    from rhagent.config import load

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = SimpleNamespace(content='{"target": 1, "reason": "ok"}')
            choice = SimpleNamespace(message=msg, finish_reason="stop")
            usage = SimpleNamespace(completion_tokens=42)
            return SimpleNamespace(choices=[choice], usage=usage)

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "OpenAI", FakeClient)

    hist = _hist([10, 11, 12])
    d = AgentEngine().decide("X", hist, 0.0)  # complete=None -> lazy nvidia client

    assert d.target == 1.0
    assert captured["max_tokens"] == load().agent.max_tokens  # not a hardcoded 256


def test_truncated_response_is_distinguishable_not_parse_fail(monkeypatch):
    """The production incident this guards against: finish_reason="length"
    used to surface as content=None -> "no JSON object in model reply" ->
    parse-fail, indistinguishable from a genuinely malformed answer. It must
    instead surface as its own "truncated" reason, naming the budget hit and
    the tokens actually used, with status == "failed"."""
    import openai

    class FakeCompletions:
        def create(self, **kwargs):
            msg = SimpleNamespace(content=None)
            choice = SimpleNamespace(message=msg, finish_reason="length")
            usage = SimpleNamespace(completion_tokens=256)
            return SimpleNamespace(choices=[choice], usage=usage)

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "OpenAI", FakeClient)

    hist = _hist([10, 11, 12])
    d = AgentEngine(max_tokens=256).decide("X", hist, 1.0)  # complete=None -> lazy nvidia client

    assert d.target == 1.0  # falls back to holding current_pos
    assert d.status == "failed"
    assert "truncated" in d.reason
    assert "parse-fail" not in d.reason
    assert "max_tokens=256" in d.reason
    assert "completion_tokens=256" in d.reason


def test_json_extraction_prefers_last_brace_span():
    """A reasoning model can echo the prompt's own example braces in its
    chain-of-thought before the real answer; extraction must not choke on
    prose sitting between an earlier stray '{...}' and the final one."""
    hist = _hist([10, 11, 12])
    raw = (
        'Reminder: reply as {"target": -1 | 0 | 1, "reason": "..."}. '
        'Thinking it over... {"target": 1, "reason": "final answer"}'
    )
    d = AgentEngine(complete=lambda p: raw).decide("X", hist, 0.0)
    assert d.target == 1.0
    assert "final answer" in d.reason


def test_rate_limit_is_distinguishable_and_holds():
    import httpx
    from openai import RateLimitError

    def always_429(p):
        req = httpx.Request("POST", "https://x/y")
        resp = httpx.Response(429, request=req, json={"error": {"message": "slow down"}})
        raise RateLimitError("slow down", response=resp, body={"error": {"message": "slow down"}})

    hist = _hist([10, 11, 12])
    d = AgentEngine(complete=always_429).decide("X", hist, 1.0)
    assert d.target == 1.0          # falls back to holding current_pos
    assert "rate-limited" in d.reason
    assert d.status == "failed"


def test_timeout_is_distinguishable():
    import httpx
    from openai import APITimeoutError

    def times_out(p):
        raise APITimeoutError(httpx.Request("POST", "https://x/y"))

    hist = _hist([10, 11, 12])
    d = AgentEngine(complete=times_out).decide("X", hist, 1.0)
    assert d.target == 1.0
    assert "timeout" in d.reason
    assert d.status == "failed"


class _Source:
    def __init__(self, frames):
        self._frames = frames

    def bars(self):
        return self._frames


def test_integration_trades_written(tmp_path):
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    frames = {
        "AAA": pd.DataFrame(
            {"open": [10, 11, 12, 13], "high": [10, 11, 12, 13],
             "low": [10, 11, 12, 13], "close": [10, 11, 12, 13]}, index=idx),
        "BBB": pd.DataFrame(
            {"open": [20, 19, 18, 17], "high": [20, 19, 18, 17],
             "low": [20, 19, 18, 17], "close": [20, 19, 18, 17]}, index=idx),
    }
    fake = lambda p: json.dumps({"target": 1, "reason": "long it"})
    trader = PaperTrader(
        engine=AgentEngine(complete=fake), source=_Source(frames),
        out_dir=tmp_path,
    )
    run_dir = trader.run()

    lines = (run_dir / "trades.jsonl").read_text().splitlines()
    assert lines
    for line in lines:
        assert json.loads(line)["trade_id"]
