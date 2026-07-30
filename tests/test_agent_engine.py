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


def test_decide_all_is_one_call_and_isolates_per_symbol_failures():
    """The batched path: one prompt for the universe, decide()'s failure
    semantics preserved per symbol."""
    hists = {s: _hist([10, 11, 12, 13, 14, 15, 16]) for s in ("AAA", "BBB", "CCC")}
    syms = list(hists)
    cur = {"AAA": 0.0, "BBB": 1.0, "CCC": -1.0}
    calls = []

    def fake(prompt):
        calls.append(prompt)
        # BBB omitted, CCC out of range -> only those two fail
        return 'here you go: {"AAA": -1, "CCC": 7}'

    out = AgentEngine(complete=fake).decide_all(syms, hists, cur)
    assert len(calls) == 1                      # ONE call for three symbols
    assert all(s in calls[0] for s in syms)     # every symbol's features in it
    assert out["AAA"].target == 0.0 and out["AAA"].status == "ok"  # -1 clamped
    assert out["BBB"].target == 1.0 and out["BBB"].status == "failed"
    assert out["CCC"].target == -1.0 and out["CCC"].status == "failed"

    # allow_short leaves the short alone
    assert AgentEngine(complete=fake, allow_short=True).decide_all(
        syms, hists, cur)["AAA"].target == -1.0

    # a failed call fails EVERY symbol, each holding its own current_pos
    def boom(_p):
        raise TimeoutError("model down")

    out = AgentEngine(complete=boom).decide_all(syms, hists, cur)
    assert {s: d.target for s, d in out.items()} == cur
    assert all(d.status == "failed" for d in out.values())


def test_decide_all_splits_truncated_chunk_at_fixed_budget():
    """The 2026-07-27 incident: a chunk that truncates whole but would fit at
    half the symbol count (same token budget -- see decide_all's docstring)
    must recover instead of failing every symbol in it forever."""
    from rhagent.engine import TruncatedResponse

    hists = {s: _hist([10, 11, 12, 13, 14, 15, 16]) for s in ("AAA", "BBB", "CCC", "DDD")}
    syms = list(hists)
    cur = {s: 0.0 for s in syms}
    calls = []

    def fake(prompt):
        # crude symbol count from the prompt's feature lines
        n = sum(1 for s in syms if f"{s}:" in prompt)
        calls.append(n)
        if n > 2:
            raise TruncatedResponse("hit max_tokens=4600 (completion_tokens=4600) before finishing")
        return json.dumps({s: 1 for s in syms if f"{s}:" in prompt})

    out = AgentEngine(complete=fake).decide_all(syms, hists, cur)
    assert all(d.status == "ok" and d.target == 1.0 for d in out.values())
    assert calls[0] == 4          # first attempt: the whole chunk
    assert 4 not in calls[1:]     # retry sends fewer symbols, not the same count


def test_decide_all_non_truncation_failure_does_not_split():
    """A rate limit / timeout / HTTP / parse failure is not caused by chunk
    size -- decide_all must fail the whole chunk in one shot, no split retry."""
    calls = []

    def boom(prompt):
        calls.append(prompt)
        raise TimeoutError("model down")

    hists = {s: _hist([10, 11, 12]) for s in ("AAA", "BBB")}
    syms = list(hists)
    cur = {"AAA": 0.0, "BBB": 1.0}
    out = AgentEngine(complete=boom).decide_all(syms, hists, cur)
    assert len(calls) == 1
    assert {s: d.target for s, d in out.items()} == cur
    assert all(d.status == "failed" for d in out.values())


def test_decide_all_truncation_down_to_one_symbol_fails_that_symbol():
    """If splitting bottoms out at a single symbol and it STILL truncates,
    that symbol must be status="failed", not silently booked as a real
    verdict -- the day is excluded from the forward record instead."""
    from rhagent.engine import TruncatedResponse

    def always_truncates(prompt):
        raise TruncatedResponse("hit max_tokens=4600 (completion_tokens=4600) before finishing")

    hist = _hist([10, 11, 12])
    hists = {"AAA": hist, "BBB": hist}
    cur = {"AAA": 0.0, "BBB": 1.0}
    out = AgentEngine(complete=always_truncates).decide_all(["AAA", "BBB"], hists, cur)
    assert out["AAA"].status == "failed" and out["AAA"].target == 0.0
    assert out["BBB"].status == "failed" and out["BBB"].target == 1.0
    assert "truncated" in out["AAA"].reason


def test_decide_all_abandons_remaining_work_after_one_confirmed_failure():
    """forward.py excludes the WHOLE bar the moment any one symbol on it isn't
    status=="ok" (_agent_positions' EXCLUSION RULE) -- so once a split has
    confirmed one symbol failed at size 1, every other chunk/split in this
    SAME decide_all call is worthless and must not spend another call. Stated
    max for one decide_all call with CHUNK_SIZE=13 when EVERY call truncates:
    the confirmed failure is found by the time the leftmost split reaches
    size 1 (13 -> 6 -> 3 -> 1, i.e. 4 calls), regardless of how many
    CHUNK_SIZE-sized chunks the universe has -- not the 2*13-1=25 an
    unbounded split tree would cost for ONE chunk, let alone 5 chunks' worth."""
    from rhagent.engine import TruncatedResponse
    from rhagent.engine import CHUNK_SIZE

    syms = [f"S{i}" for i in range(CHUNK_SIZE + 5)]  # spans 2 top-level chunks
    hists = {s: _hist([10, 11, 12]) for s in syms}
    cur = {s: 0.0 for s in syms}
    calls = []

    def always_truncates(prompt):
        calls.append(prompt)
        raise TruncatedResponse("hit max_tokens=4600 (completion_tokens=4600) before finishing")

    out = AgentEngine(complete=always_truncates).decide_all(syms, hists, cur)

    STATED_MAX = 9  # 2*ceil(log2(CHUNK_SIZE)) + 1, see decide_all's ABANDON RULE
    assert len(calls) <= STATED_MAX
    assert len(calls) < 2 * CHUNK_SIZE - 1  # nowhere near the unbounded-tree cost
    # every symbol still gets a Decision (decide_all's contract), all failed
    assert len(out) == len(syms)
    assert all(d.status == "failed" for d in out.values())
    # symbols abandoned without a call must say so honestly, not "truncated"
    abandoned = [d for d in out.values() if "abandoned" in d.reason]
    truncated = [d for d in out.values() if "truncated" in d.reason]
    assert abandoned and truncated
    assert not any("truncated" in d.reason for d in abandoned)


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
