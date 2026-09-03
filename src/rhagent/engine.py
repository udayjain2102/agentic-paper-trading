"""The decision seam between the paper-trade loop and whatever decides.

A DecisionEngine answers one question per bar: given the history up to and
including today and what we currently hold, what should the position be and
why. StrategyEngine adapts the existing rule-based strategies; an AgentEngine
wrapping the Claude loop plugs into the same protocol later.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd

from .strategies.base import Strategy


# NVIDIA's endpoint is a burst-then-~18-requests/min token bucket. Every real
# model call in this process goes through _pace() before it starts, so N
# concurrent workers can never exceed that rate no matter how fast the endpoint
# answers -- pacing at the client seam, not at the worker count, means the limit
# holds even if latency drops. Injected `complete` callables (tests) bypass this
# entirely, which is what keeps the unit tests instant.
RATE_LIMIT_PER_MIN = 18
_pace_lock = threading.Lock()
_next_call_at = 0.0


def _pace() -> None:
    global _next_call_at
    with _pace_lock:
        now = time.monotonic()
        wait = max(0.0, _next_call_at - now)
        _next_call_at = max(now, _next_call_at) + 60.0 / RATE_LIMIT_PER_MIN
    if wait:
        time.sleep(wait)


class TruncatedResponse(Exception):
    """Raised when the model hit max_tokens before finishing its answer.

    Distinct from ValueError so decide()'s except-chain can't mistake a
    budget cutoff for a malformed reply (see engine.py module docstring /
    the incident that motivated this: a 256-token cap silently produced
    content=None, logged as parse-fail, for 109 of 130 bad decisions)."""


def nvidia_complete(
    max_tokens: int | None = None, model: str = "", timeout: float = 90,
) -> Callable[[str], str]:
    """Build an NVIDIA OpenAI-compatible `complete(prompt) -> text` callable.

    Shared client-building seam: AgentEngine's decision calls and memory.reflect's
    reflection call both need "detailed thinking off" + a token cap to keep
    nemotron-super's chain-of-thought from ballooning latency (see AgentEngine
    docstring for why). max_tokens=None (like model="") defers to cfg.agent so
    config.yaml's value actually reaches the API call instead of a hardcoded default.
    timeout defaults to 90s -- measured single-symbol latency (2026-07-31) ranged
    11-60s, so 45s would fail the slow tail; 90s is 1.5x the worst observed.
    """
    from openai import OpenAI

    from .config import load

    cfg = load()
    # No custom retry layer here (a 65-symbol live tick logged 0 rate-limited
    # vs 4 timeouts out of 65 calls -- see decisions.jsonl, 2026-07-21): the
    # SDK's own default retries (max_retries, unset here) already back off on
    # 429/5xx/timeout, so there is nothing for a hand-rolled layer to add.
    client = OpenAI(
        api_key=cfg.nvidia_api_key, base_url=cfg.nvidia_base_url, timeout=timeout,
    )
    model = model or cfg.agent.model
    max_tokens = max_tokens or cfg.agent.max_tokens

    def complete(prompt: str) -> str:
        _pace()
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[
                {"role": "system", "content": "detailed thinking off"},
                {"role": "user", "content": prompt},
            ],
        )
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise TruncatedResponse(
                f"hit max_tokens={max_tokens} (completion_tokens="
                f"{resp.usage.completion_tokens}) before finishing -- raise "
                "cfg.agent.max_tokens"
            )
        return choice.message.content or ""

    return complete


@dataclass(frozen=True)
class Decision:
    target: float  # desired position in {-1, 0, +1}
    reason: str    # human-readable why
    conviction: float | None = None  # per-bar signal strength, if the strategy has one
    # "ok": a genuine model/strategy verdict. "failed": decide() couldn't get
    # one (parse failure, timeout, rate limit, API error) and fell back to
    # holding current_pos. Default "ok" so every non-agent caller (and every
    # pre-existing positional/keyword construction) is unaffected; only
    # AgentEngine's except branch sets "failed". Consumers that compute
    # agent performance/hit-rate should filter on status == "ok" -- a failed
    # tick is not a trading decision, just a forced hold.
    status: str = "ok"


class DecisionEngine(Protocol):
    name: str

    def decide(
        self, symbol: str, history: pd.DataFrame, current_pos: float
    ) -> Decision: ...


class StrategyEngine:
    """Adapt a vectorized Strategy: the last value of positions(history) is
    the target for today. history must contain only bars up to today."""

    def __init__(self, strat: Strategy) -> None:
        self.strat = strat
        self.name = strat.name

    def decide(
        self, symbol: str, history: pd.DataFrame, current_pos: float
    ) -> Decision:
        target = float(self.strat.target(history))
        close = float(history["close"].iloc[-1])
        try:
            conviction = float(self.strat.signal(history).iloc[-1])
        except (NotImplementedError, KeyError, IndexError):
            conviction = None
        reason = f"{self.name}: target={target:+.0f} close={close:.2f}"
        return Decision(target=target, reason=reason, conviction=conviction)


def _mom5(close: pd.Series) -> float:
    """5-bar momentum with the same short-history fallback _features uses."""
    k = min(5, len(close) - 1)
    return float(close.iloc[-1] / close.iloc[-1 - k] - 1.0) if k >= 1 else 0.0


def market_block(histories: dict[str, pd.DataFrame]) -> dict:
    """Cross-sectional context for ONE bar, from every symbol's history up to
    that bar. Pure; no I/O. Missing SPY -> None fields, never an exception."""
    moms = {s: _mom5(h["close"].astype(float)) for s, h in histories.items() if len(h)}
    spy = histories.get("SPY")
    if spy is not None and len(spy) >= 2:
        c = spy["close"].astype(float)
        spy_1d = float(c.iloc[-1] / c.iloc[-2] - 1.0)
        spy_5d = _mom5(c)
    else:
        spy_1d = spy_5d = None
    n = len(moms)
    breadth = (sum(m > 0 for m in moms.values()) / n) if n else None
    order = sorted(moms, key=lambda s: moms[s], reverse=True)
    return {"spy_1d": spy_1d, "spy_5d": spy_5d, "breadth": breadth,
            "rank": {s: i + 1 for i, s in enumerate(order)}, "n": n}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.2%}"


def render_context(block: dict, symbol: str) -> str:
    """The two prompt lines for `symbol`. Ends with a newline."""
    n = block["n"]
    breadth = "n/a" if block["breadth"] is None else f"{block['breadth']:.2f}"
    rank = block["rank"].get(symbol)
    rank_txt = f"{rank}/{n}" if rank is not None else f"n/a/{n}"
    return (f"Market today: SPY_1d={_pct(block['spy_1d'])} SPY_5d={_pct(block['spy_5d'])} "
            f"breadth_5d={breadth} (share of {n} names with positive 5d momentum)\n"
            f"This name: momentum_5d rank {rank_txt} (1 = strongest)\n")


class AgentEngine:
    """Let an LLM pick today's position. Same DecisionEngine protocol as
    StrategyEngine: one JSON verdict per bar from a compact, lookahead-free
    prompt. `complete(prompt) -> raw_text` is the model seam (injected in
    tests); when None it lazily builds an NVIDIA OpenAI client on first use.
    Retries on rate-limit/timeout/5xx are the SDK's own (see nvidia_complete)."""

    def __init__(
        self,
        complete: Callable[[str], str] | None = None,
        *,
        model: str = "",
        lessons: str = "",
        name: str = "agent",
        allow_short: bool = False,
        max_tokens: int | None = None,
        market_context: bool = False,
    ) -> None:
        self.complete = complete
        self.model = model
        self.lessons = lessons
        self.name = name
        self.allow_short = allow_short
        self.max_tokens = max_tokens
        self.market_context = market_context

    def _default_complete(self) -> Callable[[str], str]:
        """Lazy NVIDIA OpenAI client — built once, on first decide().

        One bar-decision is a two-field JSON, not an essay. nemotron-super is a
        hybrid reasoning model that dumps a long chain-of-thought by default
        (60-120s/call at a 16000-token budget); the "detailed thinking off"
        system directive plus a token cap keeps each call bounded while still
        returning a reasoned verdict. self.max_tokens=None (the default) defers
        to cfg.agent.max_tokens rather than silently capping lower -- tune the
        budget there, not here.
        """
        return nvidia_complete(max_tokens=self.max_tokens, model=self.model)

    def _features(self, symbol: str, history: pd.DataFrame, current_pos: float) -> str:
        """The one line of model input for a symbol."""
        close = history["close"].astype(float)
        last = float(close.iloc[-1])
        # momentum over up to 5 prior bars; fall back to the whole window when
        # history is shorter (a 6-bar minimum would zero out short runs).
        mom5 = _mom5(close)
        rets = close.pct_change().dropna()
        vol20 = float(rets.tail(20).std()) if len(rets) >= 2 else 0.0
        if pd.isna(vol20):
            vol20 = 0.0
        return (f"{symbol}: last_close={last:.2f} momentum_5d={mom5:+.4f} "
                f"vol_20d={vol20:.4f} current_pos={current_pos:+.0f}")

    def _lessons_block(self) -> str:
        return f"\nPast-loss lessons to weigh:\n{self.lessons}\n" if self.lessons else ""

    def _prompt(self, symbol: str, history: pd.DataFrame, current_pos: float,
                context: str = "") -> str:
        return (
            f"You are a trading agent deciding today's position in {symbol}.\n"
            f"{self._features(symbol, history, current_pos)}\n"
            f"{context}"
            f"{self._lessons_block()}"
            "Respond with ONLY this JSON object and nothing else -- no "
            "reasoning, no markdown fences, no text before or after it: "
            '{"target": -1 | 0 | 1, "reason": "<=15 words"} where target is '
            "the desired position (-1 short, 0 flat, 1 long)."
        )

    def _extract(self, raw: str) -> dict:
        # findall + last match, not a single greedy search: a reasoning model's
        # chain-of-thought can echo the prompt's own example braces before the
        # real answer, and a first-{-to-last-} greedy span would swallow the
        # prose between them and fail json.loads.
        matches = re.findall(r"\{[^{}]*\}", raw, re.DOTALL)
        if not matches:
            raise ValueError(f"no JSON object in model reply: {raw[:120]!r}")
        return json.loads(matches[-1])

    def _target(self, raw_target) -> float:
        target = float(int(raw_target))
        if target not in (-1.0, 0.0, 1.0):
            raise ValueError("target out of range")
        if not self.allow_short and target == -1.0:
            target = 0.0
        return target

    def decide(
        self, symbol: str, history: pd.DataFrame, current_pos: float,
        context: str = "",
    ) -> Decision:
        if self.complete is None:
            self.complete = self._default_complete()
        status = "ok"
        try:
            obj = self._extract(self.complete(
                self._prompt(symbol, history, current_pos, context)))
            target = self._target(obj["target"])
            reason = str(obj.get("reason", ""))
        except Exception as e:
            status = "failed"
            target = float(current_pos)
            reason = _fail_reason(e)
        return Decision(target=target, reason=f"agent: {reason}", status=status)

    MAX_WORKERS = 8

    def decide_all(self, symbols: list[str], histories: dict[str, pd.DataFrame],
                   current_pos: dict[str, float],
                   context_histories: dict[str, pd.DataFrame] | None = None
                   ) -> dict[str, Decision]:
        """One model call PER SYMBOL, fanned out across MAX_WORKERS threads.

        Not a batched call. Batching multiple symbols into one prompt was
        measured and rejected (revamp plan T0, 2026-07-26, git history):
        nemotron-super's response verbosity is independent of how many symbols
        you ask about -- it emits 1-2k+ tokens of preamble either way -- so a
        multi-symbol call is not cheaper per symbol, just likelier to blow
        max_tokens or the timeout. Shipping it anyway (CHUNK_SIZE=13) cost the
        forward record a week: re-measured 2026-07-31, a 13-symbol chunk failed
        1 call in 3 (one 180s timeout; the two that answered burned 2219 and
        3379 of a 4600-token budget), while 10/10 single-symbol calls succeeded
        using 240-965 of a 2000-token budget.

        The blast radius is what actually matters here. forward.py's
        _agent_positions excludes a whole bar from the record if ANY symbol on
        it isn't status=="ok", so with 13 symbols per call one bad call killed
        the day: at 5 calls/bar and a 30% per-call failure rate only 0.7**5 =
        17% of days were admitted, which is the observed 1-day-in-4. Per-symbol
        calls make one failure cost one symbol, so the same per-call failure
        rate has to beat 65 independent coin flips instead of 5 -- and the
        per-call rate itself drops, because a single-symbol prompt is the
        configuration that measures reliable.

        Concurrency is across SYMBOLS ONLY, which is safe because symbols are
        independent; bars within a symbol are sequentially dependent (pos feeds
        the next bar's current_pos) and stay sequential in forward.py's caller
        loop. Rate is bounded by _pace() at the client seam, not by MAX_WORKERS
        -- the pool size only has to be large enough to keep the paced rate
        saturated at observed latency (18/min x ~21s median = ~6.3 calls in
        flight; 8 leaves headroom for the 60s tail).

        decide() swallows its own exceptions and returns status="failed", so no
        worker can raise and every symbol always gets a Decision.

        `context_histories` is the FULL universe's history up to this bar (the
        caller may be deciding only a subset). Used only when market_context is
        set: the block is computed once here and handed to every per-symbol
        prompt. Falls back to `histories` when not given.
        """
        if self.complete is None:
            self.complete = self._default_complete()
        ctx: dict[str, str] = {}
        if self.market_context:
            block = market_block(context_histories if context_histories is not None
                                 else histories)
            ctx = {s: render_context(block, s) for s in symbols}
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
            futures = {s: pool.submit(self.decide, s, histories[s], current_pos[s],
                                      ctx.get(s, ""))
                       for s in symbols}
        return {s: f.result() for s, f in futures.items()}


def _fail_reason(e: Exception) -> str:
    """Classify a failed model call so decisions.jsonl says what actually
    happened instead of collapsing everything into "parse-fail"."""
    from openai import APIStatusError, APITimeoutError, RateLimitError

    if isinstance(e, TruncatedResponse):
        # Must precede the ValueError branch: budget exhaustion is not a parse
        # failure, and TruncatedResponse deliberately isn't a ValueError
        # subclass so it can't fall into that branch anyway.
        reason = f"truncated: {e}"
    elif isinstance(e, (json.JSONDecodeError, KeyError, ValueError, AttributeError)):
        reason = f"parse-fail: {type(e).__name__}: {e}"
    elif isinstance(e, RateLimitError):
        reason = f"rate-limited: {e}"
    elif isinstance(e, APITimeoutError):
        reason = f"timeout: {e}"
    elif isinstance(e, APIStatusError):
        reason = f"http-error {e.status_code}: {e}"
    else:
        reason = f"error: {type(e).__name__}: {e}"
    return reason[:180]
