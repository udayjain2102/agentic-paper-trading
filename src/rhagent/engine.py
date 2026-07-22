"""The decision seam between the paper-trade loop and whatever decides.

A DecisionEngine answers one question per bar: given the history up to and
including today and what we currently hold, what should the position be and
why. StrategyEngine adapts the existing rule-based strategies; an AgentEngine
wrapping the Claude loop plugs into the same protocol later.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd

from .strategies.base import Strategy

# 4s/call x 65 symbols = ~4.3 min of pure pacing per tick, before model latency.
_MIN_CALL_INTERVAL = 4.0  # seconds; NVIDIA's ~18 req/min bucket -- pace under it
_RATE_LIMIT_RETRIES = 3   # extra attempts after a 429, each with doubling backoff


def nvidia_complete(max_tokens: int = 256, model: str = "") -> Callable[[str], str]:
    """Build an NVIDIA OpenAI-compatible `complete(prompt) -> text` callable.

    Shared client-building seam: AgentEngine's decision calls and memory.reflect's
    reflection call both need "detailed thinking off" + a token cap to keep
    nemotron-super's chain-of-thought from ballooning latency (see AgentEngine
    docstring for why).
    """
    from openai import OpenAI

    from .config import load

    cfg = load()
    # max_retries=0: AgentEngine._call_model is the one retry authority for
    # rate limits (paced, backed off, tested). Letting the SDK also retry
    # here would multiply attempts -- each of our retries would silently
    # trigger another round of SDK-internal retries underneath it -- which
    # is exactly the pile-on that gets a burst of calls rate-limited harder.
    client = OpenAI(
        api_key=cfg.nvidia_api_key, base_url=cfg.nvidia_base_url,
        timeout=45, max_retries=0,
    )
    model = model or cfg.agent.model

    def complete(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[
                {"role": "system", "content": "detailed thinking off"},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    return complete


@dataclass(frozen=True)
class Decision:
    target: float  # desired position in {-1, 0, +1}
    reason: str    # human-readable why
    conviction: float | None = None  # per-bar signal strength, if the strategy has one


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


class AgentEngine:
    """Let an LLM pick today's position. Same DecisionEngine protocol as
    StrategyEngine: one JSON verdict per bar from a compact, lookahead-free
    prompt. `complete(prompt) -> raw_text` is the model seam (injected in
    tests); when None it lazily builds an NVIDIA OpenAI client on first use.
    Calls through that seam are paced and retried on rate-limit errors (see
    `_call_model`); `sleep` is injectable so tests never actually wait."""

    def __init__(
        self,
        complete: Callable[[str], str] | None = None,
        *,
        model: str = "",
        lessons: str = "",
        name: str = "agent",
        allow_short: bool = False,
        max_tokens: int = 256,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.complete = complete
        self.model = model
        self.lessons = lessons
        self.name = name
        self.allow_short = allow_short
        self.max_tokens = max_tokens
        self.sleep = sleep
        self._last_call = 0.0

    def _default_complete(self) -> Callable[[str], str]:
        """Lazy NVIDIA OpenAI client — built once, on first decide().

        One bar-decision is a two-field JSON, not an essay. nemotron-super is a
        hybrid reasoning model that dumps a long chain-of-thought by default
        (60-120s/call at cfg.agent.max_tokens=16000); the "detailed thinking
        off" system directive plus a small token cap keeps each call ~2s while
        still returning a reasoned verdict.
        """
        return nvidia_complete(max_tokens=self.max_tokens, model=self.model)

    def _prompt(self, symbol: str, history: pd.DataFrame, current_pos: float) -> str:
        close = history["close"].astype(float)
        last = float(close.iloc[-1])
        # momentum over up to 5 prior bars; fall back to the whole window when
        # history is shorter (a 6-bar minimum would zero out short runs).
        k = min(5, len(close) - 1)
        mom5 = float(close.iloc[-1] / close.iloc[-1 - k] - 1.0) if k >= 1 else 0.0
        rets = close.pct_change().dropna()
        vol20 = float(rets.tail(20).std()) if len(rets) >= 2 else 0.0
        if pd.isna(vol20):
            vol20 = 0.0
        lessons = f"\nPast-loss lessons to weigh:\n{self.lessons}\n" if self.lessons else ""
        return (
            f"You are a trading agent deciding today's position in {symbol}.\n"
            f"last_close={last:.2f} momentum_5d={mom5:+.4f} "
            f"vol_20d={vol20:.4f} current_pos={current_pos:+.0f}\n"
            f"{lessons}"
            'Reply with STRICT JSON only: {"target": -1 | 0 | 1, '
            '"reason": "<=15 words"} where target is the desired position '
            "(-1 short, 0 flat, 1 long)."
        )

    def _call_model(self, prompt: str) -> str:
        """Call self.complete, paced ~_MIN_CALL_INTERVAL apart and retried
        with doubling backoff on a rate-limit error. The only retry authority
        here (nvidia_complete's client disables its own) so the numbers below
        are the whole story, not one layer of several.

        A 65-symbol tick calling self.complete back-to-back blows straight
        through NVIDIA's burst-then-~18/min bucket; pacing keeps steady state
        under it, and the retry rides out whatever still slips through.
        ponytail: fixed interval/attempt count, not a general limiter --
        revisit if a second model provider needs different numbers.
        """
        from openai import RateLimitError

        wait = _MIN_CALL_INTERVAL - (time.monotonic() - self._last_call)
        if wait > 0:
            self.sleep(wait)
        delay = 2.0
        last_err: RateLimitError | None = None
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            self._last_call = time.monotonic()
            try:
                return self.complete(prompt)
            except RateLimitError as e:
                last_err = e
                if attempt < _RATE_LIMIT_RETRIES:
                    self.sleep(delay)
                    delay *= 2
        raise last_err

    def decide(
        self, symbol: str, history: pd.DataFrame, current_pos: float
    ) -> Decision:
        from openai import APIStatusError, APITimeoutError, RateLimitError

        if self.complete is None:
            self.complete = self._default_complete()
        prompt = self._prompt(symbol, history, current_pos)
        try:
            raw = self._call_model(prompt)
            obj = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
            target = float(int(obj["target"]))
            if target not in (-1.0, 0.0, 1.0):
                raise ValueError("target out of range")
            if not self.allow_short and target == -1.0:
                target = 0.0
            reason = str(obj.get("reason", ""))
        except Exception as e:
            target = float(current_pos)
            # Distinguish failure classes so decisions.jsonl says what actually
            # happened instead of collapsing everything into "parse-fail".
            if isinstance(e, (json.JSONDecodeError, KeyError, ValueError, AttributeError)):
                reason = f"parse-fail: {type(e).__name__}: {e}"
            elif isinstance(e, RateLimitError):
                reason = f"rate-limited: {e}"
            elif isinstance(e, APITimeoutError):
                reason = f"timeout: {e}"
            elif isinstance(e, APIStatusError):
                reason = f"http-error {e.status_code}: {e}"
            else:
                reason = f"error: {type(e).__name__}: {e}"
            reason = reason[:180]
        return Decision(target=target, reason=f"agent: {reason}")
