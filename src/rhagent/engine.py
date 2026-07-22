"""The decision seam between the paper-trade loop and whatever decides.

A DecisionEngine answers one question per bar: given the history up to and
including today and what we currently hold, what should the position be and
why. StrategyEngine adapts the existing rule-based strategies; an AgentEngine
wrapping the Claude loop plugs into the same protocol later.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd

from .strategies.base import Strategy


class TruncatedResponse(Exception):
    """Raised when the model hit max_tokens before finishing its answer.

    Distinct from ValueError so decide()'s except-chain can't mistake a
    budget cutoff for a malformed reply (see engine.py module docstring /
    the incident that motivated this: a 256-token cap silently produced
    content=None, logged as parse-fail, for 109 of 130 bad decisions)."""


def nvidia_complete(max_tokens: int | None = None, model: str = "") -> Callable[[str], str]:
    """Build an NVIDIA OpenAI-compatible `complete(prompt) -> text` callable.

    Shared client-building seam: AgentEngine's decision calls and memory.reflect's
    reflection call both need "detailed thinking off" + a token cap to keep
    nemotron-super's chain-of-thought from ballooning latency (see AgentEngine
    docstring for why). max_tokens=None (like model="") defers to cfg.agent so
    config.yaml's value actually reaches the API call instead of a hardcoded default.
    """
    from openai import OpenAI

    from .config import load

    cfg = load()
    # No custom retry layer here (a 65-symbol live tick logged 0 rate-limited
    # vs 4 timeouts out of 65 calls -- see decisions.jsonl, 2026-07-21): the
    # SDK's own default retries (max_retries, unset here) already back off on
    # 429/5xx/timeout, so there is nothing for a hand-rolled layer to add.
    client = OpenAI(
        api_key=cfg.nvidia_api_key, base_url=cfg.nvidia_base_url, timeout=45,
    )
    model = model or cfg.agent.model
    max_tokens = max_tokens or cfg.agent.max_tokens

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
    ) -> None:
        self.complete = complete
        self.model = model
        self.lessons = lessons
        self.name = name
        self.allow_short = allow_short
        self.max_tokens = max_tokens

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
            "Respond with ONLY this JSON object and nothing else -- no "
            "reasoning, no markdown fences, no text before or after it: "
            '{"target": -1 | 0 | 1, "reason": "<=15 words"} where target is '
            "the desired position (-1 short, 0 flat, 1 long)."
        )

    def decide(
        self, symbol: str, history: pd.DataFrame, current_pos: float
    ) -> Decision:
        from openai import APIStatusError, APITimeoutError, RateLimitError

        if self.complete is None:
            self.complete = self._default_complete()
        prompt = self._prompt(symbol, history, current_pos)
        status = "ok"
        try:
            raw = self.complete(prompt)
            # findall + last match, not a single greedy search: a reasoning model's
            # chain-of-thought can echo the prompt's own example braces before the
            # real answer, and a first-{-to-last-} greedy span would swallow the
            # prose between them and fail json.loads.
            matches = re.findall(r"\{[^{}]*\}", raw, re.DOTALL)
            if not matches:
                raise ValueError(f"no JSON object in model reply: {raw[:120]!r}")
            obj = json.loads(matches[-1])
            target = float(int(obj["target"]))
            if target not in (-1.0, 0.0, 1.0):
                raise ValueError("target out of range")
            if not self.allow_short and target == -1.0:
                target = 0.0
            reason = str(obj.get("reason", ""))
        except Exception as e:
            status = "failed"
            target = float(current_pos)
            # Distinguish failure classes so decisions.jsonl says what actually
            # happened instead of collapsing everything into "parse-fail".
            if isinstance(e, TruncatedResponse):
                # Must precede the ValueError branch: budget exhaustion is not
                # a parse failure, and TruncatedResponse deliberately isn't a
                # ValueError subclass so it can't fall into that branch anyway.
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
            reason = reason[:180]
        return Decision(target=target, reason=f"agent: {reason}", status=status)
