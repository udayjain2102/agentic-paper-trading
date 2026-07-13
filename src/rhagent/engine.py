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


@dataclass(frozen=True)
class Decision:
    target: float  # desired position in {-1, 0, +1}
    reason: str    # human-readable why


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
        target = float(self.strat.positions(history).iloc[-1])
        close = float(history["close"].iloc[-1])
        reason = f"{self.name}: target={target:+.0f} close={close:.2f}"
        return Decision(target=target, reason=reason)


class AgentEngine:
    """Let an LLM pick today's position. Same DecisionEngine protocol as
    StrategyEngine: one JSON verdict per bar from a compact, lookahead-free
    prompt. `complete(prompt) -> raw_text` is the model seam (injected in
    tests); when None it lazily builds an NVIDIA OpenAI client on first use."""

    def __init__(
        self,
        complete: Callable[[str], str] | None = None,
        *,
        model: str = "",
        lessons: str = "",
        name: str = "agent",
        allow_short: bool = True,
        max_tokens: int = 256,
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
        (60-120s/call at cfg.agent.max_tokens=16000); the "detailed thinking
        off" system directive plus a small token cap keeps each call ~2s while
        still returning a reasoned verdict.
        """
        from openai import OpenAI

        from .config import load

        cfg = load()
        # max_retries lets the SDK ride out transient 429s / timeouts with
        # backoff rather than dropping the bar to a held "parse-fail" decision.
        client = OpenAI(
            api_key=cfg.nvidia_api_key, base_url=cfg.nvidia_base_url,
            timeout=45, max_retries=8,
        )
        model = self.model or cfg.agent.model

        def complete(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=self.max_tokens,
                temperature=0,
                messages=[
                    {"role": "system", "content": "detailed thinking off"},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content or ""

        return complete

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

    def decide(
        self, symbol: str, history: pd.DataFrame, current_pos: float
    ) -> Decision:
        if self.complete is None:
            self.complete = self._default_complete()
        prompt = self._prompt(symbol, history, current_pos)
        try:
            raw = self.complete(prompt)
            obj = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
            target = float(int(obj["target"]))
            if target not in (-1.0, 0.0, 1.0):
                raise ValueError("target out of range")
            if not self.allow_short and target == -1.0:
                target = 0.0
            reason = str(obj.get("reason", ""))
        except Exception:
            target = float(current_pos)
            reason = "parse-fail: held"
        return Decision(target=target, reason=f"agent: {reason}")
