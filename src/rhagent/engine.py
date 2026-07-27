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


# decide_all's chunk size sits between the two failure modes measured against
# the real NVIDIA API on 2026-07-27 (see report to the task that added this):
#   - too big (65, the whole universe in one call): a single prompt asking for
#     65 verdicts never finishes inside any timeout tuned off real latency --
#     nemotron-super's hidden reasoning tokens scale with symbol count, and the
#     call either times out or exhausts max_tokens before emitting JSON.
#   - too small (1, one call per symbol): 65 sequential calls/bar against
#     NVIDIA's burst-then-~18-calls/min bucket is exactly what produced the
#     original timeouts (see decide_all's own docstring / 2026-07-21 incident).
# 13 divides the 65-symbol universe into exactly 5 calls/bar. Measured
# real-API latency for 12-13-symbol chunks: 63-91s across 4 calls (2026-07-27),
# so CHUNK_TIMEOUT_S below is 2x the worst observed, not a guess.
CHUNK_SIZE = 13
CHUNK_TIMEOUT_S = 180
# completion_tokens observed for working 12-13-symbol chunks: 1055-1602 (same
# measurement run). cfg.agent.max_tokens (2000) is sized for ONE symbol's
# preamble + JSON; a 13-symbol chunk at that cap truncated at exactly 2000
# tokens with empty content (confirmed 2026-07-27). Scale linearly off the
# single-symbol budget with headroom rather than hardcoding a second constant.
_TOKENS_PER_EXTRA_SYMBOL = 200


class TruncatedResponse(Exception):
    """Raised when the model hit max_tokens before finishing its answer.

    Distinct from ValueError so decide()'s except-chain can't mistake a
    budget cutoff for a malformed reply (see engine.py module docstring /
    the incident that motivated this: a 256-token cap silently produced
    content=None, logged as parse-fail, for 109 of 130 bad decisions)."""


def nvidia_complete(
    max_tokens: int | None = None, model: str = "", timeout: float = 45,
) -> Callable[[str], str]:
    """Build an NVIDIA OpenAI-compatible `complete(prompt) -> text` callable.

    Shared client-building seam: AgentEngine's decision calls and memory.reflect's
    reflection call both need "detailed thinking off" + a token cap to keep
    nemotron-super's chain-of-thought from ballooning latency (see AgentEngine
    docstring for why). max_tokens=None (like model="") defers to cfg.agent so
    config.yaml's value actually reaches the API call instead of a hardcoded default.
    timeout defaults to 45s (right for a single-symbol prompt); decide_all's
    chunked calls pass a larger, measured value -- see CHUNK_SIZE/CHUNK_TIMEOUT_S.
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
        self._chunk_complete: Callable[[str], str] | None = None

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

    def _default_complete_chunk(self) -> Callable[[str], str]:
        """Lazy NVIDIA client for decide_all's chunked calls: same model, but a
        larger timeout and max_tokens sized for CHUNK_SIZE symbols per call
        rather than one -- see the CHUNK_SIZE comment for the measurements
        behind both numbers. Separate from _default_complete (and cached
        separately) so decide()'s single-symbol calls keep the tighter,
        config-driven budget/timeout that already works for them."""
        from .config import load

        base_tokens = self.max_tokens or load().agent.max_tokens
        chunk_tokens = base_tokens + _TOKENS_PER_EXTRA_SYMBOL * CHUNK_SIZE
        return nvidia_complete(
            max_tokens=chunk_tokens, model=self.model, timeout=CHUNK_TIMEOUT_S,
        )

    def _features(self, symbol: str, history: pd.DataFrame, current_pos: float) -> str:
        """The one line of model input for a symbol. Shared by the single-symbol
        and whole-universe prompts so both see identical features."""
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
        return (f"{symbol}: last_close={last:.2f} momentum_5d={mom5:+.4f} "
                f"vol_20d={vol20:.4f} current_pos={current_pos:+.0f}")

    def _lessons_block(self) -> str:
        return f"\nPast-loss lessons to weigh:\n{self.lessons}\n" if self.lessons else ""

    def _prompt(self, symbol: str, history: pd.DataFrame, current_pos: float) -> str:
        return (
            f"You are a trading agent deciding today's position in {symbol}.\n"
            f"{self._features(symbol, history, current_pos)}\n"
            f"{self._lessons_block()}"
            "Respond with ONLY this JSON object and nothing else -- no "
            "reasoning, no markdown fences, no text before or after it: "
            '{"target": -1 | 0 | 1, "reason": "<=15 words"} where target is '
            "the desired position (-1 short, 0 flat, 1 long)."
        )

    def _prompt_all(self, symbols: list[str], histories: dict[str, pd.DataFrame],
                    current_pos: dict[str, float]) -> str:
        rows = "\n".join(self._features(s, histories[s], current_pos[s]) for s in symbols)
        return (
            f"You are a trading agent deciding today's position in {len(symbols)} "
            "symbols at once. One line of features per symbol:\n"
            f"{rows}\n"
            f"{self._lessons_block()}"
            "Respond with ONLY one flat JSON object and nothing else -- no "
            "reasoning, no markdown fences, no nested objects, no text before or "
            'after it: {"SYM": -1 | 0 | 1, ...} mapping every one of the '
            f"{len(symbols)} symbols above to its desired position "
            "(-1 short, 0 flat, 1 long)."
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
        self, symbol: str, history: pd.DataFrame, current_pos: float
    ) -> Decision:
        if self.complete is None:
            self.complete = self._default_complete()
        status = "ok"
        try:
            obj = self._extract(self.complete(self._prompt(symbol, history, current_pos)))
            target = self._target(obj["target"])
            reason = str(obj.get("reason", ""))
        except Exception as e:
            status = "failed"
            target = float(current_pos)
            reason = _fail_reason(e)
        return Decision(target=target, reason=f"agent: {reason}", status=status)

    def decide_all(self, symbols: list[str], histories: dict[str, pd.DataFrame],
                   current_pos: dict[str, float]) -> dict[str, Decision]:
        """One model call per CHUNK_SIZE-symbol group, same semantics as decide().

        65 sequential per-symbol calls per bar against NVIDIA's burst-then-~18/min
        bucket is what produced the original timeouts; one call for the whole
        65-symbol universe is what replaced them (#41) -- and that one call never
        finishes (measured: times out or exhausts max_tokens before emitting
        JSON). CHUNK_SIZE (see module-level comment) is the middle ground: a few
        calls/bar, each small enough to finish inside a timeout that was actually
        measured against the real API. Failure semantics are preserved exactly
        and PER CHUNK: if a chunk's call fails, every symbol in that chunk is
        status="failed" holding current_pos; if a chunk's call answers but omits
        or malforms one symbol, only that symbol fails -- other chunks are
        unaffected either way.
        """
        if self.complete is None and self._chunk_complete is None:
            self._chunk_complete = self._default_complete_chunk()
        complete = self.complete or self._chunk_complete

        out: dict[str, Decision] = {}
        for i in range(0, len(symbols), CHUNK_SIZE):
            chunk = symbols[i:i + CHUNK_SIZE]
            chunk_histories = {s: histories[s] for s in chunk}
            chunk_pos = {s: current_pos[s] for s in chunk}
            try:
                obj = self._extract(complete(self._prompt_all(chunk, chunk_histories, chunk_pos)))
            except Exception as e:
                reason = f"agent: {_fail_reason(e)}"
                for s in chunk:
                    out[s] = Decision(target=float(current_pos[s]), reason=reason,
                                      status="failed")
                continue
            for s in chunk:
                try:
                    out[s] = Decision(target=self._target(obj[s]), reason="agent: batch verdict")
                except Exception as e:
                    out[s] = Decision(target=float(current_pos[s]),
                                      reason=f"agent: {_fail_reason(e)}", status="failed")
        return out


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
