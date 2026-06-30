"""Orchestrates one cron tick: load config, check the kill switch, run the
agent, journal the result.

Usage:
    python -m rhagent.runner

Runs in dry-run unless LIVE=true. With no Robinhood MCP token configured, it
uses the in-memory mock broker so you can exercise the full path on paper.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import agent as agent_mod
from .broker import Broker, McpBroker, MockBroker
from .config import Config, load
from .executor import OrderExecutor
from .guardrails import RunState, check_halted
from .journal import Journal

HALT_FILE = Path("HALT")


def _make_broker(cfg: Config):
    """Return (broker, cleanup). Live MCP broker if a token is set, else mock."""
    if cfg.mcp_token:
        from .mcp_session import mcp_session

        ctx = mcp_session(cfg.mcp_url, cfg.mcp_token)
        session = ctx.__enter__()
        return McpBroker(session), lambda: ctx.__exit__(None, None, None)

    # Paper fallback: a small simulated account.
    broker = MockBroker(
        buying_power_usd=5_000,
        quotes={"AAPL": 200.0, "MSFT": 410.0, "SPY": 540.0},
    )
    return broker, lambda: None


def run() -> int:
    cfg = load()
    journal = Journal()
    mode = "DRY-RUN" if cfg.dry_run else "LIVE"
    journal.record("run_start", mode=mode)

    broker, cleanup = _make_broker(cfg)
    try:
        account = broker.get_account()

        halted, reason = check_halted(account, cfg.limits, HALT_FILE.exists())
        if halted:
            journal.record("run_halted", reason=reason)
            print(f"[{mode}] Halted: {reason}")
            return 0

        executor = OrderExecutor(
            broker=broker,
            account=account,
            limits=cfg.limits,
            run_state=RunState(),
            journal=journal,
            dry_run=cfg.dry_run,
        )

        if os.environ.get("MOCK_AGENT", "").strip().lower() == "true":
            # No API key needed: a scripted decision still flows through the
            # executor + guardrails, so the full dry-run pipeline is exercised.
            summary = agent_mod.run_scripted_session(
                broker=broker,
                executor=executor,
            )
        else:
            from openai import OpenAI

            if not cfg.nvidia_api_key:
                raise SystemExit(
                    "NVIDIA_API_KEY is not set. Put it in .env, or run with "
                    "MOCK_AGENT=true to exercise the pipeline without an LLM."
                )
            client = OpenAI(
                api_key=cfg.nvidia_api_key,
                base_url=cfg.nvidia_base_url,
            )
            summary = agent_mod.run_session(
                client=client,
                broker=broker,
                executor=executor,
                agent_cfg=cfg.agent,
            )
        journal.record("run_end", mode=mode, summary=summary)
        print(f"[{mode}] Run complete.\n{summary}")
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(run())
