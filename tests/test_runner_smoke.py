"""End-to-end dry-run smoke test.

Drives the real agent loop with a stubbed Anthropic client (no network), where
the model proposes a buy. Asserts the full path runs and — because we're in
dry-run — nothing reaches the broker.
"""

from types import SimpleNamespace

from rhagent.agent import run_session
from rhagent.broker import MockBroker
from rhagent.config import AgentConfig
from rhagent.executor import OrderExecutor
from rhagent.guardrails import Limits, RunState
from rhagent.journal import Journal


class _Block(SimpleNamespace):
    pass


class _FakeStream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class _FakeMessages:
    def __init__(self, scripted):
        self._scripted = scripted
        self.calls = 0

    def stream(self, **kwargs):
        msg = self._scripted[self.calls]
        self.calls += 1
        return _FakeStream(msg)


class _FakeClient:
    def __init__(self, scripted):
        self.messages = _FakeMessages(scripted)


def test_full_dry_run_places_nothing(tmp_path):
    broker = MockBroker(buying_power_usd=5_000, quotes={"AAPL": 200.0})
    journal = Journal(tmp_path / "runs.jsonl")
    executor = OrderExecutor(
        broker=broker,
        account=broker.get_account(),
        limits=Limits(
            per_trade_max_usd=250,
            total_deployed_max_usd=2000,
            max_new_positions_per_run=2,
            max_orders_per_run=5,
            daily_loss_limit_usd=200,
        ),
        run_state=RunState(),
        journal=journal,
        dry_run=True,
    )

    # Turn 1: model calls place_order. Turn 2: model finishes with text.
    scripted = [
        SimpleNamespace(
            stop_reason="tool_use",
            content=[
                _Block(
                    type="tool_use",
                    name="place_order",
                    id="t1",
                    input={"symbol": "AAPL", "side": "buy", "notional_usd": 100},
                )
            ],
        ),
        SimpleNamespace(
            stop_reason="end_turn",
            content=[_Block(type="text", text="Placed a starter position in AAPL.")],
        ),
    ]
    client = _FakeClient(scripted)

    summary = run_session(
        client=client,
        broker=broker,
        executor=executor,
        agent_cfg=AgentConfig(model="claude-opus-4-8", effort="high", max_tokens=16000),
    )

    assert "AAPL" in summary
    assert broker.placed == []  # dry-run guarantee holds end to end
