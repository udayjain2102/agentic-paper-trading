"""End-to-end dry-run smoke test.

Drives the real agent loop with a stubbed OpenAI/NVIDIA client (no network),
where the model proposes a buy. Asserts the full path runs and — because we're
in dry-run — nothing reaches the broker.
"""

import json
from types import SimpleNamespace

from rhagent.agent import run_session
from rhagent.broker import MockBroker
from rhagent.config import AgentConfig
from rhagent.executor import OrderExecutor
from rhagent.guardrails import Limits, RunState
from rhagent.journal import Journal


class _FakeMessage:
    """Mimics openai's ChatCompletionMessage (content, tool_calls, model_dump)."""

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, **_kwargs):
        return {"role": "assistant", "content": self.content}


def _tool_call(call_id, name, args):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class _FakeCompletions:
    def __init__(self, scripted):
        self._scripted = scripted
        self.calls = 0

    def create(self, **kwargs):
        msg = self._scripted[self.calls]
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _FakeClient:
    def __init__(self, scripted):
        self.chat = SimpleNamespace(completions=_FakeCompletions(scripted))


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
        _FakeMessage(
            tool_calls=[
                _tool_call(
                    "t1",
                    "place_order",
                    {"symbol": "AAPL", "side": "buy", "notional_usd": 100},
                )
            ],
        ),
        _FakeMessage(content="Placed a starter position in AAPL."),
    ]
    client = _FakeClient(scripted)

    summary = run_session(
        client=client,
        broker=broker,
        executor=executor,
        agent_cfg=AgentConfig(
            model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
            effort="high",
            max_tokens=16000,
        ),
    )

    assert "AAPL" in summary
    assert broker.placed == []  # dry-run guarantee holds end to end
