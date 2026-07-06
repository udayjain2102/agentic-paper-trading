from rhagent.broker import MockBroker


def test_mock_broker_reports_position_values_and_symbols():
    broker = MockBroker(positions={"AAPL": 180})
    account = broker.get_account()
    assert account.position_values == {"AAPL": 180}
    assert account.positions == frozenset({"AAPL"})
