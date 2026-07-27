"""Safety + behavior checks for the local control panel (rhagent.control).

Stdlib only, no network, no live LLM calls. Every filesystem-touching check
uses tmp_path -- never the real journal/ or data/.
"""

from __future__ import annotations

import inspect
import io
from http.client import HTTPMessage
from pathlib import Path

import pytest

from rhagent import control


def test_binds_loopback_only():
    assert control.HOST == "127.0.0.1"


def test_source_never_references_the_live_broker_or_the_live_env_var():
    source = inspect.getsource(control)
    assert "McpBroker" not in source
    assert "LIVE" not in source


def test_state_changing_routes_are_post_only_by_construction():
    # Every mutating action lives under do_POST; do_GET only knows "/" and a
    # 405 branch for the same path set -- there is no code path where a GET
    # reaches do_halt/do_backtest/do_paper_tick/do_config.
    assert control.ControlHandler._POST_ONLY_PATHS == {
        "/halt", "/backtest", "/paper-tick", "/config",
    }


class _FakeSocket:
    """Enough of a socket for BaseHTTPRequestHandler to read a request from
    a bytes buffer and capture the response, without a real network call.
    BaseHTTPRequestHandler writes via self.connection.sendall(), not wfile
    directly, so that's what needs stubbing here."""

    def __init__(self, request: bytes):
        self._rfile = io.BytesIO(request)
        self.out = bytearray()

    def makefile(self, mode, *a, **kw):
        return self._rfile if "r" in mode else io.BytesIO()

    def sendall(self, data: bytes) -> None:
        self.out.extend(data)


def _dispatch(request_bytes: bytes, app: control.ControlApp) -> bytes:
    handler_cls = control.ControlHandler
    orig_app = handler_cls.app
    handler_cls.app = app
    try:
        sock = _FakeSocket(request_bytes)
        handler_cls(sock, ("127.0.0.1", 0), server=None)  # runs the request in __init__
    finally:
        handler_cls.app = orig_app
    return bytes(sock.out)


def test_get_on_a_state_changing_route_is_rejected_not_executed(tmp_path):
    app = control.ControlApp(
        journal_dir=tmp_path / "journal", data_dir=tmp_path / "data",
        config_path=tmp_path / "config.yaml", halt_file=tmp_path / "HALT",
    )
    response = _dispatch(b"GET /halt HTTP/1.1\r\nHost: x\r\n\r\n", app)
    assert b" 405 " in response.splitlines()[0]
    assert not (tmp_path / "HALT").exists()  # the action never ran


def test_post_halt_toggles_the_real_halt_file_wired_to_check_halted(tmp_path):
    halt_file = tmp_path / "HALT"
    app = control.ControlApp(
        journal_dir=tmp_path / "journal", data_dir=tmp_path / "data",
        config_path=tmp_path / "config.yaml", halt_file=halt_file,
    )
    assert app.do_halt("set") == "HALT file created"
    assert halt_file.exists()
    assert app.do_halt("clear") == "HALT file removed"
    assert not halt_file.exists()


def test_page_renders_with_no_forward_records_yet(tmp_path):
    app = control.ControlApp(
        journal_dir=tmp_path / "journal", data_dir=tmp_path / "data",
        config_path=tmp_path / "config.yaml", halt_file=tmp_path / "HALT",
    )
    html = app.page()
    assert "control panel" in html
    assert "none yet" in html


def test_config_patch_edits_only_the_targeted_lines_and_keeps_comments(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "limits:\n  per_trade_max_usd: 250\n"
        "agent:\n  # a comment that must survive\n  allow_short: true\n"
        "strategy:\n  cost_bps: 7.0\n"
    )
    control._patch_config_yaml(cfg_path, {"cost_bps": "9.5", "allow_short": "false"})
    text = cfg_path.read_text()
    assert "cost_bps: 9.5" in text
    assert "allow_short: false" in text
    assert "per_trade_max_usd: 250" in text  # untouched knob preserved
    assert "a comment that must survive" in text  # comments preserved (no yaml round-trip)
