"""Connect to the Robinhood trading MCP over streamable HTTP.

Isolated here so the rest of the code (and tests) never depend on a live MCP.
The bearer token comes from ROBINHOOD_MCP_TOKEN, obtained after you authenticate
the server. Until that token exists, the runner falls back to the mock broker.
"""

from __future__ import annotations

import contextlib
from typing import Iterator


@contextlib.contextmanager
def mcp_session(url: str, token: str) -> Iterator[object]:
    """Yield an initialized mcp.ClientSession for the Robinhood MCP.

    Requires the `mcp` package. Raises if the token is missing — callers should
    check for a token before entering this context.
    """
    if not token:
        raise RuntimeError(
            "ROBINHOOD_MCP_TOKEN is not set. Authenticate the Robinhood MCP "
            "first (run `/mcp` in an interactive claude session), then export "
            "the bearer token."
        )

    import anyio
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {"Authorization": f"Bearer {token}"}

    # The MCP client API is async; bridge to the synchronous runner with a
    # dedicated portal so broker calls can be made from sync code.
    with anyio.from_thread.start_blocking_portal() as portal:
        cm = portal.wrap_async_context_manager(
            _open(streamablehttp_client, ClientSession, url, headers)
        )
        with cm as session:
            yield session


@contextlib.asynccontextmanager
async def _open(streamablehttp_client, ClientSession, url, headers):
    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session
