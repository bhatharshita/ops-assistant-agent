"""Launch the MCP server over stdio and call its tools, like an MCP client would.

    python scripts/mcp_client_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from mcp import Client, StdioServerParameters


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "ops_assistant.mcp_server"],
                                   env=dict(os.environ))
    async with Client(params) as client:
        tools = await client.list_tools()
        print("tools:", [t.name for t in tools.tools])
        result = await client.call_tool("log_stats", {"group_by": "node",
                                                      "error_signature": "OOMKilled"})
        print("log_stats(group_by=node, error_signature=OOMKilled):")
        print(result.content[0].text[:600])


if __name__ == "__main__":
    asyncio.run(main())
