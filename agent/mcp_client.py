"""MCP client: connects to the HR server and discovers its tools at runtime.

The tool list handed to the LLM is **not** written in this file. It is fetched
over the protocol with ``list_tools()`` and translated into OpenAI
function-calling schemas on the fly. Adding a tool to ``mcp_server/hr_server.py``
makes it available to the agent with no change here and no change to the
prompt -- which is the actual point of using MCP rather than wrapping local
function calls and calling it an integration.

Both transports the server supports are supported here:

* ``stdio`` -- the server is spawned as a child process. Used locally and in CI.
* ``streamable-http`` -- the server is a separate HTTP service. Used in the
  deployed app, where the agent and the tool server are independently
  addressable.

The transport is chosen by ``MCP_TRANSPORT``; nothing else in the agent knows
or cares which one is active.
"""
from __future__ import annotations

import json
import logging
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import settings

log = logging.getLogger(__name__)


class MCPUnavailable(RuntimeError):
    """The tool server could not be reached."""


@dataclass
class ToolCallResult:
    """Outcome of one tool invocation, including the failure case."""

    name: str
    arguments: dict[str, Any]
    content: Any
    is_error: bool = False
    latency_ms: float = 0.0
    raw_text: str = ""

    def as_message_content(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return json.dumps(self.content, ensure_ascii=False, default=str)


@dataclass
class MCPToolClient:
    """A live MCP session plus the tool catalogue discovered from it."""

    session: ClientSession | None = None
    tools: list[Any] = field(default_factory=list)
    server_name: str = ""
    server_version: str = ""
    transport: str = ""
    _stack: AsyncExitStack | None = None

    async def connect(self) -> MCPToolClient:
        transport = settings.mcp_transport.strip().lower()
        self._stack = AsyncExitStack()
        try:
            if transport == "stdio":
                params = StdioServerParameters(
                    command=sys.executable,
                    args=[str(settings.mcp_server_script)],
                    env=None,
                )
                read, write = await self._stack.enter_async_context(stdio_client(params))
            elif transport in ("streamable-http", "http"):
                from mcp.client.streamable_http import streamablehttp_client

                read, write, _ = await self._stack.enter_async_context(
                    streamablehttp_client(settings.mcp_server_url)
                )
            else:
                raise MCPUnavailable(f"unsupported MCP_TRANSPORT {transport!r}")

            self.session = await self._stack.enter_async_context(ClientSession(read, write))
            info = await self.session.initialize()
            self.server_name = info.server_info.name
            self.server_version = info.server_info.version
            self.transport = transport
            await self.refresh_tools()
            log.info(
                "connected to MCP server %s v%s over %s; %d tools discovered",
                self.server_name, self.server_version, transport, len(self.tools),
            )
            return self
        except Exception as exc:
            await self.aclose()
            raise MCPUnavailable(f"could not connect to the MCP server over {transport}: {exc}") from exc

    async def refresh_tools(self) -> list[Any]:
        """Re-read the catalogue. Called on connect, and available at runtime so a
        server that gains tools mid-session is picked up without a restart."""
        assert self.session is not None
        self.tools = list((await self.session.list_tools()).tools)
        return self.tools

    def openai_tools(self) -> list[dict[str, Any]]:
        """Translate discovered MCP tools into OpenAI function-calling schemas.

        MCP's ``input_schema`` is already JSON Schema, so this is a re-wrap
        rather than a re-specification: the descriptions the model sees are the
        ones the server publishes.
        """
        specs: list[dict[str, Any]] = []
        for tool in self.tools:
            schema = dict(tool.input_schema or {"type": "object", "properties": {}})
            schema.setdefault("type", "object")
            schema.setdefault("properties", {})
            specs.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": (tool.description or "").strip(),
                    "parameters": schema,
                },
            })
        return specs

    def catalogue(self) -> list[dict[str, Any]]:
        """Human-readable tool inventory, for /health and the trace panel."""
        out = []
        for tool in self.tools:
            annotations = tool.annotations
            out.append({
                "name": tool.name,
                "description": (tool.description or "").strip().split("\n")[0],
                "read_only": bool(annotations.read_only_hint) if annotations else None,
                "parameters": sorted((tool.input_schema or {}).get("properties", {})),
                "required": sorted((tool.input_schema or {}).get("required", [])),
            })
        return out

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """Invoke a tool. Protocol and server errors both come back as values.

        A raised exception here would abort the whole agent run; returning the
        error lets the model read what went wrong and try a corrected call,
        which is the self-correction behaviour the evaluation suite measures.
        """
        import time

        if self.session is None:
            raise MCPUnavailable("MCP session is not connected")

        known = {t.name for t in self.tools}
        if name not in known:
            return ToolCallResult(
                name=name,
                arguments=arguments,
                content={
                    "error": f"unknown tool {name!r}",
                    "hint": f"available tools: {', '.join(sorted(known))}",
                },
                is_error=True,
            )

        started = time.perf_counter()
        try:
            result = await self.session.call_tool(name, arguments)
        except Exception as exc:
            return ToolCallResult(
                name=name,
                arguments=arguments,
                content={"error": f"tool call failed: {exc}", "hint": "Check the argument types."},
                is_error=True,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        text = "".join(
            block.text for block in result.content if getattr(block, "type", "") == "text"
        )
        try:
            content: Any = json.loads(text)
        except json.JSONDecodeError:
            content = text

        is_error = bool(result.is_error) or (isinstance(content, dict) and "error" in content)
        return ToolCallResult(
            name=name,
            arguments=arguments,
            content=content,
            is_error=is_error,
            latency_ms=latency_ms,
            raw_text=text,
        )

    async def aclose(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception as exc:  # transport teardown races are not actionable
                log.debug("MCP teardown: %s", exc)
            self._stack = None
        self.session = None
        self.tools = []
