#!/usr/bin/env python3
"""MCP security gateway: a JSON-RPC reverse proxy with per-tool authorization.

Sits between an agent and an MCP server. Every request must carry a Bearer
token that maps to a role. `tools/list` and everything else is forwarded
untouched; `tools/call` is inspected first, and any tool whose name starts
with `admin_` is only allowed through for the admin role. Anything else gets
a JSON-RPC error back and the downstream server never hears about it.

A mock downstream server lives in the same process at /downstream/mcp so the
file runs on its own. Point DOWNSTREAM_URL at a real server otherwise.

    pip install "fastapi>=0.116,<1" "uvicorn>=0.35,<1" "httpx>=0.28,<1"
    python task2_mcp_security_gateway.py

    curl -s localhost:8002/mcp -H 'Authorization: Bearer viewer-token' \
         -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key"}}'
"""

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import httpx
import uvicorn
from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError

log = logging.getLogger("mcp-gateway")

DOWNSTREAM_URL = os.getenv("DOWNSTREAM_URL", "http://127.0.0.1:8002/downstream/mcp")
DOWNSTREAM_TIMEOUT = 5.0

# Demo credentials. In production this is a JWT verified against the identity
# provider's keys, with the role read from a signed claim. The important
# property is the same either way: the role comes from something the gateway
# verifies, never from the request body an agent controls.
ROLE_BY_TOKEN = {
    "admin-token": "admin",
    "viewer-token": "viewer",
}

# JSON-RPC reserves -32000..-32099 for server-defined errors.
UNAUTHENTICATED = -32000
UNAUTHORIZED_TOOL = -32001
DOWNSTREAM_UNAVAILABLE = -32002

# Tool names in MCP are plain identifiers. Rejecting anything else up front
# means the prefix check below never has to think about " admin_x" or
# "admin_x\n" and whether the downstream would normalise them. Checked with
# fullmatch, not `$`: in Python `$` is happy to match just before a trailing
# newline, which is exactly the kind of gap a bypass hides in.
TOOL_NAME = re.compile(r"[A-Za-z0-9_.-]{1,128}")


class JsonRpcRequest(BaseModel):
    """Just enough of the envelope to route on. Everything else passes through."""

    model_config = ConfigDict(extra="allow")
    jsonrpc: Literal["2.0"]
    id: int | str | None = None
    method: str
    params: dict[str, Any] | None = None


def rpc_error(request_id: Any, code: int, message: str) -> JSONResponse:
    # HTTP 200 on purpose: the transport worked, the JSON-RPC layer is what
    # failed, and that's where clients look for the error.
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def bearer_token(authorization: str | None) -> str | None:
    """Pull the token out of `Authorization: Bearer <token>`.

    RFC 6750 says the scheme is case-insensitive, so `bearer x` is accepted.
    """
    if not authorization:
        return None
    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def is_admin_tool(name: str) -> bool:
    return name.lower().startswith("admin_")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One pooled client for the life of the process rather than a fresh TCP
    # connection per proxied request.
    async with httpx.AsyncClient(timeout=DOWNSTREAM_TIMEOUT) as client:
        app.state.http = client
        yield


app = FastAPI(title="MCP Security Gateway", lifespan=lifespan)


@app.post("/mcp")
async def gateway(request: Request, authorization: Annotated[str | None, Header()] = None) -> Response:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return rpc_error(None, -32700, "Parse error")

    if isinstance(payload, list):
        # Batching was removed from the MCP spec in 2025-06. Say so rather
        # than forwarding a shape we did not inspect.
        return rpc_error(None, -32600, "Batch requests are not supported")

    request_id = payload.get("id") if isinstance(payload, dict) else None
    try:
        rpc = JsonRpcRequest.model_validate(payload)
    except ValidationError:
        return rpc_error(request_id, -32600, "Invalid JSON-RPC request")

    role = ROLE_BY_TOKEN.get(bearer_token(authorization) or "")
    if role is None:
        return rpc_error(rpc.id, UNAUTHENTICATED, "Authentication required")

    if rpc.method == "tools/call":
        name = (rpc.params or {}).get("name")
        if not isinstance(name, str) or not TOOL_NAME.fullmatch(name):
            return rpc_error(rpc.id, -32602, "Invalid params: tool name is missing or malformed")
        if is_admin_tool(name) and role != "admin":
            log.warning("blocked %s for role=%s", name, role)
            return rpc_error(rpc.id, UNAUTHORIZED_TOOL, "Unauthorized Tool Call")

    return await forward(request.app.state.http, rpc, payload, role)


async def forward(client: httpx.AsyncClient, rpc: JsonRpcRequest, payload: dict[str, Any], role: str) -> Response:
    """Send the original payload downstream and hand the reply back verbatim.

    The caller's own token stops here. The downstream sees the gateway's
    identity plus who the caller was, which is what its audit log wants.
    """
    headers = {"Content-Type": "application/json", "X-Gateway-Role": role}
    try:
        reply = await client.post(DOWNSTREAM_URL, content=json.dumps(payload), headers=headers)
        reply.raise_for_status()
    except httpx.HTTPError as exc:
        log.error("downstream failed for %s: %s", rpc.method, type(exc).__name__)
        return rpc_error(rpc.id, DOWNSTREAM_UNAVAILABLE, "Downstream MCP server unavailable")

    if not reply.content:
        # Notifications have no id and get no reply body.
        return Response(status_code=202)
    return Response(content=reply.content, media_type="application/json")


# --- mock downstream ---------------------------------------------------------
#
# The smallest MCP-ish server that lets the proxy be exercised end to end.
# It is deliberately naive: it trusts whatever reaches it, which is exactly
# why the gateway has to do the authorization.

@app.post("/downstream/mcp")
async def mock_downstream(request: Request) -> Response:
    payload = await request.json()
    request_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}

    if request_id is None:
        # A notification. There is nothing to answer, whatever it was.
        return Response(status_code=202)

    if method == "tools/list":
        result: dict[str, Any] = {
            "tools": [
                {"name": "get_report", "description": "Read-only report", "inputSchema": {"type": "object"}},
                {"name": "admin_reset_key", "description": "Rotate the API key", "inputSchema": {"type": "object"}},
            ]
        }
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": f"downstream ran {params.get('name')}"}], "isError": False}
    else:
        return rpc_error(request_id, -32601, "Method not found")

    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8002)
