#!/usr/bin/env python3
"""LLM gateway router: token rate limiting plus primary/secondary failover.

POST /v1/completions does three things in order:

  1. Charges the tenant's token budget. Budget is a sliding 60 second window
     of 50,000 tokens per API key, kept in an on-disk SQLite database so it
     survives restarts and is shared by every worker on the box.
  2. Calls the primary model with a hard 3 second deadline. A 429 or a
     timeout moves the request to the secondary model; any other failure
     does not.
  3. Returns either the model's answer or a small, fixed-shape error. Upstream
     bodies, exception text and stack traces stay in the server log.

Mock primary and secondary providers live in this file. The primary's
behaviour is picked by a prefix on the prompt (`!429`, `!timeout`, `!500`)
so every branch can be hit from curl.

    pip install "fastapi>=0.116,<1" "uvicorn>=0.35,<1" "httpx>=0.28,<1"
    python task4_rate_limit_fallback_router.py

    curl -s localhost:8004/v1/completions -H 'X-API-Key: tenant-a' \
         -H 'Content-Type: application/json' -d '{"prompt":"!timeout hello"}'
"""

import asyncio
import logging
import math
import os
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger("llm-router")

DB_PATH = os.getenv("RATE_LIMIT_DB", "llm_gateway_rate_limits.db")
TOKEN_LIMIT = 50_000
WINDOW_SECONDS = 60.0
PRIMARY_DEADLINE_SECONDS = 3.0

PRIMARY_MODEL_URL = os.getenv("PRIMARY_MODEL_URL", "http://127.0.0.1:8004/mock/primary")
SECONDARY_MODEL_URL = os.getenv("SECONDARY_MODEL_URL", "http://127.0.0.1:8004/mock/secondary")


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=100_000)
    max_tokens: int = Field(default=500, ge=1, le=TOKEN_LIMIT)


def estimate_tokens(request: CompletionRequest) -> int:
    """Prompt tokens plus the output budget the caller asked for.

    Four characters per token is the usual rough guide for English. The
    reservation is corrected to the provider's real count afterwards, so the
    estimate only needs to be in the right ballpark.
    """
    return max(1, math.ceil(len(request.prompt) / 4)) + request.max_tokens


# --- sliding window rate limiter --------------------------------------------

@dataclass(frozen=True)
class Reservation:
    row_id: int
    tokens: int
    remaining: int


class SlidingWindowLimiter:
    """Per-tenant token budget over a trailing window, backed by SQLite on disk.

    Each request inserts one row (tenant, tokens, timestamp). Usage is the sum
    of rows newer than `now - window`; anything older is deleted on the way
    in. The check-then-insert runs inside BEGIN IMMEDIATE so two concurrent
    requests can't both squeeze into the last bit of budget.

    Wall-clock time rather than a monotonic clock because rows outlive the
    process and have to be comparable across restarts. `clock` is injectable
    so tests can move time without sleeping.
    """

    def __init__(
        self,
        path: str,
        limit: int = TOKEN_LIMIT,
        window: float = WINDOW_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path
        self.limit = limit
        self.window = window
        self.clock = clock

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        # isolation_level=None hands transaction control to us. WAL lets
        # readers run while a writer holds the lock; busy_timeout makes a
        # second writer wait for its turn instead of failing at once.
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 5000")
            yield conn
        finally:
            conn.close()

    def setup(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant     TEXT    NOT NULL,
                    tokens     INTEGER NOT NULL,
                    created_at REAL    NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS token_usage_tenant_time ON token_usage (tenant, created_at)")

    async def reserve(self, tenant: str, tokens: int) -> Reservation | None:
        """Charge `tokens` to the tenant, or return None if that would exceed the limit."""
        return await asyncio.to_thread(self._reserve, tenant, tokens)

    async def settle(self, reservation: Reservation, actual_tokens: int) -> None:
        """Replace the estimate with what the provider says was actually used."""
        await asyncio.to_thread(self._settle, reservation.row_id, actual_tokens)

    async def used(self, tenant: str) -> int:
        return await asyncio.to_thread(self._used, tenant)

    def _reserve(self, tenant: str, tokens: int) -> Reservation | None:
        now = self.clock()
        cutoff = now - self.window
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM token_usage WHERE created_at <= ?", (cutoff,))
                used = self._sum(conn, tenant, cutoff)
                if used + tokens > self.limit:
                    conn.execute("ROLLBACK")
                    return None
                cur = conn.execute(
                    "INSERT INTO token_usage (tenant, tokens, created_at) VALUES (?, ?, ?)",
                    (tenant, tokens, now),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return Reservation(row_id=cur.lastrowid or 0, tokens=tokens, remaining=self.limit - used - tokens)

    def _settle(self, row_id: int, actual_tokens: int) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE token_usage SET tokens = ? WHERE id = ?", (max(0, actual_tokens), row_id))

    def _used(self, tenant: str) -> int:
        with self.connect() as conn:
            return self._sum(conn, tenant, self.clock() - self.window)

    @staticmethod
    def _sum(conn: sqlite3.Connection, tenant: str, cutoff: float) -> int:
        row = conn.execute(
            "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ? AND created_at > ?",
            (tenant, cutoff),
        ).fetchone()
        return int(row[0])


# --- model routing -----------------------------------------------------------

@dataclass(frozen=True)
class ModelReply:
    status: str  # "ok", "rate_limited", "timeout" or "error"
    body: dict[str, Any] | None = None


async def call_model(client: httpx.AsyncClient, url: str, payload: dict[str, Any], deadline: float) -> ModelReply:
    """One provider call, reduced to a status the router can branch on.

    `asyncio.wait_for` is the deadline, not httpx's timeout: httpx timeouts
    are per phase (connect, read, ...), so a provider that answers headers
    quickly and then dribbles the body could run well past 3 seconds. The
    wall clock is what the caller experiences, so that's what we bound.
    """
    try:
        response = await asyncio.wait_for(client.post(url, json=payload), timeout=deadline)
    except TimeoutError:
        return ModelReply("timeout")
    except httpx.HTTPError as exc:
        log.warning("%s: transport error %s", url, type(exc).__name__)
        return ModelReply("error")

    if response.status_code == 429:
        return ModelReply("rate_limited")
    if response.status_code >= 400:
        # Log the body for us; it never goes to the caller.
        log.warning("%s: HTTP %s %s", url, response.status_code, response.text[:200])
        return ModelReply("error")
    try:
        return ModelReply("ok", response.json())
    except ValueError:
        log.warning("%s: non-JSON body", url)
        return ModelReply("error")


async def route(client: httpx.AsyncClient, payload: dict[str, Any]) -> tuple[str, ModelReply, str | None]:
    """Primary first; secondary only when the primary was rate limited or timed out.

    Returns (which model answered, its reply, why we fell over if we did).
    """
    primary = await call_model(client, PRIMARY_MODEL_URL, payload, PRIMARY_DEADLINE_SECONDS)
    if primary.status in ("rate_limited", "timeout"):
        log.info("primary %s, trying secondary", primary.status)
        secondary = await call_model(client, SECONDARY_MODEL_URL, payload, PRIMARY_DEADLINE_SECONDS)
        return "secondary", secondary, primary.status
    return "primary", primary, None


# --- HTTP layer --------------------------------------------------------------

def gateway_error(status: int, code: str, message: str, request_id: str) -> JSONResponse:
    """The only error shape clients ever see. Details live in the log under request_id."""
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    limiter = SlidingWindowLimiter(DB_PATH)
    limiter.setup()
    app.state.limiter = limiter
    # Per-request deadlines come from wait_for; this is only a backstop so a
    # hung connect can't sit around forever.
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        app.state.http = client
        yield


app = FastAPI(title="LLM Gateway Router", lifespan=lifespan)


@app.post("/v1/completions")
async def completions(request: Request, body: CompletionRequest, x_api_key: str | None = Header(default=None)):
    request_id = uuid.uuid4().hex[:12]
    if not x_api_key:
        return gateway_error(401, "AUTHENTICATION_REQUIRED", "X-API-Key header is required.", request_id)

    limiter: SlidingWindowLimiter = request.app.state.limiter
    reservation = await limiter.reserve(x_api_key, estimate_tokens(body))
    if reservation is None:
        return gateway_error(
            429,
            "TOKEN_RATE_LIMIT_EXCEEDED",
            f"Token budget of {TOKEN_LIMIT} per {int(WINDOW_SECONDS)}s is exhausted for this API key.",
            request_id,
        )

    model, reply, fallback_reason = await route(request.app.state.http, body.model_dump())

    if reply.status != "ok" or reply.body is None:
        if model == "secondary":
            return gateway_error(503, "MODELS_UNAVAILABLE", "Primary and secondary models are unavailable.", request_id)
        log.error("request %s: primary failed with %s", request_id, reply.status)
        return gateway_error(502, "PRIMARY_MODEL_ERROR", "The primary model returned an error.", request_id)

    # Trade the estimate for the real count so the window reflects actual use.
    actual = reply.body.get("usage_tokens")
    if isinstance(actual, int):
        await limiter.settle(reservation, actual)

    return {
        "request_id": request_id,
        "model_used": model,
        "fallback_reason": fallback_reason,
        "result": reply.body,
        "rate_limit": {"limit": TOKEN_LIMIT, "reserved": reservation.tokens, "remaining": reservation.remaining},
    }


# --- mock providers ----------------------------------------------------------

@app.post("/mock/primary")
async def mock_primary(body: CompletionRequest):
    if body.prompt.startswith("!429"):
        return JSONResponse(status_code=429, content={"detail": "mock: rate limited"})
    if body.prompt.startswith("!timeout"):
        await asyncio.sleep(PRIMARY_DEADLINE_SECONDS + 1.0)
    if body.prompt.startswith("!500"):
        return JSONResponse(status_code=500, content={"detail": "mock: Traceback (most recent call last) ..."})
    return {"text": f"primary: {body.prompt}", "usage_tokens": 25}


@app.post("/mock/secondary")
async def mock_secondary(body: CompletionRequest):
    return {"text": f"secondary: {body.prompt}", "usage_tokens": 28}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8004)
