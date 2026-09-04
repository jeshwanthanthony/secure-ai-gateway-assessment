#!/usr/bin/env python3
"""LLM gateway with a streaming PII guardrail.

POST /v1/generate forwards the prompt to a provider, reads the provider's SSE
stream, and re-streams it to the client with emails, US social security
numbers and card numbers replaced by [REDACTED].

The hard part is that the provider chunks text wherever it likes, so
"jane.doe@" and "example.com" can arrive separately. Buffering the whole
answer would fix that and ruin time-to-first-token. Instead StreamRedactor
holds back only the shortest tail that could still turn into a match, which
on normal prose is the last word.

A mock provider at /mock-provider streams the prompt back in small pieces so
the whole thing can run and be tested in one process.

    pip install "fastapi>=0.116,<1" "uvicorn>=0.35,<1" "httpx>=0.28,<1"
    python task3_streaming_pii_guardrail.py

    curl -sN localhost:8003/v1/generate -H 'Content-Type: application/json' \
         -d '{"prompt":"Mail jane.doe@example.com, SSN 123-45-6789, card 4111 1111 1111 1111."}'
"""

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger("llm-gateway")

PROVIDER_URL = os.getenv("PROVIDER_URL", "http://127.0.0.1:8003/mock-provider")
REDACTED = "[REDACTED]"


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=10_000)


# --- PII rules ---------------------------------------------------------------
#
# Each rule has two regexes. `pattern` recognises a complete match. `partial`
# is anchored at the end of the buffer and recognises text that is not a match
# yet but could become one if more characters arrive; it is what decides how
# much to hold back. `partial` may over-match (holding a little too much only
# costs latency) but must never under-match (releasing half a card number).
#
# [0-9] instead of \d throughout because \d also matches Unicode digits.

def luhn_ok(digits: str) -> bool:
    """Luhn checksum. Real card numbers pass it, most random digit runs don't."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


@dataclass(frozen=True)
class PiiRule:
    name: str
    pattern: re.Pattern[str]
    partial: re.Pattern[str]
    accept: Callable[[str], bool] = lambda _match: True


RULES: tuple[PiiRule, ...] = (
    PiiRule(
        "email",
        re.compile(r"\b[A-Z0-9._%+-]{1,64}@[A-Z0-9.-]{1,253}\.[A-Z]{2,24}\b", re.IGNORECASE),
        re.compile(r"[A-Z0-9._%+-]{1,64}(?:@[A-Z0-9.-]{0,253})?$", re.IGNORECASE),
    ),
    PiiRule(
        "ssn",
        re.compile(r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b"),
        re.compile(r"[0-9]{1,3}(?:-(?:[0-9]{1,2}(?:-[0-9]{0,4})?)?)?$"),
    ),
    PiiRule(
        "card",
        # 13 to 19 digits, optionally separated by single spaces or hyphens,
        # not glued to other digits on either side.
        re.compile(r"(?<![0-9])(?:[0-9][ -]?){12,18}[0-9](?![0-9])"),
        re.compile(r"[0-9](?:[ -]?[0-9]){0,18}[ -]?$"),
        accept=lambda m: luhn_ok(re.sub(r"[ -]", "", m)),
    ),
)

# The widest lookbehind any rule uses. \b and (?<![0-9]) both look one
# character back, so one released character is kept for context.
LOOKBEHIND = 1


def find_spans(text: str, rules: Iterable[PiiRule] = RULES) -> list[tuple[int, int]]:
    """Every PII match in `text` as sorted, non-overlapping (start, end) spans."""
    spans = [
        (m.start(), m.end())
        for rule in rules
        for m in rule.pattern.finditer(text)
        if rule.accept(m.group())
    ]
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def apply_spans(text: str, spans: list[tuple[int, int]], start: int, end: int) -> str:
    """text[start:end] with the spans inside that window replaced."""
    out: list[str] = []
    pos = start
    for s, e in spans:
        if s < start or e > end:
            continue
        out.append(text[pos:s])
        out.append(REDACTED)
        pos = e
    out.append(text[pos:end])
    return "".join(out)


def redact(text: str, rules: Iterable[PiiRule] = RULES) -> str:
    """One-shot redaction. StreamRedactor must produce exactly this, chunked."""
    return apply_spans(text, find_spans(text, rules), 0, len(text))


class StreamRedactor:
    """Redacts PII from text that arrives in pieces of arbitrary size.

    State is one small string: a character of already-sent context (for the
    regex lookbehinds) plus whatever tail has not been released yet. Memory
    is bounded by the longest possible partial match, not by the response.
    """

    def __init__(self, rules: Iterable[PiiRule] = RULES) -> None:
        self.rules = tuple(rules)
        self.buffer = ""
        self.released = 0  # buffer[:released] has already gone to the client

    def feed(self, chunk: str) -> str:
        """Absorb a chunk and return whatever text is now safe to send."""
        self.buffer += chunk

        cut = len(self.buffer) - self.hold_length()
        spans = find_spans(self.buffer, self.rules)
        # A match that reaches into the held tail is not final yet (the next
        # chunk could extend it), so hold the whole match too. Spans are
        # sorted, so the first one that crosses the cut is the only one that
        # matters.
        for start, end in spans:
            if end > cut:
                cut = min(cut, start)
                break
        cut = max(cut, self.released)

        out = apply_spans(self.buffer, spans, self.released, cut)
        self.released = cut
        self.forget_released()
        return out

    def finish(self) -> str:
        """Provider is done; nothing can grow any more, so flush the tail."""
        out = apply_spans(self.buffer, find_spans(self.buffer, self.rules), self.released, len(self.buffer))
        self.buffer, self.released = "", 0
        return out

    def hold_length(self) -> int:
        """How many trailing characters could still be the start of a match."""
        longest = 0
        for rule in self.rules:
            m = rule.partial.search(self.buffer, self.released)
            if m:
                longest = max(longest, m.end() - m.start())
        return longest

    def forget_released(self) -> None:
        drop = self.released - LOOKBEHIND
        if drop > 0:
            self.buffer = self.buffer[drop:]
            self.released -= drop


# --- SSE plumbing ------------------------------------------------------------
#
# OpenAI-style events: `data: {"choices":[{"delta":{"content":"..."}}]}` and
# a final `data: [DONE]`. The gateway speaks the same shape back so clients
# don't know it's there.

def sse_delta(text: str) -> str:
    return f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n"


def delta_text(line: str) -> str | None:
    """The text inside one provider SSE line, or None if there isn't any."""
    if not line.startswith("data:"):
        return None
    body = line[5:].strip()
    if body == "[DONE]":
        return None
    try:
        content = json.loads(body)["choices"][0]["delta"].get("content")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


app = FastAPI(title="LLM Gateway with Streaming PII Guardrail")


@app.post("/v1/generate")
async def generate(request: GenerationRequest):
    client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=60.0))
    upstream = client.build_request("POST", PROVIDER_URL, json=request.model_dump())

    # Open the upstream connection before committing to a 200 + SSE headers,
    # so a provider outage is still a clean JSON error rather than a stream
    # that dies half way.
    try:
        response = await client.send(upstream, stream=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        await client.aclose()
        log.error("provider request failed: %s", type(exc).__name__)
        return JSONResponse(
            status_code=502,
            content={"error": {"code": "PROVIDER_UNAVAILABLE", "message": "The generation provider is unavailable."}},
        )

    async def guarded_stream() -> AsyncIterator[str]:
        redactor = StreamRedactor()
        try:
            async for line in response.aiter_lines():
                text = delta_text(line)
                if text is None:
                    continue
                safe = redactor.feed(text)
                if safe:
                    yield sse_delta(safe)
            tail = redactor.finish()
            if tail:
                yield sse_delta(tail)
            yield "data: [DONE]\n\n"
        finally:
            # Runs on normal completion and when the client hangs up.
            await response.aclose()
            await client.aclose()

    return StreamingResponse(
        guarded_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- mock provider -----------------------------------------------------------

MOCK_CHUNK_SIZE = 5


@app.post("/mock-provider")
async def mock_provider(request: GenerationRequest) -> StreamingResponse:
    """Echoes the prompt back in five character pieces, slowly enough to be a stream.

    Five characters is small enough that any PII in the prompt will be split
    across chunks, which is the case the gateway has to get right.
    """
    text = request.prompt

    async def stream() -> AsyncIterator[str]:
        for i in range(0, len(text), MOCK_CHUNK_SIZE):
            yield sse_delta(text[i : i + MOCK_CHUNK_SIZE])
            await asyncio.sleep(0.02)
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8003)
