#!/usr/bin/env python3
"""Customer refund MCP server, stdio transport.

Two tools are exposed over the Model Context Protocol:

    get_customer_record   customer_id must look like CUST-00042
    trigger_refund        customer_id, a positive amount, and a reason of
                          at least ten characters

Run it directly and speak JSON-RPC on stdin/stdout, or point any MCP client
at `python task1_customer_refund_mcp.py`. Only JSON-RPC goes to stdout;
logging goes to stderr.

    pip install "mcp>=2,<3" "pydantic>=2,<3"
"""

import asyncio
import json
import logging
import sys
from typing import Annotated, Any, TypeVar
from uuid import uuid4

import mcp.types as types
from mcp import MCPError
from mcp.server import Server
from mcp.server.stdio import stdio_server
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

log = logging.getLogger("customer-refund-mcp")


# --- input schemas -----------------------------------------------------------
#
# Everything is strict: "10" is not a float, 10 is not a string, and unknown
# fields are rejected rather than ignored. An agent that sends the wrong shape
# gets a -32602 with a list of what was wrong, which is far more useful than a
# silently coerced value.

# [0-9] rather than \d: \d also matches Unicode digits like "٤", which is not
# what anyone means by CUST-XXXXX.
CustomerId = Annotated[str, StringConstraints(strict=True, pattern=r"^CUST-[0-9]{5}$")]

# Whitespace is stripped before the length check, so ten spaces and a letter
# does not count as a ten character reason.
Reason = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=10, max_length=500)
]

# gt=0 rejects zero and negatives, allow_inf_nan rejects "Infinity" and NaN,
# which JSON can't even carry but Python callers can still hand us.
Amount = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]


class GetCustomerRecordInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: CustomerId


class TriggerRefundInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: CustomerId
    amount: Amount
    reason: Reason


# --- demo data ---------------------------------------------------------------
#
# Stands in for the billing system. Balances mutate in memory so a refund
# followed by a lookup shows the change.

CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-00001": {
        "customer_id": "CUST-00001",
        "name": "Avery Patel",
        "email": "avery.patel@example.com",
        "refundable_balance": 86.42,
        "currency": "USD",
    },
    "CUST-00002": {
        "customer_id": "CUST-00002",
        "name": "Jordan Lee",
        "email": "jordan.lee@example.com",
        "refundable_balance": 24.75,
        "currency": "USD",
    },
}


# --- helpers -----------------------------------------------------------------

ModelT = TypeVar("ModelT", bound=BaseModel)


def parse_arguments(model: type[ModelT], raw: dict[str, Any] | None, tool: str) -> ModelT:
    """Validate tool arguments, turning pydantic failures into JSON-RPC -32602.

    The `data` field carries one entry per problem so the client can see all
    of them at once instead of fixing them one round trip at a time.
    """
    try:
        return model.model_validate(raw or {})
    except ValidationError as exc:
        issues = [
            {
                "field": ".".join(str(part) for part in issue["loc"]) or "<root>",
                "problem": issue["msg"],
            }
            for issue in exc.errors()
        ]
        log.info("rejected %s call: %s", tool, issues)
        raise MCPError(types.INVALID_PARAMS, f"Invalid arguments for {tool}", {"issues": issues}) from exc


def text_result(payload: Any, *, is_error: bool = False) -> types.CallToolResult:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=is_error)


# --- tool implementations ----------------------------------------------------
#
# Two different kinds of failure, deliberately kept apart:
#   * bad input      -> JSON-RPC error (the request itself was malformed)
#   * business rule  -> result with isError=true (the request was fine, the
#                       answer is "no", and the model should read why)

def get_customer_record(args: GetCustomerRecordInput) -> types.CallToolResult:
    customer = CUSTOMERS.get(args.customer_id)
    if customer is None:
        return text_result(f"No customer with id {args.customer_id}", is_error=True)
    return text_result(customer)


def trigger_refund(args: TriggerRefundInput) -> types.CallToolResult:
    customer = CUSTOMERS.get(args.customer_id)
    if customer is None:
        return text_result(f"No customer with id {args.customer_id}", is_error=True)
    if args.amount > customer["refundable_balance"]:
        return text_result(
            f"Refund of {args.amount:.2f} exceeds refundable balance of "
            f"{customer['refundable_balance']:.2f}",
            is_error=True,
        )

    customer["refundable_balance"] = round(customer["refundable_balance"] - args.amount, 2)
    refund = {
        "refund_id": f"REF-{uuid4().hex[:12]}",
        "customer_id": args.customer_id,
        "amount": round(args.amount, 2),
        "currency": customer["currency"],
        "reason": args.reason,
        "status": "processed",
    }
    log.info("refund %s: %s for %.2f", refund["refund_id"], args.customer_id, args.amount)
    return text_result(refund)


# --- MCP wiring --------------------------------------------------------------

TOOLS = [
    types.Tool(
        name="get_customer_record",
        description="Look up a customer by id. Ids look like CUST-00042.",
        input_schema=GetCustomerRecordInput.model_json_schema(),
    ),
    types.Tool(
        name="trigger_refund",
        description="Refund a customer. Needs a positive amount and a reason of at least ten characters.",
        input_schema=TriggerRefundInput.model_json_schema(),
    ),
]


async def on_list_tools(_ctx: Any, _params: Any) -> types.ListToolsResult:
    return types.ListToolsResult(tools=TOOLS)


async def on_call_tool(_ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
    if params.name == "get_customer_record":
        return get_customer_record(parse_arguments(GetCustomerRecordInput, params.arguments, params.name))
    if params.name == "trigger_refund":
        return trigger_refund(parse_arguments(TriggerRefundInput, params.arguments, params.name))

    # The MCP spec's own example for an unknown tool uses -32602, not -32601:
    # the method (tools/call) exists, it is the params that are wrong.
    raise MCPError(types.INVALID_PARAMS, f"Unknown tool: {params.name}")


server = Server(
    "customer-refund-mcp",
    version="1.0.0",
    on_list_tools=on_list_tools,
    on_call_tool=on_call_tool,
)


async def main() -> None:
    # stdout is the wire. Anything we want a human to read goes to stderr.
    # (The SDK also repoints fd 1 at stderr while serving, so even a stray
    # print() can't corrupt the stream, but we don't rely on that.)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    log.info("starting on stdio")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
