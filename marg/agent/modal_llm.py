"""Bounded, explicit DeepSeek tool calling through Modal's shared endpoint."""

import json
import os
import sqlite3
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from .llm import LLMResponse, ProviderUnavailable, ToolCall
from .prompts import SYSTEM_PROMPT

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"


class Allowance:
    """Reserve pessimistic token cost atomically; uncertain charges stay reserved."""

    def __init__(self, path: Path, limit: int = 500_000, expires: float = 1794355200):
        self.path, self.limit, self.expires = path, min(limit, 500_000), expires
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, reserved INTEGER, receipt TEXT)"
            )

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def reserve(self, input_bound: int, output_bound: int) -> str:
        # Published uncached rates: $0.30/MTok input, $1.20/MTok output.
        reserved = (input_bound * 3 + output_bound * 12 + 9) // 10
        if time.time() >= self.expires:
            raise ProviderUnavailable(
                "The live demo allowance has expired.", "ModalAllowanceExpired"
            )
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            total, count = db.execute(
                "SELECT COALESCE(SUM(reserved),0), COUNT(*) FROM calls"
            ).fetchone()
            if count >= 100 or total + reserved > self.limit:
                raise ProviderUnavailable(
                    "The live demo allowance is exhausted.", "ModalAllowanceExhausted"
                )
            call_id = uuid4().hex
            db.execute("INSERT INTO calls VALUES (?,?,?)", (call_id, reserved, "{}"))
        return call_id

    def record(self, call_id: str, receipt: dict) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE calls SET receipt=? WHERE id=?", (json.dumps(receipt), call_id)
            )

    def status(self) -> dict:
        with self.connect() as db:
            total, count = db.execute(
                "SELECT COALESCE(SUM(reserved),0), COUNT(*) FROM calls"
            ).fetchone()
        return {
            "requests": count,
            "reserved_micro_usd": total,
            "limit_micro_usd": self.limit,
            "expires_at": self.expires,
            "available": time.time() < self.expires
            and total < self.limit
            and count < 100,
        }


def messages_for_modal(messages: list[dict]) -> list[dict]:
    converted: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in messages:
        text, calls, results = [], [], []
        for block in message.get("content", []):
            if "text" in block:
                text.append(str(block["text"]))
            if "toolUse" in block:
                call = block["toolUse"]
                calls.append(
                    {
                        "id": call["toolUseId"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["input"]),
                        },
                    }
                )
            if "toolResult" in block:
                result = block["toolResult"]
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": result["toolUseId"],
                        "content": json.dumps(result.get("content", [])),
                    }
                )
        if text or calls:
            item = {"role": message["role"], "content": "\n".join(text) or None}
            if calls:
                item["tool_calls"] = calls
            converted.append(item)
        converted.extend(results)
    return converted


class ModalLLM:
    def __init__(self, allowance: Allowance | None = None):
        self.model_id = MODEL
        self.endpoint = os.environ.get("MARG_MODAL_ENDPOINT", "")
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme != "https"
            or not (parsed.hostname or "").endswith(".modal.direct")
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "MARG_MODAL_ENDPOINT must be a credential-free Modal HTTPS origin"
            )
        if parsed.path not in {"", "/"}:
            raise ValueError("MARG_MODAL_ENDPOINT must not contain an API path")
        self.allowance = allowance or Allowance(
            Path(os.environ["MARG_MODAL_ALLOWANCE_DB"])
        )
        self.live_inference, self.requests = False, []
        self.auth_header = os.environ.get("MARG_MODAL_AUTH_HEADER", "Authorization")
        if self.auth_header not in {"Authorization", "Modal-Authorization"}:
            raise ValueError("Invalid Modal authentication header")
        self.affinity = "margai-" + uuid4().hex

    def details(self) -> dict:
        return {
            "kind": "modal",
            "model_id": MODEL,
            "region": "us-west",
            "live_inference": self.live_inference,
            "requests": self.requests,
        }

    def converse(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        if os.environ.get("MARG_MODAL_ENABLED") != "1":
            raise ProviderUnavailable("Modal live review is disabled.", "ModalDisabled")
        credential = os.environ.get("MARG_MODAL_PROXY_TOKEN", "")
        if not credential:
            raise ProviderUnavailable(
                "Modal credentials are not configured.", "ModalCredentialsMissing"
            )
        payload = {
            "model": MODEL,
            "messages": messages_for_modal(messages),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": t["toolSpec"]["name"],
                        "description": t["toolSpec"]["description"],
                        "parameters": t["toolSpec"]["inputSchema"]["json"],
                    },
                }
                for t in tools
            ],
            "temperature": 0,
            "max_tokens": 1024,
            "stream": False,
            "reasoning_effort": "none",
        }
        body = json.dumps(payload, ensure_ascii=False).encode()
        if len(body) > 65_536:
            raise ProviderUnavailable(
                "The live review reached its context limit.", "ModalContextLimit"
            )
        # Byte-level tokenization cannot exceed UTF-8 bytes; include template framing headroom.
        input_bound = len(body) + 4096
        call_id = self.allowance.reserve(input_bound, 1024)
        request = Request(
            self.endpoint.rstrip("/") + "/v1/chat/completions",
            data=body,
            headers={
                self.auth_header: "Bearer " + credential,
                "Content-Type": "application/json",
                "Modal-Routing-Affinity-Key": self.affinity,
            },
        )
        started = time.monotonic()
        try:
            with urlopen(request, timeout=45) as response:
                data = response.read(1_048_577)
                request_id = response.headers.get("x-request-id", "")
            if len(data) > 1_048_576:
                raise ValueError("Response too large")
            result = json.loads(data)
            request_id = request_id or result.get("id", "")
            if result.get("model") != MODEL:
                raise ValueError("Unexpected provider model")
            choice = result["choices"][0]
            if choice.get("finish_reason") in {"length", "content_filter"}:
                raise ValueError("Incomplete provider response")
            usage = result.get("usage", {})
            if not isinstance(usage.get("prompt_tokens"), int) or not isinstance(
                usage.get("completion_tokens"), int
            ):
                raise TypeError("Missing provider usage")
            if (
                usage["prompt_tokens"] > input_bound
                or usage["completion_tokens"] > 1024
            ):
                raise ValueError("Provider exceeded reserved bounds")
            message = choice["message"]
            calls = []
            allowed = {t["toolSpec"]["name"] for t in tools}
            for call in message.get("tool_calls") or []:
                function = call["function"]
                arguments = json.loads(function["arguments"])
                if function["name"] not in allowed or not isinstance(arguments, dict):
                    raise ValueError("Invalid tool call")
                calls.append(ToolCall(function["name"], arguments, call["id"]))
            text = message.get("content") or ""
            if not isinstance(text, str) or not (text or calls):
                raise ValueError("Empty provider response")
            self.live_inference = True
            receipt = {
                "request_id": request_id,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "seconds": round(time.monotonic() - started, 3),
            }
            self.requests.append(receipt)
            self.allowance.record(call_id, receipt)
            return LLMResponse(text=text, tool_calls=calls)
        except HTTPError as error:
            raise ProviderUnavailable(
                f"Modal rejected the request (HTTP {error.code}).",
                "ModalHTTPError",
                error.code in {429, 503},
            ) from None
        except (
            URLError,
            TimeoutError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
        ):
            raise ProviderUnavailable(
                "Modal could not return a complete, verified tool response.",
                "ModalResponseUnavailable",
            ) from None
