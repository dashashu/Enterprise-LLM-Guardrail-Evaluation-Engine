"""Provider adapters and bounded retry execution."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Protocol

import httpx

from app.config import Settings


# Log only known, fixed identifiers. Provider messages and other JSON values can
# contain submitted text or credentials and must never become diagnostics.
_ERROR_TYPES = frozenset({
    "api_error", "authentication_error", "billing_error", "conflict_error",
    "insufficient_quota", "invalid_request_error", "not_found_error",
    "overloaded_error", "permission_error", "rate_limit_error",
    "request_too_large", "server_error", "service_unavailable_error",
    "timeout_error",
})
_ERROR_CODES = frozenset({
    "access_terminated", "account_deactivated", "billing_hard_limit_reached",
    "context_length_exceeded", "credit_balance_exhausted", "insufficient_quota",
    "invalid_api_key", "invalid_request_error", "invalid_value",
    "model_not_found", "organization_spend_limit_exceeded",
    "organization_usage_limit_exceeded", "project_spend_limit_exceeded",
    "rate_limit_exceeded", "server_is_overloaded", "slow_down",
    "unsupported_parameter", "unsupported_value",
})
_ERROR_PARAMS = frozenset({
    "frequency_penalty", "max_tokens", "messages", "model",
    "presence_penalty", "response_format", "service_tier", "stream",
    "system", "temperature", "tool_choice", "tools", "top_p",
})
_NON_RETRYABLE_429_CODES = frozenset({
    "billing_hard_limit_reached", "credit_balance_exhausted", "insufficient_quota",
    "organization_spend_limit_exceeded", "organization_usage_limit_exceeded",
    "project_spend_limit_exceeded",
})
_ERROR_REASONS = frozenset({
    "deadline_exceeded", "empty_response", "malformed_response",
    "timeout", "transport_error",
})


def _allowed(value: object, choices: frozenset[str]) -> str | None:
    return value if isinstance(value, str) and value in choices else None


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = False, *,
                 status_code: int | None = None, error_type: str | None = None,
                 error_code: str | None = None, error_param: str | None = None,
                 reason: str | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code if type(status_code) is int and 400 <= status_code <= 599 else None
        self.error_type = _allowed(error_type, _ERROR_TYPES)
        self.error_code = _allowed(error_code, _ERROR_CODES)
        self.error_param = _allowed(error_param, _ERROR_PARAMS)
        self.reason = _allowed(reason, _ERROR_REASONS)

    def safe_log_details(self) -> str:
        parts = []
        if type(self.status_code) is int and 400 <= self.status_code <= 599:
            parts.append(f"status={self.status_code}")
        for name, value, choices in (
            ("type", self.error_type, _ERROR_TYPES),
            ("code", self.error_code, _ERROR_CODES),
            ("param", self.error_param, _ERROR_PARAMS),
            ("reason", self.reason, _ERROR_REASONS),
        ):
            allowed = _allowed(value, choices)
            if allowed is not None:
                parts.append(f"{name}={allowed}")
        return " " + " ".join(parts) if parts else ""


def _http_error(response: httpx.Response) -> ProviderError:
    details = {}
    try:
        body = response.json()
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            details = {
                "error_type": error.get("type"),
                "error_code": error.get("code"),
                "error_param": error.get("param"),
            }
    except ValueError:
        pass
    status = response.status_code
    error = ProviderError("Provider HTTP error", status_code=status, **details)
    error.retryable = (status in {408, 409} or status >= 500
                       or (status == 429 and error.error_code not in _NON_RETRYABLE_429_CODES))
    return error


class Provider(Protocol):
    async def complete(self, messages: list[dict[str, str]], timeout: float) -> str: ...


class HTTPProvider:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client

    async def complete(self, messages: list[dict[str, str]], timeout: float) -> str:
        if self.settings.provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.settings.api_key}"}
            payload = {"model": self.settings.model, "messages": messages,
                       "response_format": {"type": "json_object"}}
        else:
            url = "https://api.anthropic.com/v1/messages"
            headers = {"x-api-key": self.settings.api_key, "anthropic-version": "2023-06-01"}
            system = "\n".join(m["content"] for m in messages if m["role"] == "system")
            payload = {"model": self.settings.model, "max_tokens": 1024,
                       "system": system, "messages": [m for m in messages if m["role"] != "system"]}
        try:
            response = await self.client.post(url, json=payload, headers=headers, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise ProviderError("Provider request timed out", retryable=True, reason="timeout") from exc
        except httpx.TransportError as exc:
            raise ProviderError("Provider transport failure", retryable=True, reason="transport_error") from exc
        if response.status_code >= 400:
            raise _http_error(response)
        try:
            data = response.json()
            if self.settings.provider == "openai":
                content = data["choices"][0]["message"]["content"]
            else:
                content = "".join(block["text"] for block in data["content"] if block["type"] == "text")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("Provider response was malformed", reason="malformed_response") from exc
        if not isinstance(content, str):
            raise ProviderError("Provider response was malformed", reason="malformed_response")
        if not content:
            raise ProviderError("Provider returned no text", reason="empty_response")
        return content


async def complete_with_retry(provider: Provider, messages: list[dict[str, str]], settings: Settings) -> str:
    deadline = time.monotonic() + settings.request_deadline_seconds
    for attempt in range(settings.llm_max_retries + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            return await asyncio.wait_for(
                provider.complete(messages, min(settings.llm_timeout_seconds, remaining)),
                timeout=remaining,
            )
        except asyncio.TimeoutError as exc:
            raise ProviderError("Request deadline exceeded", retryable=True, reason="deadline_exceeded") from exc
        except ProviderError as exc:
            if not exc.retryable or attempt >= settings.llm_max_retries:
                raise
        delay = min(0.5 * 2**attempt + random.uniform(0, 0.25), max(0, deadline - time.monotonic()))
        if delay:
            await asyncio.sleep(delay)
    raise ProviderError("Request deadline exceeded", retryable=True, reason="deadline_exceeded")
