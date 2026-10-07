"""Safe, structured diagnostics for provider failures."""

import asyncio
from dataclasses import replace

import httpx
import pytest

from app.config import Settings
from app.provider import HTTPProvider, ProviderError, complete_with_retry
from app.schemas import GenerateRequest
from app.service import GenerationService


def settings(provider="openai"):
    return Settings(
        provider=provider, model="test-model", api_key="test-key",
        app_api_keys=(), redis_url="redis://localhost", rate_limit_per_minute=2,
        llm_timeout_seconds=1, request_deadline_seconds=2,
        llm_max_retries=0, cache_ttl_seconds=60,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status,error,retryable,details", [
    (401, {"type": "authentication_error", "code": "invalid_api_key"}, False,
     "status=401 type=authentication_error code=invalid_api_key"),
    (400, {"type": "invalid_request_error", "param": "response_format"}, False,
     "status=400 type=invalid_request_error param=response_format"),
    (429, {"type": "rate_limit_error", "code": "slow_down"}, True,
     "status=429 type=rate_limit_error code=slow_down"),
    (429, {"type": "insufficient_quota", "code": "credit_balance_exhausted"}, False,
     "status=429 type=insufficient_quota code=credit_balance_exhausted"),
])
async def test_openai_http_error_has_safe_structured_diagnostics(status, error, retryable, details):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(status, json={"error": {**error, "message": "SECRET IN MESSAGE"}})
    )) as client:
        with pytest.raises(ProviderError) as caught:
            await HTTPProvider(settings(), client).complete([], 1)
    assert caught.value.retryable is retryable
    assert caught.value.safe_log_details() == " " + details
    assert "SECRET" not in caught.value.safe_log_details()


@pytest.mark.asyncio
async def test_anthropic_error_shape_and_non_json_body():
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(401, json={"type": "error", "error": {
            "type": "authentication_error", "message": "SECRET IN MESSAGE"}})
    )) as client:
        with pytest.raises(ProviderError) as caught:
            await HTTPProvider(settings("anthropic"), client).complete([], 1)
    assert caught.value.safe_log_details() == " status=401 type=authentication_error"

    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(502, text="SECRET NON JSON BODY")
    )) as client:
        with pytest.raises(ProviderError) as caught:
            await HTTPProvider(settings(), client).complete([], 1)
    assert caught.value.safe_log_details() == " status=502"
    assert caught.value.retryable


class EmptyStore:
    async def allow(self, client_id):
        return True

    def cache_key(self, client_id, prompt, context, prompt_version):
        return "test-cache-key"

    async def get_cached(self, key):
        return None


@pytest.mark.asyncio
async def test_generation_log_never_includes_provider_message_or_unlisted_fields(caplog):
    secret = "sk-test-SECRET-123"
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(400, json={"error": {
            "type": f"bad_{secret}", "code": secret,
            "param": f"messages.0.content.{secret}",
            "message": f"The supplied key {secret} was rejected"}})
    )) as client:
        service = GenerationService(settings(), HTTPProvider(settings(), client), EmptyStore())
        with caplog.at_level("WARNING", logger="app.service"):
            result = await service.generate(GenerateRequest(prompt="Test question"), "test-client")
    assert result.source == "fallback"
    assert "Generation degraded: ProviderError status=400" in caplog.text
    assert secret not in caplog.text
    assert "message=" not in caplog.text
    assert "param=" not in caplog.text


def test_provider_error_rechecks_fields_before_logging():
    error = ProviderError("SECRET MESSAGE", status_code=401, error_code="invalid_api_key")
    error.error_code = "sk-test-SECRET-123"
    assert error.safe_log_details() == " status=401"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,reason", [
    (httpx.ReadTimeout, "timeout"),
    (httpx.ConnectError, "transport_error"),
])
async def test_transport_failures_log_only_fixed_reasons(failure, reason, caplog):
    secret = "sk-test-SECRET-123"

    def fail(request):
        raise failure(f"Sensitive provider details: {secret}", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(ProviderError) as caught:
            await HTTPProvider(settings(), client).complete([], 1)
        service = GenerationService(settings(), HTTPProvider(settings(), client), EmptyStore())
        with caplog.at_level("WARNING", logger="app.service"):
            result = await service.generate(GenerateRequest(prompt="Test question"), "test-client")

    assert caught.value.retryable
    assert caught.value.safe_log_details() == f" reason={reason}"
    assert result.source == "fallback"
    assert f"Generation degraded: ProviderError reason={reason}" in caplog.text
    assert secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,response_body,reason", [
    ("openai", {"choices": []}, "malformed_response"),
    ("openai", {"choices": [{"message": {"content": None}}]}, "malformed_response"),
    ("openai", {"choices": [{"message": {"content": ""}}]}, "empty_response"),
    ("anthropic", {"content": []}, "empty_response"),
])
async def test_invalid_provider_text_has_specific_safe_reason(provider, response_body, reason):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=response_body)
    )) as client:
        with pytest.raises(ProviderError) as caught:
            await HTTPProvider(settings(provider), client).complete([], 1)

    assert caught.value.safe_log_details() == f" reason={reason}"
    assert not caught.value.retryable


@pytest.mark.asyncio
async def test_overall_deadline_expires_even_when_provider_hangs():
    class HangingProvider:
        async def complete(self, messages, timeout):
            await asyncio.Event().wait()

    configured = replace(settings(), request_deadline_seconds=0.01)
    with pytest.raises(ProviderError) as caught:
        await complete_with_retry(HangingProvider(), [], configured)

    assert caught.value.retryable
    assert caught.value.safe_log_details() == " reason=deadline_exceeded"


def test_provider_error_rechecks_reason_before_logging():
    error = ProviderError("SECRET MESSAGE", reason="timeout")
    error.reason = "sk-test-SECRET-123"
    assert error.safe_log_details() == ""
