"""Optional hosted output moderation; errors fail closed."""

from typing import Protocol

import httpx

from app.guardrails import GuardrailError


class Moderator(Protocol):
    async def check(self, text: str) -> None: ...


class OpenAIModerator:
    def __init__(self, client: httpx.AsyncClient, api_key: str):
        self.client = client
        self.api_key = api_key

    async def check(self, text: str) -> None:
        try:
            response = await self.client.post(
                "https://api.openai.com/v1/moderations",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"input": text}, timeout=5,
            )
            response.raise_for_status()
            flagged = response.json()["results"][0]["flagged"]
            if not isinstance(flagged, bool):
                raise ValueError("Invalid moderation result")
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise GuardrailError("Output moderation unavailable") from exc
        if flagged:
            raise GuardrailError("Output failed safety moderation")
