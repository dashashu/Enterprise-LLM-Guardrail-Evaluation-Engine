"""Redis rate limiting and scoped response cache."""

from __future__ import annotations

import hashlib
import hmac
import json
import time

from redis.asyncio import Redis

from app.config import Settings


RATE_SCRIPT = """
local n = redis.call('INCR', KEYS[1])
if n == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
return n
"""


class RedisStore:
    def __init__(self, redis: Redis, settings: Settings):
        self.redis = redis
        self.settings = settings

    @staticmethod
    def client_id(api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()

    def authorized(self, presented: str) -> bool:
        return any(hmac.compare_digest(presented, key) for key in self.settings.app_api_keys)

    async def allow(self, client_id: str) -> bool:
        minute = int(time.time() // 60)
        key = f"rate:{client_id}:{minute}"
        count = await self.redis.eval(RATE_SCRIPT, 1, key, 120)
        return int(count) <= self.settings.rate_limit_per_minute

    def cache_key(self, client_id: str, prompt: str, context: str, prompt_version: str) -> str:
        payload = json.dumps([client_id, self.settings.provider, self.settings.model, prompt_version, prompt, context], separators=(",", ":"))
        return "answer:" + hashlib.sha256(payload.encode()).hexdigest()

    async def get_cached(self, key: str) -> str | None:
        value = await self.redis.get(key)
        return value.decode() if isinstance(value, bytes) else value

    async def set_cached(self, key: str, value: str) -> None:
        if self.settings.cache_ttl_seconds:
            await self.redis.set(key, value, ex=self.settings.cache_ttl_seconds)
