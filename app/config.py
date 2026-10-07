"""Environment-backed application configuration."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import dotenv_values


_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    api_key: str
    app_api_keys: tuple[str, ...]
    redis_url: str
    rate_limit_per_minute: int
    llm_timeout_seconds: float
    request_deadline_seconds: float
    llm_max_retries: int
    cache_ttl_seconds: int
    moderation_mode: str = "local"
    moderation_api_key: str = ""

    @classmethod
    def from_env(cls, *, require_app_api_keys: bool = True) -> "Settings":
        file_values = dotenv_values(_DOTENV_PATH)

        def env(name: str, default: str = "") -> str:
            # The shell always wins, even when a variable is explicitly set to empty.
            if name in os.environ:
                return os.environ[name]
            value = file_values.get(name)
            return default if value is None else value

        provider = env("LLM_PROVIDER", "openai").lower()
        if provider not in {"openai", "anthropic"}:
            raise ValueError("LLM_PROVIDER must be openai or anthropic")
        key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
        api_key = env(key_name)
        app_keys = tuple(k.strip() for k in env("APP_API_KEYS").split(",") if k.strip())
        model = env("LLM_MODEL").strip()
        missing = [name for name, value in ((key_name, api_key), ("LLM_MODEL", model)) if not value]
        if require_app_api_keys and not app_keys:
            missing.append("APP_API_KEYS")
        if missing:
            raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")
        if any(character.isspace() for character in model):
            raise ValueError("LLM_MODEL must be an API model ID without spaces (for example, gpt-6-astra)")
        moderation_mode = env("MODERATION_MODE", "local").lower()
        if moderation_mode not in {"local", "openai"}:
            raise ValueError("MODERATION_MODE must be local or openai")
        moderation_api_key = env("OPENAI_API_KEY") if moderation_mode == "openai" else ""
        if moderation_mode == "openai" and not moderation_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI moderation")
        settings = cls(
            provider=provider,
            model=model,
            api_key=api_key,
            app_api_keys=app_keys,
            redis_url=env("REDIS_URL", "redis://localhost:6379/0"),
            rate_limit_per_minute=int(env("RATE_LIMIT_PER_MINUTE", "30")),
            llm_timeout_seconds=float(env("LLM_TIMEOUT_SECONDS", "15")),
            request_deadline_seconds=float(env("REQUEST_DEADLINE_SECONDS", "30")),
            llm_max_retries=int(env("LLM_MAX_RETRIES", "2")),
            cache_ttl_seconds=int(env("CACHE_TTL_SECONDS", "300")),
            moderation_mode=moderation_mode,
            moderation_api_key=moderation_api_key,
        )
        if (settings.rate_limit_per_minute < 1 or settings.llm_timeout_seconds <= 0
                or settings.request_deadline_seconds <= 0 or settings.llm_max_retries < 0
                or settings.cache_ttl_seconds < 0):
            raise ValueError("Rate limit, timeouts, retries, or cache TTL are invalid")
        return settings
