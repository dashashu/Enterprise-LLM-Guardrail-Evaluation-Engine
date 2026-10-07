"""FastAPI entry point: uvicorn app.main:app."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
import httpx
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import Settings
from app.guardrails import GuardrailError
from app.moderation import OpenAIModerator
from app.provider import HTTPProvider
from app.schemas import GenerateRequest, GenerateResponse
from app.service import GenerationService, RateLimitError
from app.storage import RedisStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    redis = Redis.from_url(settings.redis_url)
    async with httpx.AsyncClient() as client:
        app.state.store = RedisStore(redis, settings)
        moderator = OpenAIModerator(client, settings.moderation_api_key) if settings.moderation_mode == "openai" else None
        app.state.service = GenerationService(settings, HTTPProvider(settings, client), app.state.store, moderator)
        try:
            yield
        finally:
            await redis.aclose()


app = FastAPI(title="Enterprise LLM Guardrail & Evaluation Engine", lifespan=lifespan)


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready(request: Request) -> dict[str, str]:
    try:
        await request.app.state.store.redis.ping()
    except RedisError as exc:
        raise HTTPException(status_code=503, detail="Redis unavailable") from exc
    return {"status": "ok"}


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest, request: Request, x_api_key: str = Header(default="")) -> GenerateResponse:
    store: RedisStore = request.app.state.store
    if not store.authorized(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        return await request.app.state.service.generate(body, store.client_id(x_api_key))
    except GuardrailError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": "60"}) from exc
    except RedisError as exc:
        raise HTTPException(status_code=503, detail="Rate limiter unavailable") from exc
