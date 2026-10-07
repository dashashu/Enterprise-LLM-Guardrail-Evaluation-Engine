"""Public request/response contracts."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=4000)
    context: str = Field(default="", max_length=12000)


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1, max_length=8000)


class GenerateResponse(ModelAnswer):
    source: Literal["model", "cache", "fallback"]
    request_id: str


class JudgeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    faithfulness: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=2000)
