"""Input screening, strict output parsing, and baseline output moderation.

Pattern screening is a first layer, not a complete prompt-injection or safety classifier.
"""

import re
from pydantic import ValidationError
from app.schemas import GenerateRequest, ModelAnswer


class GuardrailError(Exception):
    pass


INJECTION_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"ignore (all |any )?(previous|prior|system|developer) instructions",
    r"reveal (the |your )?(system|developer) (prompt|instructions)",
    r"(override|bypass) (the |your )?(system|safety) (prompt|rules|policy)",
))
UNSAFE_OUTPUT_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:api[_ -]?key|secret[_ -]?key)\s*[:=]\s*\S+",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
))


def validate_input(request: GenerateRequest) -> None:
    if not request.prompt.strip():
        raise GuardrailError("Prompt must contain non-whitespace text")
    if any(pattern.search(request.prompt) for pattern in INJECTION_PATTERNS):
        raise GuardrailError("Prompt contains a disallowed instruction pattern")


def parse_and_moderate(raw: str) -> ModelAnswer:
    try:
        answer = ModelAnswer.model_validate_json(raw)
    except ValidationError as exc:
        raise GuardrailError("Model output did not match the response schema") from exc
    if not answer.answer.strip() or any(p.search(answer.answer) for p in UNSAFE_OUTPUT_PATTERNS):
        raise GuardrailError("Model output failed safety checks")
    return answer


FALLBACK = ModelAnswer(answer="The service is temporarily unavailable. Please try again later.")

