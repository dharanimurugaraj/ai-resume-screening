"""Small provider adapter: one function, `call_structured`, is the entire
surface the rest of the codebase depends on. Swapping providers/models means
editing this file (and config.py's env var names) only.

Currently backed by the Gemini API (free tier) via the official `google-genai`
SDK. Every call is single-attempt and fail-soft: any error, timeout, missing
API key, or schema-validation failure returns None so the caller can fall
back to a deterministic default. One resume's LLM failure must never crash
the batch.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import BaseModel

from screening.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client = None
_client_init_failed = False


def _get_client():
    global _client, _client_init_failed
    if _client is not None or _client_init_failed:
        return _client
    try:
        from google import genai

        _client = genai.Client(api_key=settings.gemini_api_key)
    except Exception:
        logger.warning("Gemini client could not be initialized", exc_info=True)
        _client_init_failed = True
        _client = None
    return _client


def call_structured(
    prompt: str, schema: type[T], system: str | None = None
) -> T | None:
    """Call the configured LLM and parse its response into `schema`.

    Returns None on any failure (missing key, network error, invalid JSON,
    schema mismatch) -- never raises.
    """
    if not settings.llm_enabled:
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        config: dict = {
            "response_mime_type": "application/json",
            "response_schema": schema,
        }
        if system:
            config["system_instruction"] = system

        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=config,
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        if parsed is not None:
            return schema.model_validate(parsed)
        return schema.model_validate_json(response.text)
    except Exception:
        logger.warning("LLM structured call failed; falling back", exc_info=True)
        return None
