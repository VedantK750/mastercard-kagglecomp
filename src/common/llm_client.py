"""Thin wrapper around Google's Gemini API, via its OpenAI-SDK-compatible
endpoint.

Credentials come from .env (GEMINI_API_KEY) — never hardcode a key here or
print one. We call Google directly rather than the aicredits.in aggregator
(exhausted-credits — see project memory) using
`https://generativelanguage.googleapis.com/v1beta/openai/`. Note the paper's
exact model, gemini-2.5-flash, is no longer servable to new API keys (404s
as of 2026); the closest available "flash" tier is used instead. Two model
roles are exposed:

- VICTIM_MODEL: the model playing the AP2 Shopping Agent under attack.
- RED_MODEL: the model used by Red generators to write injection/content
  variants. Both default to gemini-3.1-flash-lite: it's cheap, has near-zero
  hidden-reasoning overhead, and — unlike the newer flash/preview tiers,
  which are safety-tuned enough to often refuse to write injection text at
  all — it will actually produce adversarial variants, which Red generation
  depends on. Override either independently via the env vars if needed.

Gemini 3.x models spend tokens on hidden reasoning before emitting visible
content, so `max_tokens` needs real headroom (~150-250+) above the desired
visible output length or `chat()` silently returns "".
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

VICTIM_MODEL = os.getenv("VICTIM_MODEL", "gemini-3.1-flash-lite")
RED_MODEL = os.getenv("RED_MODEL", "gemini-3.1-flash-lite")


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    api_key = os.environ["GEMINI_API_KEY"]
    return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)


def chat(
    messages: list[dict],
    model: str = VICTIM_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 600,
) -> str:
    """One chat-completion call, returns the assistant text content. Returns
    "" (never raises) if Google's safety filter blocks the response outright
    — that shows up as `message=None`, not a refusal string, so callers must
    already treat empty string as "no usable output" rather than a real
    generation."""
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    message = resp.choices[0].message
    return (message.content or "") if message is not None else ""
