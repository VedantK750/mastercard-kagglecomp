"""Thin wrapper around the project's OpenAI-SDK-compatible LLM endpoint.

Credentials come from .env (LLM_API_KEY, LLM_ENDPOINT) — never hardcode a key
here or print one. Two model roles are exposed:

- VICTIM_MODEL: the model playing the AP2 Shopping Agent under attack.
  Defaults to google/gemini-2.5-flash, the exact model Whispers of Wealth
  used, so our Branded/Vault Whisper reproduction is apples-to-apples.
- RED_MODEL: the model used by Red generators to write injection/content
  variants. Defaults to a stronger model since it needs to produce diverse,
  plausible adversarial text, not just follow a shopping task.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

VICTIM_MODEL = os.getenv("VICTIM_MODEL", "google/gemini-2.5-flash")
RED_MODEL = os.getenv("RED_MODEL", "openai/gpt-4o")


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    api_key = os.environ["LLM_API_KEY"]
    base_url = os.environ["LLM_ENDPOINT"]
    return OpenAI(api_key=api_key, base_url=base_url)


def chat(
    messages: list[dict],
    model: str = VICTIM_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """One chat-completion call, returns the assistant text content."""
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
