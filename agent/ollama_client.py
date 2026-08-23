"""
Ollama HTTP client — wraps /api/chat directly via httpx.

Why not the `ollama` Python SDK?
  - Direct HTTP gives us explicit timeout control per request.
  - We can log raw request/response for debugging without monkey-patching.
  - No version-coupling to the SDK's internal message format.

Endpoint used: POST http://localhost:11434/api/chat
Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
# llama3.1:8b confirmed available on this machine
DEFAULT_MODEL   = "llama3.1:8b"
# 120 s — local 8B model can take 60–90 s on first token with a large prompt
REQUEST_TIMEOUT = 120.0


class OllamaError(RuntimeError):
    """Raised when Ollama returns a non-200 or unparseable response."""


async def chat(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    timeout: float = REQUEST_TIMEOUT,
) -> tuple[str, dict[str, Any]]:
    """
    POST /api/chat with stream=false.

    Returns:
        (content_str, raw_response_dict)
        content_str        — the assistant message content
        raw_response_dict  — the full parsed JSON response from Ollama

    Raises OllamaError on HTTP errors or response parse failures.
    Never swallows exceptions silently — callers decide how to handle.
    """
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": temperature},
    }

    logger.debug(
        "ollama_client.chat → model=%s temperature=%s messages=%d",
        model, temperature, len(messages),
    )
    # Log user message length (not full content to keep logs readable)
    for m in messages:
        logger.debug("  [%s] %d chars", m["role"], len(m.get("content", "")))

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
        )

    logger.debug("ollama_client response: HTTP %s", resp.status_code)

    if resp.status_code != 200:
        raise OllamaError(
            f"Ollama returned HTTP {resp.status_code}: {resp.text[:500]}"
        )

    try:
        raw = resp.json()
    except Exception as exc:
        raise OllamaError(f"Ollama response is not valid JSON: {exc}") from exc

    content = raw.get("message", {}).get("content", "")
    logger.debug("ollama_client raw content (%d chars): %.300s…", len(content), content)
    return content, raw


def parse_json_response(raw_content: str) -> dict[str, Any]:
    """
    Extracts and parses a JSON object from the LLM's raw text response.

    Handles:
    - Responses wrapped in ```json ... ``` fences
    - Leading/trailing prose before or after the JSON object
    - Responses that are already bare JSON
    - Truncated responses (LLM cut off mid-output) — raises JSONDecodeError
      with a clear message including the truncation point

    Raises json.JSONDecodeError if no valid JSON object can be extracted.
    """
    if not raw_content or not raw_content.strip():
        raise json.JSONDecodeError("Empty response from LLM", "", 0)

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw_content, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    # Try to parse as-is first (most common case for well-behaved models)
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
        raise json.JSONDecodeError("Top-level value is not a JSON object", cleaned, 0)
    except json.JSONDecodeError:
        pass

    # Find the first { and scan for the matching closing }
    start = cleaned.find("{")
    if start == -1:
        raise json.JSONDecodeError(
            f"No JSON object found in response ({len(cleaned)} chars)",
            cleaned, 0,
        )

    depth = 0
    end   = -1
    in_string   = False
    escape_next = False

    for i, ch in enumerate(cleaned[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        # LLM truncated its output — report how far it got
        raise json.JSONDecodeError(
            f"LLM output appears truncated (depth={depth} at end, "
            f"{len(cleaned)-start} chars scanned from first brace)",
            cleaned, start,
        )

    candidate = cleaned[start:end]
    try:
        result = json.loads(candidate)
        if not isinstance(result, dict):
            raise json.JSONDecodeError("Extracted value is not a JSON object", candidate, 0)
        return result
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"Found balanced braces but JSON is still invalid: {exc.msg}",
            candidate, exc.pos,
        ) from exc


async def is_ollama_reachable(timeout: float = 3.0) -> bool:
    """Quick TCP + HTTP check — used at startup."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/version")
            return r.status_code == 200
    except Exception:
        return False


async def list_models() -> list[str]:
    """Returns list of pulled model names, or [] if Ollama unreachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []
