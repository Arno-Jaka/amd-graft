"""Minimal OpenAI-compatible client for AMD's Gemma endpoint.

Deliberately dependency-free. The agent layer must import and degrade even
when no provider SDK is installed, and one HTTP POST is not worth pulling in
``openai``, ``agno``, or a vendor client. Everything here is stdlib.

The endpoint AMD's Radeon "token factory" exposes is OpenAI-shaped:

    POST {base_url}/chat/completions
    Authorization: Bearer <key>
    {"model": "...", "messages": [{"role": "user", "content": "..."}]}

Only the two fields Graft needs are parsed back out: ``choices[0].message.
content``. A non-2xx response, a body that is not JSON, or an empty completion
is raised as :class:`GemmaError` with the API key kept out of the message, so
a caller can turn any failure into report text without leaking a credential.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_TIMEOUT_S = 60.0
MAX_ERROR_CHARS = 300


class GemmaError(RuntimeError):
    """The endpoint could not be reached or returned no usable completion."""


@dataclass(frozen=True)
class GemmaModel:
    """One configured chat model: where to reach it and how to authenticate."""

    base_url: str
    model: str
    api_key: str
    timeout: float = DEFAULT_TIMEOUT_S

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """One system-plus-user turn, returning the assistant's text."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """POST a chat completion and return the first message's content."""
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, object] = {"model": self.model, "messages": messages}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raise GemmaError(
                f"endpoint returned HTTP {exc.code}{_read_error_body(exc)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise GemmaError(f"could not reach the endpoint: {exc.reason}") from exc
        except TimeoutError as exc:
            raise GemmaError(
                f"endpoint did not answer within {self.timeout:.0f}s"
            ) from exc

        return _extract_content(body)


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    """A short, credential-free excerpt of an error response, for diagnosis."""
    try:
        text = exc.read().decode("utf-8", "replace").strip()
    except Exception:
        return ""
    if not text:
        return ""
    return f": {text[:MAX_ERROR_CHARS]}"


def _extract_content(body: str) -> str:
    """Pull the assistant text out of an OpenAI-shaped completion body."""
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise GemmaError("endpoint returned a non-JSON body") from exc
    if not isinstance(parsed, dict):
        raise GemmaError("endpoint returned an unexpected body")

    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        raise GemmaError("endpoint returned no choices")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise GemmaError("endpoint returned an empty completion")
    return content.strip()


__all__ = ["DEFAULT_TIMEOUT_S", "GemmaError", "GemmaModel"]
