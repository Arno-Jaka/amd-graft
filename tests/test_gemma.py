"""Gemma client, AI conversion, and AI review.

No test here touches the network. The HTTP call is faked at the urlopen
boundary so the request shape (OpenAI chat completions, Bearer auth) is
asserted without a live endpoint, and the model's reply is a fixed string so
the parsing and normalisation paths are deterministic.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from graft.agent import convert, gemma, review


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> bool:
        return False


def _fake_urlopen(body: bytes):
    captured: dict = {}

    def fake(request, timeout=None):
        captured["url"] = request.full_url
        captured["data"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = {k.lower(): v for k, v in request.header_items()}
        captured["timeout"] = timeout
        return _Response(body)

    return fake, captured


def _completion(content: str) -> bytes:
    return json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": content}}]}
    ).encode("utf-8")




def test_chat_posts_the_openai_shape(monkeypatch):
    fake, captured = _fake_urlopen(_completion("hello"))
    monkeypatch.setattr(gemma.urllib.request, "urlopen", fake)

    model = gemma.GemmaModel(
        base_url="https://example.test/v1", model="gemma-4", api_key="secret", timeout=7
    )
    assert model.complete("system", "prompt") == "hello"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["data"]["model"] == "gemma-4"
    assert captured["data"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]
    assert captured["timeout"] == 7


def test_the_api_key_is_sent_as_a_bearer_token(monkeypatch):
    fake, captured = _fake_urlopen(_completion("ok"))
    monkeypatch.setattr(gemma.urllib.request, "urlopen", fake)

    gemma.GemmaModel("https://h/v1", "m", "top-secret").complete("", "hi")
    assert captured["headers"]["authorization"] == "Bearer top-secret"


def test_a_non_json_body_is_a_gemma_error(monkeypatch):
    fake, _ = _fake_urlopen(b"<html>not json</html>")
    monkeypatch.setattr(gemma.urllib.request, "urlopen", fake)

    with pytest.raises(gemma.GemmaError):
        gemma.GemmaModel("https://h/v1", "m", "k").complete("", "hi")


def test_an_empty_completion_is_a_gemma_error(monkeypatch):
    fake, _ = _fake_urlopen(_completion("   "))
    monkeypatch.setattr(gemma.urllib.request, "urlopen", fake)

    with pytest.raises(gemma.GemmaError):
        gemma.GemmaModel("https://h/v1", "m", "k").complete("", "hi")


def test_an_http_error_keeps_the_key_out_of_the_message(monkeypatch):
    def fake(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad key")
        )

    monkeypatch.setattr(gemma.urllib.request, "urlopen", fake)
    with pytest.raises(gemma.GemmaError) as excinfo:
        gemma.GemmaModel("https://h/v1", "m", "top-secret").complete("", "hi")
    assert "401" in str(excinfo.value)
    assert "top-secret" not in str(excinfo.value)




def _patch_complete(monkeypatch, reply: str):
    monkeypatch.setattr(
        gemma.GemmaModel, "complete", lambda self, system, prompt, **_: reply
    )


def test_convert_parses_code_filename_and_comments(monkeypatch):
    _patch_complete(
        monkeypatch,
        json.dumps(
            {
                "summary": "Port done.",
                "filename": "kernel_gemma.cu",
                "code": "// a\n// b\n",
                "comments": [
                    {"line": 2, "severity": "review", "title": "T", "comment": "C"},
                    {"line": 99, "severity": "nonsense", "title": "X", "comment": "Y"},
                ],
            }
        ),
    )
    result = convert.convert_with_gemma(
        "src",
        {},
        base_url="https://h/v1",
        model_id="gemma-4",
        api_key="k",
        filename="kernel.cu",
    )
    assert result["available"] is True
    assert result["filename"] == "kernel_gemma.cu"
    assert result["provider"] == "gemma"
    assert result["comments"][0]["line"] == 2
    assert result["comments"][1]["line"] is None
    assert result["comments"][1]["severity"] == "review"


def test_convert_rejects_a_path_escaping_filename(monkeypatch):
    _patch_complete(
        monkeypatch,
        json.dumps({"filename": "../../etc/passwd", "code": "x = 1\n", "comments": []}),
    )
    result = convert.convert_with_gemma(
        "src",
        {},
        base_url="https://h/v1",
        model_id="m",
        api_key="k",
        filename="a.py",
    )
    assert result["filename"] == "passwd"


def test_convert_recovers_code_when_the_reply_is_not_json(monkeypatch):
    _patch_complete(monkeypatch, "```python\nprint('hi')\n```")
    result = convert.convert_with_gemma(
        "src",
        {},
        base_url="https://h/v1",
        model_id="m",
        api_key="k",
        filename="a.py",
    )
    assert result["available"] is True
    assert "print('hi')" in result["code"]
    assert "not valid JSON" in result["summary"]


def test_convert_needs_an_endpoint():
    result = convert.convert_with_gemma(
        "src", {}, base_url="", model_id="", api_key="k"
    )
    assert result["available"] is False
    assert "endpoint" in result["reason"].lower()


def test_convert_refuses_an_oversized_file():
    result = convert.convert_with_gemma(
        "x" * (convert.MAX_CONVERT_CHARS + 1),
        {},
        base_url="https://h/v1",
        model_id="m",
        api_key="k",
    )
    assert result["available"] is False
    assert "limit" in result["reason"]




def test_review_normalises_recommendation_and_line_numbers(monkeypatch):
    _patch_complete(
        monkeypatch,
        json.dumps(
            {
                "recommendation": "not_a_real_recommendation",
                "summary": "Needs a look.",
                "comments": [
                    {
                        "file": "converted",
                        "line": 1,
                        "severity": "blocker",
                        "title": "T",
                        "comment": "C",
                        "suggestion": "S",
                    },
                    {"file": "original", "line": 50, "severity": "info"},
                ],
            }
        ),
    )
    result = review.review_conversion(
        "a\nb\n",
        "c\n",
        {},
        base_url="https://h/v1",
        model_id="gemma-4",
        api_key="k",
    )
    assert result["available"] is True
    assert result["provider"] == "gemma"
    assert result["recommendation"] == "engineer_review_required"
    assert result["comments"][0]["line"] == 1
    assert result["comments"][1]["line"] is None


def test_review_without_a_converted_file_is_unavailable():
    result = review.review_conversion(
        "a\n", "", {}, base_url="https://h/v1", model_id="m", api_key="k"
    )
    assert result["available"] is False
    assert "did not produce" in result["reason"]
