"""Gemini wrapper internals that do not require network access."""

from __future__ import annotations

from types import SimpleNamespace

from orion.config import OrionConfig
from orion.llm import (
    FunctionCall,
    GeminiLLM,
    LLMResponse,
    Usage,
    normalize_model,
)


def test_model_aliases() -> None:
    assert normalize_model("flash") == "gemini-3.5-flash"
    assert normalize_model("pro") == "gemini-3.1-pro-preview"
    assert normalize_model("flash-lite") == "gemini-3.5-flash-lite"
    assert normalize_model("models/gemini-2.0-flash") == "gemini-2.0-flash"
    assert normalize_model("custom-model") == "custom-model"


def test_usage_accumulates() -> None:
    usage = Usage()
    usage.add(10, 5)
    usage.add(2, 3)
    assert usage.total_tokens == 20
    assert usage.calls == 2
    other = Usage(prompt_tokens=1, output_tokens=1, calls=1)
    usage.merge(other)
    assert usage.total_tokens == 22
    assert usage.as_dict()["calls"] == 3


def test_message_conversion(config: OrionConfig) -> None:
    llm = GeminiLLM(config)
    contents = llm._to_contents(
        [
            {"role": "user", "text": "hello"},
            {"role": "assistant", "text": "hi"},
            {
                "role": "model",
                "function_calls": [{"name": "read_file", "args": {"path": "a.py"}}],
            },
            {
                "role": "tool",
                "function_responses": [
                    {"name": "read_file", "response": {"ok": True, "result": "x"}}
                ],
            },
        ]
    )
    assert [content.role for content in contents] == ["user", "model", "model", "user"]
    assert contents[0].parts[0].text == "hello"
    assert contents[2].parts[0].function_call.name == "read_file"
    assert contents[3].parts[0].function_response.name == "read_file"


def test_retry_delay_is_read_from_the_error(config: OrionConfig) -> None:
    llm = GeminiLLM(config)
    assert llm._retry_delay(RuntimeError("Please retry in 21.25s")) == 21.25
    assert llm._retry_delay(RuntimeError("{'retryDelay': '3s'}")) == 3.0
    assert llm._retry_delay(RuntimeError("invalid argument")) is None


def test_parse_response_reads_text_parts(config: OrionConfig) -> None:
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="hello", function_call=None)])
            )
        ],
        usage_metadata=None,
    )
    parsed = GeminiLLM(config)._parse_response(response, "test-model")
    assert parsed.text == "hello"
    assert parsed.function_calls == []


def test_thought_signature_is_preserved(config: OrionConfig) -> None:
    llm = GeminiLLM(config)
    signature = b"\x12\x34signature"
    function_call = SimpleNamespace(name="greet", args={"name": "Orion"}, id="call-1")
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            text=None,
                            function_call=function_call,
                            thought_signature=signature,
                        )
                    ]
                )
            )
        ],
        usage_metadata=None,
    )
    parsed = llm._parse_response(response, "test-model")
    assert parsed.function_calls[0].thought_signature == signature

    contents = llm._to_contents(
        [
            {
                "role": "model",
                "function_calls": [
                    {
                        "name": "greet",
                        "args": {"name": "Orion"},
                        "id": "call-1",
                        "thought_signature": signature,
                    }
                ],
            }
        ]
    )
    assert contents[0].parts[0].thought_signature == signature


def test_llm_response_helpers() -> None:
    response = LLMResponse(text="hi", function_calls=[FunctionCall("read_file", {})])
    assert response.has_tool_calls
    assert not LLMResponse(text="plain").has_tool_calls


def test_error_classification(config: OrionConfig) -> None:
    llm = GeminiLLM(config)
    assert llm._retryable(RuntimeError("503 UNAVAILABLE"))
    assert llm._retryable(RuntimeError("rate limit exceeded"))
    assert not llm._retryable(RuntimeError("invalid argument"))
    wrapped = llm._wrap_error(RuntimeError("API key not valid"))
    assert "GEMINI_API_KEY" in str(wrapped)


def test_auto_mode_falls_back_when_vertex_api_disabled(config: OrionConfig) -> None:
    vertex_config = OrionConfig(
        _env_file=None,
        workspace=config.workspace,
        gemini_api_key="AQ.exampleKeyThatIsLongEnough",
        vertex="auto",
    )
    llm = GeminiLLM(vertex_config)
    llm._client = object()  # pretend a Vertex client was already built
    llm._vertex_active = True
    exc = RuntimeError(
        "403 PERMISSION_DENIED: aiplatform.googleapis.com has not been used in project 123"
    )
    assert llm._maybe_switch_mode(exc) is True
    assert llm._vertex_active is False
    assert llm._client is None
    assert llm._mode_switched is True
    assert llm.mode_switched is True
    assert llm.active_backend == "ai-studio"
    # Only one automatic switch is allowed.
    assert llm._maybe_switch_mode(exc) is False


def test_auto_mode_does_not_switch_when_forced_or_injected(config: OrionConfig) -> None:
    exc = RuntimeError("permission denied")
    forced = OrionConfig(
        _env_file=None,
        workspace=config.workspace,
        gemini_api_key="AQ.exampleKeyThatIsLongEnough",
        vertex="true",
    )
    llm = GeminiLLM(forced)
    llm._vertex_active = True
    assert llm._maybe_switch_mode(exc) is False

    injected = GeminiLLM(config, client=object())
    injected._vertex_active = True
    assert injected._maybe_switch_mode(exc) is False


def test_mode_mismatch_markers(config: OrionConfig) -> None:
    llm = GeminiLLM(config)
    assert llm.active_backend == "ai-studio"
    assert llm.mode_switched is False
    assert llm._looks_like_mode_mismatch(RuntimeError("API key not valid"))
    assert llm._looks_like_mode_mismatch(RuntimeError("SERVICE_DISABLED"))
    assert not llm._looks_like_mode_mismatch(RuntimeError("invalid argument"))

    first = RuntimeError("aiplatform.googleapis.com has not been used in project")
    second = RuntimeError("API key not valid")
    llm._mode_switched = True
    llm._first_mode_error = first
    message = str(llm._error_for(second))
    assert "both Vertex AI and the Gemini API" in message
    assert "Detected-mode error" in message
