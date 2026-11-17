"""Gemini wrapper internals that do not require network access."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from orion.config import OrionConfig
from orion.llm import (
    FunctionCall,
    GCloudLLM,
    GeminiLLM,
    LLMResponse,
    LocalLLM,
    Usage,
    build_llm,
    detect_vertex_project,
    normalize_model,
)


def test_model_aliases() -> None:
    assert normalize_model("flash") == "gemini-3.5-flash"
    assert normalize_model("pro") == "gemini-3.1-pro-preview"
    assert normalize_model("flash-lite") == "gemini-3.5-flash-lite"
    assert normalize_model("models/gemini-2.0-flash") == "gemini-2.0-flash"
    assert normalize_model("custom-model") == "custom-model"


def test_vertex_project_falls_back_to_adc_quota_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("orion.llm._gcloud_config_project", lambda: None)
    for key in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "PROJECT_ID"):
        monkeypatch.delenv(key, raising=False)
    credentials = SimpleNamespace(quota_project_id="project-6a0ab983-d480-4954-99f")
    assert detect_vertex_project(credentials, None) == "project-6a0ab983-d480-4954-99f"
    assert detect_vertex_project(credentials, "direct-project") == "direct-project"


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


class _FakeOpener:
    """Minimal urllib opener stand-in that records the request and returns bytes."""

    def __init__(self, payload: object) -> None:
        if isinstance(payload, bytes):
            self.body = payload
        elif isinstance(payload, str):
            self.body = payload.encode("utf-8")
        else:
            self.body = json.dumps(payload).encode("utf-8")
        self.request: object | None = None

    def open(self, request: object, timeout: float | None = None) -> io.BytesIO:
        self.request = request
        return io.BytesIO(self.body)


def _local_config(config: OrionConfig) -> OrionConfig:
    return config.with_overrides(
        llm_provider="local",
        local_model="qwen2.5:7b-instruct",
        local_base_url="http://localhost:11434/v1",
    )


def test_build_llm_selects_provider(config: OrionConfig) -> None:
    assert isinstance(build_llm(config), GCloudLLM)
    gemini = build_llm(config.with_overrides(llm_provider="gemini", gemini_api_key="AIzaTest"))
    assert isinstance(gemini, GeminiLLM)
    local = build_llm(_local_config(config))
    assert isinstance(local, LocalLLM)
    assert local.model == "qwen2.5:7b-instruct"
    assert local.active_backend == "ollama"
    assert local.mode_switched is False


def test_build_llm_selects_gcloud(config: OrionConfig) -> None:
    gcloud = build_llm(config.with_overrides(llm_provider="gcloud", gcloud_project="p"))
    assert isinstance(gcloud, GCloudLLM)
    assert gcloud.model == "gemini-3.5-flash"


def test_gcloud_model_fallback_chain(config: OrionConfig) -> None:
    llm = GCloudLLM(config.with_overrides(llm_provider="gcloud"))
    assert llm._model_candidates("claude-opus-5-5") == [
        "claude-opus-5-5",
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3-flash-preview",
    ]
    assert llm._model_candidates("glm-5-maas")[:2] == ["glm-5-maas", "gemini-3.5-flash"]
    assert llm._model_unavailable(llm._wrap_error(RuntimeError("404 NOT_FOUND")))
    assert not llm._model_unavailable(llm._wrap_error(RuntimeError("429 rate limit")))


def test_gcloud_backend_metadata_and_errors(config: OrionConfig) -> None:
    gcloud_config = config.with_overrides(
        llm_provider="gcloud",
        gcloud_project="project-6a0ab983-d480-4954-99f",
        gcloud_region="us-east5",
        gcloud_endpoint="us-east5-aiplatform.googleapis.com",
    )
    llm = GCloudLLM(gcloud_config)
    assert llm.active_backend == "gcloud-vertex (us-east5)"
    assert llm.mode_switched is False
    assert llm.base_url == "https://us-east5-aiplatform.googleapis.com/"

    wrapped = llm._wrap_error(RuntimeError("default credentials were not found"))
    assert "gcloud auth application-default login" in str(wrapped)
    forbidden = llm._wrap_error(RuntimeError("403 permission denied"))
    assert "roles/aiplatform.user" in str(forbidden)


def test_gcloud_client_uses_adc_and_regional_endpoint(
    config: OrionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import google.auth
    import google.genai

    calls: dict[str, object] = {}

    class FakeCredentials:
        pass

    def fake_default(*, scopes: object = None) -> tuple[object, str]:
        calls["scopes"] = scopes
        return FakeCredentials(), "adc-project"

    def fake_client(**kwargs: object) -> object:
        calls["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(google.auth, "default", fake_default)
    monkeypatch.setattr(google.genai, "Client", fake_client)

    gcloud_config = config.with_overrides(
        llm_provider="gcloud",
        gcloud_project="project-6a0ab983-d480-4954-99f",
        gcloud_region="us-east5",
        gcloud_endpoint="us-east5-aiplatform.googleapis.com",
    )
    llm = GCloudLLM(gcloud_config)
    assert llm.client is not None
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["vertexai"] is True
    assert kwargs["project"] == "project-6a0ab983-d480-4954-99f"
    assert kwargs["location"] == "us-east5"
    assert isinstance(kwargs["credentials"], FakeCredentials)
    assert kwargs["http_options"] == {"base_url": "https://us-east5-aiplatform.googleapis.com/"}


def test_local_message_conversion(config: OrionConfig) -> None:
    llm = LocalLLM(_local_config(config))
    messages = llm._to_messages(
        [
            {"role": "user", "text": "hello"},
            {
                "role": "model",
                "text": "reading",
                "function_calls": [
                    {"name": "read_file", "args": {"path": "a.py"}, "id": "call-1"}
                ],
            },
            {
                "role": "tool",
                "function_responses": [
                    {"name": "read_file", "id": "call-1", "response": {"ok": True}}
                ],
            },
        ],
        system="be brief",
    )
    assert messages[0] == {"role": "system", "content": "be brief"}
    assert messages[1] == {"role": "user", "content": "hello"}
    assert messages[2]["role"] == "assistant"
    tool_call = messages[2]["tool_calls"][0]
    assert tool_call["id"] == "call-1"
    assert tool_call["function"]["name"] == "read_file"
    assert json.loads(tool_call["function"]["arguments"]) == {"path": "a.py"}
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"ok": true}',
    }


def test_local_generate_parses_tools_and_usage(config: OrionConfig) -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": "ok",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "a.py"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    opener = _FakeOpener(payload)
    llm = LocalLLM(_local_config(config), client=opener)
    response = llm.generate(
        [{"role": "user", "text": "hi"}],
        system="sys",
        tools=[{"name": "read_file", "description": "Read", "parameters": {"type": "object"}}],
        response_mime_type="application/json",
    )
    assert response.text == "ok"
    assert response.function_calls[0].name == "read_file"
    assert response.function_calls[0].args == {"path": "a.py"}
    assert response.prompt_tokens == 10
    assert llm.usage.output_tokens == 3

    request = opener.request
    assert request is not None
    body = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
    assert body["model"] == "qwen2.5:7b-instruct"
    assert body["stream"] is False
    assert body["response_format"] == {"type": "json_object"}
    assert body["tools"][0]["type"] == "function"


def test_local_stream_yields_sse_content(config: OrionConfig) -> None:
    first = {"choices": [{"delta": {"content": "Hel"}}]}
    second = {"choices": [{"delta": {"content": "lo"}}], "usage": {"prompt_tokens": 4}}
    sse = (
        f"data: {json.dumps(first)}\n\n"
        f"data: {json.dumps(second)}\n\n"
        "data: [DONE]\n\n"
    )
    llm = LocalLLM(_local_config(config), client=_FakeOpener(sse))
    assert "".join(llm.stream([{"role": "user", "text": "hi"}])) == "Hello"
    assert llm.usage.calls == 1
    assert llm.usage.prompt_tokens == 4


def test_local_embed_is_offline(config: OrionConfig) -> None:
    llm = LocalLLM(_local_config(config), client=_FakeOpener({}))
    vectors = llm.embed(["hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 512
    assert llm.embed([]) == []


def test_local_error_hints(config: OrionConfig) -> None:
    llm = LocalLLM(_local_config(config))
    offline = llm._wrap_error(
        RuntimeError("Connection refused"), model=llm.model, base_url=llm.base_url
    )
    assert "local model server" in str(offline)
    missing = llm._wrap_error(
        RuntimeError("404 model not found"), model=llm.model, base_url=llm.base_url
    )
    assert "not found by the local server" in str(missing)
