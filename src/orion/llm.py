"""LLM backends used by the agents and the embedding pipeline.

Two providers are supported:

* ``gemini`` (default) talks to Gemini through the official ``google-genai``
  SDK. Both Google AI Studio keys (``AIza...``) and Vertex AI express-mode keys
  (``AQ....``) are supported. In ``auto`` mode the mode is inferred from the key
  prefix and, if that guess is rejected, Orion retries once with the other
  backend. Set ``ORION_VERTEX=true``/``false`` to pin a mode.
* ``local`` talks to any OpenAI-compatible local server (Ollama by default).
  It needs no API key and uses the offline hash embedder for retrieval.

Both backends expose the same small interface: ``generate``, ``stream``,
``embed``, ``ping``, ``usage``, and ``active_backend``.
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from orion.config import OrionConfig
from orion.errors import LLMError

MODEL_ALIASES = {
    "flash": "gemini-3.5-flash",
    "pro": "gemini-3.1-pro-preview",
    "flash-lite": "gemini-3.5-flash-lite",
    "3.5-flash": "gemini-3.5-flash",
    "3.5-flash-lite": "gemini-3.5-flash-lite",
    "3.1-pro": "gemini-3.1-pro-preview",
    "2.5-flash": "gemini-2.5-flash",
    "2.5-flash-lite": "gemini-2.5-flash-lite",
    "2.5-pro": "gemini-2.5-pro",
    "2.0-flash": "gemini-2.0-flash",
}

EMBEDDING_DIMENSIONS = 768
# Gemini's free tier allows 100 embedding requests per minute and counts every
# input in a batch as one request. Keep a small safety margin and pace batches.
EMBED_REQUESTS_PER_MINUTE = 90
EMBED_WINDOW_SECONDS = 60.0
_RETRY_DELAY_RE = re.compile(
    r"retry(?:Delay)?[^0-9]{0,40}(\d+(?:\.\d+)?)\s*s",
    re.IGNORECASE,
)

RETRYABLE_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "INTERNAL",
    "overloaded",
    "rate limit",
    "timed out",
    "timeout",
)

# Errors that usually mean "the key belongs to the other backend" rather than
# "the request itself is wrong". Used only in ``ORION_VERTEX=auto`` mode.
MODE_SWITCH_MARKERS = (
    "service_disabled",
    "service disabled",
    "has not been used in project",
    "api is not enabled",
    "api has not been enabled",
    "aiplatform.googleapis.com",
    "api key not valid",
    "api_key_invalid",
    "invalid api key",
    "unauthenticated",
    "permission_denied",
    "permission denied",
)


@dataclass
class Usage:
    """Accumulated token usage for a run."""

    prompt_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def add(self, prompt_tokens: int = 0, output_tokens: int = 0, calls: int = 1) -> None:
        self.prompt_tokens += max(0, int(prompt_tokens or 0))
        self.output_tokens += max(0, int(output_tokens or 0))
        self.calls += calls

    def merge(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


@dataclass
class FunctionCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    # Gemini 3 requires the original signature when a function call is replayed
    # in the conversation history.
    thought_signature: bytes | None = None


@dataclass
class LLMResponse:
    text: str = ""
    function_calls: list[FunctionCall] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    raw: Any | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.function_calls)


def normalize_model(name: str) -> str:
    value = (name or "").strip()
    if value.startswith("models/"):
        value = value.split("/", 1)[1]
    return MODEL_ALIASES.get(value.lower(), value)


def _part_text(part: Any) -> str:
    text = getattr(part, "text", None)
    return text if isinstance(text, str) else ""


def _response_text(response: Any) -> str:
    """Collect text parts without triggering google-genai's mixed-parts warning.

    ``response.text`` logs a warning when a response also contains function
    calls. Reading the candidate parts directly keeps ordinary tool-use turns
    quiet.
    """
    candidates = getattr(response, "candidates", None) or []
    parts_text: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = _part_text(part)
            if text:
                parts_text.append(text)
    if candidates:
        return "".join(parts_text)
    try:
        return response.text or ""
    except (ValueError, AttributeError):
        return ""


class GeminiLLM:
    """Thin, retrying wrapper around ``google.genai``."""

    def __init__(
        self,
        config: OrionConfig,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = normalize_model(model or config.model)
        self.embedding_model = config.embedding_model
        self._client = client
        self._client_injected = client is not None
        self._vertex_active: bool | None = None
        self._mode_switched = False
        self._first_mode_error: Exception | None = None
        self._embed_times: list[float] = []
        self.usage = Usage()

    # ------------------------------------------------------------------
    # Client plumbing
    # ------------------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _make_client(self) -> Any:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency is required
            raise LLMError(
                "google-genai is not installed. Run: pip install 'orion-ai'"
            ) from exc

        api_key = self.config.require_api_key()
        vertex = self.config.vertex_enabled if self._vertex_active is None else self._vertex_active
        kwargs: dict[str, Any] = {"api_key": api_key}
        if vertex:
            kwargs["vertexai"] = True
        try:
            client = genai.Client(**kwargs)
        except Exception as exc:
            raise LLMError(
                "Could not create the Gemini client. Check GEMINI_API_KEY and "
                f"ORION_VERTEX ({self.config.vertex}). Details: {exc}"
            ) from exc
        self._vertex_active = vertex
        return client

    @staticmethod
    def _types() -> Any:
        from google.genai import types

        return types

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------
    def _to_contents(self, messages: Sequence[dict[str, Any]]) -> list[Any]:
        types = self._types()
        contents: list[Any] = []
        for message in messages:
            role = "model" if message.get("role") in {"model", "assistant"} else "user"
            parts: list[Any] = []
            text = message.get("text")
            if text:
                parts.append(types.Part.from_text(text=str(text)))
            for call in message.get("function_calls", []) or []:
                signature = call.get("thought_signature")
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=call.get("name"),
                            args=call.get("args") or {},
                            id=call.get("id"),
                        ),
                        thought_signature=signature or None,
                    )
                )
            for response in message.get("function_responses", []) or []:
                payload = response.get("response")
                if not isinstance(payload, dict):
                    payload = {"result": payload}
                parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=response.get("name"),
                            response=payload,
                            id=response.get("id"),
                        )
                    )
                )
            if parts:
                contents.append(types.Content(role=role, parts=parts))
        return contents

    def _build_tools(self, tools: Sequence[dict[str, Any]] | None) -> list[Any] | None:
        if not tools:
            return None
        types = self._types()
        declarations = []
        for tool in tools:
            declarations.append(
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters_json_schema=tool.get("parameters") or {"type": "object"},
                )
            )
        return [types.Tool(function_declarations=declarations)]

    def _generate_config(
        self,
        *,
        system: str | None,
        tools: Sequence[dict[str, Any]] | None,
        temperature: float | None,
        max_output_tokens: int | None,
        response_mime_type: str | None,
    ) -> Any:
        types = self._types()
        kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_output_tokens": max_output_tokens or self.config.max_output_tokens,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        tool_objects = self._build_tools(tools)
        if tool_objects:
            kwargs["tools"] = tool_objects
        if response_mime_type:
            kwargs["response_mime_type"] = response_mime_type
        return types.GenerateContentConfig(**{k: v for k, v in kwargs.items() if v is not None})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_mime_type: str | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Run one non-streaming generation, including any tool calls."""
        model_name = normalize_model(model or self.model)
        contents = self._to_contents(messages)
        config = self._generate_config(
            system=system,
            tools=tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type,
        )

        def call() -> Any:
            return self.client.models.generate_content(
                model=model_name, contents=contents, config=config
            )

        response = self._retry(call, model=model_name)
        result = self._parse_response(response, model_name)
        self.usage.add(result.prompt_tokens, result.output_tokens)
        return result

    def stream(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        """Yield text chunks from a streaming generation (tools disabled)."""
        model_name = normalize_model(model or self.model)
        contents = self._to_contents(messages)
        config = self._generate_config(
            system=system,
            tools=None,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type=None,
        )
        attempts = 0
        while True:
            emitted = False
            try:
                stream = self.client.models.generate_content_stream(
                    model=model_name, contents=contents, config=config
                )
                for chunk in stream:
                    text = _part_text(chunk)
                    if not text:
                        continue
                    usage = getattr(chunk, "usage_metadata", None)
                    if usage is not None:
                        self.usage.add(
                            getattr(usage, "prompt_token_count", 0) or 0,
                            getattr(usage, "candidates_token_count", 0) or 0,
                            calls=0,
                        )
                    emitted = True
                    yield text
                if emitted:
                    self.usage.calls += 1
                return
            except Exception as exc:
                if not emitted and self._maybe_switch_mode(exc):
                    continue
                attempts += 1
                if emitted or attempts >= 4 or not self._retryable(exc):
                    raise self._error_for(exc, model=model_name) from exc
                self._wait_before_retry(exc, attempts)

    def embed(
        self,
        texts: Sequence[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: int = EMBEDDING_DIMENSIONS,
    ) -> list[list[float]]:
        """Embed documents or a query with Gemini embeddings."""
        if not texts:
            return []
        types = self._types()
        results: list[list[float]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])

            def call(batch: list[str] = batch) -> Any:
                config = types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=output_dimensionality,
                )
                return self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch,
                    config=config,
                )

            self._throttle_embeddings(len(batch))
            response = self._retry(call, model=self.embedding_model)
            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != len(batch):
                raise LLMError(
                    f"Embedding API returned {len(embeddings)} vectors for {len(batch)} inputs"
                )
            for embedding in embeddings:
                values = getattr(embedding, "values", None)
                if not values:
                    raise LLMError("Embedding API returned an empty vector")
                results.append([float(value) for value in values])
        return results

    def _throttle_embeddings(self, count: int) -> None:
        """Pace embedding calls to respect the free-tier per-minute quota.

        The quota counts every input in a batch, so a 64-chunk batch consumes
        64 of the 100 free requests. Waiting is far better than failing an
        index build halfway through.
        """
        while True:
            now = time.monotonic()
            while self._embed_times and now - self._embed_times[0] > EMBED_WINDOW_SECONDS:
                self._embed_times.pop(0)
            if len(self._embed_times) + count <= EMBED_REQUESTS_PER_MINUTE:
                break
            wait = EMBED_WINDOW_SECONDS - (now - self._embed_times[0]) + 0.2
            time.sleep(max(0.2, min(wait, EMBED_WINDOW_SECONDS)))
        now = time.monotonic()
        self._embed_times.extend([now] * count)

    def ping(self) -> str:
        """Cheap connectivity check used by ``orion doctor``."""
        response = self.generate(
            [{"role": "user", "text": "Reply with the single word: ready"}],
            max_output_tokens=64,
            temperature=0.0,
        )
        return (response.text or "").strip() or "connected"

    @property
    def active_backend(self) -> str:
        """The backend currently selected: ``vertex`` or ``ai-studio``.

        Before the first request this reflects the configured setting; after a
        request it reflects the mode that actually worked (including an
        automatic fallback in ``ORION_VERTEX=auto`` mode).
        """
        vertex = (
            self.config.vertex_enabled if self._vertex_active is None else self._vertex_active
        )
        return "vertex" if vertex else "ai-studio"

    @property
    def mode_switched(self) -> bool:
        """True once auto-detection fell back to the other backend."""
        return self._mode_switched

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _parse_response(self, response: Any, model_name: str) -> LLMResponse:
        text = _response_text(response)

        calls: list[FunctionCall] = []
        pending_signature: bytes | None = None
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                signature = getattr(part, "thought_signature", None)
                if signature:
                    pending_signature = signature
                function_call = getattr(part, "function_call", None)
                if function_call is None:
                    continue
                calls.append(
                    FunctionCall(
                        name=getattr(function_call, "name", "") or "",
                        args=dict(getattr(function_call, "args", None) or {}),
                        id=getattr(function_call, "id", None),
                        thought_signature=signature or pending_signature,
                    )
                )
                if signature:
                    pending_signature = None

        if not calls:
            for call in getattr(response, "function_calls", None) or []:
                calls.append(
                    FunctionCall(
                        name=getattr(call, "name", "") or "",
                        args=dict(getattr(call, "args", None) or {}),
                        id=getattr(call, "id", None),
                    )
                )

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        return LLMResponse(
            text=text.strip(),
            function_calls=[call for call in calls if call.name],
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            model=model_name,
            raw=response,
        )

    def _retry(self, operation: Any, *, model: str) -> Any:
        attempts = 0
        while True:
            try:
                return operation()
            except Exception as exc:
                if self._maybe_switch_mode(exc):
                    continue
                attempts += 1
                if attempts >= 6 or not self._retryable(exc):
                    raise self._error_for(exc, model=model) from exc
                self._wait_before_retry(exc, attempts)

    @staticmethod
    def _looks_like_mode_mismatch(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(marker in text for marker in MODE_SWITCH_MARKERS)

    def _maybe_switch_mode(self, exc: Exception) -> bool:
        """Retry once with the other backend when ``vertex='auto'`` guessed wrong."""
        if self._client_injected or self._mode_switched:
            return False
        if self.config.vertex != "auto" or self._vertex_active is None:
            return False
        if not self._looks_like_mode_mismatch(exc):
            return False
        self._first_mode_error = exc
        self._vertex_active = not self._vertex_active
        self._client = None
        self._mode_switched = True
        return True

    def _error_for(self, exc: Exception, *, model: str | None = None) -> LLMError:
        if (
            self._mode_switched
            and self._first_mode_error is not None
            and exc is not self._first_mode_error
        ):
            return LLMError(
                "Orion tried both Vertex AI and the Gemini API, but neither worked.\n"
                f"Detected-mode error: {self._first_mode_error}\n"
                f"Fallback-mode error: {exc}"
            )
        return self._wrap_error(exc, model=model)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}"
        code = getattr(exc, "code", None)
        if isinstance(code, int) and code in {429, 500, 502, 503, 504}:
            return True
        return any(marker.lower() in text.lower() for marker in RETRYABLE_MARKERS)

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(12.0, (2 ** (attempt - 1)) + random.uniform(0.1, 0.6))

    @staticmethod
    def _retry_delay(exc: Exception) -> float | None:
        """Read the server's RetryInfo hint (``retryDelay: '21s'``) if present."""
        match = _RETRY_DELAY_RE.search(f"{exc}")
        if not match:
            return None
        try:
            return max(0.0, float(match.group(1)))
        except ValueError:  # pragma: no cover - regex guarantees a float
            return None

    def _wait_before_retry(self, exc: Exception, attempt: int) -> None:
        delay = self._retry_delay(exc)
        time.sleep(min(60.0, delay) if delay is not None else self._backoff(attempt))

    @staticmethod
    def _wrap_error(exc: Exception, model: str | None = None) -> LLMError:
        message = str(exc)
        lowered = message.lower()
        if "api key" in lowered or "unauthenticated" in lowered or "permission" in lowered:
            hint = (
                "Gemini rejected the credentials. Check GEMINI_API_KEY in .env. "
                "AI Studio keys start with 'AIza'; Vertex express keys start with 'AQ.'. "
                "Orion tries the detected mode and falls back automatically in auto mode; "
                "set ORION_VERTEX=false to force AI Studio or true to force Vertex."
            )
            return LLMError(f"{hint}\n\nDetails: {message}")
        if "not found" in lowered or "not supported" in lowered:
            return LLMError(
                f"Model '{model or 'unknown'}' is not available for this key. "
                "Try ORION_MODEL=gemini-3.5-flash or list models with `orion doctor`.\n"
                f"Details: {message}"
            )
        return LLMError(f"Gemini request failed: {message}")


def _tool_call_id(call: dict[str, Any], index: int) -> str:
    """Stable id used to pair assistant tool calls with tool responses."""
    value = call.get("id") or call.get("name")
    return str(value) if value else f"call-{index}"


def _json_content(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class LocalLLM:
    """OpenAI-compatible local backend (Ollama by default).

    Speaks ``POST {base_url}/chat/completions`` with the same message/tool
    shapes as the OpenAI API. No key is sent unless ``ORION_LOCAL_API_KEY`` is
    set (useful for local gateways that still require a token). Embeddings are
    handled by the deterministic offline hash embedder, so enabling this backend
    never makes an embedding API call.
    """

    def __init__(
        self,
        config: OrionConfig,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.model = self._resolve_model(model or config.local_model)
        self.embedding_model = "local-hash-v1"
        self.base_url = config.local_base_url.strip().rstrip("/")
        self._opener = client
        self._mode_switched = False
        self.usage = Usage()

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------
    def _resolve_model(self, name: str | None) -> str:
        candidate = normalize_model(name or "")
        # ORION_MODEL keeps its Gemini default for backwards compatibility; in
        # local mode that value is only a sentinel meaning "use local_model".
        if not candidate or candidate.startswith("gemini-"):
            return self.config.local_model
        return candidate

    @property
    def chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    @property
    def active_backend(self) -> str:
        lowered = self.base_url.lower()
        return "ollama" if "11434" in lowered or "ollama" in lowered else "openai-compatible"

    @property
    def mode_switched(self) -> bool:
        """Local mode never falls back to a different provider."""
        return False

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.config.local_api_key_value
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _post(self, payload: dict[str, Any], *, model: str) -> Any:
        request = urllib.request.Request(
            self.chat_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        opener = self._opener or urllib.request.build_opener()
        try:
            return opener.open(request, timeout=self.config.local_timeout)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:  # pragma: no cover - defensive for custom openers
                detail = ""
            raise self._wrap_error(
                RuntimeError(detail or f"HTTP {exc.code} {exc.reason}"),
                model=model,
                base_url=self.base_url,
            ) from exc
        except OSError as exc:  # URLError and socket errors both subclass OSError
            raise self._wrap_error(exc, model=model, base_url=self.base_url) from exc

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------
    def _to_messages(
        self,
        messages: Sequence[dict[str, Any]],
        system: str | None = None,
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        if system:
            converted.append({"role": "system", "content": system})

        for message in messages:
            role = "assistant" if message.get("role") in {"model", "assistant"} else "user"
            text = str(message.get("text") or "")
            calls = list(message.get("function_calls") or [])
            responses = list(message.get("function_responses") or [])

            if calls:
                tool_calls = [
                    {
                        "id": _tool_call_id(call, index),
                        "type": "function",
                        "function": {
                            "name": str(call.get("name") or ""),
                            "arguments": _json_content(call.get("args") or {}),
                        },
                    }
                    for index, call in enumerate(calls)
                ]
                converted.append(
                    {"role": "assistant", "content": text, "tool_calls": tool_calls}
                )
            elif text:
                converted.append({"role": role, "content": text})

            for index, response in enumerate(responses):
                payload = response.get("response")
                if not isinstance(payload, (dict, list)):
                    payload = {"result": payload}
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": _tool_call_id(response, index),
                        "content": _json_content(payload),
                    }
                )
        return converted

    @staticmethod
    def _build_tools(tools: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {"type": "object"},
                },
            }
            for tool in tools
            if tool.get("name")
        ] or None

    @staticmethod
    def _usage_from(raw: dict[str, Any]) -> tuple[int, int]:
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        prompt = usage.get("prompt_tokens") or usage.get("prompt_eval_count") or 0
        output = usage.get("completion_tokens") or usage.get("eval_count") or 0
        try:
            return int(prompt or 0), int(output or 0)
        except (TypeError, ValueError):  # pragma: no cover - malformed server response
            return 0, 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_mime_type: str | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Run one non-streaming generation, including any tool calls."""
        model_name = self._resolve_model(model or self.model)
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": self._to_messages(messages, system),
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": max_output_tokens or self.config.max_output_tokens,
            "stream": False,
        }
        tool_objects = self._build_tools(tools)
        if tool_objects:
            payload["tools"] = tool_objects
        if response_mime_type:
            payload["response_format"] = {"type": "json_object"}

        with self._post(payload, model=model_name) as response:
            body = response.read().decode("utf-8", errors="replace")
        try:
            raw = json.loads(body)
        except ValueError as exc:
            raise LLMError(
                f"Local model at {self.base_url} returned non-JSON data: "
                f"{truncate_for_error(body)}"
            ) from exc
        result = self._parse_response(raw, model_name)
        self.usage.add(result.prompt_tokens, result.output_tokens)
        return result

    def stream(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        """Yield text chunks from a streaming generation (tools disabled)."""
        model_name = self._resolve_model(model or self.model)
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": self._to_messages(messages, system),
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": max_output_tokens or self.config.max_output_tokens,
            "stream": True,
        }
        emitted = False
        with self._post(payload, model=model_name) as response:
            for raw_line in response:
                line = (
                    raw_line.decode("utf-8", errors="replace")
                    if isinstance(raw_line, bytes)
                    else str(raw_line)
                ).strip()
                if not line.startswith("data:"):
                    continue
                data_part = line[5:].strip()
                if data_part == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_part)
                except ValueError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    raise self._wrap_error(
                        RuntimeError(str(chunk["error"])),
                        model=model_name,
                        base_url=self.base_url,
                    )
                prompt, output = self._usage_from(chunk)
                if prompt or output:
                    self.usage.add(prompt, output, calls=0)
                choices = chunk.get("choices") or []
                for choice in choices:
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    if text:
                        emitted = True
                        yield text
        if emitted:
            self.usage.calls += 1

    def embed(
        self,
        texts: Sequence[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: int = EMBEDDING_DIMENSIONS,
    ) -> list[list[float]]:
        """Embed locally with the offline hash embedder (no API calls)."""
        if not texts:
            return []
        # Imported lazily: rag.embeddings imports this module at import time.
        from orion.rag.embeddings import LocalHashEmbedder

        return LocalHashEmbedder().embed_documents([str(text) for text in texts])

    def ping(self) -> str:
        """Cheap connectivity check used by ``orion doctor``."""
        response = self.generate(
            [{"role": "user", "text": "Reply with the single word: ready"}],
            max_output_tokens=64,
            temperature=0.0,
        )
        return (response.text or "").strip() or "connected"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _parse_response(self, raw: Any, model_name: str) -> LLMResponse:
        if not isinstance(raw, dict):
            raise LLMError(f"Local model returned an unexpected response: {raw!r}")
        if raw.get("error"):
            raise self._wrap_error(
                RuntimeError(str(raw["error"])),
                model=model_name,
                base_url=self.base_url,
            )
        choices = raw.get("choices") or []
        message = choices[0].get("message") if choices else (raw.get("message") or {})
        if not isinstance(message, dict):
            message = {}
        text = message.get("content") or ""

        calls: list[FunctionCall] = []
        for index, tool_call in enumerate(message.get("tool_calls") or []):
            function = tool_call.get("function") or {}
            name = function.get("name") or ""
            args = function.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    args = {}
            if not isinstance(args, dict):
                args = {"value": args}
            calls.append(
                FunctionCall(
                    name=str(name),
                    args=args,
                    id=str(tool_call.get("id") or name or f"call-{index}"),
                )
            )

        prompt, output = self._usage_from(raw)
        return LLMResponse(
            text=str(text).strip(),
            function_calls=[call for call in calls if call.name],
            prompt_tokens=prompt,
            output_tokens=output,
            model=model_name,
            raw=raw,
        )

    @staticmethod
    def _wrap_error(exc: Exception, *, model: str, base_url: str) -> LLMError:
        text = str(exc)
        lowered = text.lower()
        if any(
            marker in lowered
            for marker in (
                "connection refused",
                "name or service not known",
                "timed out",
                "timeout",
                "unreachable",
                "cannot reach",
            )
        ):
            hint = (
                f"Cannot reach the local model server at {base_url}. "
                "Start Ollama first (for example: ollama serve), then retry."
            )
        elif "404" in text or "not found" in lowered:
            hint = (
                f"Model '{model}' was not found by the local server. "
                f"Pull it with: ollama pull {model}"
            )
        elif "400" in text and ("tool" in lowered or "function" in lowered):
            hint = (
                f"The local model '{model}' rejected tool calling. Use a tool-capable "
                "model such as qwen2.5:7b-instruct, or reduce the tool schema."
            )
        else:
            hint = f"Local model request failed for '{model}'."
        return LLMError(f"{hint}\n\nDetails: {text}")


def truncate_for_error(text: str, limit: int = 300) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


LLM = GeminiLLM | LocalLLM


def build_llm(config: OrionConfig, *, client: Any | None = None) -> LLM:
    """Create the configured LLM backend.

    ``client`` is an optional injected client/opener used by tests.
    """
    if config.llm_provider == "local":
        return LocalLLM(config, client=client)
    return GeminiLLM(config, client=client)


__all__ = [
    "EMBEDDING_DIMENSIONS",
    "LLM",
    "MODEL_ALIASES",
    "FunctionCall",
    "GeminiLLM",
    "LLMResponse",
    "LocalLLM",
    "Usage",
    "build_llm",
    "normalize_model",
]
