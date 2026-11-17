"""Gemini client wrapper used by the agents and the embedding pipeline.

Orion talks to Gemini through the official ``google-genai`` SDK. Both Google AI
Studio keys (``AIza...``) and Vertex AI express-mode keys (``AQ....``) are
supported. In ``auto`` mode the mode is inferred from the key prefix and, if
that guess is rejected, Orion retries once with the other backend. Set
``ORION_VERTEX=true``/``false`` to pin a mode.
"""

from __future__ import annotations

import random
import re
import time
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


__all__ = [
    "EMBEDDING_DIMENSIONS",
    "MODEL_ALIASES",
    "FunctionCall",
    "GeminiLLM",
    "LLMResponse",
    "Usage",
    "normalize_model",
]
