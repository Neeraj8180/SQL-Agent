"""OpenAI provider — wraps the existing langchain_openai.ChatOpenAI usage.

Preserves today's behavior byte-for-byte: the object this provider returns is
a thin proxy around a real ``ChatOpenAI`` instance, so every pre-existing
structured-output call (``with_structured_output(cls).invoke(msgs)``) works
exactly as it did before phase 2. No agent code changes.

Phase 8.3: the proxy additionally records OpenAI token usage per call into
the ``token_usage_var`` request context, so tracking can surface input /
output token counts as MLflow metrics + Prometheus counters.
"""

from __future__ import annotations

from typing import Any, Optional

from sql_agent.config import get_logger, settings
from sql_agent.request_context import record_token_usage

from .base import ProviderUnavailableError


_log = get_logger("llm_serving.openai")


# ---------------------------------------------------------------------------
# Token-counting proxy around ChatOpenAI.
#
# LangChain's `ChatOpenAI.invoke` returns AIMessage objects with
# ``usage_metadata`` populated. `.with_structured_output(cls)` by default
# returns a parsed Pydantic model and DISCARDS the AIMessage — we set
# ``include_raw=True`` to recover the raw message and report usage, then
# hand the parsed model back to the agent (same type the agent expected).
# ---------------------------------------------------------------------------


def _usage_from_ai_message(msg: Any) -> tuple[int, int]:
    usage = getattr(msg, "usage_metadata", None) or {}
    return (
        int(usage.get("input_tokens", 0) or 0),
        int(usage.get("output_tokens", 0) or 0),
    )


def _record_call(model_id: str, latency_ms: float, i: int, o: int) -> None:
    """Emit both the request-context token record AND Prometheus metrics."""
    record_token_usage(
        provider="openai", model=model_id, input_tokens=i, output_tokens=o
    )
    try:
        from sql_agent.observability.metrics import get_metrics

        get_metrics().record_llm_call(
            "openai", model_id, latency_ms, input_tokens=i, output_tokens=o
        )
    except Exception:  # pragma: no cover
        pass


class _StructuredInvokerProxy:
    """Mirrors LangChain's RunnableSerializable[...].invoke(...) signature but
    reports token usage + latency before returning the parsed model."""

    def __init__(self, raw_invoker, parsed_cls, model_id: str) -> None:
        self._raw = raw_invoker
        self._parsed_cls = parsed_cls
        self._model_id = model_id

    def invoke(self, messages):
        import time as _time

        t0 = _time.perf_counter()
        result = self._raw.invoke(messages)
        latency_ms = (_time.perf_counter() - t0) * 1000.0
        # include_raw=True returns a dict; otherwise a raw Pydantic model.
        if isinstance(result, dict):
            raw = result.get("raw")
            parsed = result.get("parsed")
            i, o = (_usage_from_ai_message(raw) if raw is not None else (0, 0))
            _record_call(self._model_id, latency_ms, i, o)
            return parsed
        # No usage info available; still record latency.
        _record_call(self._model_id, latency_ms, 0, 0)
        return result


class _ChatOpenAIProxy:
    """Proxy around langchain_openai.ChatOpenAI that reports token usage + latency."""

    def __init__(self, inner, model_id: str) -> None:
        self._inner = inner
        self._model_id = model_id

    @property
    def temperature(self) -> float:
        return float(getattr(self._inner, "temperature", 0.0) or 0.0)

    def with_structured_output(self, model_cls):
        # include_raw=True lets us see AIMessage.usage_metadata.
        raw_invoker = self._inner.with_structured_output(
            model_cls, include_raw=True
        )
        return _StructuredInvokerProxy(raw_invoker, model_cls, self._model_id)

    def invoke(self, messages):
        import time as _time

        t0 = _time.perf_counter()
        result = self._inner.invoke(messages)
        latency_ms = (_time.perf_counter() - t0) * 1000.0
        i, o = _usage_from_ai_message(result)
        _record_call(self._model_id, latency_ms, i, o)
        return result

    def __getattr__(self, item):
        # Pass-through for any other attribute access (e.g. `.model_name`).
        return getattr(self._inner, item)


class OpenAIProvider:
    """LLMProvider backed by OpenAI via langchain_openai.

    Matches the structural ``LLMProvider`` protocol (no inheritance required).
    """

    name: str = "openai"
    device: str = "remote"

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or settings.openai_chat_model
        self._api_key = api_key if api_key is not None else settings.openai_api_key
        self._cache: dict = {}

        if not self._api_key:
            # Deferred failure: some code paths (mock, tests) may never call
            # ``chat_model``. Only raise when actually attempting to construct.
            _log.debug("OpenAIProvider constructed without an API key")

    # ------------------------------------------------------------------
    def chat_model(self, temperature: float = 0.0):  # -> ChatOpenAI
        cache_key = ("chat", round(float(temperature), 3))
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self._api_key:
            raise ProviderUnavailableError(
                "OpenAIProvider requires OPENAI_API_KEY to be set."
            )

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover -- langchain_openai is in core reqs
            raise ProviderUnavailableError(
                f"langchain_openai is not installed: {exc}"
            ) from exc

        chat = ChatOpenAI(
            model=self.model_id,
            api_key=self._api_key,
            temperature=temperature,
            timeout=60,
            max_retries=2,
        )
        # Wrap in a proxy that reports token usage — behavior is otherwise
        # byte-identical to returning the ChatOpenAI instance directly.
        wrapped = _ChatOpenAIProxy(chat, self.model_id)
        self._cache[cache_key] = wrapped
        _log.debug(
            "OpenAIProvider: built ChatOpenAI model=%s temperature=%s",
            self.model_id,
            temperature,
        )
        return wrapped
