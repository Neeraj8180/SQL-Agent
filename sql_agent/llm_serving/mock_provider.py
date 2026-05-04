"""Deterministic mock provider — for tests, demos, and fully-offline runs.

Factories are registered per structured-output schema class name. If an
agent asks for a schema that has not been registered, we raise so the test
surfaces the gap immediately.

The built-in defaults cover the two schemas the orchestrator uses:
    - ``Intent``     (from sql_agent.models.intent)
    - ``ParamPlan``  (local class inside sql_agent.agents.param_builder_agent)

Callers can override or extend via ``MockProvider.register(name, factory)``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Type

from pydantic import BaseModel

from sql_agent.config import get_logger


_log = get_logger("llm_serving.mock")


# ---------------------------------------------------------------------------
# Default canned responses (sufficient to run the demo pipeline end-to-end).
# ---------------------------------------------------------------------------


def _default_intent(cls: Type[BaseModel]) -> BaseModel:
    return cls(
        metrics=["order_count"],
        dimensions=[],
        filters=[],
        time_range=None,
        output_type="count",
        visualize=False,
        notes="count of orders",
    )


def _default_param_plan(cls: Type[BaseModel]) -> BaseModel:
    # Build lazily so ``from sql_agent.llm_serving ...`` doesn't force an
    # import of the FetchParams pydantic model for users who don't need it.
    from sql_agent.models import FetchParams

    return cls(
        reasoning="(mock) Count all rows in 'orders'.",
        params=FetchParams(
            table_names=["orders"],
            aggregations=[{"func": "count", "column": "*", "alias": "count"}],
            limit=100,
        ),
    )


_DEFAULT_FACTORIES: Dict[str, Callable[[Type[BaseModel]], BaseModel]] = {
    "Intent": _default_intent,
    "ParamPlan": _default_param_plan,
}


# ---------------------------------------------------------------------------
# Shim chat model — matches ChatModel protocol.
# ---------------------------------------------------------------------------


class _MockResp:
    def __init__(self, content: str = "(mock summary)") -> None:
        self.content = content


class _MockStructuredInvoker:
    def __init__(
        self, model_cls: Type[BaseModel], factories: Dict[str, Callable]
    ) -> None:
        self._model_cls = model_cls
        self._factories = factories

    def invoke(self, _messages: List[Any]) -> BaseModel:
        fn = self._factories.get(self._model_cls.__name__)
        if fn is None:
            raise NotImplementedError(
                f"MockProvider: no factory registered for "
                f"'{self._model_cls.__name__}'. Call "
                f"MockProvider.register('{self._model_cls.__name__}', fn)."
            )
        return fn(self._model_cls)


class _MockChatModel:
    temperature: float = 0.0

    def __init__(self, factories: Dict[str, Callable]) -> None:
        self._factories = factories

    def with_structured_output(self, model_cls: Type[BaseModel]):
        return _MockStructuredInvoker(model_cls, self._factories)

    def invoke(self, _messages: List[Any]):
        return _MockResp()


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class MockProvider:
    name: str = "mock"
    model_id: str = "mock"
    device: str = "cpu"

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[Type[BaseModel]], BaseModel]] = dict(
            _DEFAULT_FACTORIES
        )
        _log.info(
            "MockProvider initialized (%d built-in factories)",
            len(self._factories),
        )

    def register(
        self, schema_name: str, factory: Callable[[Type[BaseModel]], BaseModel]
    ) -> None:
        """Register / override a factory for a structured-output schema."""
        self._factories[schema_name] = factory

    def chat_model(self, temperature: float = 0.0):  # -> ChatModel
        return _MockChatModel(self._factories)
