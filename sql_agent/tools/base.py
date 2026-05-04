"""BaseTool abstract class enforcing the Pydantic IO contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

from pydantic import BaseModel

from sql_agent.config import get_logger


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ToolExecutionError(Exception):
    """Raised by tools on predictable, user-facing failures."""


class BaseTool(ABC, Generic[InputT, OutputT]):
    """Every tool declares its pydantic input / output types and a ``run``
    method. No tool is allowed to emit free-form SQL; DB access flows through
    SQLAlchemy expression language inside concrete subclasses."""

    name: str = "base_tool"
    description: str = ""

    #: pydantic class for the input payload
    input_schema: Type[BaseModel]
    #: pydantic class for the output payload
    output_schema: Type[BaseModel]

    def __init__(self) -> None:
        self._log = get_logger(f"tool.{self.name}")

    @abstractmethod
    def _execute(self, payload: InputT) -> OutputT:
        """Concrete per-tool logic. Receives validated input, returns output."""

    def run(self, payload: InputT | dict) -> OutputT:
        """Validate input, execute, and log at DEBUG level."""
        if isinstance(payload, dict):
            payload = self.input_schema(**payload)  # type: ignore[assignment]
        if not isinstance(payload, self.input_schema):
            raise ToolExecutionError(
                f"{self.name}: expected {self.input_schema.__name__}, got {type(payload).__name__}"
            )
        self._log.debug("run input: %s", payload.model_dump())
        try:
            result = self._execute(payload)  # type: ignore[arg-type]
        except ToolExecutionError:
            raise
        except Exception as exc:
            self._log.exception("%s failed", self.name)
            raise ToolExecutionError(f"{self.name}: {exc}") from exc
        if not isinstance(result, self.output_schema):
            raise ToolExecutionError(
                f"{self.name}: returned {type(result).__name__}, "
                f"expected {self.output_schema.__name__}"
            )
        return result  # type: ignore[return-value]
