"""SchemaDiscoveryTool — inspects the DB and returns a typed SchemaInfo."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel
from sqlalchemy import Engine, inspect
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.types import (
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    Numeric,
    String,
    Time,
)

from sql_agent.models import ColumnSchema, ColumnType, SchemaInfo, TableSchema
from sql_agent.services.db import get_engine
from sql_agent.services.schema_cache import schema_cache

from .base import BaseTool, ToolExecutionError


class SchemaDiscoveryInput(BaseModel):
    force_refresh: bool = False


class SchemaDiscoveryOutput(BaseModel):
    # Named ``db_schema`` internally (the ``schema`` name shadows a
    # deprecated BaseModel attribute in pydantic 2.x). Phase-8 rename;
    # all internal callers have been updated.
    db_schema: SchemaInfo


def _map_type(sa_type) -> ColumnType:  # type: ignore[no-untyped-def]
    try:
        if isinstance(sa_type, Boolean):
            return ColumnType.BOOLEAN
        if isinstance(sa_type, (DateTime, Date, Time)):
            return ColumnType.DATETIME
        if isinstance(sa_type, Integer):
            return ColumnType.INTEGER
        if isinstance(sa_type, (Float, Numeric)):
            return ColumnType.FLOAT
        if isinstance(sa_type, String):
            return ColumnType.STRING
    except Exception:
        pass
    # Fallback: inspect python_type if SQLAlchemy provides it.
    try:
        pt = sa_type.python_type  # type: ignore[attr-defined]
        if pt is bool:
            return ColumnType.BOOLEAN
        if pt is int:
            return ColumnType.INTEGER
        if pt is float:
            return ColumnType.FLOAT
        if pt is str:
            return ColumnType.STRING
    except Exception:
        pass
    return ColumnType.OTHER


class SchemaDiscoveryTool(BaseTool[SchemaDiscoveryInput, SchemaDiscoveryOutput]):
    """Reflect and cache DB schema using SQLAlchemy's inspector."""

    name = "schema_discovery"
    description = "Return tables/columns/types/PKs/FKs for the configured DB."
    input_schema = SchemaDiscoveryInput
    output_schema = SchemaDiscoveryOutput

    def __init__(self, engine: Optional[Engine] = None) -> None:
        super().__init__()
        self._engine = engine or get_engine()
        self._cache_key = str(self._engine.url)

    def _execute(self, payload: SchemaDiscoveryInput) -> SchemaDiscoveryOutput:
        if not payload.force_refresh:
            cached = schema_cache.get(self._cache_key)
            if cached is not None:
                self._log.debug("schema cache hit")
                return SchemaDiscoveryOutput(db_schema=cached)

        try:
            inspector: Inspector = inspect(self._engine)
            schema = self._reflect(inspector)
        except Exception as exc:
            raise ToolExecutionError(
                f"Could not reflect database schema: {exc}"
            ) from exc

        if not schema.tables:
            raise ToolExecutionError(
                "Database is reachable but has no tables. "
                "Seed a DB first (see seed_demo.py)."
            )

        schema_cache.set(self._cache_key, schema)
        return SchemaDiscoveryOutput(db_schema=schema)

    def _reflect(self, inspector: Inspector) -> SchemaInfo:
        tables: Dict[str, TableSchema] = {}
        for table_name in inspector.get_table_names():
            cols = inspector.get_columns(table_name)
            pk_cols = set(
                inspector.get_pk_constraint(table_name).get("constrained_columns") or []
            )
            fk_map: Dict[str, str] = {}
            for fk in inspector.get_foreign_keys(table_name):
                ref_table = fk.get("referred_table")
                ref_cols = fk.get("referred_columns") or []
                cons_cols = fk.get("constrained_columns") or []
                for local, remote in zip(cons_cols, ref_cols):
                    if ref_table and remote:
                        fk_map[local] = f"{ref_table}.{remote}"

            col_map: Dict[str, ColumnSchema] = {}
            for col in cols:
                col_map[col["name"]] = ColumnSchema(
                    type=_map_type(col["type"]),
                    nullable=bool(col.get("nullable", True)),
                    primary_key=col["name"] in pk_cols,
                    foreign_key=fk_map.get(col["name"]),
                )
            tables[table_name] = TableSchema(columns=col_map)
        return SchemaInfo(tables=tables)
