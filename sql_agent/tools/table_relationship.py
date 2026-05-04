"""TableRelationshipTool — infer FK relationships + shortest join paths."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from sql_agent.models import SchemaInfo

from .base import BaseTool, ToolExecutionError


class JoinEdge(BaseModel):
    left_table: str
    left_column: str
    right_table: str
    right_column: str


class TableRelationshipInput(BaseModel):
    db_schema: SchemaInfo
    tables: List[str] = Field(
        default_factory=list,
        description="If empty, returns *all* relationships; else returns a join "
        "path connecting exactly these tables (order-independent).",
    )


class TableRelationshipOutput(BaseModel):
    edges: List[JoinEdge]
    join_path: List[JoinEdge] = Field(
        default_factory=list,
        description="Ordered edges linking the requested tables (if any).",
    )


class TableRelationshipTool(
    BaseTool[TableRelationshipInput, TableRelationshipOutput]
):
    name = "table_relationship"
    description = "Derive FK edges and shortest join paths from the schema."
    input_schema = TableRelationshipInput
    output_schema = TableRelationshipOutput

    def _execute(
        self, payload: TableRelationshipInput
    ) -> TableRelationshipOutput:
        edges = self._collect_edges(payload.db_schema)

        join_path: List[JoinEdge] = []
        if len(payload.tables) >= 2:
            for t in payload.tables:
                if not payload.db_schema.has_table(t):
                    raise ToolExecutionError(
                        f"Unknown table '{t}' — cannot build join path."
                    )
            join_path = self._connect(edges, payload.tables)

        return TableRelationshipOutput(edges=edges, join_path=join_path)

    @staticmethod
    def _collect_edges(schema: SchemaInfo) -> List[JoinEdge]:
        edges: List[JoinEdge] = []
        for table_name, tbl in schema.tables.items():
            for col_name, col in tbl.columns.items():
                if col.foreign_key:
                    try:
                        ref_table, ref_col = col.foreign_key.split(".", 1)
                    except ValueError:
                        continue
                    edges.append(
                        JoinEdge(
                            left_table=table_name,
                            left_column=col_name,
                            right_table=ref_table,
                            right_column=ref_col,
                        )
                    )
        return edges

    @staticmethod
    def _connect(edges: List[JoinEdge], tables: List[str]) -> List[JoinEdge]:
        # Build undirected adjacency keyed by table names.
        adj: Dict[str, List[Tuple[str, JoinEdge]]] = {}
        for e in edges:
            adj.setdefault(e.left_table, []).append((e.right_table, e))
            adj.setdefault(e.right_table, []).append((e.left_table, e))

        visited_globally: Set[str] = {tables[0]}
        ordered: List[JoinEdge] = []

        for target in tables[1:]:
            if target in visited_globally:
                continue
            path = TableRelationshipTool._bfs(adj, visited_globally, target)
            if path is None:
                raise ToolExecutionError(
                    f"No FK path found to join '{target}' with "
                    f"{sorted(visited_globally)}"
                )
            for edge in path:
                ordered.append(edge)
                visited_globally.update({edge.left_table, edge.right_table})
        return ordered

    @staticmethod
    def _bfs(
        adj: Dict[str, List[Tuple[str, JoinEdge]]],
        start_set: Set[str],
        target: str,
    ) -> Optional[List[JoinEdge]]:
        """BFS from *any* node in ``start_set`` to ``target``."""
        queue: deque = deque()
        parents: Dict[str, Tuple[str, JoinEdge]] = {}
        seen: Set[str] = set(start_set)
        for s in start_set:
            queue.append(s)

        while queue:
            node = queue.popleft()
            if node == target:
                # Reconstruct.
                path: List[JoinEdge] = []
                cur = node
                while cur in parents:
                    prev, edge = parents[cur]
                    path.append(edge)
                    cur = prev
                return list(reversed(path))
            for neighbor, edge in adj.get(node, []):
                if neighbor not in seen:
                    seen.add(neighbor)
                    parents[neighbor] = (node, edge)
                    queue.append(neighbor)
        return None
