"""
SnapDB Query Engine — SQL-like filtering, sorting, and pagination
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from .core import SnapDB


# ── Operators ──────────────────────────────────────────────────────────────────

_OPS = {
    "eq": operator.eq, "ne": operator.ne,
    "gt": operator.gt, "gte": operator.ge,
    "lt": operator.lt, "lte": operator.le,
}


def _compile_filter(conditions: Dict[str, Any]) -> Callable:
    """Compile a dict of column conditions into a predicate.

    Examples:
        {"age": 25}              → age == 25
        {"age": {"gt": 25}}      → age > 25
        {"age": {"gte": 25, "lt": 65}} → age >= 25 and age < 65
    """
    checks = []
    for col, spec in conditions.items():
        if isinstance(spec, dict):
            # Range/comparison: {"gt": 10, "lt": 100}
            for op_name, val in spec.items():
                op_fn = _OPS[op_name]
                checks.append(lambda r, c=col, v=val, op=op_fn: op(r.get(c), v))
        else:
            # Exact match
            checks.append(lambda r, c=col, v=spec: r.get(c) == v)

    def predicate(row: Dict[str, Any]) -> bool:
        return all(check(row) for check in checks)

    return predicate


# ── Query Builder ──────────────────────────────────────────────────────────────

@dataclass
class Query:
    """Immutable query specification."""
    db: SnapDB
    where: Optional[Callable[[Dict[str, Any]], bool]] = None
    order_by: Optional[Tuple[str, bool]] = None  # (column, desc)
    limit: Optional[int] = None
    offset: int = 0

    def filter(self, **conditions) -> "Query":
        """Add WHERE conditions."""
        pred = _compile_filter(conditions)
        new_where = pred if self.where is None else lambda r: self.where(r) and pred(r)
        return Query(self.db, new_where, self.order_by, self.limit, self.offset)

    def where_fn(self, fn: Callable[[Dict[str, Any]], bool]) -> "Query":
        """Add custom WHERE predicate."""
        new_where = fn if self.where is None else lambda r: self.where(r) and fn(r)
        return Query(self.db, new_where, self.order_by, self.limit, self.offset)

    def order(self, column: str, desc: bool = False) -> "Query":
        """Add ORDER BY."""
        return Query(self.db, self.where, (column, desc), self.limit, self.offset)

    def slice(self, limit: int, offset: int = 0) -> "Query":
        """Add LIMIT/OFFSET."""
        return Query(self.db, self.where, self.order_by, limit, offset)

    def execute(self) -> List[Tuple[int, Dict[str, Any]]]:
        """Run the query and return (idx, row) results."""
        # 1. Filter
        if self.where:
            rows = [(idx, row) for idx, row in self.db if self.where(row)]
        else:
            rows = list(self.db)

        # 2. Sort
        if self.order_by:
            col, desc = self.order_by
            rows.sort(key=lambda x: x[1].get(col, 0), reverse=desc)

        # 3. Slice
        start = self.offset
        end = start + self.limit if self.limit else len(rows)
        return rows[start:end]

    def first(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Return first match or None."""
        results = self.slice(1).execute()
        return results[0] if results else None

    def count(self) -> int:
        """Return count of matching rows."""
        if self.where:
            return sum(1 for _, row in self.db if self.where(row))
        return len(self.db)

    def __iter__(self):
        yield from self.execute()


# ── Convenience ──────────────────────────────────────────────────────────────────

def query(db: SnapDB) -> Query:
    """Start a new query on a SnapDB instance."""
    return Query(db)
