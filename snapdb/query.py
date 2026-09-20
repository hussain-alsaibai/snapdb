"""
SnapDB Query Engine — SQL-like filtering, sorting, and pagination
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Tuple

from .core import SnapDB


# ── Operators ──────────────────────────────────────────────────────────────────

_OPS = {
    "eq": operator.eq, "ne": operator.ne,
    "gt": operator.gt, "gte": operator.ge,
    "lt": operator.lt, "lte": operator.le,
}

_OP_SYMBOLS = {
    "eq": "==", "ne": "!=",
    "gt": ">", "gte": ">=",
    "lt": "<", "lte": "<=",
}


def _compile_filter(conditions: Dict[str, Any]) -> Callable:
    """Compile a dict of column conditions into a predicate.

    Examples:
        {"age": 25}              → age == 25
        {"age": {"gt": 25}}      → age > 25
        {"age": {"gte": 25, "lt": 65}} → age >= 25 and age < 65

    Generates and compiles a single boolean expression instead of chaining N
    per-condition closures through all() — filter() runs this once and
    execute() then applies it to every row in the scan, so collapsing N
    Python function calls (closure + operator.*) per row per condition into
    one evaluated expression pays for its own compile() cost on any table
    past a handful of rows.

    Column names and comparison values are never interpolated into the
    generated source (only their generated local-variable names are) — they
    reach the expression purely through the exec() namespace, so arbitrary
    (non-literal-safe) values are fine and there is no injection surface.
    """
    exprs = []
    ns: Dict[str, Any] = {}
    n = 0
    for col, spec in conditions.items():
        col_key = f"_c{n}"
        ns[col_key] = col
        if isinstance(spec, dict):
            # Range/comparison: {"gt": 10, "lt": 100}
            for op_name, val in spec.items():
                sym = _OP_SYMBOLS[op_name]  # KeyError on an unknown op, same as before
                val_key = f"_v{n}"
                ns[val_key] = val
                exprs.append(f"(r.get({col_key}) {sym} {val_key})")
                n += 1
        else:
            # Exact match
            val_key = f"_v{n}"
            ns[val_key] = spec
            exprs.append(f"(r.get({col_key}) == {val_key})")
            n += 1

    if not exprs:
        return lambda r: True

    src = "def _predicate(r):\n    return " + " and ".join(exprs)
    exec(compile(src, "<snapdb-query-filter>", "exec"), ns)
    return ns["_predicate"]


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
        if self.limit is not None and self.limit <= 0:
            return []

        # 1. Filter (lazily — only materialized when sorting requires it)
        if self.where:
            matches = ((idx, row) for idx, row in self.db if self.where(row))
        else:
            matches = iter(self.db)

        # 2a. Sort: needs every match, then slice
        if self.order_by:
            rows = list(matches)
            col, desc = self.order_by
            rows.sort(key=lambda x: x[1].get(col, 0), reverse=desc)
            end = self.offset + self.limit if self.limit is not None else None
            return rows[self.offset:end]

        # 2b. No sort: stop scanning as soon as offset+limit matches are seen,
        # so first()/small-limit queries don't decode the whole table.
        stop = self.offset + self.limit if self.limit is not None else None
        return list(islice(matches, self.offset, stop))

    def first(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Return first match or None."""
        results = self.slice(1).execute()
        return results[0] if results else None

    def count(self) -> int:
        """Return count of matching rows."""
        if self.where:
            return sum(1 for _, row in self.db if self.where(row))
        return len(self.db)

    # ── Aggregations (v0.16.1) ──────────────────────────────────────────────
    #
    # All aggregations stream rows through the same WHERE predicate the
    # rest of the query uses, so they compose with .filter() / .where_fn()
    # exactly like count() does. Missing keys are skipped (None / absent
    # columns are not counted as zero, which matches SQL NULL semantics and
    # avoids the common footgun of averaging over a sentinel value).

    def _iter_matching_values(self, column: str):
        """Yield non-None values of `column` for rows that match WHERE."""
        if self.where:
            for _, row in self.db:
                if self.where(row):
                    v = row.get(column)
                    if v is not None:
                        yield v
        else:
            for _, row in self.db:
                v = row.get(column)
                if v is not None:
                    yield v

    def sum(self, column: str, default: Any = 0) -> Any:
        """Sum of `column` over matching rows.

        Returns `default` (default 0) when there are no rows. Non-numeric
        values raise TypeError so callers don't silently accumulate strings.
        """
        total = default
        for v in self._iter_matching_values(column):
            total = total + v
        return total

    def min(self, column: str, default: Any = None) -> Any:
        """Minimum of `column` over matching rows, or `default` if empty."""
        it = self._iter_matching_values(column)
        try:
            first = next(it)
        except StopIteration:
            return default
        return min(first, *(v for v in it))

    def max(self, column: str, default: Any = None) -> Any:
        """Maximum of `column` over matching rows, or `default` if empty."""
        it = self._iter_matching_values(column)
        try:
            first = next(it)
        except StopIteration:
            return default
        return max(first, *(v for v in it))

    def avg(self, column: str, default: Any = None) -> Optional[float]:
        """Arithmetic mean of `column` over matching rows.

        Returns `default` (None) when there are no rows. Numerator and
        denominator are accumulated in the same loop so we don't iterate
        twice; intermediate `total` is held in a Python int/float so this
        is exact for ints and IEEE-754 for floats, matching numpy.mean.
        """
        total = 0
        n = 0
        for v in self._iter_matching_values(column):
            total = total + v
            n += 1
        if n == 0:
            return default
        return total / n

    def count_distinct(self, column: str) -> int:
        """Count of distinct values of `column` over matching rows.

        Uses an insertion-ordered dict to dedupe without an extra hash
        table; values must be hashable (same constraint as a dict key).
        Order is preserved for stable iteration downstream.
        """
        seen: Dict[Any, None] = {}
        for v in self._iter_matching_values(column):
            seen.setdefault(v, None)
        return len(seen)

    def __iter__(self):
        yield from self.execute()


# ── Convenience ──────────────────────────────────────────────────────────────────

def query(db: SnapDB) -> Query:
    """Start a new query on a SnapDB instance."""
    return Query(db)
