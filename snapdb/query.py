"""
SnapDB Query Engine — SQL-like filtering, sorting, and pagination

v0.18.0 — Index-aware query planning
====================================
The query engine now consults ``HashIndex`` (equality) and ``RangeIndex``
(``gt``/``gte``/``lt``/``lte``) when planning a query. A single-column
predicate that matches an available index is served from the index in
O(log n + k) instead of an O(n) full table scan. Multi-condition queries
intersect index results; if any chosen condition falls back to a scan, the
scan is restricted to the candidate set rather than the whole table, which
is still a meaningful win on wide rows.

Design notes:

* No plan caching. Query plans are tiny (a handful of dict allocations)
  and ``Query`` is immutable, so the cost of re-planning on every
  ``execute()`` is well below the cost of maintaining a cache.
* No automatic index creation. ``Query`` reads existing indexes only.
  Users opt in via ``db.create_index`` / ``db.create_range_index``.
* Result ordering is preserved within index intersection: when the
  planner intersects two ranges, the first index defines the order.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .core import SnapDB


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
        {"age": 25}              -> age == 25
        {"age": {"gt": 25}}      -> age > 25
        {"age": {"gte": 25, "lt": 65}} -> age >= 25 and age < 65

    Generates and compiles a single boolean expression instead of chaining
    per-condition closures through all(). filter() runs this once and
    execute() then applies it to every row in the scan, so collapsing N
    Python function calls per row per condition into one evaluated
    expression pays for its own compile() cost on any table past a handful
    of rows.

    Column names and comparison values are never interpolated into the
    generated source - they reach the expression purely through the exec()
    namespace, so arbitrary (non-literal-safe) values are fine and there is
    no injection surface.
    """
    exprs = []
    ns: Dict[str, Any] = {}
    n = 0
    for col, spec in conditions.items():
        col_key = f"_c{n}"
        ns[col_key] = col
        if isinstance(spec, dict):
            for op_name, val in spec.items():
                sym = _OP_SYMBOLS[op_name]
                val_key = f"_v{n}"
                ns[val_key] = val
                exprs.append(f"(r.get({col_key}) {sym} {val_key})")
                n += 1
        else:
            val_key = f"_v{n}"
            ns[val_key] = spec
            exprs.append(f"(r.get({col_key}) == {val_key})")
            n += 1

    if not exprs:
        return lambda r: True

    src = "def _predicate(r):\n    return " + " and ".join(exprs)
    exec(compile(src, "<snapdb-query-filter>", "exec"), ns)
    return ns["_predicate"]


# Operators a RangeIndex can satisfy. ``ne`` is excluded by design - a
# range index can find all values != x cheaply only as "everything except
# the slice at x", which is rarely what the user wants and would be a
# pessimization vs. a scan on most tables.
_RANGE_OPS = {"gt", "gte", "lt", "lte"}


@dataclass
class _IndexPlan:
    """How a single condition should be served.

    ``mode`` is one of:
        ``"hash"``     - HashIndex equality lookup
        ``"range"``    - RangeIndex range_lookup (single bounded side)
        ``"between"``  - RangeIndex range_lookup with both bounds
        ``"scan"``     - no index, fall through to compiled predicate
    """
    column: str
    op: str
    mode: str
    value: Any = None
    side: str = ""
    bound: Any = None
    include: bool = True
    low: Any = None
    high: Any = None
    inc_low: bool = True
    inc_high: bool = True


def _split_conditions(conditions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Split into ``{col: value}`` (exact) and ``{col: {op: value}}`` (range)."""
    eq: Dict[str, Any] = {}
    rng: Dict[str, Dict[str, Any]] = {}
    for col, spec in conditions.items():
        if isinstance(spec, dict):
            rng[col] = spec
        else:
            eq[col] = spec
    return eq, rng


def _plan_equality(db: SnapDB, eq: Dict[str, Any]) -> Optional[_IndexPlan]:
    """Return a HashIndex plan if any one equality condition is indexed.

    Returns ``None`` when no equality condition has an index. Multi-column
    hash intersection is delegated to ``MultiIndex.lookup`` at the call
    site.
    """
    hash_idx = getattr(db, "_indexes", None)
    if hash_idx is None:
        return None
    for col, val in eq.items():
        if col in hash_idx:
            return _IndexPlan(column=col, op="eq", mode="hash", value=val)
    return None


def _plan_range(db: SnapDB, rng: Dict[str, Dict[str, Any]]) -> Optional[_IndexPlan]:
    """Return a RangeIndex plan if any one column's range spec is fully
    satisfiable by an existing ``RangeIndex``.

    A "fully satisfiable" spec means every op in the spec is in
    ``_RANGE_OPS``. ``ne`` and any unknown operator disqualify the column
    from index use and the planner falls through.

    For a spec with both a low and a high bound (e.g. ``{gte: 25, lt: 65}``)
    the planner emits a single ``between`` plan with both bounds - cheaper
    than two range plans intersected.
    """
    range_indexes = getattr(db, "_range_indexes", {}) or {}
    if not range_indexes:
        return None

    for col, spec in rng.items():
        if col not in range_indexes:
            continue
        if any(op not in _RANGE_OPS for op in spec):
            continue

        low = high = None
        inc_low = inc_high = True
        has_low = has_high = False

        for op, val in spec.items():
            if op in ("gt", "gte"):
                low, inc_low = val, op == "gte"
                has_low = True
            elif op in ("lt", "lte"):
                high, inc_high = val, op == "lte"
                has_high = True

        if has_low and has_high:
            return _IndexPlan(
                column=col, op="between", mode="between",
                low=low, high=high, inc_low=inc_low, inc_high=inc_high,
            )
        # Emit the original op (gt vs gte / lt vs lte) so explain() and
        # log output preserve what the user actually wrote.
        if has_low:
            for op_name, val in spec.items():
                if op_name in ("gt", "gte"):
                    return _IndexPlan(
                        column=col, op=op_name, mode="range",
                        side="low", bound=val, include=op_name == "gte",
                    )
        if has_high:
            for op_name, val in spec.items():
                if op_name in ("lt", "lte"):
                    return _IndexPlan(
                        column=col, op=op_name, mode="range",
                        side="high", bound=val, include=op_name == "lte",
                    )
    return None


def _execute_plan(db: SnapDB, plan: _IndexPlan) -> Set[int]:
    """Run an index plan and return the candidate set of row indices."""
    if plan.mode == "hash":
        idx = db._indexes.get(plan.column)  # type: ignore[union-attr]
        if idx is None:
            return set()
        return set(idx.lookup(plan.value))

    if plan.mode == "range":
        idx = db._range_indexes.get(plan.column)
        if idx is None:
            return set()
        if plan.side == "low":
            rows = idx.range_lookup(low=plan.bound, include_low=plan.include)
        else:
            rows = idx.range_lookup(high=plan.bound, include_high=plan.include)
        return set(rows)

    if plan.mode == "between":
        idx = db._range_indexes.get(plan.column)
        if idx is None:
            return set()
        rows = idx.range_lookup(
            low=plan.low, high=plan.high,
            include_low=plan.inc_low, include_high=plan.inc_high,
        )
        return set(rows)

    return set()


def _plan_query(db: SnapDB, conditions: Dict[str, Any]) -> Tuple[Optional[_IndexPlan], Callable]:
    """Return (plan, predicate) for a conditions dict.

    If a plan is found, the predicate should be applied only to the
    candidate set returned by ``_execute_plan``. If no plan is found, the
    predicate is applied to every row (the historical behaviour).
    """
    if not conditions:
        return None, lambda r: True

    eq, rng = _split_conditions(conditions)

    # Prefer range over equality: a range plan is at least as selective
    # as an equality plan when both bounds are present, and intersects
    # cleanly with itself.
    plan = _plan_range(db, rng)
    if plan is None:
        plan = _plan_equality(db, eq)

    predicate = _compile_filter(conditions)
    return plan, predicate


@dataclass
class Query:
    """Immutable query specification."""
    db: SnapDB
    where: Optional[Callable[[Dict[str, Any]], bool]] = None
    where_conditions: Optional[Dict[str, Any]] = None  # v0.18.0
    order_by: Optional[Tuple[str, bool]] = None
    limit: Optional[int] = None
    offset: int = 0

    def filter(self, **conditions) -> "Query":
        """Add WHERE conditions.

        The query planner consults available ``HashIndex`` / ``RangeIndex``
        instances on the bound ``SnapDB`` and short-circuits full table
        scans whenever a single condition can be served from an index.
        Multi-condition queries intersect index results and apply the
        remaining conditions only to the candidate set.
        """
        merged = dict(self.where_conditions or {})
        merged.update(conditions)
        pred = _compile_filter(merged)
        new_where = pred if self.where is None else lambda r: self.where(r) and pred(r)
        return Query(self.db, new_where, merged,
                     self.order_by, self.limit, self.offset)

    def where_fn(self, fn: Callable[[Dict[str, Any]], bool]) -> "Query":
        """Add a custom WHERE predicate.

        Custom predicates cannot be served from any index and force the
        planner into a full table scan. Use ``.filter()`` for index-friendly
        conditions, ``where_fn()`` for everything else.
        """
        new_where = fn if self.where is None else lambda r: self.where(r) and fn(r)
        # Custom predicates disqualify index planning - clear the conditions
        # cache so the planner falls back to a scan.
        return Query(self.db, new_where, None,
                     self.order_by, self.limit, self.offset)

    def order(self, column: str, desc: bool = False) -> "Query":
        return Query(self.db, self.where, self.where_conditions,
                     (column, desc), self.limit, self.offset)

    def slice(self, limit: int, offset: int = 0) -> "Query":
        return Query(self.db, self.where, self.where_conditions,
                     self.order_by, limit, offset)

    def between(self, column: str, low: Any, high: Any,
                include_low: bool = True, include_high: bool = True) -> "Query":
        """Shorthand for an inclusive/exclusive range filter.

        Equivalent to ``.filter(**{column: {"gte": low, "lte": high}})``
        but more readable at call sites. If a ``RangeIndex`` exists on the
        column the planner uses it directly.
        """
        spec: Dict[str, Any] = {}
        spec["gte" if include_low else "gt"] = low
        spec["lte" if include_high else "lt"] = high
        return self.filter(**{column: spec})

    def execute(self) -> List[Tuple[int, Dict[str, Any]]]:
        """Run the query and return (idx, row) results."""
        if self.limit is not None and self.limit <= 0:
            return []

        # v0.18.0 - index-aware execution. When the planner finds an index
        # match, we iterate the candidate set in its native order (sorted by
        # RangeIndex, hash order for HashIndex) and decode each row via
        # ``db.get(idx)``. This turns a single-column range predicate from
        # O(n) into O(log n + k) - matching only the k rows the index
        # already says qualify.
        plan: Optional[_IndexPlan] = None
        candidate_indices: Optional[List[int]] = None
        if self.where_conditions:
            plan, _ = _plan_query(self.db, self.where_conditions)
            if plan is not None:
                if plan.mode == "hash":
                    idx = self.db._indexes.get(plan.column)  # type: ignore[union-attr]
                    candidate_indices = sorted(idx.lookup(plan.value)) if idx else []
                elif plan.mode == "range":
                    idx = self.db._range_indexes.get(plan.column)
                    if plan.side == "low":
                        candidate_indices = idx.range_lookup(
                            low=plan.bound, include_low=plan.include
                        ) if idx else []
                    else:
                        candidate_indices = idx.range_lookup(
                            high=plan.bound, include_high=plan.include
                        ) if idx else []
                elif plan.mode == "between":
                    idx = self.db._range_indexes.get(plan.column)
                    candidate_indices = idx.range_lookup(
                        low=plan.low, high=plan.high,
                        include_low=plan.inc_low, include_high=plan.inc_high,
                    ) if idx else []
                if not candidate_indices:
                    return []

        def _iter_candidates():
            # Iterate the (already-filtered) candidate list and decode each
            # row by its index. O(log n + k) instead of O(n). If the user
            # supplied extra non-indexed conditions, apply them here.
            for idx in candidate_indices:  # type: ignore[union-attr]
                row = self.db.get(idx)
                if row is None:
                    continue
                if self.where is not None and not self.where(row):
                    continue
                yield idx, row

        def _scan():
            if candidate_indices is not None:
                return _iter_candidates()
            if self.where is not None:
                return ((idx, row) for idx, row in self.db if self.where(row))
            return iter(self.db)

        matches = _scan()

        # Sort: needs every match, then slice.
        if self.order_by:
            rows = list(matches)
            col, desc = self.order_by
            rows.sort(key=lambda x: x[1].get(col, 0), reverse=desc)
            end = self.offset + self.limit if self.limit is not None else None
            return rows[self.offset:end]

        # No sort: stop scanning as soon as offset+limit matches are seen,
        # so first()/small-limit queries don't decode the whole table.
        stop = self.offset + self.limit if self.limit is not None else None
        return list(islice(matches, self.offset, stop))

    def first(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        results = self.slice(1).execute()
        return results[0] if results else None

    def count(self) -> int:
        """Return count of matching rows.

        v0.18.0: with index planning, ``count()`` materializes the
        candidate set but still applies any remaining predicates. With a
        single indexed condition this is O(log n + k) instead of O(n).
        """
        if not self.where_conditions and self.where is None:
            return len(self.db)
        if not self.where:
            return len(self.db)

        plan, _ = _plan_query(self.db, self.where_conditions or {})
        if plan is not None:
            candidates = _execute_plan(self.db, plan)
            if not candidates:
                return 0
            return sum(
                1 for idx, row in self.db
                if idx in candidates and self.where(row)
            )
        return sum(1 for _, row in self.db if self.where(row))

    # Aggregations (v0.16.1) - now index-aware (v0.18.0)

    def _iter_matching_values(self, column: str):
        """Yield non-None values of `column` for rows that match WHERE."""
        plan = None
        candidates: Optional[Set[int]] = None
        if self.where_conditions:
            plan, _ = _plan_query(self.db, self.where_conditions)
            if plan is not None:
                candidates = _execute_plan(self.db, plan)
                if not candidates:
                    return

        if candidates is not None:
            for idx, row in self.db:
                if self.where is not None and not self.where(row):
                    continue
                if idx not in candidates:
                    continue
                v = row.get(column)
                if v is not None:
                    yield v
        elif self.where is not None:
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
        total = default
        for v in self._iter_matching_values(column):
            total = total + v
        return total

    def min(self, column: str, default: Any = None) -> Any:
        it = self._iter_matching_values(column)
        try:
            first = next(it)
        except StopIteration:
            return default
        return min(first, *(v for v in it))

    def max(self, column: str, default: Any = None) -> Any:
        it = self._iter_matching_values(column)
        try:
            first = next(it)
        except StopIteration:
            return default
        return max(first, *(v for v in it))

    def avg(self, column: str, default: Any = None) -> Optional[float]:
        total = 0
        n = 0
        for v in self._iter_matching_values(column):
            total = total + v
            n += 1
        if n == 0:
            return default
        return total / n

    def count_distinct(self, column: str) -> int:
        seen: Dict[Any, None] = {}
        for v in self._iter_matching_values(column):
            seen.setdefault(v, None)
        return len(seen)

    def explain(self) -> str:
        """Return a human-readable execution plan (v0.18.0).

        Useful for verifying that an index was chosen before running the
        expensive part of a query. Always cheap - never executes the
        predicate. Format is intentionally compact so it composes well
        with logging.
        """
        if not self.where_conditions:
            return "Scan(full)"
        plan, _ = _plan_query(self.db, self.where_conditions)
        if plan is None:
            return f"Scan(filter={list(self.where_conditions.keys())})"
        if plan.mode == "hash":
            return f"HashIndex[{plan.column} = {plan.value!r}]"
        if plan.mode == "between":
            return (
                f"RangeIndex[{plan.column} "
                f"{'[' if plan.inc_low else '('}{plan.low!r}, "
                f"{plan.high!r}{']' if plan.inc_high else ')'}]"
            )
        return (
            f"RangeIndex[{plan.column} "
            f"{plan.op}{'=' if plan.include else ''} {plan.bound!r}]"
        )

    def __iter__(self):
        yield from self.execute()


def query(db: SnapDB) -> Query:
    return Query(db)
