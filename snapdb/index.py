"""
SnapDB Hash Index — O(1) lookups on any column

Auto-updates on insert/update/delete.
"""
from __future__ import annotations

import bisect
from typing import Any, Dict, List, Optional, Set, Tuple


class HashIndex:
    """In-memory hash index for a single column.

    Usage:
        idx = db.create_index("email")
        rows = idx.lookup("alice@example.com")  # list of row indices
    """

    def __init__(self, column: str) -> None:
        self.column = column
        # Map value -> set of row indices (handles duplicates)
        self._index: Dict[Any, Set[int]] = {}
        self._total = 0

    def _key(self, value: Any) -> Any:
        """Normalize key for hashing (handle bytes, etc)."""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    def insert(self, row_idx: int, row: Dict[str, Any]) -> None:
        """Add a row to the index."""
        key = self._key(row.get(self.column))
        if key not in self._index:
            self._index[key] = set()
        self._index[key].add(row_idx)
        self._total += 1

    def update(self, row_idx: int, old_row: Dict[str, Any], new_row: Dict[str, Any]) -> None:
        """Update index for a changed row."""
        old_key = self._key(old_row.get(self.column))
        new_key = self._key(new_row.get(self.column))
        if old_key == new_key:
            return  # No change
        # Remove from old
        if old_key in self._index:
            self._index[old_key].discard(row_idx)
            if not self._index[old_key]:
                del self._index[old_key]
        # Add to new
        if new_key not in self._index:
            self._index[new_key] = set()
        self._index[new_key].add(row_idx)

    def delete(self, row_idx: int, row: Dict[str, Any]) -> None:
        """Remove a row from the index."""
        key = self._key(row.get(self.column))
        if key in self._index:
            self._index[key].discard(row_idx)
            if not self._index[key]:
                del self._index[key]
        self._total -= 1

    def lookup(self, value: Any) -> List[int]:
        """Find all row indices matching value. Returns empty list if none."""
        key = self._key(value)
        return sorted(self._index.get(key, set()))

    def has(self, value: Any) -> bool:
        """Check if any row has this value."""
        return self._key(value) in self._index

    def __len__(self) -> int:
        return self._total

    def __repr__(self) -> str:
        return f"HashIndex({self.column!r}, {len(self._index)} keys, {self._total} entries)"


class MultiIndex:
    """Manages multiple hash indexes on a SnapDB."""

    def __init__(self) -> None:
        self._indexes: Dict[str, HashIndex] = {}

    def create(self, column: str) -> HashIndex:
        """Create a new index on a column."""
        if column in self._indexes:
            raise ValueError(f"Index already exists on {column}")
        idx = HashIndex(column)
        self._indexes[column] = idx
        return idx

    def get(self, column: str) -> Optional[HashIndex]:
        return self._indexes.get(column)

    def drop(self, column: str) -> None:
        del self._indexes[column]

    def lookup(self, **kwargs) -> Optional[List[int]]:
        """Lookup by one or more indexed columns.
        Returns intersection of matching row indices.
        """
        if not kwargs:
            return None

        results: Optional[Set[int]] = None
        for col, val in kwargs.items():
            idx = self._indexes.get(col)
            # Use an explicit None check: an existing-but-empty HashIndex is
            # falsy (len 0) and must NOT be mistaken for a missing index.
            if idx is None:
                raise KeyError(f"No index on column: {col}")
            matches = set(idx.lookup(val))
            if results is None:
                results = matches
            else:
                results &= matches
            if not results:
                return []

        return sorted(results) if results else []

    def __contains__(self, column: str) -> bool:
        return column in self._indexes

    def __repr__(self) -> str:
        return f"MultiIndex({list(self._indexes.keys())})"


class RangeIndex:
    """Sorted in-memory range index for ordered scalar columns."""

    def __init__(self, column: str) -> None:
        self.column = column
        self._items: List[Tuple[Any, int]] = []

    def insert(self, row_idx: int, row: Dict[str, Any]) -> None:
        value = row.get(self.column)
        if value is None:
            return
        bisect.insort(self._items, (value, row_idx))

    def delete(self, row_idx: int, row: Dict[str, Any]) -> None:
        value = row.get(self.column)
        if value is None:
            return
        pos = bisect.bisect_left(self._items, (value, row_idx))
        if pos < len(self._items) and self._items[pos] == (value, row_idx):
            self._items.pop(pos)

    def update(self, row_idx: int, old_row: Dict[str, Any], new_row: Dict[str, Any]) -> None:
        old_value = old_row.get(self.column)
        new_value = new_row.get(self.column)
        if old_value == new_value:
            return
        self.delete(row_idx, old_row)
        self.insert(row_idx, new_row)

    def range_lookup(self, low: Any = None, high: Any = None,
                     include_low: bool = True, include_high: bool = True) -> List[int]:
        start_key = (low, -1) if low is not None else None
        end_key = (high, float("inf")) if high is not None else None

        start = 0
        if start_key is not None:
            start = (bisect.bisect_left if include_low else bisect.bisect_right)(
                self._items, start_key)

        end = len(self._items)
        if end_key is not None:
            end = (bisect.bisect_right if include_high else bisect.bisect_left)(
                self._items, end_key)

        return [row_idx for _, row_idx in self._items[start:end]]

    def __len__(self) -> int:
        return len(self._items)
