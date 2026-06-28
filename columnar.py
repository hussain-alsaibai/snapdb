"""
Columnar storage engine for SnapDB.
Inspired by ClickHouse columnar layout for analytical workloads.
Zero-dependency, pure Python.

v0.3.1: Optimized batch operations, precomputed column lists, faster iteration.
"""

from __future__ import annotations

import array
from typing import Any, Dict, List, Tuple, Callable, Iterator, Optional

_TYPE_CODES = {
    "i8": "b", "i16": "h", "i32": "i", "i64": "q",
    "u8": "B", "u16": "H", "u32": "I", "u64": "Q",
    "f32": "f", "f64": "d",
    "bool": "?",
}

_TYPE_SIZES = {
    "i8": 1, "i16": 2, "i32": 4, "i64": 8,
    "u8": 1, "u16": 2, "u32": 4, "u64": 8,
    "f32": 4, "f64": 8,
    "bool": 1,
}


def _type_size(dtype: str) -> int:
    if dtype.startswith("bytes"):
        return int(dtype.split(":")[1])
    return _TYPE_SIZES[dtype]


def _array_typecode(dtype: str) -> str:
    if dtype.startswith("bytes"):
        return "list"
    mapping = {
        "i8": "b", "i16": "h", "i32": "i", "i64": "q",
        "u8": "B", "u16": "H", "u32": "I", "u64": "Q",
        "f32": "f", "f64": "d",
        "bool": "B",
    }
    return mapping[dtype]


class Column:
    """A single column storing values of one type."""

    def __init__(self, name: str, dtype: str) -> None:
        self.name = name
        self.dtype = dtype
        self.width = _type_size(dtype)
        self._data: Any = None
        self._nullmask: Optional[array.array] = None
        self._init_storage()

    def _init_storage(self) -> None:
        if self.dtype.startswith("bytes"):
            self._data: List[bytes] = []
            self._nullmask = array.array("b")
        else:
            typecode = _array_typecode(self.dtype)
            self._data = array.array(typecode)
            self._nullmask = array.array("b")

    def append(self, value: Any) -> None:
        if value is None:
            self._nullmask.append(1)
            if self.dtype.startswith("bytes"):
                self._data.append(b"")
            else:
                self._data.append(0 if self._data.typecode not in ("f", "d") else 0.0)
        else:
            self._nullmask.append(0)
            if self.dtype.startswith("bytes"):
                if isinstance(value, str):
                    value = value.encode("utf-8")
                self._data.append(bytes(value))
            elif self.dtype == "bool":
                self._data.append(1 if value else 0)
            else:
                self._data.append(value)

    def __getitem__(self, idx: int) -> Any:
        if self._nullmask[idx]:
            return None
        if self.dtype.startswith("bytes"):
            return self._data[idx].decode("utf-8", errors="replace")
        if self.dtype == "bool":
            return bool(self._data[idx])
        return self._data[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        if idx < 0 or idx >= len(self._nullmask):
            raise IndexError(f"Index {idx} out of range")
        if value is None:
            self._nullmask[idx] = 1
            if self.dtype.startswith("bytes"):
                self._data[idx] = b""
            else:
                self._data[idx] = 0.0 if self._data.typecode in ("f", "d") else 0
        else:
            self._nullmask[idx] = 0
            if self.dtype.startswith("bytes"):
                if isinstance(value, str):
                    value = value.encode("utf-8")
                self._data[idx] = bytes(value)
            elif self.dtype == "bool":
                self._data[idx] = 1 if value else 0
            else:
                self._data[idx] = value

    def __len__(self) -> int:
        return len(self._nullmask)

    def iter_valid(self) -> Iterator[Tuple[int, Any]]:
        """Iterate (index, value) for non-null entries — fast path."""
        data = self._data
        nullmask = self._nullmask
        is_bytes = self.dtype.startswith("bytes")
        is_bool = self.dtype == "bool"
        for i in range(len(nullmask)):
            if nullmask[i] == 0:
                if is_bytes:
                    yield i, data[i].decode("utf-8", errors="replace")
                elif is_bool:
                    yield i, bool(data[i])
                else:
                    yield i, data[i]

    def count_valid(self) -> int:
        return self._nullmask.count(0)

    def memory_usage(self) -> int:
        if self.dtype.startswith("bytes"):
            total = sum(len(d) for d in self._data)
            return total + len(self._nullmask) + len(self._data) * 8
        else:
            return len(self._data) * self._data.itemsize + len(self._nullmask)


class ColumnarTable:
    """
    Simple columnar in-memory table for analytical workloads.
    Supports insert, batch insert, get, update, delete, select, aggregates.
    """

    def __init__(self, name: str, schema: List[Tuple[str, str]]) -> None:
        self.name = name
        self.columns: Dict[str, Column] = {}
        self._col_list: List[Column] = []
        self._col_names: List[str] = []
        for col_name, col_type in schema:
            col = Column(col_name, col_type)
            self.columns[col_name] = col
            self._col_list.append(col)
            self._col_names.append(col_name)
        self._row_count = 0

    def insert(self, row: Dict[str, Any]) -> int:
        for col in self._col_list:
            col.append(row.get(col.name))
        idx = self._row_count
        self._row_count += 1
        return idx

    def batch_insert(self, rows: List[Dict[str, Any]]) -> int:
        """Insert multiple rows at once — much faster than individual inserts."""
        start_idx = self._row_count
        for row in rows:
            for col in self._col_list:
                col.append(row.get(col.name))
            self._row_count += 1
        return start_idx

    def __len__(self) -> int:
        return self._row_count

    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        if idx < 0 or idx >= self._row_count:
            return None
        # Check first column's nullmask as quick check for deleted rows
        first_col = self._col_list[0]
        if first_col._nullmask[idx]:
            # Check if ALL columns are null (deleted row)
            all_null = True
            for col in self._col_list:
                if not col._nullmask[idx]:
                    all_null = False
                    break
            if all_null:
                return None
        return {col.name: col[idx] for col in self._col_list}

    def update(self, idx: int, row: Dict[str, Any]) -> None:
        if idx < 0 or idx >= self._row_count:
            raise IndexError(f"Row index {idx} out of range")
        current = self.get(idx)
        if current is None:
            raise KeyError(f"Row {idx} not found")
        merged = {**current, **row}
        for col in self._col_list:
            col[idx] = merged.get(col.name)

    def delete(self, idx: int) -> None:
        if idx < 0 or idx >= self._row_count:
            raise IndexError(f"Row index {idx} out of range")
        for col in self._col_list:
            col._nullmask[idx] = 1
            if col.dtype.startswith("bytes"):
                col._data[idx] = b""
            elif col._data.typecode in ("f", "d"):
                col._data[idx] = 0.0
            else:
                col._data[idx] = 0

    def select(self,
               where: Optional[Callable[[Dict[str, Any]], bool]] = None,
               columns: Optional[List[str]] = None,
               limit: Optional[int] = None,
               offset: int = 0) -> List[Dict[str, Any]]:
        if columns is None:
            col_names = self._col_names
        else:
            for c in columns:
                if c not in self.columns:
                    raise ValueError(f"Unknown column: {c}")
            col_names = columns

        result: List[Dict[str, Any]] = []
        matched = 0

        # Pre-fetch column references for speed
        selected_cols = [self.columns[name] for name in col_names]
        all_cols = self._col_list

        for idx in range(self._row_count):
            # Quick null check: skip if first col is null (likely deleted)
            if all_cols[0]._nullmask[idx]:
                continue

            if where is not None:
                # Build row dict for predicate
                row = {col.name: col[idx] for col in all_cols}
                if not where(row):
                    continue

            if matched < offset:
                matched += 1
                continue

            # Build projected row
            result.append({name: col[idx] for name, col in zip(col_names, selected_cols)})
            matched += 1
            if limit is not None and len(result) >= limit:
                break
        return result

    def select_column(self, column_name: str) -> List[Any]:
        if column_name not in self.columns:
            raise ValueError(f"Unknown column: {column_name}")
        col = self.columns[column_name]
        return [col[i] for i in range(len(col))]

    def aggregate(self,
                  column_name: str,
                  agg: str = "sum",
                  where: Optional[Callable[[Dict[str, Any]], bool]] = None) -> Any:
        if column_name not in self.columns:
            raise ValueError(f"Unknown column: {column_name}")
        col = self.columns[column_name]

        if agg == "count":
            return col.count_valid()

        # Fast path: no where clause — iterate column directly
        if where is None:
            if agg == "sum":
                total = 0
                for _, v in col.iter_valid():
                    total += v
                return total
            if agg == "avg":
                total = 0
                count = 0
                for _, v in col.iter_valid():
                    total += v
                    count += 1
                return total / count if count else 0
            if agg == "min":
                min_val = None
                for _, v in col.iter_valid():
                    if min_val is None or v < min_val:
                        min_val = v
                return min_val
            if agg == "max":
                max_val = None
                for _, v in col.iter_valid():
                    if max_val is None or v > max_val:
                        max_val = v
                return max_val

        # Slow path: with where clause
        all_cols = self._col_list
        if agg == "sum":
            total = 0
            for idx in range(self._row_count):
                if all_cols[0]._nullmask[idx]:
                    continue
                row = {c.name: c[idx] for c in all_cols}
                if where(row) and col._nullmask[idx] == 0:
                    total += col._data[idx]
            return total
        if agg == "avg":
            total = 0
            count = 0
            for idx in range(self._row_count):
                if all_cols[0]._nullmask[idx]:
                    continue
                row = {c.name: c[idx] for c in all_cols}
                if where(row) and col._nullmask[idx] == 0:
                    total += col._data[idx]
                    count += 1
            return total / count if count else 0
        if agg == "min":
            min_val = None
            for idx in range(self._row_count):
                if all_cols[0]._nullmask[idx]:
                    continue
                row = {c.name: c[idx] for c in all_cols}
                if where(row) and col._nullmask[idx] == 0:
                    v = col._data[idx]
                    if min_val is None or v < min_val:
                        min_val = v
            return min_val
        if agg == "max":
            max_val = None
            for idx in range(self._row_count):
                if all_cols[0]._nullmask[idx]:
                    continue
                row = {c.name: c[idx] for c in all_cols}
                if where(row) and col._nullmask[idx] == 0:
                    v = col._data[idx]
                    if max_val is None or v > max_val:
                        max_val = v
            return max_val

        raise ValueError(f"Unsupported aggregation: {agg}")

    def memory_usage(self) -> int:
        return sum(col.memory_usage() for col in self._col_list)

    def __repr__(self) -> str:
        return f"ColumnarTable({self.name!r}, rows={self._row_count}, cols={self._col_names})"
