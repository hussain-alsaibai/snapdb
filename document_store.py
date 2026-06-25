"""
SnapDB Document Store — JSON document storage on top of SnapDB.

Wraps the binary SnapDB with a simple JSON-like API:
    db = DocumentStore("docs.snap")
    db.insert({"name": "Alice", "age": 30})
    db.insert({"name": "Bob", "age": 25})
    results = db.query({"age": {"$gt": 25}})
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Union

from snapdb import SnapDB, Schema, ColumnDef


class DocumentStore:
    """
    JSON document store backed by SnapDB binary storage.

    Auto-infers schema from first document. Supports querying,
    indexing, and transactions.
    """

    def __init__(self, path: str, max_field_len: int = 256) -> None:
        self.path = path
        self.max_field_len = max_field_len
        self._db: Optional[SnapDB] = None
        self._field_types: Dict[str, str] = {}

        if os.path.exists(path) and os.path.getsize(path) > 0:
            self._load()
        else:
            self._db = None

    def _infer_schema(self, doc: Dict[str, Any]) -> Schema:
        """Infer fixed-width schema from a JSON document."""
        columns = []
        for key, val in doc.items():
            if isinstance(val, bool):
                dtype = "bool"
            elif isinstance(val, int):
                # Use i32 for integers (-2B to +2B)
                dtype = "i32"
            elif isinstance(val, float):
                dtype = "f64"
            elif isinstance(val, (str, bytes)):
                dtype = f"bytes:{self.max_field_len}"
            else:
                # Complex types: store as JSON string
                dtype = f"bytes:{self.max_field_len * 2}"
            columns.append(ColumnDef(key, dtype))
            self._field_types[key] = dtype
        return Schema(columns)

    def _coerce(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce a document to match the fixed-width schema."""
        result = {}
        for key, dtype in self._field_types.items():
            val = doc.get(key)
            if val is None:
                if dtype == "bool":
                    result[key] = False
                elif dtype in ("i8", "i16", "i32", "i64"):
                    result[key] = 0
                elif dtype in ("f32", "f64"):
                    result[key] = 0.0
                elif dtype.startswith("bytes"):
                    result[key] = b""
                else:
                    result[key] = b""
            elif dtype.startswith("bytes"):
                if isinstance(val, bytes):
                    result[key] = val
                else:
                    result[key] = str(val).encode("utf-8")
            else:
                result[key] = val
        return result

    def _decode(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Decode a row back to JSON-friendly types."""
        result = {}
        for key, val in row.items():
            if isinstance(val, bytes):
                # Try decode as JSON first, then string
                try:
                    result[key] = json.loads(val.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    try:
                        result[key] = val.decode("utf-8").rstrip("\x00")
                    except UnicodeDecodeError:
                        result[key] = val
            else:
                result[key] = val
        return result

    def _load(self) -> None:
        """Load existing document store."""
        # For now, just create with a basic schema and load
        # In v0.3.1: store schema metadata in the file header
        schema = Schema([ColumnDef("__json", f"bytes:{self.max_field_len * 4}")])
        try:
            self._db = SnapDB(self.path, schema)
        except (ValueError, OSError):
            # File exists but is empty or corrupted — start fresh
            self._db = None

    def insert(self, doc: Dict[str, Any]) -> int:
        """Insert a document. Returns row index."""
        if self._db is None:
            # First document — infer schema
            schema = self._infer_schema(doc)
            self._db = SnapDB(self.path, schema)

        row = self._coerce(doc)
        return self._db.insert(row)

    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get document by index."""
        if self._db is None:
            return None
        row = self._db.get(idx)
        if row is None:
            return None
        return self._decode(row)

    def update(self, idx: int, doc: Dict[str, Any]) -> None:
        """Update a document."""
        if self._db is None:
            raise RuntimeError("No documents stored yet")
        row = self._coerce(doc)
        self._db.update(idx, row)

    def delete(self, idx: int) -> None:
        """Delete a document."""
        if self._db is None:
            raise RuntimeError("No documents stored yet")
        self._db.delete(idx)

    def query(self, filter_spec: Optional[Dict[str, Any]] = None,
              select: Optional[List[str]] = None,
              sort: Optional[str] = None,
              desc: bool = False,
              limit: Optional[int] = None,
              offset: int = 0) -> List[Dict[str, Any]]:
        """
        Query documents with MongoDB-style filters.

        Args:
            filter_spec: {"age": {"$gt": 25}, "$or": [...]}
            select: Fields to include ["name", "age"]
            sort: Field to sort by
            desc: Sort descending
            limit: Max results
            offset: Skip first N
        """
        if self._db is None:
            return []

        # Build predicate from filter_spec
        if filter_spec:
            pred = self._build_predicate(filter_spec)
            rows = [(idx, self._decode(row)) for idx, row in self._db if pred(row)]
        else:
            rows = [(idx, self._decode(row)) for idx, row in self._db]

        # Sort
        if sort:
            rows.sort(key=lambda x: x[1].get(sort, 0), reverse=desc)

        # Slice
        start = offset
        end = start + limit if limit else len(rows)
        rows = rows[start:end]

        # Select fields
        if select:
            results = []
            for idx, row in rows:
                filtered = {k: v for k, v in row.items() if k in select}
                results.append(filtered)
            return results

        return [row for _, row in rows]

    def _build_predicate(self, spec: Dict[str, Any]) -> Any:
        """Build a predicate function from a filter spec."""
        checks = []
        for key, val in spec.items():
            if key == "$or":
                or_preds = [self._build_predicate({k: v}) for item in val for k, v in item.items()]
                checks.append(lambda r: any(p(r) for p in or_preds))
            elif key == "$and":
                and_preds = [self._build_predicate({k: v}) for item in val for k, v in item.items()]
                checks.append(lambda r: all(p(r) for p in and_preds))
            elif isinstance(val, dict):
                # Comparison operators
                for op, cmp_val in val.items():
                    if op == "$gt":
                        checks.append(lambda r, k=key, v=cmp_val: r.get(k, 0) > v)
                    elif op == "$gte":
                        checks.append(lambda r, k=key, v=cmp_val: r.get(k, 0) >= v)
                    elif op == "$lt":
                        checks.append(lambda r, k=key, v=cmp_val: r.get(k, 0) < v)
                    elif op == "$lte":
                        checks.append(lambda r, k=key, v=cmp_val: r.get(k, 0) <= v)
                    elif op == "$ne":
                        checks.append(lambda r, k=key, v=cmp_val: r.get(k) != v)
                    elif op == "$in":
                        checks.append(lambda r, k=key, v=cmp_val: r.get(k) in v)
                    elif op == "$contains":
                        checks.append(lambda r, k=key, v=cmp_val: v in str(r.get(k, "")))
                    elif op == "$exists":
                        checks.append(lambda r, k=key: k in r)
            else:
                # Exact match
                checks.append(lambda r, k=key, v=val: r.get(k) == v)

        def predicate(row: Dict[str, Any]) -> bool:
            return all(check(row) for check in checks)

        return predicate

    def create_index(self, field: str) -> None:
        """Create an index on a field."""
        if self._db is None:
            raise RuntimeError("No documents stored yet")
        self._db.create_index(field)

    def count(self) -> int:
        """Return total document count."""
        if self._db is None:
            return 0
        return len(self._db)

    def __len__(self) -> int:
        return self.count()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self._db is None:
            return iter([])
        return iter(self._decode(row) for _, row in self._db)

    def export_json(self, path: str) -> int:
        """Export all documents to JSON. Returns count."""
        docs = list(self)
        with open(path, "w") as f:
            json.dump(docs, f, indent=2, default=str)
        return len(docs)

    def import_json(self, path: str) -> int:
        """Import documents from JSON. Returns count."""
        with open(path, "r") as f:
            docs = json.load(f)
        count = 0
        for doc in docs:
            self.insert(doc)
            count += 1
        return count

    def close(self) -> None:
        """Close the store."""
        if self._db is not None:
            self._db.close()

    def __repr__(self) -> str:
        return f"DocumentStore({self.path!r}, docs={self.count()}, fields={list(self._field_types.keys())})"
