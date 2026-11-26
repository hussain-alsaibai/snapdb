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
from typing import Any, Dict, Iterator, List, Optional

from .core import SnapDB, Schema, ColumnDef


def _infer_dtype(val: Any) -> Optional[str]:
    """Map a Python value to a SnapDB column type, or None if width-dependent.

    ``None`` means "this needs ``max_field_len`` to size the column" — strings
    and complex values — not "untypeable".
    """
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "i32"
    if isinstance(val, float):
        return "f64"
    return None


def _sort_key(value: Any) -> Any:
    """Order a heterogeneous field without raising, and without lying.

    ``rows.sort(key=lambda x: x[1].get(sort, 0))`` mixes an absent field
    (``0``) with real values. When the real values are strings, Python 3
    raises ``TypeError: '<' not supported between 'str' and 'int'``; when
    they are floats, ``0`` silently sorts as an ordinary value, so a missing
    timestamps column interleaves with the present ones and the order is
    wrong but plausible. Neither is acceptable in a store whose selling point
    is a durable agent audit trail.

    The key puts every value into one total order: things that sort are
    ordered among themselves, and ``None`` (absent / unorderable) always
    sorts last, so a caller reading ``sort="ts", limit=10`` never gets an
    empty-valued row before a real one.
    """
    if value is None:
        return (1, 0, "")
    if isinstance(value, bool):
        return (0, 0, int(value))
    if isinstance(value, (int, float)):
        return (0, 0, value)
    if isinstance(value, str):
        return (0, 1, value)
    return (0, 2, repr(value))


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
        # Per-call record of which doc keys were absent (see _coerce).
        self._unset_seen: Optional[frozenset] = None
        self._unset_token: Optional[int] = None

        if os.path.exists(path) and os.path.getsize(path) > 0:
            self._load()
        else:
            self._db = None

    def _rebuild_for_schema_growth(self) -> None:
        """Widen the on-disk schema so a newly-seen document field persists.

        The schema used to be fixed by the first document: a later
        ``insert({"text": ..., "ts": 3})`` silently dropped ``ts``, and the
        only field that ever existed was the one the very first record named.

        The rewrite itself is ``SnapDB.replace_schema`` — the row format is
        positional, so this cannot be an in-place assignment. Rows written
        before the column existed are re-encoded with the null placeholder, so
        ``query(sort="ts")`` does not present a legacy record as ``ts == 0``;
        the caller's insert then runs normally against the widened schema.
        """
        db = self._db
        if db is None:
            return
        db.replace_schema(Schema(self._schema_columns()))

    def _unset_fields(self, token: Optional[int]) -> frozenset:
        """Keys the current ``_coerce`` call must leave NULL.

        ``_coerce`` receives only a dict, so the key set of the caller's
        original document is stashed by identity right before the call. If the
        token does not match (a bare ``_coerce`` call, e.g. from an update
        path that already has a full row), nothing is treated as absent and
        every field keeps the "fill with the type default" behaviour.
        """
        if token is None or token != self._unset_token:
            return frozenset()
        return self._unset_seen or frozenset()

    def _coerce_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce ``doc``, recording which keys it did not mention."""
        if self._db is not None:
            self._unset_seen = frozenset(self._field_types) - frozenset(doc)
            self._unset_token = id(doc)
        try:
            return self._coerce(doc)
        finally:
            self._unset_token = None
            self._unset_seen = None

    def _dtype_for(self, val: Any) -> str:
        """Column type for one value, including the width-dependent types."""
        inferred = _infer_dtype(val)
        if inferred is not None:
            return inferred
        if isinstance(val, (str, bytes)):
            return f"bytes:{self.max_field_len}"
        return f"bytes:{self.max_field_len * 2}"

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
        return Schema(self._schema_columns())

    def _schema_columns(self) -> List[ColumnDef]:
        # ``not_null=False`` is what makes an absent field storable: the row
        # format is fixed-width, and a caller who never wrote ``ts`` has no
        # value to encode for it. Without this the widening rewrite and every
        # subsequent insert into a sparse document both raise.
        return [
            ColumnDef(name, dtype, not_null=False)
            for name, dtype in self._field_types.items()
        ]

    def _coerce(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce a document to match the fixed-width schema.

        A key that is present in ``doc`` but absent from the schema is added
        before coercion. Without this, the schema was fixed by the first
        document forever: ``insert({"text": ..., "ts": 3})`` against a store
        whose first document had only ``text`` silently dropped ``ts``.
        """
        missing = [k for k in doc if k not in self._field_types]
        if missing:
            for key in missing:
                self._field_types[key] = self._dtype_for(doc[key])
            if self._db is not None:
                self._rebuild_for_schema_growth()

        result = {}
        unset = self._unset_fields(id(doc))
        for key, dtype in self._field_types.items():
            val = doc.get(key)
            if key in unset:
                # The caller did not mention this field: keep it NULL rather
                # than writing the type default. Otherwise a record inserted
                # before a column existed comes back indistinguishable from a
                # record that explicitly said 0 / "" / False, and a sort or a
                # [:5] tail scan cannot tell an absent value from a real one.
                # Absent: keep it NULL in storage and drop it on the way out
                # (see _decode), so it is neither "" / 0 / False nor a
                # caller-visible key the caller never wrote.
                result[key] = b"" if dtype.startswith("bytes") else None
            elif val is None:
                if dtype == "bool":
                    result[key] = False
                elif dtype in ("i8", "i16", "i32", "i64"):
                    result[key] = 0
                elif dtype in ("f32", "f64"):
                    result[key] = 0.0
                elif dtype.startswith("bytes"):
                    result[key] = b""
                else:
                    result[key] = 0
            elif dtype.startswith("bytes"):
                if isinstance(val, bytes):
                    result[key] = val
                else:
                    # JSON-encode strings too (not just complex values) so
                    # _decode round-trips types exactly: a stored "2134" must
                    # come back as the string "2134", not the int 2134.
                    result[key] = json.dumps(val).encode("utf-8")
            else:
                result[key] = val
        return result

    def _decode(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Decode a row back to JSON-friendly types.

        Keys the writer never supplied are omitted, not returned as "" / 0 /
        False. The placeholders exist so the fixed-width row can be encoded;
        handing them back would make an absent field indistinguishable from a
        value the caller actually wrote, which is the difference between "this
        agent step recorded no failure reason" and "it recorded an empty one".
        """
        result = {}
        for key, val in row.items():
            if val is None or (isinstance(val, bytes) and val == b"") or val == "":
                # Absent — see the docstring. A stored ``b""`` is the
                # fixed-width placeholder for a field the caller never
                # supplied; '' / 0 / False the caller wrote are returned as
                # written, because those are JSON-encoded, not empty.
                continue
            if isinstance(val, bytes):
                # Try decode as JSON first, then string
                try:
                    result[key] = json.loads(val.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    try:
                        result[key] = val.decode("utf-8").rstrip("\x00")
                    except UnicodeDecodeError:
                        result[key] = val
            elif isinstance(val, str):
                try:
                    result[key] = json.loads(val)
                except json.JSONDecodeError:
                    result[key] = val
            else:
                result[key] = val
        return result

    def _load(self) -> None:
        """Load an existing document store.

        The placeholder schema passed here is replaced by the real schema
        persisted in the file header (SnapDB._load restores it), so the field
        map must be rebuilt from the loaded schema — without it, _coerce()
        returns empty rows and every insert/update after a reopen fails.
        """
        schema = Schema([ColumnDef("__json", f"bytes:{self.max_field_len * 4}")])
        self._db = SnapDB(self.path, schema)
        self._field_types = {c.name: c.dtype for c in self._db.schema.columns}

    def insert(self, doc: Dict[str, Any]) -> int:
        """Insert a document. Returns row index."""
        if self._db is None:
            # First document — infer schema
            schema = self._infer_schema(doc)
            self._db = SnapDB(self.path, schema)

        row = self._coerce_document(doc)
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
        """Update a document. Named fields replace; unnamed fields are kept.

        ``update`` used to write ``doc`` as the whole record. Because the row
        format is fixed-width and ``_decode`` drops the NULL / ``b""``
        placeholders, every field the caller did not name came back **absent**
        rather than unchanged:

            db.insert({"name": "alice", "age": 30, "role": "admin"})
            db.update(0, {"age": 31})
            db.get(0)   # -> {"age": 31}   the name and role are gone

        Nothing raised. The row still existed and the field the caller meant
        to change was correct, so the call looked like it had worked -- while
        the rest of the record had been erased. For a store whose purpose is
        an append-only agent audit trail, a silent erase is the worst
        possible failure: the surviving row reads as if the record never had
        those fields.

        Semantics now match the name: ``update`` is a patch. Fields present in
        ``doc`` replace their stored value; fields absent from ``doc`` keep
        the value already on disk. To clear a field, pass it explicitly with
        ``None`` (or ``drop`` it, if the caller wants it gone from the schema,
        which a fixed-width row does not support) -- an explicit ``None``
        stays an explicit null, an unmentioned key stays untouched.

        Use :meth:`replace` to overwrite a whole record.
        """
        if self._db is None:
            raise RuntimeError("No documents stored yet")
        current = self.get(idx)
        if current is None:
            # An update to a row that does not exist used to write a
            # placeholder row into the store. Failing loudly is the only
            # honest answer: the caller's intent was to modify something.
            raise IndexError(f"no document at index {idx}")
        merged = dict(current)
        merged.update(doc)
        row = self._coerce_document(merged)
        self._db.update(idx, row)

    def replace(self, idx: int, doc: Dict[str, Any]) -> None:
        """Overwrite a document entirely. Fields not in ``doc`` are removed.

        This is the behaviour ``update`` used to have. It is spelled
        separately so that clearing fields is something a caller asks for.
        """
        if self._db is None:
            raise RuntimeError("No documents stored yet")
        if self.get(idx) is None:
            raise IndexError(f"no document at index {idx}")
        row = self._coerce_document(doc)
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
        if limit is not None and limit <= 0:
            return []

        # Build predicate from filter_spec. Filter on the DECODED document —
        # raw rows hold JSON-encoded strings, so comparisons against them
        # would never match the caller's values.
        if filter_spec:
            pred = self._build_predicate(filter_spec)
            rows = [(idx, doc) for idx, row in self._db
                    if pred(doc := self._decode(row))]
        else:
            rows = [(idx, self._decode(row)) for idx, row in self._db]

        # Sort. The key is total (see _sort_key): a mixed or absent field
        # used to raise TypeError or interleave silently-wrong rows.
        #
        # ``desc`` is applied only to the value band, never to the
        # absent/unorderable band, so an empty field cannot be promoted to the
        # top of a descending sort. Empty rows stay last in both directions.
        if sort:
            rows.sort(key=lambda x: _sort_key(x[1].get(sort)))
            if desc:
                present = [r for r in rows if _sort_key(r[1].get(sort))[0] == 0]
                absent = [r for r in rows if _sort_key(r[1].get(sort))[0] != 0]
                present.reverse()
                rows = present + absent

        # Slice
        start = offset
        end = start + limit if limit is not None else len(rows)
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
                        # A silently-ignored unknown operator would make the
                        # condition match everything (filter bypass).
                        raise ValueError(f"Unsupported operator: {op!r}")
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

    def search(self, term: str, fields: Optional[List[str]] = None,
               limit: int = 20) -> List[Dict[str, Any]]:
        """Case-insensitive substring search across string fields.

        Designed as a lightweight agent-memory / lexical retrieval primitive:
        find the documents whose text fields contain ``term``, without needing
        a full-text engine. Pass ``fields`` to restrict which columns are
        searched (defaults to all string columns).
        """
        if self._db is None or limit is None or limit <= 0:
            return []
        term_l = term.lower()
        candidates = fields if fields else list(self._field_types.keys())
        results: List[Dict[str, Any]] = []
        for _, row in self._db:
            doc = self._decode(row)
            for f in candidates:
                val = doc.get(f)
                if isinstance(val, str) and term_l in val.lower():
                    results.append(doc)
                    break
                elif isinstance(val, (dict, list)) and term_l in str(val).lower():
                    results.append(doc)
                    break
            if len(results) >= limit:
                break
        return results

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
