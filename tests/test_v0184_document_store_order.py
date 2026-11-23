"""DocumentStore sorting, search limits, and schema growth (v0.18.4).

Found by pointing the agent-memory pattern at the ordinary next step: the
first record has fields {text}, the second adds {ts}. Four defects, all in
the store an agent uses as an audit trail:

1. ``query(sort=<str field>)`` raised ``TypeError`` — the sort key mixed an
   absent field's ``0`` with real strings.
2. ``query(sort=<float field>)`` returned the *wrong order* without raising,
   because ``0.0`` is a legal float and interleaved with real values.
3. ``search(term, limit=0)`` returned one document — the limit was checked
   after the first append.
4. **A field introduced after the first document was silently dropped.**
   The schema was fixed by document #1's key set, so every later ``ts`` was
   discarded by ``_coerce`` and never stored.

Each test fails on the pre-fix behaviour and passes only if the fix holds;
the "must still work" cases guard the other direction.
"""

import os
import tempfile

import pytest

from snapdb import ColumnDef, Schema, SnapDB
from snapdb.document_store import DocumentStore


@pytest.fixture()
def store():
    fd, path = tempfile.mkstemp(suffix=".snap")
    os.close(fd)
    os.unlink(path)
    db = DocumentStore(path)
    try:
        yield db
    finally:
        try:
            db.close()
        except Exception:
            pass
        for suffix in ("", ".wal", ".lock"):
            p = path + suffix
            if os.path.exists(p):
                os.unlink(p)


def test_sort_string_field_does_not_raise(store):
    """Pre-fix: TypeError: '<' not supported between 'str' and 'int'."""
    store.insert({"name": "carol"})
    store.insert({"name": "alice"})
    store.insert({"name": "bob"})

    assert [d["name"] for d in store.query(sort="name")] == ["alice", "bob", "carol"]
    assert [d["name"] for d in store.query(sort="name", desc=True)] == [
        "carol",
        "bob",
        "alice",
    ]


def test_sort_float_field_is_numeric_not_positional(store):
    """Pre-fix: 0.0 for an absent field interleaved with real floats."""
    store.insert({"text": "c", "ts": 3.5})
    store.insert({"text": "a", "ts": 1.2})
    store.insert({"text": "b", "ts": 2.7})

    assert [d["ts"] for d in store.query(sort="ts")] == [1.2, 2.7, 3.5]
    assert [d["ts"] for d in store.query(sort="ts", desc=True)] == [3.5, 2.7, 1.2]


def test_sort_absent_value_does_not_win_the_top(store):
    """A record that never had the sort field must not outrank real values.

    Pre-fix, ``sort="ts", limit=1`` returned the row whose ``ts`` was the
    default ``0`` — an empty record presented as the oldest real event.
    """
    store.insert({"text": "legacy"})  # no ts
    store.insert({"text": "ts3", "ts": 3})
    store.insert({"text": "ts1", "ts": 1})

    assert [d["text"] for d in store.query(sort="ts")] == ["ts1", "ts3", "legacy"]
    assert store.query(sort="ts", limit=1)[0]["text"] == "ts1"
    # ...and the absent row is absent, not a zero the caller never wrote.
    assert "ts" not in store.query(sort="ts")[-1]


def test_absent_value_sorts_last_in_both_directions(store):
    """``sort`` must not hoist an absent value to the top on ``desc``.

    Pre-fix the absent field was encoded as the type default (0 / "") and was
    then indistinguishable from a real value, so ``desc`` put it first.
    """
    store.insert({"text": "no-ts"})
    store.insert({"text": "one", "ts": 1})
    store.insert({"text": "two", "ts": 2})

    desc_rows = store.query(sort="ts", desc=True)
    assert desc_rows[-1]["text"] == "no-ts"
    assert [d.get("ts") for d in desc_rows] == [2, 1, None]

    # Ascending: present values first, absent last -- so a limited read never
    # spends its first slot on a row that has no value at all.
    assert store.query(sort="ts", limit=1)[0]["ts"] == 1


def test_absent_field_is_omitted_from_returned_document(store):
    """An absent field must not come back as '' / 0 / False.

    The store cannot tell "the caller wrote an empty string" from "the caller
    never wrote this key" once both are encoded; the sentinel column keeps them
    apart so a report can say "no failure reason recorded" honestly.
    """
    store.insert({"name": "a", "reason": "timeout"})
    store.insert({"name": "b"})
    store.insert({"name": "c", "reason": ""})

    rows = {d["name"]: d for d in store.query()}
    assert "reason" not in rows["b"]
    assert rows["a"]["reason"] == "timeout"
    assert rows["c"]["reason"] == ""


def test_sort_is_applied_before_limit_and_offset(store):
    for ts in (5, 3, 9, 1, 7):
        store.insert({"text": f"t{ts}", "ts": ts})

    assert [d["ts"] for d in store.query(sort="ts", limit=2, offset=1)] == [3, 5]


def test_search_limit_zero_and_negative_return_nothing(store):
    """Pre-fix limit=0 returned one hit; a negative limit meant 'no limit'."""
    store.insert({"body": "agent recalled the earlier task"})
    store.insert({"body": "agent decided to skip the retry"})

    assert store.search("agent", limit=0) == []
    assert store.search("agent", limit=-1) == []
    assert len(store.search("agent", limit=1)) == 1


def test_search_still_finds_nested_values(store):
    store.insert({"name": "a", "tags": ["retry", "backoff"]})
    store.insert({"name": "b", "tags": ["cache"]})

    hits = store.search("backoff")
    assert len(hits) == 1 and hits[0]["name"] == "a"


def test_field_added_after_first_document_is_stored(store):
    """Pre-fix: the schema was document #1's key set. ``ts`` vanished."""
    store.insert({"text": "legacy"})
    store.insert({"text": "later", "ts": 42})

    row = store.query({"ts": 42})
    assert len(row) == 1 and row[0]["text"] == "later"

    # The legacy row is readable and simply has no value for the new field.
    assert store.get(0)["text"] == "legacy"


def test_schema_growth_survives_reopen(store):
    """The widened schema is persisted, not just held in memory."""
    path = store.path
    store.insert({"text": "legacy"})
    store.insert({"text": "later", "ts": 42})
    store.close()

    reopened = DocumentStore(path)
    try:
        assert len(reopened) == 2
        assert reopened.query({"ts": 42})[0]["text"] == "later"
    finally:
        reopened.close()


def test_schema_growth_keeps_all_earlier_rows(store):
    """Rewriting the file must not lose or duplicate rows."""
    for i in range(10):
        store.insert({"text": f"t{i}"})
    store.insert({"text": "with-ts", "ts": 7})

    assert len(store) == 11
    texts = sorted(d["text"] for d in store.query())
    assert texts == sorted([f"t{i}" for i in range(10)] + ["with-ts"])


def test_replace_schema_refuses_inside_a_transaction():
    """The WAL holds rows in the old shape; widening mid-transaction corrupts."""
    fd, path = tempfile.mkstemp(suffix=".snap")
    os.close(fd)
    os.unlink(path)
    db = SnapDB(path, Schema([ColumnDef("text", "bytes:32")]))
    try:
        with db.transaction():
            with pytest.raises(RuntimeError):
                db.replace_schema(
                    Schema([ColumnDef("text", "bytes:32"), ColumnDef("ts", "i32")])
                )
            db.insert({"text": "still works"})
        assert len(db) == 1
    finally:
        db.close()
        for suffix in ("", ".wal", ".lock"):
            p = path + suffix
            if os.path.exists(p):
                os.unlink(p)


def test_replace_schema_bumps_version_on_disk():
    """A reopen detects the new column — the change is in the header."""
    fd, path = tempfile.mkstemp(suffix=".snap")
    os.close(fd)
    os.unlink(path)
    db = SnapDB(path, Schema([ColumnDef("text", "bytes:32")]))
    db.insert({"text": "a"})
    db.replace_schema(Schema([ColumnDef("text", "bytes:32"), ColumnDef("ts", "i32")]))
    db.close()

    reopened = SnapDB(path, Schema([ColumnDef("placeholder", "i32")]))
    try:
        assert [c.name for c in reopened.schema.columns] == ["text", "ts"]
    finally:
        reopened.close()
        for suffix in ("", ".wal", ".lock"):
            p = path + suffix
            if os.path.exists(p):
                os.unlink(p)


def test_replace_schema_marks_preexisting_rows_absent(tmp_path):
    """A column added later holds ``None`` for rows written before it.

    Not the type default: ``ts=0`` would claim those rows happened at the unix
    epoch and would win a ``sort="ts"`` read against genuinely early rows.
    """
    from snapdb import SnapDB, Schema, ColumnDef

    path = str(tmp_path / "widen.snap")
    db = SnapDB(path, Schema([ColumnDef("a", "i32")]))
    db.insert({"a": 1})
    db.insert({"a": 5})

    grown = db.replace_schema(Schema([
        ColumnDef("a", "i32"),
        ColumnDef("ts", "i32", not_null=False),
    ]))
    assert isinstance(grown, int)

    assert db.get(0) == {"a": 1, "ts": None}
    db.insert({"a": 2, "ts": 7})
    assert db.get(2)["ts"] == 7
    db.close()

    reopened = SnapDB(path, Schema([ColumnDef("a", "i32")]))
    assert [(r["a"], r["ts"]) for r in (row for _, row in reopened)] == [
        (1, None), (5, None), (2, 7),
    ]
    reopened.close()
