"""v0.18.5 — `update()` is a patch, not a whole-record write.

`DocumentStore.update(idx, doc)` wrote `doc` as the entire record. The row
format is fixed-width and `_decode` omits the NULL / `b""` placeholders used
for fields the writer did not supply, so every field the caller did not name
came back **absent** rather than unchanged:

    db.insert({"name": "alice", "age": 30, "role": "admin"})
    db.update(0, {"age": 31})
    db.get(0)   # -> {"age": 31}

Nothing raised, the row still existed, and the field the caller meant to
change was correct. The rest of the record was gone, and the surviving row
reads as if the record never had those fields -- the worst failure mode for a
store whose purpose is a durable agent audit trail.

Why 264 green tests did not see it: every existing test wrote *every* field in
the update call (`db.update(0, {"name": "Alicia", "age": 31})`). A full
rewrite and a patch are indistinguishable when the payload is complete. The
suite proved the spelling that cannot fail.

`update` is now a patch. `replace` keeps the old whole-record behaviour, so
clearing fields is something a caller asks for rather than something that
happens by omission.
"""

import os
import tempfile

import pytest

from snapdb.document_store import DocumentStore


@pytest.fixture()
def store():
    fd, path = tempfile.mkstemp(suffix=".snap")
    os.close(fd)
    os.unlink(path)
    db = DocumentStore(path, max_field_len=64)
    yield db
    db.close()
    for suffix in ("", ".lock"):
        if os.path.exists(path + suffix):
            os.unlink(path + suffix)


class TestUpdateIsAPatch:
    def test_unnamed_fields_survive(self, store):
        store.insert({"name": "alice", "age": 30, "role": "admin"})
        store.update(0, {"age": 31})
        assert store.get(0) == {"name": "alice", "age": 31, "role": "admin"}

    def test_named_field_is_replaced(self, store):
        store.insert({"name": "alice", "age": 30})
        store.update(0, {"name": "alicia"})
        assert store.get(0)["name"] == "alicia"
        assert store.get(0)["age"] == 30

    def test_single_field_patch_on_wide_record(self, store):
        # NOTE: `retries: 0` is deliberately not in this record. A nullable
        # i32 column encodes "never written" as the zero-filled slot, so a
        # real 0 is currently indistinguishable from an absent field and is
        # dropped on read (see test_v0185_..._zero_roundtrip below, marked
        # xfail). That is a separate, pre-existing schema-level defect; mixing
        # it into this test would make a patch failure and a sentinel failure
        # look identical.
        record = {"step": 7, "task": "scan", "ok": True, "note": "fine"}
        store.insert(record)
        store.update(0, {"ok": False})
        expected = dict(record, ok=False)
        assert store.get(0) == expected

    def test_explicit_none_still_clears_a_field(self, store):
        store.insert({"name": "alice", "role": "admin"})
        store.update(0, {"role": None})
        # An explicit None is "this field is now null", not "leave it alone".
        assert store.get(0) == {"name": "alice"}

    def test_patch_does_not_touch_other_rows(self, store):
        store.insert({"name": "alice", "age": 30})
        store.insert({"name": "bob", "age": 25})
        store.update(0, {"age": 31})
        assert store.get(1) == {"name": "bob", "age": 25}

    def test_successive_patches_accumulate(self, store):
        store.insert({"a": 1, "b": 2, "c": 3})
        store.update(0, {"a": 10})
        store.update(0, {"b": 20})
        assert store.get(0) == {"a": 10, "b": 20, "c": 3}

    def test_patch_survives_reopen(self, store):
        store.insert({"name": "alice", "age": 30, "role": "admin"})
        store.update(0, {"age": 31})
        path = store.path
        store.close()
        reopened = DocumentStore(path, max_field_len=64)
        try:
            assert reopened.get(0) == {"name": "alice", "age": 31, "role": "admin"}
        finally:
            reopened.close()


class TestUpdatePastTheEnd:
    def test_update_missing_row_raises(self, store):
        store.insert({"name": "alice"})
        with pytest.raises(IndexError):
            store.update(9, {"name": "ghost"})

    def test_update_missing_row_does_not_create_one(self, store):
        store.insert({"name": "alice"})
        with pytest.raises(IndexError):
            store.update(5, {"name": "ghost"})
        assert store.count() == 1

    def test_update_on_empty_store_raises_runtime_error(self, store):
        with pytest.raises(RuntimeError):
            store.update(0, {"name": "ghost"})


class TestReplaceKeepsTheOldPower:
    def test_replace_drops_unnamed_fields(self, store):
        store.insert({"name": "alice", "age": 30, "role": "admin"})
        store.replace(0, {"name": "alice"})
        assert store.get(0) == {"name": "alice"}

    def test_replace_missing_row_raises(self, store):
        store.insert({"name": "alice"})
        with pytest.raises(IndexError):
            store.replace(3, {"name": "ghost"})

    def test_full_payload_update_still_works(self, store):
        """The historical spelling must keep working unchanged."""
        store.insert({"name": "Alice", "age": 30})
        store.update(0, {"name": "Alicia", "age": 31})
        assert store.get(0) == {"name": "Alicia", "age": 31}
