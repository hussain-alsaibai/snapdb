import os
import subprocess
import sys
import tempfile
import threading

import pytest

from snapdb import ColumnDef, Schema, SnapDB
from snapdb.document_store import DocumentStore
from snapdb.query import query


def _schema():
    return Schema([
        ColumnDef("id", "i32"),
        ColumnDef("name", "bytes:16"),
        ColumnDef("score", "f32"),
    ])


def _tmp():
    fd, path = tempfile.mkstemp(suffix=".snap")
    os.close(fd)
    os.remove(path)
    return path


def _cleanup(path):
    for p in (
        path,
        path.replace(".snap", ".wal"),
        path + ".lock",
        path + ".compact",
        path + ".tmp",
    ):
        if os.path.exists(p):
            os.remove(p)


def test_close_inside_transaction_rolls_back():
    path = _tmp()
    try:
        db = SnapDB(path, _schema())
        with db.transaction():
            db.insert({"id": 1, "name": b"alice", "score": 1.0})
            db.close()

        db = SnapDB(path, _schema())
        try:
            assert len(db) == 0
        finally:
            db.close()
    finally:
        _cleanup(path)


def test_nested_transaction_fails_loudly():
    path = _tmp()
    try:
        db = SnapDB(path, _schema())
        try:
            with pytest.raises(RuntimeError, match="Nested transactions"):
                with db.transaction():
                    with db.transaction():
                        pass
        finally:
            db.close()
    finally:
        _cleanup(path)


def test_double_open_same_file_is_rejected():
    path = _tmp()
    try:
        db = SnapDB(path, _schema())
        try:
            with pytest.raises(RuntimeError, match="already open|locked"):
                SnapDB(path, _schema())
        finally:
            db.close()
    finally:
        _cleanup(path)


def test_threaded_inserts_on_one_instance_do_not_drop_rows():
    path = _tmp()
    try:
        db = SnapDB(path, _schema())
        try:
            def worker(base):
                for i in range(250):
                    db.insert({"id": base + i, "name": f"u{base+i}".encode(), "score": float(i)})

            threads = [threading.Thread(target=worker, args=(n * 250,)) for n in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(db) == 1000
            assert sum(1 for _ in db) == 1000
        finally:
            db.close()
    finally:
        _cleanup(path)


def test_second_process_open_is_rejected_while_locked():
    path = _tmp()
    try:
        db = SnapDB(path, _schema())
        code = f"""
from snapdb import ColumnDef, Schema, SnapDB
schema = Schema([
    ColumnDef("id", "i32"),
    ColumnDef("name", "bytes:16"),
    ColumnDef("score", "f32"),
])
try:
    SnapDB({path!r}, schema)
except RuntimeError:
    raise SystemExit(0)
raise SystemExit(1)
"""
        try:
            subprocess.check_call(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONPATH": os.getcwd()},
            )
        finally:
            db.close()
    finally:
        _cleanup(path)


def test_limit_zero_returns_no_rows():
    path = _tmp()
    try:
        db = SnapDB(path, _schema())
        try:
            db.insert({"id": 1, "name": b"alice", "score": 1.0})
            assert query(db).slice(0).execute() == []
            assert query(db).slice(-1).execute() == []
        finally:
            db.close()
    finally:
        _cleanup(path)


def test_columnar_file_reopens_with_rows():
    path = _tmp()
    try:
        db = SnapDB(path, _schema(), storage_type="columnar")
        db.insert({"id": 1, "name": b"alice", "score": 1.0})
        db.close()

        db = SnapDB(path, _schema(), storage_type="columnar")
        try:
            assert len(db) == 1
            assert db.get(0)["name"] == "alice"
        finally:
            db.close()
    finally:
        _cleanup(path)


def test_committed_transaction_recovers_after_abrupt_exit():
    path = _tmp()
    code = f"""
import os
from snapdb import ColumnDef, Schema, SnapDB
schema = Schema([
    ColumnDef("id", "i32"),
    ColumnDef("name", "bytes:16"),
    ColumnDef("score", "f32"),
])
db = SnapDB({path!r}, schema)
with db.transaction():
    for i in range(25):
        db.insert({{"id": i, "name": f"u{{i}}".encode(), "score": float(i)}})
os._exit(0)
"""
    try:
        subprocess.check_call(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONPATH": os.getcwd()},
        )
        db = SnapDB(path, _schema())
        try:
            assert len(db) == 25
            assert db.get(24)["id"] == 24
        finally:
            db.close()
    finally:
        _cleanup(path)


def test_missing_required_column_and_unique_constraint_fail_cleanly():
    path = _tmp()
    schema = Schema([
        ColumnDef("id", "i32", primary_key=True),
        ColumnDef("name", "bytes:16"),
        ColumnDef("score", "f32"),
    ])
    try:
        db = SnapDB(path, schema)
        try:
            with pytest.raises(KeyError, match="Missing required column"):
                db.insert({"id": 1, "name": b"alice"})
            db.insert({"id": 1, "name": b"alice", "score": 1.0})
            with pytest.raises(ValueError, match="UNIQUE constraint"):
                db.insert({"id": 1, "name": b"bob", "score": 2.0})
        finally:
            db.close()
    finally:
        _cleanup(path)


def test_compact_reclaims_deleted_row_space():
    path = _tmp()
    try:
        db = SnapDB(path, _schema())
        try:
            db.batch_insert([
                {"id": i, "name": f"u{i}".encode(), "score": float(i)}
                for i in range(600)
            ])
            db.flush()
            before = os.path.getsize(path)
            for i in range(300, 600):
                db.delete(i)
            db.flush()
            assert os.path.getsize(path) == before
            reclaimed = db.compact()
            after = os.path.getsize(path)
            assert reclaimed > 0
            assert after < before
            assert len(db) == 300
            assert list(db)[-1][1]["id"] == 299
        finally:
            db.close()
    finally:
        _cleanup(path)


def test_fsck_and_repair_report_clean_database():
    path = _tmp()
    try:
        db = SnapDB(path, _schema())
        try:
            db.batch_insert([
                {"id": i, "name": f"u{i}".encode(), "score": float(i)}
                for i in range(50)
            ])
            for i in range(25, 50):
                db.delete(i)
            before = db.fsck()
            assert before["ok"] is True
            assert before["rows"] == 25
            repaired = db.repair()
            assert repaired["ok"] is True
            assert repaired["rows"] == 25
            assert len(db) == 25
        finally:
            db.close()
    finally:
        _cleanup(path)


def test_hot_backup_reopens_without_manual_flush():
    path = _tmp()
    backup = path.replace(".snap", ".backup.snap")
    try:
        db = SnapDB(path, _schema())
        try:
            db.insert({"id": 1, "name": b"alice", "score": 1.0})
            db.backup(backup)
        finally:
            db.close()

        restored = SnapDB(backup, _schema())
        try:
            assert len(restored) == 1
            assert restored.get(0)["name"] == "alice"
        finally:
            restored.close()
    finally:
        _cleanup(path)
        _cleanup(backup)


def test_batch_update_group_by_and_join():
    left_path = _tmp()
    right_path = _tmp()
    try:
        left = SnapDB(left_path, _schema())
        right = SnapDB(right_path, Schema([
            ColumnDef("id", "i32"),
            ColumnDef("dept", "bytes:16"),
        ]))
        try:
            left.batch_insert([
                {"id": 1, "name": b"a", "score": 1.0},
                {"id": 2, "name": b"b", "score": 2.0},
                {"id": 3, "name": b"a", "score": 3.0},
            ])
            right.batch_insert([
                {"id": 1, "dept": b"eng"},
                {"id": 3, "dept": b"ops"},
            ])
            assert left.batch_update(lambda r: r["name"] == "a", {"score": 10.0}) == 2
            assert left.group_by("name", "score", "sum") == {"a": 20.0, "b": 2.0}
            joined = left.join(right, "id", "id")
            assert [(l["id"], r["dept"]) for l, r in joined] == [(1, "eng"), (3, "ops")]
        finally:
            left.close()
            right.close()
    finally:
        _cleanup(left_path)
        _cleanup(right_path)


def test_row_range_index_tracks_update_delete_and_compact():
    path = _tmp()
    try:
        db = SnapDB(path, _schema())
        try:
            db.batch_insert([
                {"id": i, "name": f"u{i}".encode(), "score": float(i)}
                for i in range(20)
            ])
            db.create_range_index("score")
            assert [r["id"] for r in db.range_find("score", 5.0, 8.0)] == [5, 6, 7, 8]

            db.update(8, {"score": 100.0})
            db.delete(5)
            assert [r["id"] for r in db.range_find("score", 5.0, 8.0)] == [6, 7]

            db.compact()
            assert [r["id"] for r in db.range_find("score", 18.0, 100.0)] == [18, 19, 8]
        finally:
            db.close()
    finally:
        _cleanup(path)


def test_columnar_batch_update_and_group_by_use_columnar_helpers():
    path = _tmp()
    try:
        db = SnapDB(path, _schema(), storage_type="columnar")
        try:
            db.batch_insert([
                {"id": 1, "name": b"a", "score": 1.0},
                {"id": 2, "name": b"b", "score": 2.0},
                {"id": 3, "name": b"a", "score": 3.0},
            ])
            assert db.batch_update(lambda r: r["name"] == "a", {"score": 10.0}) == 2
            assert db.group_by("name", "score", "sum") == {"a": 20.0, "b": 2.0}
            assert db.group_by("name", "score", "avg") == {"a": 10.0, "b": 2.0}
        finally:
            db.close()
    finally:
        _cleanup(path)


def test_document_store_json_lists_round_trip():
    path = _tmp()
    json_path = path.replace(".snap", ".json")
    restored_path = path.replace(".snap", ".restored.snap")
    try:
        db = DocumentStore(path, max_field_len=128)
        db.insert({"name": "Alice", "tags": ["dev", "python"]})
        assert db.get(0)["tags"] == ["dev", "python"]
        db.export_json(json_path)

        restored = DocumentStore(restored_path, max_field_len=128)
        restored.import_json(json_path)
        assert restored.get(0)["tags"] == ["dev", "python"]
    finally:
        for obj in ("db", "restored"):
            try:
                locals()[obj].close()
            except Exception:
                pass
        _cleanup(path)
        _cleanup(restored_path)
        if os.path.exists(json_path):
            os.remove(json_path)


def test_encryption_key_hides_row_and_wal_plaintext():
    path = _tmp()
    secret = b"top-secret-token"
    try:
        db = SnapDB(path, _schema(), encryption_key="pw")
        try:
            with db.transaction():
                db.insert({"id": 1, "name": secret, "score": 1.0})
        finally:
            db.close()

        with open(path, "rb") as f:
            assert secret not in f.read()
        wal_path = path.replace(".snap", ".wal")
        if os.path.exists(wal_path):
            with open(wal_path, "rb") as f:
                assert secret not in f.read()

        reopened = SnapDB(path, _schema(), encryption_key="pw")
        try:
            assert reopened.get(0)["name"] == secret.decode()
        finally:
            reopened.close()
    finally:
        _cleanup(path)
