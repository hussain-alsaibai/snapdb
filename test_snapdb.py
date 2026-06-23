"""Quick test of SnapDB basic functionality."""

import tempfile
from pathlib import Path

from snapdb import SnapDB, Schema, ColumnDef


def test_basic():
    schema = Schema([
        ColumnDef("id", "i32"),
        ColumnDef("temp", "f32"),
        ColumnDef("active", "bool"),
    ])

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.snap"
        
        # Create
        db = SnapDB(db_path, schema)
        print(f"Created: {db}")
        
        # Insert
        idx0 = db.insert({"id": 1, "temp": 25.5, "active": True})
        idx1 = db.insert({"id": 2, "temp": 30.0, "active": False})
        idx2 = db.insert({"id": 3, "temp": 15.0, "active": True})
        print(f"Inserted 3 rows, indices: {idx0}, {idx1}, {idx2}")
        print(f"Total rows: {len(db)}")
        
        # Read decoded
        row0 = db.get(0)
        print(f"Row 0: {row0}")
        assert row0["id"] == 1
        assert abs(row0["temp"] - 25.5) < 0.1
        assert row0["active"] == True
        
        # Read zero-copy raw
        raw0 = db.get_raw(0)
        print(f"Row 0 raw (memoryview): {raw0}")
        assert raw0 is not None
        assert len(raw0) == schema.row_width
        
        # Update
        db.update(1, {"id": 2, "temp": 99.9, "active": False})
        row1 = db.get(1)
        print(f"Row 1 after update: {row1}")
        assert abs(row1["temp"] - 99.9) < 0.1
        
        # Delete
        db.delete(2)
        row2 = db.get(2)
        print(f"Row 2 after delete: {row2}")
        assert row2 is None
        
        # Iteration
        print("\n--- Iterate all ---")
        count = 0
        for idx, row in db:
            print(f"  [{idx}] id={row['id']} temp={row['temp']:.1f} active={row['active']}")
            count += 1
        assert count == 2  # 3 inserted - 1 deleted
        
        # Query
        print("\n--- Query active rows ---")
        for idx, row in db.query(lambda r: r["active"]):
            print(f"  [{idx}] id={row['id']} temp={row['temp']:.1f}")
        
        # Close and reopen
        db.close()
        print("\n--- Closed ---")
        
        db2 = SnapDB(db_path, schema)
        print(f"Reopened: {db2}")
        print(f"Total rows after reopen: {len(db2)}")
        assert len(db2) == 2
        
        row0_reopened = db2.get(0)
        assert row0_reopened["id"] == 1
        
        # Bulk insert
        print("\n--- Bulk insert 1000 rows ---")
        for i in range(1000):
            db2.insert({"id": i + 100, "temp": float(i), "active": i % 2 == 0})
        print(f"Total rows: {len(db2)}")
        assert len(db2) == 1002
        
        # Query
        evens = list(db2.query(lambda r: r["id"] % 2 == 0))
        print(f"Even ID rows: {len(evens)}")
        
        db2.close()
        print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_basic()
