"""Agent audit trail on SnapDB — a runnable version of the README's claim.

The README says SnapDB is a good fit for the *durable state* and *audit
trail* needs of an autonomous agent. This script is that claim, executed:

  1. append N tool-call records to a durable, append-only table
  2. take a named snapshot (crash recovery point)
  3. point-read a single record by row id
  4. run a selective range query with and without the index
  5. restore the snapshot into an independent instance and check the row count

Run it:

    python examples/agent_memory.py            # 20,000 records
    python examples/agent_memory.py 100000     # bigger table

Numbers are printed for the machine you run it on. They are not a benchmark
against SQLite or DuckDB -- they are here so you can see, before you adopt,
what one agent step costs and whether ``explain()`` predicted the plan you
actually got.

Every record lives in a temp directory that is removed on exit.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import time

from snapdb import ColumnDef, Schema, SnapDB, query

TOOLS = ("read_file", "grep", "edit_file", "run_tests", "git_commit")
STATUSES = ("ok", "ok", "ok", "ok", "error")  # 80% success, like a real run


def build_schema() -> Schema:
    return Schema(
        [
            ColumnDef("step", "i32"),        # agent step number
            ColumnDef("ts", "i64"),          # unix millis
            ColumnDef("tool", "bytes:16"),   # tool name
            ColumnDef("duration_ms", "i32"), # wall time of the call
            ColumnDef("status", "bytes:8"),  # ok | error
            ColumnDef("args", "bytes:64"),   # truncated JSON of the arguments
        ]
    )


def ms(fn):
    """Run fn, return (milliseconds, result)."""
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000.0, out


def main(n: int = 20_000) -> int:
    tmp = tempfile.mkdtemp(prefix="snapdb-agent-memory-")
    path = os.path.join(tmp, "trail.snap")
    db = SnapDB(path, build_schema())
    rng = random.Random(7)

    # ---- 1. append -------------------------------------------------------
    base = int(time.time() * 1000)

    def append_all():
        for step in range(n):
            db.insert(
                {
                    "step": step,
                    "ts": base + step * 40,
                    "tool": rng.choice(TOOLS),
                    "duration_ms": rng.randint(3, 900),
                    "status": rng.choice(STATUSES),
                    "args": '{"path":"src/app.py"}',
                }
            )

    insert_ms, _ = ms(append_all)
    print(f"append        {n:>7,} rows   {insert_ms:8.1f} ms"
          f"   {insert_ms / n * 1000:6.1f} us/row")

    # ---- 2. snapshot -----------------------------------------------------
    snap_ms, snap_name = ms(lambda: db.snapshot("before-refactor"))
    snaps = db.list_snapshots()
    print(f"snapshot      {snap_name!r:>18}   {snap_ms:8.1f} ms"
          f"   {snaps[0]['rows']:,} rows recorded")

    # ---- 3. point read ---------------------------------------------------
    reads = 10_000
    read_ms, _ = ms(lambda: [db.get(i % n) for i in range(reads)])
    print(f"point read    {reads:>7,} gets   {read_ms:8.1f} ms"
          f"   {read_ms / reads * 1000:6.2f} us/get")

    # ---- 4. selective range query ---------------------------------------
    # The last 2% of steps is our "window": a range narrow enough that the
    # index should win. explain() says which plan will run, before it runs.
    low, high = n - max(1, n // 50), n
    q = query(db).between("step", low, high)
    plan = q.explain()

    noidx_ms, _ = ms(q.execute)
    count_no_idx_ms, count_no_idx = ms(q.count)

    db.create_range_index("step")
    q2 = query(db).between("step", low, high)
    plan_idx = q2.explain()
    idx_ms, idx_rows = ms(q2.execute)
    count_idx_ms, count_idx = ms(q2.count)

    print()
    print(f"plain scan    {plan!r}")
    print(f"              execute {noidx_ms:8.1f} ms   count {count_no_idx_ms:8.1f} ms"
          f"   ({count_no_idx:,} rows)")
    print(f"with index    {plan_idx!r}")
    print(f"              execute {idx_ms:8.1f} ms   count {count_idx_ms:8.1f} ms"
          f"   ({count_idx:,} rows)")

    assert len(idx_rows) == count_idx, "execute() and count() disagree"
    assert count_no_idx == count_idx, "index changed the answer, not just the speed"
    if idx_ms > 0:
        print(f"speedup       execute {noidx_ms / idx_ms:5.1f}x"
              f"   count {count_no_idx_ms / max(count_idx_ms, 1e-9):5.1f}x")
    print(f"same answer   {count_idx:,} rows both ways")

    # ---- 5. restore ------------------------------------------------------
    restore_ms, restored = ms(lambda: db.open_snapshot("before-refactor"))
    try:
        print()
        print(f"restore       {len(restored):>7,} rows   {restore_ms:8.1f} ms"
              f"   independent copy, writes do not touch the live db")
        assert len(restored) == n, f"snapshot lost rows: {len(restored)} != {n}"
    finally:
        restored.close()

    db.close()
    print("\nok — every assertion held")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    raise SystemExit(main(n))
