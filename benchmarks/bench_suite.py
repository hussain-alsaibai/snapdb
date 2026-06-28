#!/usr/bin/env python3
"""
SnapDB benchmark suite — reproducible, cross-platform, machine-readable.

Compares SnapDB (row + columnar) against in-process Python alternatives on a
fixed synthetic workload and emits:

  * a human-readable table on stdout,
  * ``--json PATH``  : structured results (for CI artifacts / trend tracking),
  * ``--markdown PATH`` : a Markdown table (for injection into the README).

Competitors are optional — any that aren't installed are simply skipped, so the
suite runs anywhere with just the stdlib (SnapDB + sqlite3 + dict are always
available; pandas / numpy / duckdb are used when present).

Usage:
    python benchmarks/bench_suite.py --rows 100000
    python benchmarks/bench_suite.py --rows 50000 --json results.json --markdown bench.md

Numbers are reported honestly: SnapDB is a zero-dependency, lightweight store —
it wins on memory footprint, point reads and beating SQLite for simple
workloads, and it is explicit where a NumPy-backed engine is faster at pure
vectorized math.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sqlite3
import statistics
import sys
import tempfile
import time
import tracemalloc
from typing import Callable, Dict, List, Optional

# Make `snapdb` importable when run from a checkout without installing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snapdb import SnapDB, Schema, ColumnDef  # noqa: E402
from snapdb.columnar import ColumnarTable  # noqa: E402

try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    import duckdb
except ImportError:
    duckdb = None


# ── Timing helpers ──────────────────────────────────────────────────────────

def _bench(fn: Callable[[], object], repeat: int = 5, warmup: int = 1) -> float:
    """Return the best (min) wall-clock seconds over `repeat` runs."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return min(samples)


def _rows(n: int) -> List[dict]:
    rnd = random.Random(42)
    statuses = [b"active", b"idle", b"banned"]
    return [
        {
            "id": i,
            "ts": 1_700_000_000 + i,            # monotonic -> delta-friendly
            "sensor": i % 16,                   # low-cardinality
            "status": statuses[i % 3],          # low-cardinality string
            "temp": 20.0 + rnd.random() * 30.0,
        }
        for i in range(n)
    ]


# ── Workload definitions ────────────────────────────────────────────────────
#
# Each engine implements a small adapter exposing the workloads it supports.
# Missing workloads are reported as "n/a".

class Engine:
    name = "?"
    available = True

    def setup(self, rows: List[dict]):
        ...

    def teardown(self):
        ...

    # workloads (override what you support) -----------------------------------
    def insert(self, rows: List[dict]):
        raise NotImplementedError

    def point_read(self, keys: List[int]):
        raise NotImplementedError

    def scan_sum(self):
        raise NotImplementedError

    def filter_multi(self):
        raise NotImplementedError

    def memory_bytes(self) -> Optional[int]:
        return None


class SnapDBColumnar(Engine):
    name = "SnapDB (columnar)"

    def __init__(self):
        self.path = _tmp()
        self.db = None

    def _schema(self):
        return Schema([
            ColumnDef("id", "i32"), ColumnDef("ts", "i64"),
            ColumnDef("sensor", "u8"), ColumnDef("status", "bytes:8"),
            ColumnDef("temp", "f64"),
        ])

    def insert(self, rows):
        self.db = SnapDB(self.path, self._schema(), storage_type="columnar",
                         dict_columns=["status"], delta_columns=["ts"])
        self.db.batch_insert(rows)

    def point_read(self, keys):
        g = self.db.get
        for k in keys:
            g(k)

    def scan_sum(self):
        return self.db.aggregate("temp", "sum")

    def filter_multi(self):
        return self.db.select_where(
            [("sensor", ">", 8), ("status", "==", b"active"), ("temp", "<", 35.0)],
            columns=["id"])

    def memory_bytes(self):
        return self.db.memory_usage()

    def teardown(self):
        if self.db:
            self.db.close()
        _cleanup(self.path)


class SnapDBRow(Engine):
    name = "SnapDB (row)"

    def __init__(self):
        self.path = _tmp()
        self.db = None

    def insert(self, rows):
        schema = Schema([
            ColumnDef("id", "i32"), ColumnDef("ts", "i64"),
            ColumnDef("sensor", "u8"), ColumnDef("status", "bytes:8"),
            ColumnDef("temp", "f64"),
        ])
        self.db = SnapDB(self.path, schema, storage_type="row")
        self.db.batch_insert(rows)

    def point_read(self, keys):
        g = self.db.get
        for k in keys:
            g(k)

    def scan_sum(self):
        total = 0.0
        for _, row in self.db:
            total += row["temp"]
        return total

    def filter_multi(self):
        return [row for _, row in self.db
                if row["sensor"] > 8 and row["status"] == "active" and row["temp"] < 35.0]

    def teardown(self):
        if self.db:
            self.db.close()
        _cleanup(self.path)


class SQLiteMem(Engine):
    name = "sqlite3 (:memory:)"

    def insert(self, rows):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, ts INTEGER, sensor INTEGER, "
            "status TEXT, temp REAL)")
        self.conn.executemany(
            "INSERT INTO t VALUES (?,?,?,?,?)",
            [(r["id"], r["ts"], r["sensor"], r["status"].decode(), r["temp"]) for r in rows])
        self.conn.commit()

    def point_read(self, keys):
        cur = self.conn.cursor()
        for k in keys:
            cur.execute("SELECT * FROM t WHERE id=?", (k,)).fetchone()

    def scan_sum(self):
        return self.conn.execute("SELECT SUM(temp) FROM t").fetchone()[0]

    def filter_multi(self):
        return self.conn.execute(
            "SELECT id FROM t WHERE sensor>8 AND status='active' AND temp<35.0").fetchall()

    def memory_bytes(self):
        # SQLite reports pages * page_size used by the in-memory DB.
        try:
            used = self.conn.execute("PRAGMA page_count").fetchone()[0]
            size = self.conn.execute("PRAGMA page_size").fetchone()[0]
            return used * size
        except Exception:
            return None

    def teardown(self):
        self.conn.close()


class PandasEngine(Engine):
    name = "pandas"
    available = pd is not None

    def insert(self, rows):
        self.df = pd.DataFrame(rows)
        self.df["status"] = self.df["status"].str.decode("utf-8")
        self.df = self.df.set_index("id", drop=False)

    def point_read(self, keys):
        loc = self.df.loc
        for k in keys:
            _ = loc[k]

    def scan_sum(self):
        return float(self.df["temp"].sum())

    def filter_multi(self):
        d = self.df
        return d[(d["sensor"] > 8) & (d["status"] == "active") & (d["temp"] < 35.0)]["id"]

    def memory_bytes(self):
        return int(self.df.memory_usage(deep=True).sum())


class DictEngine(Engine):
    name = "dict (baseline)"

    def insert(self, rows):
        self.data = {r["id"]: r for r in rows}

    def point_read(self, keys):
        g = self.data.get
        for k in keys:
            g(k)

    def scan_sum(self):
        return sum(r["temp"] for r in self.data.values())

    def filter_multi(self):
        return [r["id"] for r in self.data.values()
                if r["sensor"] > 8 and r["status"] == b"active" and r["temp"] < 35.0]

    def memory_bytes(self):
        tracemalloc.start()
        d = {r["id"]: dict(r) for r in self.data.values()}
        cur, _peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del d
        return cur


class DuckDBEngine(Engine):
    name = "duckdb"
    available = duckdb is not None

    def insert(self, rows):
        self.con = duckdb.connect()
        self._df = pd.DataFrame(rows) if pd is not None else None
        if self._df is not None:
            self._df["status"] = self._df["status"].str.decode("utf-8")
            self.con.execute("CREATE TABLE t AS SELECT * FROM _df", {"_df": self._df})
        else:
            self.available = False

    def scan_sum(self):
        return self.con.execute("SELECT SUM(temp) FROM t").fetchone()[0]

    def filter_multi(self):
        return self.con.execute(
            "SELECT id FROM t WHERE sensor>8 AND status='active' AND temp<35.0").fetchall()

    def teardown(self):
        try:
            self.con.close()
        except Exception:
            pass


def _tmp(suffix=".snap"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    os.remove(path)
    return path


def _cleanup(path):
    for p in (path, path.replace(".snap", ".wal")):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


# ── Runner ──────────────────────────────────────────────────────────────────

WORKLOADS = [
    ("insert", "Bulk insert", "rows/s"),
    ("point_read", "Point read (PK)", "ops/s"),
    ("scan_sum", "Full scan + SUM", "rows/s"),
    ("filter_multi", "3-cond filter", "rows/s"),
    ("memory", "Memory footprint", "MB"),
]

ENGINES = [SnapDBColumnar, SnapDBRow, SQLiteMem, PandasEngine, DuckDBEngine, DictEngine]


def run(rows_n: int, read_ops: int, repeat: int) -> dict:
    rows = _rows(rows_n)
    rnd = random.Random(7)
    keys = [rnd.randrange(rows_n) for _ in range(read_ops)]

    results: Dict[str, Dict[str, Optional[float]]] = {w[0]: {} for w in WORKLOADS}
    engine_names: List[str] = []

    for cls in ENGINES:
        eng = cls()
        if not getattr(eng, "available", True):
            continue
        engine_names.append(eng.name)
        try:
            # insert is also the setup; time it.
            t = _bench(lambda e=eng, r=rows: e.insert(r), repeat=1, warmup=0)
            results["insert"][eng.name] = rows_n / t

            for attr in ("point_read", "scan_sum", "filter_multi"):
                fn = getattr(eng, attr, None)
                if fn is None or type(eng).__dict__.get(attr) is None:
                    results.setdefault(attr, {})[eng.name] = None
                    continue
                try:
                    if attr == "point_read":
                        t = _bench(lambda f=fn: f(keys), repeat=repeat)
                        results["point_read"][eng.name] = read_ops / t
                    else:
                        t = _bench(fn, repeat=repeat)
                        results[attr][eng.name] = rows_n / t
                except NotImplementedError:
                    results.setdefault(attr, {})[eng.name] = None

            mem = eng.memory_bytes()
            results["memory"][eng.name] = (mem / 1024 / 1024) if mem else None
        finally:
            eng.teardown()
            gc.collect()

    return {
        "config": {
            "rows": rows_n, "read_ops": read_ops, "repeat": repeat,
            "python": sys.version.split()[0],
            "platform": sys.platform,
        },
        "engines": engine_names,
        "results": results,
    }


# ── Formatting ──────────────────────────────────────────────────────────────

def _fmt(val: Optional[float], unit: str) -> str:
    if val is None:
        return "n/a"
    if unit == "MB":
        return f"{val:,.1f}"
    return f"{val:,.0f}"


def to_table(data: dict) -> str:
    engines = data["engines"]
    w = max(18, *(len(e) for e in engines))
    head = "Workload".ljust(20) + "Unit".ljust(8) + "".join(e.ljust(w + 2) for e in engines)
    lines = [head, "-" * len(head)]
    for key, label, unit in WORKLOADS:
        row = label.ljust(20) + unit.ljust(8)
        for e in engines:
            row += _fmt(data["results"].get(key, {}).get(e), unit).ljust(w + 2)
        lines.append(row)
    return "\n".join(lines)


def to_markdown(data: dict) -> str:
    engines = data["engines"]
    cfg = data["config"]
    out = [
        f"_{cfg['rows']:,} rows · {cfg['read_ops']:,} point reads · "
        f"best of {cfg['repeat']} · Python {cfg['python']} · {cfg['platform']}. "
        f"Higher is better except Memory (lower is better)._",
        "",
        "| Workload | Unit | " + " | ".join(engines) + " |",
        "|---|---|" + "|".join(["---"] * len(engines)) + "|",
    ]
    for key, label, unit in WORKLOADS:
        cells = [_fmt(data["results"].get(key, {}).get(e), unit) for e in engines]
        out.append(f"| {label} | {unit} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="SnapDB benchmark suite")
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--read-ops", type=int, default=50_000)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--markdown", type=str, default=None)
    args = ap.parse_args()

    print(f"Running SnapDB benchmark — {args.rows:,} rows, "
          f"{args.read_ops:,} reads, best of {args.repeat} ...\n")
    data = run(args.rows, args.read_ops, args.repeat)
    print(to_table(data))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\nwrote {args.json}")
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(to_markdown(data) + "\n")
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
