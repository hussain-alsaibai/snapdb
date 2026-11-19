"""The package-level `query` name must be the function, not the submodule.

`snapdb/query.py` defines a function called `query()` and lives at a path
whose last segment is also `query`. If `__init__.py` does not import the
function explicitly, Python's `from snapdb import query` finds no attribute
on the package, falls through to the submodule loader, and binds the
**module**. Nothing raises. The README's own index example then fails with
`TypeError: 'module' object is not callable`, and the failure looks like the
user's mistake.

This was invisible for four releases because every test imports the function
from its defining module (`from snapdb.query import query`), which always
works. Only the documented import path was broken.
"""

import pathlib
import types


import snapdb


def test_package_query_is_callable():
    assert callable(snapdb.query), (
        "snapdb.query is a module, not the query() function - "
        "__init__.py must import it explicitly"
    )
    assert not isinstance(snapdb.query, types.ModuleType)


def test_documented_import_path_works():
    """Exactly the line the README tells users to write."""
    from snapdb import query

    assert isinstance(query, types.FunctionType)


def test_query_class_is_exported():
    from snapdb import Query

    assert isinstance(Query, type)


def test_query_is_in_dunder_all():
    assert "query" in snapdb.__all__
    assert "Query" in snapdb.__all__


def test_readme_example_runs(tmp_path):
    """The README block, executed end to end."""
    from snapdb import ColumnDef, Schema, SnapDB, query

    db = SnapDB(str(tmp_path / "q.snap"),
                Schema([ColumnDef("id", "i32"), ColumnDef("age", "i32")]))
    for i in range(100):
        db.insert({"id": i, "age": i % 70})

    db.create_range_index("age")
    q = query(db).between("age", 25, 30)
    assert "RangeIndex" in q.explain()
    rows = [r for _, r in q.execute()]
    assert rows, "documented query returned nothing"
    # between() defaults to inclusive on BOTH ends (gte/lte) - see the
    # docstring. An exclusive upper bound needs include_high=False.
    assert all(25 <= r["age"] <= 30 for r in rows)
    assert q.count() == len(rows)

    exclusive = [r for _, r in query(db).between("age", 25, 30,
                                                include_high=False).execute()]
    assert all(r["age"] < 30 for r in exclusive)
    assert len(exclusive) < len(rows), "include_high=False changed nothing"
    db.close()


def test_agent_memory_example_is_executable():
    """examples/agent_memory.py must be syntax-valid and importable as source.

    The example is documentation that pretends to be code, so it drifts the
    same way a README does. Compiling it catches that at PR time.
    """
    path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "agent_memory.py"
    assert path.exists(), "examples/agent_memory.py is gone"
    compile(path.read_text(), str(path), "exec")
