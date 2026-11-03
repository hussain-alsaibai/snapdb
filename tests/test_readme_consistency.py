"""Guard against README/API drift.

The README is a promise about the public API. If a documented constructor
argument or method does not exist, every reader who copies the snippet gets
an AttributeError/TypeError at runtime. This test parses every Python block
in the README and checks it against the real, introspected API.

Issue #37.
"""

import ast
import inspect
import re
import pathlib

import pytest

import snapdb
from snapdb import SnapDB

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"

# Constructors the README is allowed to call, mapped to the class they mean.
CONSTRUCTORS = {
    "SnapDB": SnapDB,
    "Schema": snapdb.Schema,
    "ColumnDef": snapdb.ColumnDef,
    "ColumnarTable": snapdb.ColumnarTable,
    "VectorIndex": snapdb.VectorIndex,
    "Metrics": snapdb.Metrics,
}

# Methods that live on a helper object rather than on SnapDB itself.
HELPER_TYPES = (snapdb.ColumnarTable, snapdb.VectorIndex, snapdb.Schema, snapdb.Metrics)

# Names bound by the README snippets to a SnapDB/ColumnarTable instance.
KNOWN_VARS = {"db", "store", "audit", "tool_cache", "old", "restored", "table"}

FENCE = chr(96) * 3


def _blocks():
    text = README.read_text(encoding="utf-8")
    blocks = re.findall(FENCE + r"python\n(.*?)" + FENCE, text, re.S)
    assert blocks, "README contains no python code blocks"
    return blocks


def test_readme_exists():
    assert README.is_file(), f"{README} is missing"


def test_readme_python_blocks_parse():
    """Every python block must at least be valid Python."""
    for idx, block in enumerate(_blocks()):
        try:
            ast.parse(block)
        except SyntaxError as exc:
            pytest.fail(f"README python block {idx} does not parse: {exc}")


def test_readme_imports_are_exported():
    """Every 'from snapdb import X' in the README must resolve."""
    for idx, block in enumerate(_blocks()):
        tree = ast.parse(block)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "snapdb":
                for alias in node.names:
                    assert hasattr(snapdb, alias.name), (
                        f"README block {idx} imports {alias.name!r} from snapdb, "
                        "but it is not exported"
                    )


def test_readme_constructor_kwargs_exist():
    """Every keyword argument in a README constructor call must be real."""
    for idx, block in enumerate(_blocks()):
        tree = ast.parse(block)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else None
            target = CONSTRUCTORS.get(name)
            if target is None:
                continue
            params = set(inspect.signature(target.__init__).parameters) - {"self"}
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                assert kw.arg in params, (
                    f"README block {idx}: {name}(...) accepts no keyword "
                    f"{kw.arg!r}; valid: {sorted(params)}"
                )


def test_readme_methods_exist():
    """Every method the README calls on a db/table must exist somewhere real."""
    for idx, block in enumerate(_blocks()):
        tree = ast.parse(block)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            value = func.value
            if not isinstance(value, ast.Name) or value.id not in KNOWN_VARS:
                continue
            attr = func.attr
            if hasattr(SnapDB, attr):
                continue
            if any(hasattr(t, attr) for t in HELPER_TYPES):
                continue
            pytest.fail(
                f"README block {idx} calls {value.id}.{attr}(), but no public "
                "SnapDB/ColumnarTable/VectorIndex/Schema/Metrics member has "
                "that name (documented API drift)"
            )


def test_readme_mentions_no_removed_apis():
    """Guard the specific regressions this test was written for."""
    text = README.read_text(encoding="utf-8")
    for gone in (
        "memory_store",
        "row_store",
        "columnar_store",
        "db.restore()",
        "top_k=",
    ):
        assert gone not in text, (
            f"README still references {gone!r}, which is not part of the "
            "shipped API"
        )


VALID_DTYPES = {
    "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64",
    "f32", "f64",
    "bool",
}


def _dtype_is_real(dtype):
    """bytes:N is accepted for any N; the rest must be a known engine dtype."""
    if dtype.startswith("bytes:"):
        return dtype.split(":")[1].isdigit()
    return dtype in VALID_DTYPES


def test_readme_documented_dtypes_are_real():
    """Every dtype string used in a README ColumnDef/ColumnarTable must exist."""
    for idx, block in enumerate(_blocks()):
        tree = ast.parse(block)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else None

            if name == "ColumnDef":
                # ColumnDef(name, dtype, ...) - the dtype is the 2nd argument.
                if len(node.args) < 2:
                    continue
                dtype_node = node.args[1]
                if not (
                    isinstance(dtype_node, ast.Constant)
                    and isinstance(dtype_node.value, str)
                ):
                    continue
                dtype = dtype_node.value
                assert _dtype_is_real(dtype), (
                    f"README block {idx} uses dtype {dtype!r}, which the "
                    f"engine does not support (valid: {sorted(VALID_DTYPES)})"
                )

            elif name == "ColumnarTable":
                # ColumnarTable(name, schema) where schema is a list of
                # (column_name, dtype) tuples.
                if len(node.args) < 2:
                    continue
                schema_node = node.args[1]
                if not isinstance(schema_node, (ast.List, ast.Tuple)):
                    continue
                for entry in schema_node.elts:
                    if not isinstance(entry, (ast.List, ast.Tuple)):
                        continue
                    if len(entry.elts) < 2:
                        continue
                    dtype_node = entry.elts[1]
                    if not (
                        isinstance(dtype_node, ast.Constant)
                        and isinstance(dtype_node.value, str)
                    ):
                        continue
                    dtype = dtype_node.value
                    assert _dtype_is_real(dtype), (
                        f"README block {idx} uses dtype {dtype!r}, which the "
                        f"engine does not support (valid: {sorted(VALID_DTYPES)})"
                    )
