"""Pytest configuration.

A few legacy test modules are written as standalone scripts (they define a
``def test(...)`` helper and a ``__main__`` runner, and assert via a custom
counter). Pytest would mis-collect the helper, so we skip those files here and
run them directly as scripts in CI instead. The unittest / pytest-style modules
(test_optimizations, test_features, test_v02, test_document_store) collect
normally.
"""

collect_ignore = [
    "test_snapdb.py",
    "test_schema_fast.py",
    "test_delta_encoding.py",
    "test_dict_encoding.py",
]
