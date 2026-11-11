"""Guard: the sdist must be self-consistent and its suite must run as shipped.

Three defects shipped in pysnapdb 0.18.0 on PyPI and none of them was visible
in a checkout, because a checkout has files an sdist does not:

1. ``tests/test_version_consistency.py`` read ``CHANGELOG.md`` and
   ``package.json`` from the repo root, and neither is in the sdist
   (2 failures after ``pip download && pytest``).
2. ``tests/test_delta_encoding.py`` and ``tests/test_dict_encoding.py`` define
   a helper called ``test()`` whose first parameter is ``name``, which pytest
   collects as a test nobody can call (``fixture 'name' not found``).
   ``pytest tests/ -q`` reported 2 errors on every CI run; they were tolerated
   because the run was not ``-x`` and nothing read the collected-error count.
3. Nothing compared the two halves, so (1) and (2) both survived a green build.

This file is that comparison. The packaging assertions only run inside an
unpacked sdist (``PKG-INFO`` beside the package); in a checkout they would be
asserting something the checkout cannot see.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
PKG_INFO = ROOT / "PKG-INFO"
IS_SDIST = PKG_INFO.is_file()


def _field(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\s*=\s*(.+)$", text, re.MULTILINE)
    assert match, f"pyproject.toml has no {name}"
    return match.group(1).strip().strip('"')


class TestSdistCompleteness:
    """Only meaningful inside an unpacked sdist."""

    @pytest.mark.skipif(not IS_SDIST, reason="not running from an unpacked sdist")
    def test_files_the_suite_reads_are_shipped(self) -> None:
        """Anything the suite reads from the repo root must be in the tarball."""
        missing = [
            name
            for name in ("CHANGELOG.md", "package.json")
            if not (ROOT / name).is_file()
        ]
        assert not missing, (
            "sdist is missing files the test suite reads: "
            f"{missing}. Ship them in MANIFEST.in or stop reading them from "
            "the repo root."
        )

    @pytest.mark.skipif(not IS_SDIST, reason="not running from an unpacked sdist")
    def test_sdist_version_matches_pkg_info(self) -> None:
        text = PKG_INFO.read_text(encoding="utf-8")
        match = re.search(r"^Version:\s*(\S+)", text, re.MULTILINE)
        assert match, "PKG-INFO has no Version field"
        assert _field(PYPROJECT.read_text(encoding="utf-8"), "version") == match.group(1)


class TestSuiteHasNoCollectorSmoke:
    """A module-level helper named ``test`` is collected by pytest, not called."""

    def test_no_collectable_helper_shadows_pytest(self) -> None:
        offenders = []
        for path in sorted((ROOT / "tests").glob("test_*.py")):
            text = path.read_text(encoding="utf-8")
            # `def test(...)` at module level is collected with its first
            # parameter as a fixture name. That is never a test — it is a
            # helper that happens to be named like one.
            if re.search(r"^def test\(", text, re.MULTILINE):
                offenders.append(path.name)
        assert not offenders, (
            "these test modules define a pytest-collectable helper named "
            f"`test(...)`: {offenders}. Rename it to `check(...)`."
        )
