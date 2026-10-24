"""Version consistency: the published package version must match the code.

A pyproject.toml version that lags the module's __version__ ships a
package that reports the wrong version at runtime, and breaks every
"pip show" / importlib.metadata check a downstream user makes.

This asserts the contract rather than a literal version, so it keeps
working across future bumps instead of breaking on every release.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


def _find_pyproject() -> Path:
    """Walk up from this file until a pyproject.toml shows up."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        pyproject = candidate / "pyproject.toml"
        if pyproject.exists():
            return pyproject
    raise AssertionError("no pyproject.toml found above " + str(here))


PYPROJECT = _find_pyproject()


def _pyproject_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert match, "pyproject.toml has no version field"
    return match.group(1)


class TestVersionConsistency(unittest.TestCase):
    def test_pyproject_matches_module_version(self) -> None:
        import snapdb

        self.assertEqual(
            _pyproject_version(),
            snapdb.__version__,
            "pyproject.toml version does not match snapdb.__version__",
        )

    def test_version_is_semver(self) -> None:
        version = _pyproject_version()
        self.assertRegex(
            version,
            r"^\d+\.\d+\.\d+$",
            "version must be plain semver, got " + version,
        )
