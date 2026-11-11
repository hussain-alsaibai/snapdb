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

class TestTrackedVersionSurfaces(unittest.TestCase):
    """package.json is tracked but nothing reads it.

    It is not the Python build, not the release workflow, not any test - so its
    version drifted seven releases behind (0.11.0 against 0.18.0) with no
    failure anywhere. A version surface that no consumer reads is still a
    surface a reader trusts.

    Both surfaces are *repo* metadata. ``package.json`` and ``CHANGELOG.md``
    are not in the sdist, so inside an unpacked sdist (the bytes a user gets
    from PyPI) these checks cannot run and must skip rather than fail. That is
    not a licence to stop checking them: the checkout half of the same suite
    still enforces both, and CI runs from a checkout.
    """

    def _optional_text(self, name: str) -> str:
        """Read a repo-root file, skipping when it is not in this distribution."""
        path = PYPROJECT.parent / name
        if not path.exists():
            self.skipTest(f"{name} is not part of the sdist")
        return path.read_text(encoding="utf-8")

    def _package_json_version(self) -> str:
        import json

        return json.loads(self._optional_text("package.json"))["version"]

    def test_package_json_matches_pyproject(self) -> None:
        self.assertEqual(
            self._package_json_version(),
            _pyproject_version(),
            "package.json version does not match pyproject.toml",
        )

    def test_changelog_documents_the_shipped_version(self) -> None:
        changelog = self._optional_text("CHANGELOG.md")
        version = _pyproject_version()
        self.assertRegex(
            changelog,
            r"(?m)^## \[%s\]" % re.escape(version),
            "CHANGELOG.md has no ## [%s] entry" % version,
        )

    def test_shipped_version_is_not_behind_the_changelog(self) -> None:
        changelog = self._optional_text("CHANGELOG.md")
        released = re.findall(r"(?m)^## \[([0-9]+\.[0-9]+\.[0-9]+)\]", changelog)

        def key(v: str) -> tuple:
            return tuple(int(part) for part in v.split("."))

        newest = max(released, key=key)
        self.assertGreaterEqual(
            key(_pyproject_version()),
            key(newest),
            "shipped %s is behind CHANGELOG %s" % (_pyproject_version(), newest),
        )
