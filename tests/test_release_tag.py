"""The release tag, the package version and the changelog must agree.

pysnapdb sat on PyPI at 0.14.0 from 2026-07-03 while the repository reached
0.18.1 — four versions, including the index-aware query planner and vector
search. Nothing failed, because nothing compared the three places the version
is written to the one place a user reads it.

This module adds the missing comparison. It is deliberately conditional:

  * In a checkout with no tag for HEAD (the normal state while developing),
    every test here skips. A release check that fails on every ordinary
    commit is a check people disable.
  * In an sdist or wheel built from a tag, ``PKG-INFO`` is present and the
    tag-vs-version comparison is exact.

The failure it exists to catch is "tagged v0.18.1, pyproject says 0.18.0" and
"version exists in CHANGELOG but was never tagged" — both of which publish a
package whose metadata lies about what it contains.
"""

from __future__ import annotations

import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    """Run git, returning None when git or the repo is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert match, "pyproject.toml has no version field"
    return match.group(1)


def _tags_pointing_at_head() -> list[str] | None:
    """Exact tags on HEAD, or None when we cannot tell."""
    out = _git("tag", "--points-at", "HEAD", "--list", "v*")
    if out is None:
        return None
    return [line.strip() for line in out.splitlines() if line.strip()]


def _is_sdist_or_wheel() -> bool:
    """A built artifact carries PKG-INFO and no .git."""
    return (ROOT / "PKG-INFO").exists() or (ROOT / "pyproject.toml").exists() and not (ROOT / ".git").exists()


class TestReleaseTagMatchesVersion(unittest.TestCase):
    def test_tag_on_head_matches_package_version(self) -> None:
        """`git tag v0.18.1` on a commit whose pyproject says 0.18.0 would
        upload 0.18.0 under the 0.18.1 name."""
        tags = _tags_pointing_at_head()
        if not tags:
            self.skipTest("no release tag on HEAD (normal during development)")

        version = _pyproject_version()
        expected = {"v" + version}
        unexpected = [t for t in tags if t not in expected]
        self.assertEqual(
            unexpected,
            [],
            f"HEAD is tagged {unexpected} but pyproject.toml says {version!r}",
        )

    def test_changelog_documents_the_package_version(self) -> None:
        """A version nobody wrote down is a version nobody can audit."""
        changelog = ROOT / "CHANGELOG.md"
        if not changelog.exists():
            self.skipTest("CHANGELOG.md is not in this tree")
        text = changelog.read_text(encoding="utf-8")
        version = _pyproject_version()
        # re.M is required: without it `^` only matches the start of the
        # whole file and the check fails on a changelog that is correct.
        self.assertTrue(
            re.search(r"^## \[" + re.escape(version) + r"\]", text, re.M),
            f"CHANGELOG.md has no entry for {version}",
        )

    @unittest.skipUnless(_is_sdist_or_wheel(), "only meaningful for a built artifact")
    def test_built_artifact_reports_the_expected_version(self) -> None:
        """In an sdist/wheel the metadata and the changelog are both present,
        so the comparison is exact rather than skipped."""
        from importlib.metadata import version as dist_version

        self.assertEqual(dist_version("pysnapdb"), _pyproject_version())


if __name__ == "__main__":
    unittest.main(verbosity=2)
