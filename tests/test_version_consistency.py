"""Every place that states snapdb's version must agree.

`package.json` sat at 0.11.0 for seven releases while `pyproject.toml`,
`snapdb.__version__` and the CHANGELOG were at 0.18.0. Nothing read the file,
so nothing failed - which is exactly why this test exists.
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def pyproject_version():
    return re.search(r'^version\s*=\s*"([^"]+)"', read("pyproject.toml"), re.M).group(1)


def module_version():
    path = os.path.join(ROOT, "snapdb", "__init__.py")
    with open(path, encoding="utf-8") as fh:
        return re.search(r'^__version__\s*=\s*"([^"]+)"', fh.read(), re.M).group(1)


def package_json_version():
    return json.loads(read("package.json"))["version"]


def release_history():
    """Every `## [x.y.z]` heading in the CHANGELOG."""
    return re.findall(r"^## \[([0-9]+\.[0-9]+\.[0-9]+)\]", read("CHANGELOG.md"), re.M)


def test_pyproject_matches_module():
    assert pyproject_version() == module_version()


def test_package_json_matches_pyproject():
    # The file is not read by the Python build or by any workflow, so a stale
    # value is invisible in CI. That is the whole failure mode.
    assert package_json_version() == pyproject_version()


def test_package_json_name_matches_pyproject():
    assert json.loads(read("package.json"))["name"] == "pysnapdb"


def test_changelog_documents_the_shipped_version():
    assert pyproject_version() in release_history(), (
        "CHANGELOG.md has no ## [%s] entry" % pyproject_version()
    )


def test_no_version_regression():
    # A released version must never be lower than the newest changelog entry.
    def key(v):
        return tuple(int(x) for x in v.split("."))

    newest = max(release_history(), key=key)
    assert key(pyproject_version()) >= key(newest), (
        "shipped %s is behind CHANGELOG %s" % (pyproject_version(), newest)
    )
