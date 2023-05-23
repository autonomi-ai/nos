import re

from nos.version import __version__  # noqa: F401


def test_semver_syntax():
    """Test that the version string is a valid semantic version.

    Allow pre-release versions (e.g. 0.0.4a) but not build metadata (e.g. 0.0.4+1).
    """
    assert re.match(r"^\d+\.\d+\.\d+[a-z]?$", __version__)

    # Get major, minor, and patch version tuple
    major, minor, patch = tuple(x for x in __version__.split("."))
    assert int(major) == 0
    assert int(minor) >= 0
