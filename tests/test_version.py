import re

from nos.version import __version__  # noqa: F401


def test_semver_syntax():
    """Test that the version string is a valid semantic version."""
    assert re.match(r"^\d+\.\d+\.\d+$", __version__)

    # Get major, minor, and patch version tuple
    major, minor, patch = tuple(int(x) for x in __version__.split("."))
    assert major == 0
    assert minor >= 0
    assert patch >= 0
