from nos.logging import logger  # noqa: F401
from nos.logging import build_logger  # noqa: F401


def test_default_logger():
    """Test that the default logger is not None."""
    assert logger is not None


def test_build_logger():
    """Test that the build_logger function returns a logger."""
    assert build_logger(__name__) is not None
