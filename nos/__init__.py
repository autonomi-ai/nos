import importlib
import sys

from nos.version import __version__  # noqa: F401

from .client import Client  # noqa: F401
from .logging import logger  # noqa: F401
from .server import init, shutdown  # noqa: F401


def internal_libs_available():
    """Check if the internal module is available."""
    from .common.runtime import is_package_available  # noqa: F401

    return is_package_available("autonomi.nos._internal")
