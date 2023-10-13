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


# Check if the internal module is available
try:
    if internal_libs_available():
        from autonomi.nos._internal.version import __version__ as _internal_version  # noqa: F401, F403

        sys.modules["nos._internal"] = importlib.import_module("autonomi.nos._internal")
        logger.debug(f"`nos._internal` module [version={_internal_version}].")
except ModuleNotFoundError:
    logger.debug("Failed to load `nos._internal` module: ModuleNotFoundError")
except Exception as e:
    logger.debug(f"Failed to load `nos._internal` module: {e}")
