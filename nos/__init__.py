import importlib
import sys

import docker
from nos.version import __version__  # noqa: F401

from .client import InferenceClient  # noqa: F401
from .logging import logger  # noqa: F401
from .server import init, shutdown  # noqa: F401


# Check if the internal module is available
_internal_libs_available = False
try:
    from autonomi.nos._internal import compile  # noqa: F401, F403
    from autonomi.nos._internal.version import __version__ as internal_version  # noqa: F401, F403

    sys.modules["nos._internal"] = importlib.import_module("autonomi.nos._internal")
    logger.debug(f"`nos._internal` module [version={internal_version}].")

    _internal_libs_available = True
except ImportError as e:
    logger.debug(f"Failed to import internal module [e={e}]")
    compile = None


def internal_libs_available():
    """Check if the internal module is available."""
    return _internal_libs_available
