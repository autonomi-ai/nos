import docker
from nos.version import __version__  # noqa: F401

from .client import InferenceClient  # noqa: F401
from .server import init, shutdown  # noqa: F401

# Check if the internal module is available
_internal_available = False
try:
    from autonomi.nos._internal import compile  # noqa: F401, F403
    _internal_available = True
except ImportError:
    pass