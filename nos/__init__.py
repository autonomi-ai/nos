import docker
from nos.version import __version__  # noqa: F401

from .client import InferenceClient  # noqa: F401
from .server import init, shutdown  # noqa: F401
