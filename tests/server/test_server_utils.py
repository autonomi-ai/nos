import logging

import psutil
import pytest

import nos
from nos.common.system import has_docker
from nos.server import InferenceServiceRuntime


# Skip this entire test if docker is not installed
num_cpus = psutil.cpu_count(logical=False)
pytestmark = pytest.mark.skipif(not has_docker() or num_cpus < 4, reason="docker is not installed")


def test_nos_init():
    """Test the NOS server daemon initialization.

    See tests/client/test_client_integration.py for the end-to-end integration with client.
    """

    # Initialize the server
    container_1 = nos.init(logging_level=logging.DEBUG)
    assert container_1 is not None
    container_2 = nos.init(logging_level="DEBUG")
    assert container_2 is not None
    assert container_1.id == container_2.id

    containers = InferenceServiceRuntime.list()
    assert len(containers) == 1

    # Shutdown the server
    nos.shutdown()

    containers = InferenceServiceRuntime.list()
    assert len(containers) == 0

    # Shutdown the server again (should not raise an error)
    nos.shutdown()


def test_nos_init_variants():
    """Test the NOS server daemon initialization variants."""
    import logging

    with pytest.raises(ValueError):
        nos.init(runtime="invalid")

    for utilization in [-1, 0.0, 1.1]:
        with pytest.raises(ValueError):
            nos.init(utilization=utilization)

    with pytest.raises(ValueError):
        nos.init(logging_level="invalid")

    with pytest.raises(ValueError):
        nos.init(logging_level=0)

    with pytest.raises(ValueError):
        nos.init(tag=0)

    with pytest.raises(NotImplementedError):
        nos.init(tag="latest")

    with pytest.raises(ValueError):
        nos.init(logging_level="invalid")

    nos.init(runtime="cpu", utilization=0.5, logging_level=logging.INFO)
    nos.shutdown()
