import logging
from pathlib import Path

import numpy as np
import psutil
import pytest
from fastapi.responses import FileResponse
from PIL import Image

import nos
from nos.common.system import has_docker
from nos.server import InferenceServiceRuntime
from nos.server.http._utils import decode_item, encode_item
from nos.test.utils import AVAILABLE_RUNTIMES


# Skip this entire test if docker is not installed
NUM_CPUS = psutil.cpu_count(logical=False)
pytestmark = pytest.mark.skipif(not has_docker() or NUM_CPUS < 4, reason="docker is not installed")


def test_encode_decode_item():
    # Test encoding a dictionary
    input_dict = {"key1": "value1", "key2": "value2"}
    expected_dict = {"key1": "value1", "key2": "value2"}
    assert encode_item(input_dict) == expected_dict

    # Test encoding a list
    input_list = [1, 2, 3]
    expected_list = [1, 2, 3]
    assert encode_item(input_list) == expected_list

    # Test encoding a tuple
    input_tuple = (4, 5, 6)
    expected_tuple = [4, 5, 6]
    assert encode_item(input_tuple) == expected_tuple

    # Test encoding an Image object
    input_image = Image.new("RGB", (100, 100))
    assert encode_item(input_image).startswith("data:image/")
    assert (decode_item(encode_item(input_image)) == input_image).all()

    # Test encoding an ndarray object
    input_ndarray = np.array([1, 2, 3])
    expected_ndarray = [1, 2, 3]
    assert encode_item(input_ndarray) == expected_ndarray

    # Test encoding a 3D ndarray object
    input_ndarray = np.random.rand(3, 3, 3)
    assert encode_item(input_ndarray).startswith("data:application/numpy;base64,")
    assert (decode_item(encode_item(input_ndarray)) == input_ndarray).all()

    # Test encoding a Path object
    input_path = Path("/path/to/file.txt")
    assert isinstance(encode_item(input_path), FileResponse)


@pytest.mark.parametrize("runtime", AVAILABLE_RUNTIMES)
def test_nos_init(runtime):
    """Test the NOS server daemon initialization.

    See tests/client/test_client_integration.py for the end-to-end integration with client.
    """

    # Initialize the server
    container_1 = nos.init(runtime=runtime, logging_level=logging.DEBUG)
    assert container_1 is not None

    containers = InferenceServiceRuntime.list()
    assert len(containers) == 1

    container_2 = nos.init(runtime=runtime, logging_level="DEBUG")
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


def test_nos_init_local():
    import ray

    nos.init(runtime="local", logging_level=logging.DEBUG)
    assert ray.is_initialized(), "Ray should be initialized."


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
