import time

import pytest

import nos
from nos.client import Client
from nos.constants import DEFAULT_GRPC_HOST
from nos.logging import logger
from nos.server import InferenceServiceRuntime
from nos.test.conftest import grpc_client  # noqa: F401
from nos.test.utils import AVAILABLE_RUNTIMES


logger.debug(f"AVAILABLE_RUNTIMES={AVAILABLE_RUNTIMES}")


@pytest.mark.client
def test_client_cloudpickle_serialization(grpc_client):  # noqa: F811
    """Test cloudpickle serialization."""
    from nos.common.cloudpickle import dumps

    stub = grpc_client.stub  # noqa: F841

    def predict_wrap():
        return grpc_client.Run(
            "openai/clip-vit-base-patch32",
            inputs={"texts": "This is a test"},
        )

    predict_fn = dumps(predict_wrap)
    assert isinstance(predict_fn, bytes)

    def predict_module_wrap():
        module = grpc_client.Module("openai/clip-vit-base-patch32")
        return module(inputs={"prompts": ["This is a test"]})

    predict_fn = dumps(predict_module_wrap)
    assert isinstance(predict_fn, bytes)

    def train_wrap():
        return grpc_client.Train(
            method="stable-diffusion-dreambooth-lora",
            inputs={
                "model_name": "stabilityai/stable-diffusion-2-1",
                "instance_directory": "/tmp",
                "instance_prompt": "A photo of a bench on the moon",
            },
            metadata={
                "name": "sdv21-dreambooth-lora-test-bench",
            },
        )

    train_fn = dumps(train_wrap)
    assert isinstance(train_fn, bytes)


@pytest.mark.client
@pytest.mark.parametrize("runtime", AVAILABLE_RUNTIMES)
def test_grpc_client_init(runtime):  # noqa: F811
    """Test the NOS server daemon initialization."""
    GRPC_PORT = 50055

    # Initialize the server
    container = nos.init(runtime=runtime, port=GRPC_PORT, utilization=0.5)
    assert container is not None
    assert container.id is not None
    containers = InferenceServiceRuntime.list()
    assert len(containers) == 1

    # Test waiting for server to start
    # This call should be instantaneous as the server is already ready for the test
    client = Client(f"{DEFAULT_GRPC_HOST}:{GRPC_PORT}")
    assert client.WaitForServer(timeout=180, retry_interval=5)
    assert client.IsHealthy()

    # Test re-initializing the server
    st = time.time()
    container_ = nos.init(runtime=runtime, port=GRPC_PORT, utilization=0.5)
    assert container_.id == container.id
    assert time.time() - st <= 0.5, "Re-initializing the server should be instantaneous, instead took > 0.5s"

    # Shutdown the server
    nos.shutdown()
    containers = InferenceServiceRuntime.list()
    assert len(containers) == 0
