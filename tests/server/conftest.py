import pytest

from nos.logging import logger
from nos.server.train._runtime import CustomServiceRuntime
from nos.test.conftest import DEFAULT_GRPC_PORT, GRPC_TEST_PORT_GPU, ray_executor  # noqa: F401
from nos.test.utils import skip_if_no_torch_cuda


pytestmark = pytest.mark.server


def custom_runtime(runtime_name: str):
    runtime = CustomServiceRuntime(runtime=runtime_name, name=f"nos-custom-service-runtime-{runtime_name}-test")
    assert runtime is not None

    containers = runtime.list()
    assert containers is not None

    logger.debug(f"Starting custom service runtime: {runtime_name}")
    runtime.start(ports={DEFAULT_GRPC_PORT: GRPC_TEST_PORT_GPU, 8265: 8265})
    assert runtime.get_container() is not None
    assert runtime.get_container_id() is not None
    assert runtime.get_container_name() is not None
    assert runtime.get_container_status() is not None

    yield runtime

    logger.debug(f"Stopping custom service runtime: {runtime_name}")
    runtime.stop()
    logger.debug(f"Stopped custom service runtime: {runtime_name}")


@skip_if_no_torch_cuda
@pytest.fixture(scope="session")
def grpc_server_docker_runtime_mmdet_gpu():
    """Fixture for starting a gRPC server with the mmdet-gpu docker runtime."""
    yield from custom_runtime("mmdet-gpu")


@skip_if_no_torch_cuda
@pytest.fixture(scope="session")
def grpc_server_docker_runtime_diffusers_gpu():
    """Fixture for starting a gRPC server with the diffusers-gpu docker runtime."""
    yield from custom_runtime("diffusers-gpu")
