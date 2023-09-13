import pytest

from nos.logging import logger
from nos.server.train._runtime import CustomServiceRuntime
from nos.test.conftest import DEFAULT_GRPC_PORT, GRPC_TEST_PORT_GPU, ray_executor  # noqa: F401
from nos.test.utils import skip_if_no_torch_cuda


@skip_if_no_torch_cuda
@pytest.fixture(scope="session")
def grpc_server_docker_runtime_mmdet_gpu():
    """Fixture for starting a gRPC server with the mmdet-gpu docker runtime."""
    RUNTIME = "mmdet-gpu"
    runtime = CustomServiceRuntime(runtime=RUNTIME, name=f"nos-custom-service-runtime-{RUNTIME}-test")
    assert runtime is not None

    containers = runtime.list()
    assert containers is not None

    logger.debug(f"Starting custom service runtime: {RUNTIME}")
    runtime.start(ports={DEFAULT_GRPC_PORT: GRPC_TEST_PORT_GPU, 8265: 8265})
    assert runtime.get_container() is not None
    assert runtime.get_container_id() is not None
    assert runtime.get_container_name() is not None
    assert runtime.get_container_status() is not None

    yield runtime
    logger.debug(f"Stopping custom service runtime: {RUNTIME}")
    runtime.stop()
    logger.debug(f"Stopped custom service runtime: {RUNTIME}")


@pytest.fixture(scope="session")
def grpc_client_with_mmdet_gpu_backend(grpc_server_docker_runtime_mmdet_gpu, grpc_client_gpu):  # noqa: F811
    """Fixture for starting a gRPC client with the mmdet-gpu docker runtime."""
    # Wait for server to start
    grpc_client_gpu.WaitForServer(timeout=180, retry_interval=5)
    logger.info("Server started!")

    # Yield the gRPC client once the server is up and initialized
    yield grpc_client_gpu
