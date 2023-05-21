import time

import pytest
from loguru import logger

from nos.test.conftest import (  # noqa: F401, E402  # noqa: F401, E402
    grpc_client_cpu,
    grpc_client_gpu,
    grpc_server_docker_runtime_cpu,
    grpc_server_docker_runtime_gpu,
)


@pytest.fixture(scope="session")
def grpc_client_with_cpu_backend(grpc_server_docker_runtime_cpu, grpc_client_cpu):  # noqa: F811
    """Test gRPC client with initialized CPU docker runtime (Port: 50053)."""
    # Wait for server to start
    st = time.time()
    while time.time() - st <= 180:
        try:
            # TODO (spillai): Replace with a better health check
            grpc_client_cpu.ListModels()
            break
        except Exception:
            logger.warning("Waiting for server to start... (elapsed={:.0f}s)".format(time.time() - st))
            time.sleep(5)
    logger.info("Server started!")

    # Yield the gRPC client once the server is up and initialized
    yield grpc_client_cpu


@pytest.fixture(scope="session")
def grpc_client_with_gpu_backend(grpc_server_docker_runtime_gpu, grpc_client_gpu):  # noqa: F811
    """Test gRPC client with initialized CPU docker runtime (Port: 50054)."""
    # Wait for server to start
    st = time.time()
    while time.time() - st <= 180:
        try:
            # TODO (spillai): Replace with a better health check
            grpc_client_gpu.ListModels()
            break
        except Exception:
            logger.warning("Waiting for server to start... (elapsed={:.0f}s)".format(time.time() - st))
            time.sleep(5)
    logger.info("Server started!")

    # Yield the gRPC client once the server is up and initialized
    yield grpc_client_gpu
