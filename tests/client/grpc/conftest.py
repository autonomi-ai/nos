import pytest
from loguru import logger

from nos.test.conftest import (  # noqa: F401, E402  # noqa: F401, E402
    grpc_client,
    grpc_client_cpu,
    grpc_client_gpu,
    grpc_server,
    grpc_server_docker_runtime_cpu,
    grpc_server_docker_runtime_gpu,
    ray_executor,
)


@pytest.fixture(scope="session")
def local_grpc_client_with_server(grpc_server, grpc_client):  # noqa: F811
    """Test local gRPC client with local runtime (Port: 50052)."""
    # Wait for server to start
    grpc_client.WaitForServer(timeout=180, retry_interval=5)
    logger.info("Server started!")

    # Yield the gRPC client once the server is up and initialized
    yield grpc_client


@pytest.fixture(scope="session")
def grpc_client_with_cpu_backend(grpc_server_docker_runtime_cpu, grpc_client_cpu):  # noqa: F811
    """Test gRPC client with initialized CPU docker runtime (Port: 50053)."""
    # Wait for server to start
    grpc_client_cpu.WaitForServer(timeout=180, retry_interval=5)
    logger.info("Server started!")

    # Yield the gRPC client once the server is up and initialized
    yield grpc_client_cpu


@pytest.fixture(scope="session")
def grpc_client_with_gpu_backend(grpc_server_docker_runtime_gpu, grpc_client_gpu):  # noqa: F811
    """Test gRPC client with initialized CPU docker runtime (Port: 50054)."""
    # Wait for server to start
    grpc_client_gpu.WaitForServer(timeout=180, retry_interval=5)
    logger.info("Server started!")

    # Yield the gRPC client once the server is up and initialized
    yield grpc_client_gpu
