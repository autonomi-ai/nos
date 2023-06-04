import pytest
from loguru import logger

from nos.test.conftest import (  # noqa: F401, E402  # noqa: F401, E402
    grpc_client,
    grpc_server,
    ray_executor,
)


@pytest.fixture(scope="session")
def grpc_client_with_server(grpc_server, grpc_client):  # noqa: F811
    """Test gRPC client with local gRPC server (Port: 50052)."""
    # Wait for server to start
    grpc_client.WaitForServer(timeout=180, retry_interval=5)
    logger.info("Server started!")

    # Yield the gRPC client once the server is up and initialized
    yield grpc_client
