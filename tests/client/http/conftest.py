import pytest

from nos.test.conftest import (  # noqa: F401, E402
    GRPC_TEST_PORT,
    grpc_server,
)


@pytest.fixture(scope="session")
def http_server(grpc_server):  # noqa: F811
    """Start a local uvicorn HTTP server process alonside the gRPC server."""
    from multiprocessing import Process

    import uvicorn

    from nos.server.http._service import app

    def _run_uvicorn_server():
        uvicorn.run(app(), host="localhost", port=8000, workers=1, log_level="info")

    proc = Process(target=_run_uvicorn_server, args=(), daemon=True)
    proc.start()
    yield
    proc.kill()


@pytest.fixture(scope="session")
def http_client():
    from fastapi.testclient import TestClient

    from nos.server.http._service import app

    # Create a test HTTP client fixture
    with TestClient(app(grpc_port=GRPC_TEST_PORT)) as _client:
        yield _client


@pytest.fixture(scope="session")
def local_http_client_with_server(grpc_server, http_client):  # noqa: F811
    """Test local HTTP client with local runtime."""
    # Yield the HTTP client once the server is up and initialized
    yield http_client
