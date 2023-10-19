import pytest

from nos.test.conftest import (  # noqa: F401, E402
    GRPC_TEST_PORT,
    GRPC_TEST_PORT_CPU,
    GRPC_TEST_PORT_GPU,
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
