# import pytest
# from nos.test.conftest import (  # noqa: F401, E402  # noqa: F401, E402
#     GRPC_TEST_PORT,
#     http_client,
#     local_grpc_client_with_server,
#     local_http_client_with_server,
#     ray_executor,
# )


# @pytest.fixture(scope="session")
# def http_client():
#     from fastapi.testclient import TestClient

#     from nos.server.http._service import app

#     # Create a test HTTP client fixture
#     with TestClient(app(grpc_port=GRPC_TEST_PORT)) as _client:
#         yield _client


# @pytest.fixture(scope="session")
# def local_http_client_with_server(grpc_server, http_client):  # noqa: F811
#     """Test local HTTP client with local runtime."""
#     # Yield the HTTP client once the server is up and initialized
#     yield http_client
