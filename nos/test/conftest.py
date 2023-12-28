import pytest
from loguru import logger

from nos.constants import DEFAULT_GRPC_PORT
from nos.protoc import import_module


GRPC_TEST_PORT = DEFAULT_GRPC_PORT + 1
GRPC_TEST_PORT_CPU = DEFAULT_GRPC_PORT + 2
GRPC_TEST_PORT_GPU = DEFAULT_GRPC_PORT + 3

HTTP_TEST_PORT = 8001


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


@pytest.fixture(scope="session")
def ray_executor():
    from nos.executors.ray import RayExecutor

    executor = RayExecutor.get()
    executor.init()

    yield executor

    executor.stop()


@pytest.fixture
def model_manager(ray_executor):  # noqa: F811
    """Model manager fixture for testing purposes.

    Note (spillai): This is currently scoped to the object-level to avoid
    issues with the Ray runtime. This will be scoped to the session-level
    once we have a proper Ray runtime for testing.
    """
    from nos.managers import ModelManager

    manager = ModelManager()
    assert manager is not None

    yield manager


@pytest.fixture(scope="session")
async def grpc_server(ray_executor):
    """Test gRPC server (Port: 50052)."""
    from grpc import aio
    from loguru import logger

    from nos.constants import (  # noqa F401
        GRPC_MAX_MESSAGE_LENGTH,
    )
    from nos.server._service import InferenceServiceImpl

    logger.info(f"Starting gRPC test server on port: {GRPC_TEST_PORT}")
    options = [
        ("grpc.max_message_length", GRPC_MAX_MESSAGE_LENGTH),
        ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
        ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ]
    server = aio.server(options=options)
    address = f"[::]:{GRPC_TEST_PORT}"
    nos_service_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServiceImpl(), server)
    server.add_insecure_port(address)

    await server.start()
    assert server is not None

    yield server
    server.stop(grace=None)


@pytest.fixture(scope="session")
def grpc_client():
    """Test gRPC client (Port: 50052)."""
    from nos.client import Client

    yield Client(f"[::]:{GRPC_TEST_PORT}")


@pytest.fixture(scope="session")
def grpc_client_cpu():
    """Test gRPC client to be used with CPU docker runtime (Port: 50053)."""
    from nos.client import Client

    yield Client(f"[::]:{GRPC_TEST_PORT_CPU}")


@pytest.fixture(scope="session")
def grpc_client_gpu():
    """Test gRPC client to be used with GPU docker runtime (Port: 50054)."""
    from nos.client import Client

    yield Client(f"[::]:{GRPC_TEST_PORT_GPU}")


@pytest.fixture(scope="session")
def grpc_server_docker_runtime_cpu():
    """Test DockerRuntime CPU (Port: 50053)."""
    from nos.server import InferenceServiceRuntime

    CPU_CONTAINER_NAME = "nos-inference-service-runtime-cpu-e2e-test"
    runtime = InferenceServiceRuntime(runtime="cpu", name=CPU_CONTAINER_NAME)

    # Force stop any existing containers
    try:
        runtime.stop()
    except Exception:
        logger.info(f"Killing any existing container with name: {CPU_CONTAINER_NAME}")

    # Start grpc server runtime (CPU)
    container = runtime.start(
        ports={f"{DEFAULT_GRPC_PORT}/tcp": GRPC_TEST_PORT_CPU},
        environment={
            "NOS_LOGGING_LEVEL": "DEBUG",
        },
    )
    assert container is not None
    assert container.id is not None
    status = runtime.get_container_status()
    assert status is not None and status == "running"

    # Yield the running container
    yield runtime

    # Tear down
    try:
        runtime.stop()
    except Exception:
        logger.info(f"Failed to stop existing container with name: {CPU_CONTAINER_NAME}")


@pytest.fixture(scope="session")
def grpc_server_docker_runtime_gpu():
    """Test DockerRuntime GPU (Port: 50054)."""
    from nos.server import InferenceServiceRuntime

    GPU_CONTAINER_NAME = "nos-inference-service-runtime-gpu-e2e-test"
    runtime = InferenceServiceRuntime(runtime="gpu", name=GPU_CONTAINER_NAME)

    # Force stop any existing containers
    try:
        runtime.stop()
    except Exception:
        logger.info(f"Killing any existing container with name: {GPU_CONTAINER_NAME}")

    # Start grpc server runtime (GPU)
    container = runtime.start(
        ports={f"{DEFAULT_GRPC_PORT}/tcp": GRPC_TEST_PORT_GPU},
        environment={
            "NOS_LOGGING_LEVEL": "DEBUG",
        },
    )
    assert container is not None
    assert container.id is not None
    status = runtime.get_container_status()
    assert status is not None and status == "running"

    # Yield the running container
    yield runtime

    # Tear down
    try:
        runtime.stop()
    except Exception:
        logger.info(f"Failed to stop existing container with name: {GPU_CONTAINER_NAME}")


# Note: These fixtures need to be named the same as the following variables
# so that we reference them by value in the test functions.
GRPC_CLIENT_WITH_LOCAL = "local_grpc_client_with_server"


@pytest.fixture(scope="session")
def local_grpc_client_with_server(grpc_server, grpc_client):  # noqa: F811
    """Test local gRPC client with local runtime (Port: 50052)."""
    # Wait for server to start
    grpc_client.WaitForServer(timeout=180, retry_interval=5)
    logger.info("Server started!")

    # Yield the gRPC client once the server is up and initialized
    yield grpc_client


# Note: These fixtures need to be named the same as the following variables
# so that we reference them by value in the test functions.
GRPC_CLIENT_WITH_CPU = "grpc_client_with_cpu_backend"


@pytest.fixture(scope="session")
def grpc_client_with_cpu_backend(grpc_server_docker_runtime_cpu, grpc_client_cpu):  # noqa: F811
    """Test gRPC client with initialized CPU docker runtime (Port: 50053)."""
    # Wait for server to start
    grpc_client_cpu.WaitForServer(timeout=180, retry_interval=5)
    logger.info("Server started!")

    # Yield the gRPC client once the server is up and initialized
    yield grpc_client_cpu


# Note: These fixtures need to be named the same as the following variables
# so that we reference them by value in the test functions.
GRPC_CLIENT_WITH_GPU = "grpc_client_with_gpu_backend"


@pytest.fixture(scope="session")
def grpc_client_with_gpu_backend(grpc_server_docker_runtime_gpu, grpc_client_gpu):  # noqa: F811
    """Test gRPC client with initialized CPU docker runtime (Port: 50054)."""
    # Wait for server to start
    grpc_client_gpu.WaitForServer(timeout=180, retry_interval=5)
    logger.info("Server started!")

    # Yield the gRPC client once the server is up and initialized
    yield grpc_client_gpu


# Note: These fixtures need to be named the same as the following variables
# so that we reference them by value in the test functions.
HTTP_CLIENT_WITH_LOCAL = "local_http_client_with_server"


@pytest.fixture(scope="session")
def local_http_client_with_server(grpc_server):  # noqa: F811
    """Test local HTTP client with local runtime."""
    from fastapi.testclient import TestClient

    from nos.server.http._service import app_factory

    # Yield the HTTP client once the server is up and initialized
    with TestClient(app_factory(address=f"[::]:{GRPC_TEST_PORT}")) as _client:
        yield _client


# Note: These fixtures need to be named the same as the following variables
# so that we reference them by value in the test functions.
HTTP_CLIENT_WITH_CPU = "http_client_with_cpu_backend"


@pytest.fixture(scope="session")
def http_client_with_cpu_backend(grpc_server_docker_runtime_cpu):  # noqa: F811
    """Test HTTP client with initialized CPU docker runtime (Port: 50053)."""
    from fastapi.testclient import TestClient

    from nos.server.http._service import app_factory

    # Yield the HTTP client once the server is up and initialized
    with TestClient(app_factory(address=f"[::]:{GRPC_TEST_PORT_CPU}")) as _client:
        yield _client


# Note: These fixtures need to be named the same as the following variables
# so that we reference them by value in the test functions.
HTTP_CLIENT_WITH_GPU = "http_client_with_gpu_backend"


@pytest.fixture(scope="session")
def http_client_with_gpu_backend(grpc_server_docker_runtime_gpu):  # noqa: F811
    """Test HTTP client with initialized CPU docker runtime (Port: 50054)."""
    from fastapi.testclient import TestClient

    from nos.server.http._service import app_factory

    # Yield the HTTP client once the server is up and initialized
    with TestClient(app_factory(address=f"[::]:{GRPC_TEST_PORT_GPU}")) as _client:
        yield _client


# Needed for referencing relevant pytest fixtures
HTTP_CLIENT_SERVER_CONFIGURATIONS = [
    # HTTP_CLIENT_WITH_LOCAL,
    # HTTP_CLIENT_WITH_CPU,
    HTTP_CLIENT_WITH_GPU
]

# Needed for referencing relevant pytest fixtures
GRPC_CLIENT_SERVER_CONFIGURATIONS = [
    # GRPC_CLIENT_WITH_LOCAL,
    # GRPC_CLIENT_WITH_CPU,
    GRPC_CLIENT_WITH_GPU
]


@pytest.fixture(scope="session")
def http_server_with_gpu_backend(grpc_client_with_gpu_backend):  # noqa: F811
    """Test HTTP gateway server with GPU docker runtime."""
    import time
    from multiprocessing import Process

    import requests
    import uvicorn

    from nos.server.http._service import app_factory

    # Wait for server to start, then start uvicorn server
    grpc_client = grpc_client_with_gpu_backend
    grpc_client.WaitForServer(timeout=180, retry_interval=5)
    logger.info("Server started!")

    def _run_uvicorn_server():
        uvicorn.run(
            app_factory(address=f"[::]:{GRPC_TEST_PORT_GPU}", env="dev"),
            host="localhost",
            port=HTTP_TEST_PORT,
            workers=1,
            log_level="info",
        )

    # Start uvicorn server
    proc = Process(target=_run_uvicorn_server, args=(), daemon=True)
    proc.start()
    time.sleep(5)

    # Wait for the HTTP server by polling the health endpoint
    def _wait_for_server(attempts: int = 10, base_url=f"http://localhost:{HTTP_TEST_PORT}/v1"):
        if attempts <= 0:
            raise RuntimeError("Failed to start HTTP server!")
        try:
            response = requests.get(f"{base_url}/health", timeout=1)
            while response.status_code != 200 and attempts > 0:
                logger.debug(f"Waiting for HTTP server to start up ... [pending_attempts={attempts}]")
                time.sleep(1)
                response = requests.get(f"{base_url}/health")
                attempts -= 1
            logger.info("HTTP server started!")
        except Exception:
            logger.debug("Failed to start HTTP server, retrying in 1 second")
            time.sleep(1)
            _wait_for_server(attempts=attempts - 1)

    _wait_for_server()

    # Yield once the server is up and initialized
    yield

    # Stop the HTTP server
    proc.kill()
