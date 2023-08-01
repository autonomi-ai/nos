from concurrent import futures

import grpc
import pytest
from loguru import logger

from nos.constants import DEFAULT_GRPC_PORT
from nos.protoc import import_module


GRPC_TEST_PORT = DEFAULT_GRPC_PORT + 1
GRPC_TEST_PORT_CPU = DEFAULT_GRPC_PORT + 2
GRPC_TEST_PORT_GPU = DEFAULT_GRPC_PORT + 3


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
    from nos.managers import ModelManager

    manager = ModelManager()
    assert manager is not None

    yield manager


@pytest.fixture(scope="session")
def grpc_server(ray_executor):
    """Test gRPC server (Port: 50052)."""
    from loguru import logger

    from nos.server._service import InferenceServiceImpl

    logger.info(f"Starting gRPC test server on port: {GRPC_TEST_PORT}")
    options = [
        ("grpc.max_message_length", 512 * 1024 * 1024),
        ("grpc.max_send_message_length", 512 * 1024 * 1024),
        ("grpc.max_receive_message_length", 512 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=options)
    nos_service_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServiceImpl(), server)
    server.add_insecure_port(f"[::]:{GRPC_TEST_PORT}")
    server.start()
    yield server
    server.stop(grace=None)


@pytest.fixture(scope="session")
def grpc_client():
    """Test gRPC client (Port: 50052)."""
    from nos.client import InferenceClient

    yield InferenceClient(f"[::]:{GRPC_TEST_PORT}")


@pytest.fixture(scope="session")
def grpc_client_cpu():
    """Test gRPC client to be used with CPU docker runtime (Port: 50053)."""
    from nos.client import InferenceClient

    yield InferenceClient(f"[::]:{GRPC_TEST_PORT_CPU}")


@pytest.fixture(scope="session")
def grpc_client_gpu():
    """Test gRPC client to be used with GPU docker runtime (Port: 50054)."""
    from nos.client import InferenceClient

    yield InferenceClient(f"[::]:{GRPC_TEST_PORT_GPU}")


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
