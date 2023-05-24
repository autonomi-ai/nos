import time
from concurrent import futures
from pathlib import Path

import grpc
import pytest
from loguru import logger

from nos.constants import DEFAULT_GRPC_PORT
from nos.protoc import import_module


GRPC_TEST_PORT = DEFAULT_GRPC_PORT + 1
GRPC_TEST_PORT_CPU = DEFAULT_GRPC_PORT + 2
GRPC_TEST_PORT_GPU = DEFAULT_GRPC_PORT + 3
CPU_CONTAINER_NAME = "nos-grpc-server-cpu-test"
GPU_CONTAINER_NAME = "nos-grpc-server-gpu-test"

nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


@pytest.fixture(scope="session")
def ray_executor():
    from nos.executors.ray import RayExecutor

    executor = RayExecutor.get()
    executor.init()

    yield executor

    executor.stop()


@pytest.fixture(scope="session")
def grpc_server():
    """Test gRPC server (Port: 50052)."""
    from loguru import logger

    from nos.server.service import InferenceServiceImpl

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

    yield InferenceClient(f"localhost:{GRPC_TEST_PORT}")


@pytest.fixture(scope="session")
def grpc_client_cpu():
    """Test gRPC client to be used with CPU docker runtime (Port: 50053)."""
    from nos.client import InferenceClient

    yield InferenceClient(f"localhost:{GRPC_TEST_PORT_CPU}")


@pytest.fixture(scope="session")
def grpc_client_gpu():
    """Test gRPC client to be used with GPU docker runtime (Port: 50054)."""
    from nos.client import InferenceClient

    yield InferenceClient(f"localhost:{GRPC_TEST_PORT_GPU}")


@pytest.fixture(scope="session")
def grpc_server_docker_runtime_cpu():
    """Test DockerRuntime CPU (Port: 50053)."""
    from nos.server.docker import DockerRuntime
    from nos.server.runtime import NOS_DOCKER_IMAGE_CPU, NOS_GRPC_SERVER_CMD

    docker_runtime = DockerRuntime.get()

    # Force stop any existing containers
    try:
        docker_runtime.stop(CPU_CONTAINER_NAME)
    except Exception:
        logger.info(f"Killing any existing container with name: {CPU_CONTAINER_NAME}")

    # Start grpc server runtime (CPU)
    container = docker_runtime.start(
        image=NOS_DOCKER_IMAGE_CPU,
        container_name=CPU_CONTAINER_NAME,
        command=[NOS_GRPC_SERVER_CMD],
        ports={DEFAULT_GRPC_PORT: GRPC_TEST_PORT_CPU},
        environment={
            "NOS_LOGGING_LEVEL": "DEBUG",
        },
        volumes={
            str(Path.home() / ".nosd_test"): {"bind": "/app/.nos", "mode": "rw"},
            "/tmp": {"bind": "/tmp", "mode": "rw"},
        },
        detach=True,
        gpu=False,
    )
    assert container is not None
    assert container.id is not None

    # Get container and status
    container_ = docker_runtime.get_container(CPU_CONTAINER_NAME)
    status = docker_runtime.get_container_status(CPU_CONTAINER_NAME)
    assert container_.id == container.id
    assert status is not None
    assert status == "running"

    yield container

    # Tear down (raise errors if this fails)
    docker_runtime.stop(CPU_CONTAINER_NAME)

    # Wait for container to stop (up to 20 seconds)
    st = time.time()
    while time.time() - st <= 20:
        status = docker_runtime.get_container_status(CPU_CONTAINER_NAME)
        if status == "exited" or status is None:
            break
        time.sleep(1)
    assert status is None or status == "exited"


@pytest.fixture(scope="session")
def grpc_server_docker_runtime_gpu():
    """Test DockerRuntime GPU (Port: 50054)."""
    from nos.server.docker import DockerRuntime
    from nos.server.runtime import NOS_DOCKER_IMAGE_GPU, NOS_GRPC_SERVER_CMD

    docker_runtime = DockerRuntime.get()

    # Force stop any existing containers
    try:
        docker_runtime.stop(GPU_CONTAINER_NAME)
    except Exception:
        logger.info(f"Killing any existing container with name: {GPU_CONTAINER_NAME}")

    # Start grpc server runtime (GPU)
    container = docker_runtime.start(
        image=NOS_DOCKER_IMAGE_GPU,
        container_name=GPU_CONTAINER_NAME,
        command=[NOS_GRPC_SERVER_CMD],
        ports={DEFAULT_GRPC_PORT: GRPC_TEST_PORT_GPU},
        environment={
            "NOS_LOGGING_LEVEL": "DEBUG",
        },
        volumes={
            str(Path.home() / ".nosd_test"): {"bind": "/app/.nos", "mode": "rw"},
            "/tmp": {"bind": "/tmp", "mode": "rw"},
        },
        detach=True,
        gpu=True,
    )
    assert container is not None
    assert container.id is not None

    # Get container and status
    container_ = docker_runtime.get_container(GPU_CONTAINER_NAME)
    status = docker_runtime.get_container_status(GPU_CONTAINER_NAME)
    assert container_.id == container.id
    assert status is not None
    assert status == "running"

    yield container

    # Tear down (raise errors if this fails)
    docker_runtime.stop(GPU_CONTAINER_NAME)

    # Wait for container to stop (up to 20 seconds)
    st = time.time()
    while time.time() - st <= 20:
        status = docker_runtime.get_container_status(GPU_CONTAINER_NAME)
        if status == "exited" or status is None:
            break
        time.sleep(1)
    assert status is None or status == "exited"
