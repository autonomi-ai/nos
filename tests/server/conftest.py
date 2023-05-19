import time

import pytest
from loguru import logger

from nos.server.docker import DockerRuntime
from nos.server.runtime import NOS_DOCKER_IMAGE_CPU, NOS_DOCKER_IMAGE_GPU, NOS_GRPC_SERVER_CMD


CPU_CONTAINER_NAME = "nos-cpu-test"
GPU_CONTAINER_NAME = "nos-gpu-test"


@pytest.fixture(scope="session")
def grpc_server_runtime_cpu_container():
    """Test DockerRuntime CPU."""
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
def grpc_server_runtime_gpu_container():
    """Test DockerRuntime GPU."""
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
