import time

import pytest
from loguru import logger

from nos.server.docker import DockerRuntime
from nos.server.runtime import NOS_DOCKER_IMAGE_CPU, NOS_DOCKER_IMAGE_GPU, NOS_GRPC_SERVER_CMD
from nos.test.conftest import docker_runtime  # noqa: F401
from nos.test.utils import skip_if_no_torch_cuda


pytestmark = pytest.mark.e2e


def test_docker_runtime_singleton(docker_runtime: DockerRuntime):  # noqa: F811
    """Test DockerRuntime singleton."""
    docker_runtime_ = DockerRuntime.get()
    assert docker_runtime is docker_runtime_


def test_docker_runtime_cpu(docker_runtime: DockerRuntime):  # noqa: F811
    """Test DockerRuntime for CPU."""
    container_name = "nos-cpu-test"

    # Force stop any existing containers
    try:
        docker_runtime.stop(container_name)
    except Exception:
        logger.info(f"Killing any existing container with name: {container_name}")

    # Test CPU support
    command = [NOS_GRPC_SERVER_CMD]
    container = docker_runtime.start(
        image=NOS_DOCKER_IMAGE_CPU,
        container_name=container_name,
        command=command,
        detach=True,
        gpu=False,
    )
    assert container is not None
    assert container.id is not None

    # Get container and status
    container_ = docker_runtime.get_container(container_name)
    status = docker_runtime.get_container_status(container_name)
    assert container_.id == container.id
    assert status is not None
    assert status == "running"

    # Try re-starting the container
    # This should not fail, and should return the existing container
    container_ = docker_runtime.start(
        image=NOS_DOCKER_IMAGE_CPU,
        container_name=container_name,
        command=command,
        detach=True,
        gpu=True,
    )
    assert container_ is not None
    assert container_.id is not None
    assert container_.id == container.id

    # Tear down
    container = docker_runtime.stop(container_name)
    assert container is not None

    # Wait for container to stop (up to 20 seconds)
    st = time.time()
    while time.time() - st <= 20:
        status = docker_runtime.get_container_status(container_name)
        if status == "exited" or status is None:
            break
        time.sleep(1)
    assert status is None or status == "exited"


@skip_if_no_torch_cuda
def test_docker_runtime_gpu(docker_runtime: DockerRuntime):  # noqa: F811
    """Test DockerRuntime for GPU."""
    # Test GPU support
    container_name = "nos-gpu-test"
    command = [NOS_GRPC_SERVER_CMD]
    container = docker_runtime.start(
        image=NOS_DOCKER_IMAGE_GPU,
        container_name=container_name,
        command=command,
        detach=True,
        gpu=True,
    )
    assert container is not None
    assert container.id is not None

    # Get container and status
    status = docker_runtime.get_container_status(container_name)
    assert status is not None
    assert status == "running"

    # Tear down
    container = docker_runtime.stop(container_name)
    assert container is not None
    status = container.status

    # Wait for container to stop (up to 20 seconds)
    st = time.time()
    while time.time() - st <= 20:
        status = docker_runtime.get_container_status(container_name)
        if status == "exited" or status is None:
            break
        time.sleep(1)
    assert status is None or status == "exited"


def test_docker_runtime_logs(docker_runtime: DockerRuntime):  # noqa: F811
    """Test Dockerdocker_runtime logs."""
    # Test CPU support
    container_name = "nos-cpu-test"
    command = [NOS_GRPC_SERVER_CMD]
    container = docker_runtime.start(
        image=NOS_DOCKER_IMAGE_GPU,
        container_name=container_name,
        command=command,
        detach=True,
        gpu=False,
    )
    assert container is not None
    assert container.id is not None

    # Get container logs
    container = docker_runtime.get_container(container_name)
    logs = docker_runtime.get_logs(container_name)
    assert logs is not None
