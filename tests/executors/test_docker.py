import time

import pytest
from loguru import logger

from nos.executors.docker import DockerExecutor
from nos.experimental.grpc.server.runtime import NOS_DOCKER_IMAGE_CPU, NOS_DOCKER_IMAGE_GPU, NOS_GRPC_SERVER_CMD
from nos.test.utils import requires_torch_cuda


@pytest.mark.e2e
def test_docker_executor_singleton():
    """Test DockerExecutor singleton."""
    executor = DockerExecutor.get()
    executor_ = DockerExecutor.get()
    assert executor is executor_


@pytest.mark.e2e
def test_docker_executor_cpu():
    """Test DockerExecutor for CPU."""
    executor = DockerExecutor.get()
    container_name = "nos-cpu-test"

    # Force stop any existing containers
    try:
        executor.stop(container_name)
    except Exception:
        logger.info(f"Killing any existing container with name: {container_name}")

    # Test CPU support
    command = [NOS_GRPC_SERVER_CMD]
    container = executor.start(
        image=NOS_DOCKER_IMAGE_CPU,
        container_name=container_name,
        command=command,
        detach=True,
        gpu=False,
    )
    assert container is not None
    assert container.id is not None

    # Get container and status
    container_ = executor.get_container(container_name)
    status = executor.get_container_status(container_name)
    assert container_.id == container.id
    assert status is not None
    assert status == "running"

    # Try re-starting the container
    # This should not fail, and should return the existing container
    container_ = executor.start(
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
    container = executor.stop(container_name)
    assert container is not None

    # Wait for container to stop (up to 20 seconds)
    st = time.time()
    while time.time() - st <= 20:
        status = executor.get_container_status(container_name)
        if status == "exited" or status is None:
            break
        time.sleep(1)
    assert status is None or status == "exited"


@pytest.mark.e2e
@requires_torch_cuda
def test_docker_executor_gpu():
    """Test DockerExecutor for GPU."""
    executor = DockerExecutor.get()

    # Test GPU support
    container_name = "nos-gpu-test"
    command = [NOS_GRPC_SERVER_CMD]
    container = executor.start(
        image=NOS_DOCKER_IMAGE_GPU,
        container_name=container_name,
        command=command,
        detach=True,
        gpu=True,
    )
    assert container is not None
    assert container.id is not None

    # Get container and status
    status = executor.get_container_status(container_name)
    assert status is not None
    assert status == "running"

    # Tear down
    container = executor.stop(container_name)
    assert container is not None
    status = container.status

    # Wait for container to stop (up to 20 seconds)
    st = time.time()
    while time.time() - st <= 20:
        status = executor.get_container_status(container_name)
        if status == "exited" or status is None:
            break
        time.sleep(1)
    assert status is None or status == "exited"


@pytest.mark.e2e
def test_docker_executor_logs():
    """Test DockerExecutor logs."""
    executor = DockerExecutor.get()

    # Test CPU support
    container_name = "nos-cpu-test"
    command = [NOS_GRPC_SERVER_CMD]
    container = executor.start(
        image=NOS_DOCKER_IMAGE_GPU,
        container_name=container_name,
        command=command,
        detach=True,
        gpu=False,
    )
    assert container is not None
    assert container.id is not None

    # Get container logs
    container = executor.get_container(container_name)
    logs = executor.get_logs(container_name)
    assert logs is not None
