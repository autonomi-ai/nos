import time

import pytest
from loguru import logger

from nos.server._docker import DockerRuntime
from nos.server._runtime import NOS_DOCKER_IMAGE_CPU, NOS_DOCKER_IMAGE_GPU
from nos.test.utils import skip_if_no_torch_cuda


pytestmark = pytest.mark.server


def test_docker_runtime_singleton():
    """Test DockerRuntime singleton."""
    docker_runtime = DockerRuntime.get()
    docker_runtime_ = DockerRuntime.get()
    assert docker_runtime is docker_runtime_

    # List containers
    containers = docker_runtime.list()
    assert containers is not None


@pytest.fixture
def docker_runtime_cpu_container():
    """Test DockerRuntime CPU."""
    docker_runtime = DockerRuntime.get()

    container_name = "nos-docker-runtime-cpu-test"

    # Start CPU container
    container = docker_runtime.start(
        image=NOS_DOCKER_IMAGE_CPU,
        name=container_name,
        command=r"""bash -c 'echo CPU test && sleep 5'""",
        detach=True,
    )
    logger.debug(f"Starting test CPU container: name={container_name}")

    # Wait for container to start
    st = time.time()
    while time.time() - st <= 60:
        status = docker_runtime.get_container_status(container_name)
        if status == "running" or status is None:
            break
        time.sleep(1)
        logger.debug(
            f"Waiting for container to start: name={container_name}, status={status}, elapsed={time.time() - st:.0f}s"
        )
    assert status == "running" or status is None
    logger.debug(f"Started test CPU container: name={container_name}")

    yield container

    # Get container logs
    logs = list(docker_runtime.get_container_logs(container_name))
    assert len(logs) > 0
    logger.debug(f"CPU container logs:\n{''.join(logs)}")

    # Stop container
    # Note: Once the command is finished, the container will stop automatically.
    # This just ensures that the container is stopped and removed.
    logger.debug(f"Stopping test CPU container: name={container_name}")
    docker_runtime.stop(container_name)

    # Wait for container to stop
    st = time.time()
    while time.time() - st <= 60:
        status = docker_runtime.get_container_status(container_name)
        if status != "running":
            break
        time.sleep(1)
    assert status != "running" or status is None
    logger.debug(f"Stopped test CPU container: name={container_name}")


@skip_if_no_torch_cuda
@pytest.fixture
def docker_runtime_gpu_container():
    """Test DockerRuntime GPU."""
    docker_runtime = DockerRuntime.get()

    container_name = "nos-docker-runtime-gpu-test"

    # Start CPU container
    container = docker_runtime.start(
        image=NOS_DOCKER_IMAGE_GPU,
        name=container_name,
        command=r"""bash -c 'echo GPU test && nvidia-smi && sleep 5'""",
        detach=True,
        device="gpu",
    )
    logger.debug(f"Starting test GPU container: name={container_name}")

    # Wait for container to start
    st = time.time()
    while time.time() - st <= 60:
        status = docker_runtime.get_container_status(container_name)
        if status == "running" or status is None:
            break
        time.sleep(1)
        logger.debug(
            f"Waiting for container to start: name={container_name}, status={status}, elapsed={time.time() - st:.0f}s"
        )
    assert status == "running" or status is None
    logger.debug(f"Started test GPU container: name={container_name}")

    yield container

    # Get container logs
    logs = list(docker_runtime.get_container_logs(container_name))
    assert len(logs) > 0
    logger.debug(f"GPU container logs:\n{''.join(logs)}")

    # Stop container
    # Note: Once the command is finished, the container will stop automatically.
    # This just ensures that the container is stopped and removed.
    logger.debug(f"Stopping test GPU container: name={container_name}")
    docker_runtime.stop(container_name)

    # Wait for container to stop
    st = time.time()
    while time.time() - st <= 60:
        status = docker_runtime.get_container_status(container_name)
        if status != "running":
            break
        time.sleep(1)
    assert status != "running" or status is None
    logger.debug(f"Stopped test GPU container: name={container_name}")


def test_docker_runtime_cpu(docker_runtime_cpu_container):
    """Test DockerRuntime for CPU."""
    docker_runtime = DockerRuntime.get()

    container = docker_runtime_cpu_container
    container_name = container.name
    assert container is not None
    assert container.id is not None
    assert docker_runtime.get_container_id(container_name) == container.id

    # Get CPU container
    container_ = docker_runtime.get_container(container_name)
    assert container_ is not None
    assert container_.id == container.id


@skip_if_no_torch_cuda
def test_docker_runtime_gpu(docker_runtime_gpu_container):
    """Test DockerRuntime for GPU."""
    docker_runtime = DockerRuntime.get()

    container = docker_runtime_gpu_container
    container_name = container.name
    assert container is not None
    assert container.id is not None
    assert docker_runtime.get_container_id(container_name) == container.id

    # Get GPU container
    container_ = docker_runtime.get_container(container_name)
    assert container_ is not None
    assert container_.id == container.id
