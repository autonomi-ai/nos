import pytest

from nos.server.docker import DockerRuntime
from nos.server.runtime import NOS_DOCKER_IMAGE_CPU, NOS_DOCKER_IMAGE_GPU, NOS_GRPC_SERVER_CMD
from nos.test.utils import skip_if_no_torch_cuda


pytestmark = pytest.mark.e2e


def test_docker_runtime_singleton():
    """Test DockerRuntime singleton."""
    docker_runtime = DockerRuntime.get()
    docker_runtime_ = DockerRuntime.get()
    assert docker_runtime is docker_runtime_


def test_docker_runtime_cpu(grpc_server_docker_runtime_cpu):
    """Test DockerRuntime for CPU."""
    # Try re-starting the container
    # This should not fail, and should return the existing container
    docker_runtime = DockerRuntime.get()
    container_ = docker_runtime.start(
        image=NOS_DOCKER_IMAGE_CPU,
        container_name=grpc_server_docker_runtime_cpu.name,
        command=[NOS_GRPC_SERVER_CMD],
        detach=True,
        gpu=False,
    )
    assert container_ is not None
    assert container_.id is not None
    assert container_.id == grpc_server_docker_runtime_cpu.id


@skip_if_no_torch_cuda
def test_docker_runtime_gpu(grpc_server_docker_runtime_gpu):
    """Test DockerRuntime for GPU."""
    # Try re-starting the container
    # This should not fail, and should return the existing container
    docker_runtime = DockerRuntime.get()
    container_ = docker_runtime.start(
        image=NOS_DOCKER_IMAGE_GPU,
        container_name=grpc_server_docker_runtime_gpu.name,
        command=[NOS_GRPC_SERVER_CMD],
        detach=True,
        gpu=True,
    )
    assert container_ is not None
    assert container_.id is not None
    assert container_.id == grpc_server_docker_runtime_gpu.id


def test_docker_runtime_logs(grpc_server_docker_runtime_cpu):  # noqa: F811
    """Test Dockerdocker_runtime logs."""
    docker_runtime = DockerRuntime.get()
    docker_runtime.get_container(grpc_server_docker_runtime_cpu.name)
    logs = docker_runtime.get_logs(grpc_server_docker_runtime_cpu.name)
    assert logs is not None
