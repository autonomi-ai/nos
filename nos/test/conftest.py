import pytest

from nos.executors.ray import RayExecutor
from nos.server.docker import DockerRuntime


@pytest.fixture(scope="session")
def docker_runtime():
    runtime = DockerRuntime.get()
    yield runtime
    runtime.stop()


@pytest.fixture(scope="session")
def ray_executor():
    executor = RayExecutor.get()
    executor.init()

    yield executor

    executor.stop()
