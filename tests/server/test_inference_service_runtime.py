import pytest

from nos.server import InferenceServiceRuntime
from nos.server.docker import DockerRuntime
from nos.test.conftest import docker_runtime  # noqa: F401


pytestmark = pytest.mark.e2e


def test_inference_service_runtime(docker_runtime: DockerRuntime):  # noqa: F811
    runtime = InferenceServiceRuntime()
    assert runtime is not None
