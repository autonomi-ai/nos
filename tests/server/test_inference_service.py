import pytest

from nos.executors.ray import RayExecutor
from nos.server import InferenceService
from nos.test.conftest import ray_executor  # noqa: F401


pytestmark = pytest.mark.e2e


def test_inference_service(ray_executor: RayExecutor):  # noqa: F811
    service = InferenceService()
    assert service is not None
