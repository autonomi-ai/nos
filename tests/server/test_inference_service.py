import pytest

from nos.executors.ray import RayExecutor
from nos.server.service import InferenceServiceImpl
from nos.test.conftest import ray_executor  # noqa: F401


pytestmark = pytest.mark.server


@pytest.mark.skip(reason="This test is not ready yet.")
def test_inference_service_impl(ray_executor: RayExecutor):  # noqa: F811
    service = InferenceServiceImpl()
    assert service is not None
