import pytest

from nos.server.runtime import InferenceServiceRuntime


pytestmark = pytest.mark.e2e


def test_inference_service_runtime():  # noqa: F811
    runtime = InferenceServiceRuntime()
    assert runtime is not None

    runtime.start()
    assert runtime.id is not None
    runtime.stop()
