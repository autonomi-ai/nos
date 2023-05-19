import pytest

from nos.server import InferenceServiceRuntime


pytestmark = pytest.mark.e2e


def test_inference_service_runtime(grpc_server_runtime_cpu_container):  # noqa: F811
    runtime = InferenceServiceRuntime()
    assert runtime is not None
