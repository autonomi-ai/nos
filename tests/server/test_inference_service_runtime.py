import time

import pytest
from loguru import logger

from nos.constants import DEFAULT_GRPC_PORT
from nos.server._runtime import InferenceServiceRuntime
from nos.test.utils import skip_if_no_torch_cuda


pytestmark = pytest.mark.server


@pytest.fixture
def inference_service_runtime_cpu():
    runtime = InferenceServiceRuntime(runtime="cpu", name="nos-inference-service-runtime-cpu-test")
    assert runtime is not None

    containers = runtime.list()
    assert containers is not None

    logger.debug("Starting inference service runtime: cpu")
    runtime.start(ports={f"{DEFAULT_GRPC_PORT-1}/tcp": DEFAULT_GRPC_PORT - 1})
    assert runtime.get_container() is not None
    assert runtime.get_container_id() is not None
    assert runtime.get_container_name() is not None
    assert runtime.get_container_status() is not None

    yield runtime

    logger.debug("Stopping inference service runtime: cpu")
    runtime.stop()
    logger.debug("Stopped inference service runtime: cpu")


@skip_if_no_torch_cuda
@pytest.fixture
def inference_service_runtime_gpu():
    runtime = InferenceServiceRuntime(runtime="gpu", name="nos-inference-service-runtime-gpu-test")
    assert runtime is not None

    containers = runtime.list()
    assert containers is not None

    logger.debug("Starting inference service runtime: gpu")
    runtime.start(ports={f"{DEFAULT_GRPC_PORT-2}/tcp": DEFAULT_GRPC_PORT - 2})
    assert runtime.get_container() is not None
    assert runtime.get_container_id() is not None
    assert runtime.get_container_name() is not None
    assert runtime.get_container_status() is not None

    yield runtime
    logger.debug("Stopping inference service runtime: gpu")
    runtime.stop()
    logger.debug("Stopped inference service runtime: gpu")


def test_inference_service_runtime_cpu(inference_service_runtime_cpu):  # noqa: F811
    assert inference_service_runtime_cpu is not None
    time.sleep(5)


@skip_if_no_torch_cuda
def test_inference_service_runtime_gpu(inference_service_runtime_gpu):  # noqa: F811
    assert inference_service_runtime_gpu is not None
    time.sleep(5)
