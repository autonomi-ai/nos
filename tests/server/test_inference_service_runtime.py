import time

import pytest
from loguru import logger

from nos.common.system import has_gpu, is_aws_inf2
from nos.constants import DEFAULT_GRPC_PORT
from nos.server._runtime import InferenceServiceRuntime
from nos.test.utils import skip_if_no_torch_cuda


pytestmark = pytest.mark.server


def test_supported_inference_service_runtime():
    available_runtimes = InferenceServiceRuntime.supported_runtimes()
    assert available_runtimes is not None
    assert "cpu" in available_runtimes
    assert "gpu" in available_runtimes
    assert "inf2" in available_runtimes


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


@pytest.fixture
def inference_service_runtime_inf2():
    runtime = InferenceServiceRuntime(runtime="inf2", name="nos-inference-service-runtime-inf2-test")
    assert runtime is not None

    containers = runtime.list()
    assert containers is not None

    logger.debug("Starting inference service runtime: gpu")
    runtime.start(ports={f"{DEFAULT_GRPC_PORT-4}/tcp": DEFAULT_GRPC_PORT - 4})
    assert runtime.get_container() is not None
    assert runtime.get_container_id() is not None
    assert runtime.get_container_name() is not None
    assert runtime.get_container_status() is not None

    yield runtime
    logger.debug("Stopping inference service runtime: gpu")
    runtime.stop()
    logger.debug("Stopped inference service runtime: gpu")


def test_inference_service_runtime_utils():
    runtime = InferenceServiceRuntime.detect()
    # Note (spillai): Currently, we're only testing these on CPU/GPU instances, so only check for those
    assert runtime == "gpu" if has_gpu() else "cpu", "Invalid runtime detected."
    assert InferenceServiceRuntime.list() is not None

    devices = InferenceServiceRuntime.devices()
    assert devices is not None
    assert isinstance(devices, list)
    assert len(devices) >= 0, "Invalid number of devices detected."


def test_inference_service_runtime_cpu(inference_service_runtime_cpu):  # noqa: F811
    assert inference_service_runtime_cpu is not None
    time.sleep(5)


@skip_if_no_torch_cuda
def test_inference_service_runtime_gpu(inference_service_runtime_gpu):  # noqa: F811
    assert inference_service_runtime_gpu is not None
    time.sleep(5)


@pytest.mark.skipif(not is_aws_inf2(), reason="No inf2 / Inferentia2 device detected.")
def test_inference_service_runtime_inf2(inference_service_runtime_inf2):  # noqa: F811
    assert inference_service_runtime_inf2 is not None
    time.sleep(5)
