import contextlib

import numpy as np
import pytest
from loguru import logger
from PIL import Image

from nos import hub
from nos.common import TaskType, tqdm
from nos.common.shm import NOS_SHM_ENABLED
from nos.executors.ray import RayExecutor
from nos.managers import ModelHandle, ModelManager
from nos.test.conftest import ray_executor  # noqa: F401
from nos.test.utils import NOS_TEST_IMAGE


pytestmark = pytest.mark.server


@contextlib.contextmanager
def warns(warn_cls=UserWarning, shm_enabled=NOS_SHM_ENABLED):
    """Catch warnings if shared memory is enabled."""
    if shm_enabled:
        yield pytest.warns(warn_cls)
    else:
        yield


def test_model_manager(ray_executor: RayExecutor):  # noqa: F811
    manager = ModelManager()
    assert manager is not None

    with pytest.raises(NotImplementedError):
        manager = ModelManager(policy=ModelManager.EvictionPolicy.LRU)

    # Test adding several models back to back with the same manager.
    # This should not raise any OOM errors as models are evicted
    # from the manager's cache.
    for idx, spec in enumerate(hub.list()):
        handler: ModelHandle = manager.get(spec)
        assert handler is not None
        assert isinstance(handler, ModelHandle)

        logger.info(">" * 80)
        logger.info(f"idx: {idx}")
        logger.info(f"Model manager: {manager}, spec: {spec}")


@pytest.mark.benchmark
def test_model_manager_inference(ray_executor: RayExecutor):  # noqa: F811
    """Benchmark inference with a model manager."""

    manager = ModelManager()
    assert manager is not None

    # Load a model spec
    spec = hub.load_spec("openai/clip-vit-base-patch32", task=TaskType.IMAGE_EMBEDDING)

    # Add the model to the manager (or via `manager.add()`)
    handle: ModelHandle = manager.get(spec)
    assert handle is not None

    img = Image.open(NOS_TEST_IMAGE)
    for _ in tqdm(duration=5, desc="Inference (5s warmup)"):
        result = handle.remote(images=[img] * 8)
        assert result is not None

    # Run inference
    img = Image.open(NOS_TEST_IMAGE)
    for _ in tqdm(duration=20, desc="Inference (20s benchmark)"):
        result = handle.remote(images=[img] * 8)
        assert result is not None


@pytest.mark.skipif(not NOS_SHM_ENABLED, reason="Shared memory transport is not enabled.")
@pytest.mark.parametrize(
    "client_with_server",
    ("local_grpc_client_with_server", "grpc_client_with_cpu_backend", "grpc_client_with_gpu_backend"),
)
def test_shm_registry(client_with_server, request):  # noqa: F811
    """Test shm registry with local server."""
    shm_enabled = NOS_SHM_ENABLED

    client = request.getfixturevalue(client_with_server)
    assert client is not None
    assert client.IsHealthy()

    # Load dummy image
    img = Image.open(NOS_TEST_IMAGE)

    # Load noop model
    task, model_name = TaskType.CUSTOM, "noop/process-images"
    model = client.Module(task=task, model_name=model_name)
    assert model is not None
    assert model.GetModelInfo() is not None

    # Test manually registering/unregistering shared memory regions
    for _shape in [(224, 224), (640, 480), (1280, 720)]:
        images = np.stack([np.asarray(img.resize(_shape)) for _ in range(8)])
        inputs = {"images": images}
        if shm_enabled:
            model.RegisterSystemSharedMemory(inputs)
            response = model(**inputs)
            assert isinstance(response, dict)
            model.UnregisterSystemSharedMemory()


@pytest.mark.parametrize(
    "client_with_server",
    ("local_grpc_client_with_server", "grpc_client_with_cpu_backend", "grpc_client_with_gpu_backend"),
)
def test_inference_service_noop(client_with_server, request):  # noqa: F811
    """Test inference service with shared memory transport."""
    shm_enabled = NOS_SHM_ENABLED

    client = request.getfixturevalue(client_with_server)
    assert client is not None
    assert client.IsHealthy()

    # Load dummy image
    img = Image.open(NOS_TEST_IMAGE)

    # Load noop model
    task, model_name = TaskType.CUSTOM, "noop/process-images"
    model = client.Module(task=task, model_name=model_name)
    assert model is not None
    assert model.GetModelInfo() is not None

    # Inference (single forward pass)
    # Change image sizes on the fly to test if the client
    # registers new shared memory regions.
    for shape in [(224, 224), (640, 480), (1280, 720)]:
        # __call__(images: np.ndarray, 3-dim)
        inputs = {"images": np.asarray(img.resize(shape))}
        response = model(**inputs)
        assert isinstance(response, dict)
        assert "result" in response

        # __call__(images: List[np.ndarray], List of 3-dim)
        # This call should register a new shared memory region
        # and raise a user warning.
        inputs = {"images": [np.asarray(img.resize(shape))]}
        with warns(UserWarning, shm_enabled=shm_enabled) as warn:
            response = model(**inputs)
        # This call should not raise any warnings.
        with warns(None, shm_enabled=shm_enabled) as warn:
            response = model(**inputs)
        if warn:
            assert len(warn) == 0, "Expected no warnings, but warnings were raised"
        assert isinstance(response, dict)
        assert "result" in response

        # __call__(image: Image.Image)
        # This will not use shared memory transport
        # and force the client to unregister shm objects,
        # and send the image over the wire.
        inputs = {"images": img.resize(shape)}
        response = model(**inputs)

        # __call__(image: List[Image.Image])
        # This will not use shared memory transport
        # and force the client to send the image over the wire.
        # Note: Since we've already unregistered the shm objects,
        # no new shm region changes should happen here.
        inputs = {"images": [img.resize(shape)]}
        response = model(**inputs)

    # TODO (spillai) Compare round-trip-times with and without shared memory transport
    # Note: This test is only valid for the local server.


BENCHMARK_IMAGE_SHAPES = [(224, 224), (640, 480), (1280, 720), (1920, 1080), (2880, 1620), (3840, 2160)]


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "client_with_server",
    ("local_grpc_client_with_server", "grpc_client_with_cpu_backend", "grpc_client_with_gpu_backend"),
)
@pytest.mark.parametrize("shape", BENCHMARK_IMAGE_SHAPES)
@pytest.mark.parametrize("image_type", [Image.Image, np.ndarray])
def test_benchmark_inference_service_noop(client_with_server, shape, image_type, request):  # noqa: F811
    """Benchmark shared memory transport and inference between the client-server.

    Tests with 3 client-server configurations:
      - local client, local server
      - local client, CPU docker server
      - local client, GPU docker server

    Note: This test is only valid for the local server.
    """
    shm_enabled = NOS_SHM_ENABLED

    client = request.getfixturevalue(client_with_server)
    assert client is not None
    assert client.IsHealthy()

    # Load dummy image
    img = Image.open(NOS_TEST_IMAGE)
    img = img.resize(shape)
    if image_type == np.ndarray:
        img = np.asarray(img)
        assert img.shape[:2][::-1] == shape
    elif image_type == Image.Image:
        assert img.size == shape
        if shm_enabled:
            logger.warning("Shared memory transport is not supported for Image.Image")
    else:
        raise TypeError(f"Invalid image type: {image_type}")

    # Load noop model
    task, model_name = TaskType.CUSTOM, "noop/process-images"
    model = client.Module(task=task, model_name=model_name)
    assert model is not None
    assert model.GetModelInfo() is not None

    # Benchmark (10s)
    for b in range(8):
        B = 2**b

        # Prepare inputs
        if isinstance(img, np.ndarray):
            images = np.stack([img for _ in range(B)])
        elif isinstance(img, Image.Image):
            images = [img for _ in range(B)]
        else:
            raise TypeError(f"Invalid image type: {type(img)}")
        inputs = {"images": images}

        # Register shared memory regions
        if shm_enabled:
            model.RegisterSystemSharedMemory(inputs)

        # Warmup
        for _ in tqdm(duration=2, desc="Warmup", disable=True):
            try:
                response = model(**inputs)
            except Exception as e:
                logger.error(f"Exception: {e}")
                continue

        # Benchmark no-op inference
        for _ in tqdm(
            duration=10,
            desc=f"Benchmark model={model_name}, task={task} [B={B}, shape={shape}, type={image_type}]",
            unit="images",
            unit_scale=B,
            total=0,
        ):
            try:
                response = model(**inputs)
            except Exception as e:
                logger.error(f"Exception: {e}")
                continue
            assert isinstance(response, dict)
            assert "result" in response

        if shm_enabled:
            model.UnregisterSystemSharedMemory()
