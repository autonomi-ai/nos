import contextlib
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from loguru import logger
from PIL import Image

import nos
from nos import hub
from nos.client.exceptions import NosInputValidationException
from nos.common import TaskType, TimingInfo, tqdm
from nos.common.shm import NOS_SHM_ENABLED
from nos.executors.ray import RayExecutor
from nos.managers import ModelHandle, ModelManager
from nos.test.conftest import ray_executor  # noqa: F401
from nos.test.utils import NOS_TEST_IMAGE
from nos.version import __version__ as nos_version


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
        result = handle(images=[img] * 8)
        assert result is not None

    # Run inference
    img = Image.open(NOS_TEST_IMAGE)
    for _ in tqdm(duration=20, desc="Inference (20s benchmark)"):
        result = handle(images=[img] * 8)
        assert result is not None


@pytest.mark.skipif(not NOS_SHM_ENABLED, reason="Shared memory transport is not enabled.")
@pytest.mark.parametrize(
    "client_with_server",
    ("local_grpc_client_with_server", "grpc_client_with_cpu_backend", "grpc_client_with_gpu_backend"),
)
def test_shm_registry(client_with_server, request):  # noqa: F811
    """Test shm registry."""
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
    p_namespace, p_object_id = None, None
    for _shape in [(224, 224), (640, 480), (1280, 720)]:
        images = np.stack([np.asarray(img.resize(_shape)) for _ in range(8)])
        inputs = {"images": images}
        if shm_enabled:
            model.RegisterSystemSharedMemory(inputs)
            # Test if the object_ids change with each RegisterSystemSharedMemory call
            if p_namespace is not None:
                assert p_namespace != model.namespace
                assert p_object_id != model.object_id
            p_namespace, p_object_id = model.namespace, model.object_id
        response = model(**inputs)
        assert isinstance(response, dict)
        if shm_enabled:
            model.UnregisterSystemSharedMemory()

    # Repeatedly register/unregister shared memory regions
    # so that we can test the shared memory registry.
    for _ in range(10):
        model.RegisterSystemSharedMemory(inputs)
        model.UnregisterSystemSharedMemory()
        # shm_files = list(Path("/dev/shm/").rglob("nos_psm_*"))
        # assert len(shm_files) == 0, "Expected no shared memory regions, but found some."


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


BENCHMARK_BATCH_SIZES = [2**b for b in (0, 4, 8)]
BENCHMARK_IMAGE_SHAPES = [(224, 224), (640, 480), (1280, 720), (2880, 1620)]
BENCHMARK_MODELS = [
    (TaskType.CUSTOM, "noop/process-images", [(224, 224), (640, 480), (1280, 720), (2880, 1620)]),
    (TaskType.IMAGE_EMBEDDING, "openai/clip-vit-base-patch32", [(224, 224), (640, 480)]),
    (TaskType.OBJECT_DETECTION_2D, "yolox/medium", [(640, 480), (1280, 720), (2880, 1620)]),
    (
        TaskType.OBJECT_DETECTION_2D,
        "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",
        [(640, 480), (1280, 720), (2880, 1620)],
    ),
]
BENCHMARK_IMAGE_TYPES = [np.ndarray, Image.Image]


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "client_with_server",
    ("grpc_client_with_gpu_backend",),
    # ("local_grpc_client_with_server", "grpc_client_with_cpu_backend", "grpc_client_with_gpu_backend"),
)
def test_benchmark_inference_service_noop(client_with_server, request):  # noqa: F811
    """Benchmark shared memory transport and inference between the client-server.

    Tests with 3 client-server configurations:
      - local client, local server
      - local client, CPU docker server
      - local client, GPU docker server

    Note: This test is only valid for the local server.
    """
    pd.set_option("display.max_rows", 500)

    shm_enabled = NOS_SHM_ENABLED

    client = request.getfixturevalue(client_with_server)
    assert client is not None
    assert client.IsHealthy()

    # Load model
    timing_records = []
    for (task, model_name, shapes) in BENCHMARK_MODELS:
        model = client.Module(task=task, model_name=model_name)
        assert model is not None
        assert model.GetModelInfo() is not None

        # Benchmark
        for (image_type, shape, B) in product(BENCHMARK_IMAGE_TYPES, shapes, BENCHMARK_BATCH_SIZES):
            # Load dummy image
            img = Image.open(NOS_TEST_IMAGE)
            img = img.resize(shape)
            W, H = shape
            nbytes = H * W * 3
            if image_type == np.ndarray:
                img = np.asarray(img)
                assert img.shape[:2][::-1] == shape
            elif image_type == Image.Image:
                assert img.size == shape
            else:
                raise TypeError(f"Invalid image type: {image_type}")

            # Skip if batched images are >= 512 MB
            if B * nbytes >= 512 * 1024**2:
                continue

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
            duration = 5.0
            for nbatches in tqdm(  # noqa: B007
                duration=duration,
                desc=f"Benchmark model={model_name}, task={task} [B={B}, shape={shape}, size={(B * nbytes) / 1024 ** 2:.1f}MB, type={image_type}]",
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
            timing_records.append(
                TimingInfo(
                    desc=f"{model_name}_{B}x{shape[0]}x{shape[1]}x3",
                    b=B,
                    n=B * nbatches,
                    elapsed=duration,
                    shape=shape,
                    image_type=image_type.__name__,
                )
            )
            if shm_enabled:
                model.UnregisterSystemSharedMemory()

    # Save timing records
    backend = "gpu" if "gpu" in client_with_server else "cpu"
    date_str = datetime.utcnow().strftime("%Y%m%d")

    # Print timing records
    timing_df = pd.DataFrame(
        [r.to_dict() for r in timing_records], columns=["desc", "b", "n", "elapsed", "shape", "image_type", "backend"]
    )
    timing_df = timing_df.assign(
        elapsed=lambda x: x.elapsed.round(2),
        latency_ms=lambda x: ((x.elapsed / x.n) * 1000).round(2),
        fps=lambda x: (1 / (x.elapsed / x.n)).round(2),
        date=date_str,
        backend=backend,
        version=nos_version,
    )
    logger.info(f"\nTiming records\n{timing_df}")

    NOS_DIR = Path(nos.__file__).parent.parent / ".nos"
    NOS_BENCHMARK_DIR = NOS_DIR / "benchmark"
    NOS_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    version_str = nos.__version__.replace(".", "-")
    profile_path = Path(NOS_BENCHMARK_DIR) / f"nos-{backend}-inference-benchmark--{version_str}--{date_str}.json"
    timing_df.to_json(str(profile_path), orient="records", indent=2)
    logger.info(f"Saved timing records to {str(profile_path)}")


def test_memray_tracking(request):  # noqa: F811
    client = request.getfixturevalue("grpc_client_with_gpu_backend")
    assert client is not None
    assert client.IsHealthy()

    # Load dummy image
    img = Image.open(NOS_TEST_IMAGE)

    # Load noop model (This should still trigger memray tracking)
    task, model_name = TaskType.CUSTOM, "noop/process-images"
    model = client.Module(task=task, model_name=model_name)
    assert model is not None
    assert model.GetModelInfo() is not None

    # Make an inference request and confirm that this doesn't break inference
    shape = (224, 224)
    images = [np.asarray(img.resize(shape))]
    inputs = {"images": images}
    response = model(**inputs)
    assert isinstance(response, dict)


def test_client_exception_types(request):
    # Inference request with malformed input.
    client = request.getfixturevalue("grpc_client_with_gpu_backend")
    assert client is not None
    assert client.IsHealthy()

    task, model_name = TaskType.CUSTOM, "noop/process-images"
    model = client.Module(task=task, model_name=model_name)
    assert model is not None
    assert model.GetModelInfo() is not None

    # TODO(scott): We only validate input count and not the types themselves. When
    # we finish input validation the test should change accordingly.
    inputs = {}
    with pytest.raises(NosInputValidationException):
        response = model(**inputs)
        assert isinstance(response, dict)
