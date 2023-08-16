"""
Test the model manager (nos.managers.ModelManager).

See `model-manager-benchmarks.md` for full benchmark results.

NOTE: The noop benchmarks are only valid if the noop models have some overhead (i.e. 10ms+).
The following benchmarks were obtained with a noop model that sleeps for 10ms.

NOTE: Using OMP_NUM_THREADS=`psutil.cpu_percent(logical=False)`
OMP_NUM_THREADS=32 ray start --head

CPU benchmarks are run as:
`CUDA_VISIBLE_DEVICES="" pytest -sv tests/managers/test_model_manager.py -k test_model_manager_inference -m benchmark`

GPU benchmarks are run as:
`pytest -sv tests/managers/test_model_manager.py -k test_model_manager_inference -m benchmark`

"""
import numpy as np
import pytest
from loguru import logger

from nos import hub
from nos.common import ModelSpec, TaskType
from nos.managers import ModelHandle, ModelManager
from nos.test.conftest import model_manager as manager  # noqa: F401, F811


def test_model_manager(manager):  # noqa: F811
    # Test adding several models back to back with the same manager.
    # This should not raise any OOM errors as models are evicted
    # from the manager's cache.
    for idx, spec in enumerate(hub.list()):
        # Note: `manager.get()` is a wrapper around `manager.add()`
        # and creates a single replica of the model.
        handler: ModelHandle = manager.get(spec)
        assert handler is not None
        assert isinstance(handler, ModelHandle)

        logger.debug(">" * 80)
        logger.debug(f"idx: {idx}")
        logger.debug(f"Model manager: {manager}, spec: {spec}")

    # Check if the model manager contains the model.
    assert spec in manager

    # Test noop with model manager
    spec = hub.load_spec("noop/process-images", task=TaskType.CUSTOM)
    noop: ModelHandle = manager.get(spec)
    assert noop is not None
    assert isinstance(noop, ModelHandle)

    B = 1
    img = (np.random.rand(B, 480, 640, 3) * 255).astype(np.uint8)

    # NoOp: __call__
    result = noop(images=img)
    assert isinstance(result, list)
    assert len(result) == B


def test_model_manager_errors(manager):  # noqa: F811
    # Get model specs
    spec = None
    for _, _spec in enumerate(hub.list()):
        spec = _spec
        break

    # Re-adding the same model twice should raise a `ValueError`.
    manager.add(spec)
    assert spec in manager
    with pytest.raises(ValueError):
        manager.add(spec)

    # Creating a model with num_replicas > 1
    ModelHandle(spec, num_replicas=2)

    # Creating a model with an invalid eviction policy should raise a `NotImplementedError`.
    with pytest.raises(NotImplementedError):
        ModelManager(policy=ModelManager.EvictionPolicy.LRU)


@pytest.mark.server
def test_model_manager_custom_model_inference_with_custom_runtime(manager):  # noqa: F811
    """Test wrapping custom models for remote execution.

    See also: tests/common/test_common_spec.py for a similar test that
    simply wraps a custom model for execution purposes.
    """
    from pathlib import Path
    from typing import List, Union

    import numpy as np

    class CustomModel:
        """Custom inference model with onnx-runtime."""

        def __init__(self, model_name: str = "resnet18"):
            """Initialize the model."""
            import onnxruntime as ort
            from onnx import hub as onnx_hub

            # Load onnx model resnet18 from hub
            _ = onnx_hub.load(model_name)  # force download the model weights
            info = onnx_hub.get_model_info(model=model_name)

            # Load the model from the local cache
            _path = Path(onnx_hub.get_dir()) / info.model_path
            path = _path.parent / f"{info.metadata['model_sha']}_{_path.name}"
            self.session = ort.InferenceSession(str(path), providers=ort.get_available_providers())

        def __call__(self, images: Union[np.ndarray, List[np.ndarray]], n: int = 1) -> np.ndarray:
            if isinstance(images, np.ndarray) and images.ndim == 3:
                images = [images]
            (output,) = self.session.get_outputs()
            (input,) = self.session.get_inputs()
            (result,) = self.session.run(
                [output.name], {input.name: np.stack(images * n).astype(np.float32).transpose(0, 3, 1, 2)}
            )
            return result

    # Get the model spec for remote execution
    spec = ModelSpec.from_cls(
        CustomModel,
        init_args=(),
        init_kwargs={"model_name": "resnet18"},
        pip=["onnx", "onnxruntime", "pydantic<2"],
    )
    assert spec is not None
    assert isinstance(spec, ModelSpec)

    # Check if the model can be loaded with the ModelManager
    # Note: This will be executed as a Ray actor within a custom runtime env.
    model_handle = manager.get(spec)
    assert model_handle is not None
    assert isinstance(model_handle, ModelHandle)

    # Check if the model can be called
    images = [np.random.rand(224, 224, 3).astype(np.uint8)]
    result = model_handle(images=images)
    assert len(result) == 1
    assert isinstance(result, np.ndarray)

    # Check if the model can be called with keyword arguments
    result = model_handle(images=images, n=2)
    assert len(result) == 2
    assert isinstance(result, np.ndarray)

    # Check if the model can NOT be called with positional arguments
    # We expect this to raise an exception, as the model only accepts keyword arguments.
    with pytest.raises(Exception):
        result = model_handle(images, 2)


@pytest.mark.benchmark
def test_model_manager_noop_inference(manager):  # noqa: F811
    """Test inference with a no-op model."""

    from nos.common import tqdm

    spec = hub.load_spec("noop/process-images", task=TaskType.CUSTOM)
    noop: ModelHandle = manager.get(spec)
    assert noop is not None
    assert isinstance(noop, ModelHandle)

    B = 16
    img = (np.random.rand(B, 480, 640, 3) * 255).astype(np.uint8)

    # NoOp: __call__
    result = noop(images=img)
    assert isinstance(result, list)
    assert len(result) == B

    # NoOp: __call__ (perf.)
    pbar = tqdm(duration=5, unit_scale=B, desc=f"noop [B={B}]", total=0)
    for idx in pbar:
        result = noop(images=img)
        desc = f"noop [B={B}, idx={idx}, pending={len(noop.pending)}, queue={len(noop.results)}]"
        pbar.set_description(desc)

        assert isinstance(result, list)
        assert len(result) == B

    # NoOp: submit() + get_next()
    def noop_gen(_noop, _pbar, B):
        for idx in _pbar:
            _noop.submit(images=img)
            desc = f"noop async [B={B}, replicas={_noop.num_replicas}, idx={idx}, pending={len(_noop.pending)}, queue={len(_noop.results)}]"
            _pbar.set_description(desc)
            if _noop.results.ready():
                yield _noop.get_next()
        while _noop.has_next():
            yield _noop.get_next()

    # NoOp scaling with replicas: submit + get_next (perf.)
    for replicas in [1, 2, 4, 8]:
        # scale the mode
        noop = noop.scale(replicas)

        # test: __call__
        result = noop(images=img)  # init / warmup
        assert isinstance(result, list)
        assert len(result) == B

        logger.debug(f"NoOp ({replicas}): {noop}")
        pbar = tqdm(duration=10, unit_scale=B, desc=f"noop async [B={B}, replicas={noop.num_replicas}]", total=0)

        # warmup: submit()
        for result in noop_gen(noop, tqdm(duration=1, disable=True), B):
            assert isinstance(result, list)
            assert len(result) == B

        # benchmark: submit()
        idx = 0
        for _ in noop_gen(noop, pbar, B):
            idx += 1

        assert idx == pbar.n
        assert len(noop.results) == 0
        assert len(noop.pending) == 0


BENCHMARK_BATCH_SIZES = [2**b for b in (0, 4, 7)]  # [1, 16, 128]
BENCHMARK_MODELS = [
    (TaskType.CUSTOM, "noop/process-images", [(224, 224), (640, 480), (1280, 720), (2880, 1620)]),
    (TaskType.IMAGE_EMBEDDING, "openai/clip-vit-base-patch32", [(224, 224), (640, 480)]),
    (TaskType.OBJECT_DETECTION_2D, "yolox/medium", [(640, 480), (1280, 720), (2880, 1620)]),
    (
        TaskType.OBJECT_DETECTION_2D,
        "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",
        [(640, 480), (1280, 960), (2880, 1620)],
    ),
]
BENCHMARK_WARMUP_SEC = 2
BENCHMARK_DURATION_SEC = 10


@pytest.mark.benchmark
def test_model_manager_inference(manager):  # noqa: F811
    """Benchmark the model manager with a single model."""
    from datetime import datetime
    from itertools import product
    from pathlib import Path

    import pandas as pd

    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 1000)

    from PIL import Image

    from nos.common import timer, tqdm
    from nos.constants import NOS_CACHE_DIR
    from nos.test.utils import NOS_TEST_IMAGE
    from nos.version import __version__

    timing_records = []

    # Benchmark: for each model, and set of image-shapes
    for task, model_name, image_shapes in BENCHMARK_MODELS:
        # Load a model spec
        spec = hub.load_spec(model_name, task=task)

        # Add the model to the manager (or via `manager.add()`)
        model: ModelHandle = manager.get(spec)
        assert model is not None

        # Benchmark: for each image-shape and batch-size
        for (shape, B) in product(image_shapes, BENCHMARK_BATCH_SIZES):
            img = Image.open(NOS_TEST_IMAGE)
            W, H = shape
            nbytes = W * H * 3
            img = np.asarray(img.resize((W, H)))
            assert img.shape[:2] == (H, W)

            # Skip if batched images are >= 512 MB
            if B * nbytes >= 512 * 1024**2:
                continue

            # Prepare inputs
            inputs = {"images": np.stack([img for _ in range(B)])}

            # Benchmark: for each model, image-shape, and batch-size
            try:
                # Warmup (sync)
                model = model.scale(1)
                for _ in tqdm(
                    duration=BENCHMARK_WARMUP_SEC,
                    desc=f"Warmup SYNC [model={model_name}, B={B}, shape={shape}]",
                    total=0,
                ):
                    result = model(**inputs)
                    assert result is not None
                    assert isinstance(result, (np.ndarray, list, dict))
                    if isinstance(result, dict):
                        for _k, v in result.items():
                            assert isinstance(v, (np.ndarray, list))
                            assert len(v) == B
                    else:
                        assert len(result) == B

                # Benchmark (30s, sync)
                with timer(f"{model_name}_{B}x{W}x{H}", replicas=1, B=B, shape=shape) as info:
                    for n in tqdm(  # noqa: B007
                        duration=BENCHMARK_DURATION_SEC,
                        desc=f"Benchmark SYNC [model={model_name}, B={B}, shape={shape}]",
                        unit="images",
                        unit_scale=B,
                        total=0,
                    ):
                        result = model(**inputs)
                info.niters = n + 1
                info.n = (n + 1) * B
                logger.info(info)
                timing_records.append(info)

                # Benchmark (async)
                def handle_gen(_handle, _inputs, _pbar):
                    for _idx in _pbar:
                        _handle.submit(**_inputs)
                        if _handle.results.ready():
                            yield _handle.get_next()
                    while _handle.has_next():
                        yield _handle.get_next()

                # Model scaling with replicas: submit + get (perf.)
                for replicas in [2, 4]:
                    # Skip if total bytes processed are >= 256 MB
                    # to avoid GPU OOM errors.
                    if B * nbytes * replicas >= 256 * 1024**2:
                        continue

                    # Scale up the model
                    model = model.scale(replicas)

                    # Warmup (async)
                    pbar = tqdm(
                        duration=BENCHMARK_WARMUP_SEC,
                        unit_scale=B,
                        desc=f"Warmup ASYNC [model={model_name}, B={B}, shape={shape}, replicas={model.num_replicas}]",
                        total=0,
                    )
                    for result in handle_gen(model, inputs, pbar):
                        assert result is not None
                        assert isinstance(result, (np.ndarray, list, dict))
                        if isinstance(result, dict):
                            for _k, v in result.items():
                                assert isinstance(v, (np.ndarray, list))
                                assert len(v) == B
                        else:
                            assert len(result) == B

                    # Benchmark (30s, async)
                    pbar = tqdm(
                        duration=BENCHMARK_DURATION_SEC,
                        unit_scale=B,
                        desc=f"Benchmark ASYNC [model={model_name}, B={B}, shape={shape}, replicas={model.num_replicas}]",
                        total=0,
                    )
                    with timer(f"{model_name}_{B}x{W}x{H}_async", replicas=replicas, B=B, shape=shape) as info:
                        for n, result in enumerate(handle_gen(model, inputs, pbar)):  # noqa: B007
                            assert result is not None
                    info.niters = n + 1
                    info.n = (n + 1) * B
                    logger.info(info)
                    timing_records.append(info)
            except Exception as e:
                logger.error(f"Failed to run model [model={model_name}, B={B}, shape={shape}]: {e}]")
                continue

    timing_df = pd.DataFrame(
        [r.to_dict() for r in timing_records],
        columns=["desc", "elapsed", "n", "niters", "replicas", "B", "shape", "cpu_util"],
    )
    timing_df = timing_df.assign(
        elapsed=lambda x: x.elapsed.round(2),
        latency_ms=lambda x: ((x.elapsed / x.n) * 1000).round(2),
        throughput=lambda x: (1 / (x.elapsed / x.n)).round(2),
    )
    logger.info(f"\nTiming records\n{timing_df}")

    NOS_BENCHMARK_DIR = Path(NOS_CACHE_DIR) / "benchmarks"
    NOS_BENCHMARK_DIR.mkdir(exist_ok=True, parents=True)

    # Save timing records
    version_str = __version__.replace(".", "-")
    date_str = datetime.utcnow().strftime("%Y%m%d")
    profile_path = Path(NOS_BENCHMARK_DIR) / f"nos-model-manager-benchmark--{version_str}--{date_str}.json"
    timing_df.to_json(str(profile_path), orient="records", indent=2)
    logger.info(f"Saved timing records to: {str(profile_path)}")
