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

    # Creating a model with num_replicas > 1 should raise a `NotImplementedError`.
    with pytest.raises(NotImplementedError):
        ModelHandle(spec, num_replicas=2)

    # Creating a model with an invalid eviction policy should raise a `NotImplementedError`.
    with pytest.raises(NotImplementedError):
        ModelManager(policy=ModelManager.EvictionPolicy.LRU)


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

    # NoOp: submit + get (perf.)
    def noop_gen(_noop, _pbar, B):
        for idx in _pbar:
            _noop.submit(images=img)
            desc = f"noop async [B={B}, replicas={_noop.num_replicas}, idx={idx}, pending={len(_noop.pending)}, queue={len(_noop.results)}]"
            _pbar.set_description(desc)
            if _noop.full():
                yield _noop.get()
        while _noop.has_next():
            yield _noop.get()

    for replicas in [2, 4, 8]:
        noop = noop.scale(replicas)
        logger.debug(f"NoOp ({replicas}): {noop}")
        pbar = tqdm(duration=10, unit_scale=B, desc=f"noop async [B={B}, replicas={noop.num_replicas}]", total=0)

        idx = 0
        for result in noop_gen(noop, pbar, B):
            assert isinstance(result, list)
            assert len(result) == B
            idx += 1

        assert idx == pbar.n
        assert len(noop.results) == 0
        assert len(noop.pending) == 0


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
@pytest.mark.parametrize(
    "model_name",
    [
        "openai/clip-vit-base-patch32",
    ],
)
@pytest.mark.parametrize("scale", [1, 8])
def test_model_manager_inference(model_name, scale, manager):  # noqa: F811
    """Benchmark the model manager with a single model."""
    import time

    from PIL import Image
    from tqdm import tqdm

    from nos.common import TaskType
    from nos.test.utils import NOS_TEST_IMAGE

    img = Image.open(NOS_TEST_IMAGE)
    W, H = 224, 224
    img = img.resize((W * scale, H * scale))
    img = np.asarray(img)

    # Load a model spec
    task = TaskType.IMAGE_EMBEDDING
    spec = hub.load_spec(model_name, task=task)

    # Add the model to the manager (or via `manager.add()`)
    handle: ModelHandle = manager.get(spec)
    assert handle is not None

    # Warmup
    for _ in tqdm(range(10), desc=f"Warmup model={model_name}, B=1", total=0):
        images = [img]
        result = handle(images=images)
        assert result is not None
        assert isinstance(result, np.ndarray)

    # Benchmark (10s)
    for b in range(0, 8):
        B = 2**b
        images = [img for _ in range(B)]
        st = time.time()
        for _ in tqdm(
            range(100_000),
            desc=f"Benchmark model={model_name}, task={task} [B={B}, shape={img.shape}]",
            unit="images",
            unit_scale=B,
            total=0,
        ):
            embedding = handle(images=images)
            assert embedding is not None
            assert isinstance(embedding, np.ndarray)
            N, _ = embedding.shape
            assert N == B
            if time.time() - st > 10.0:
                break
