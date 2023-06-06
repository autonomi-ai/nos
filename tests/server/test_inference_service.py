import pytest
from loguru import logger

from nos.executors.ray import RayExecutor
from nos.server.service import InferenceServiceImpl
from nos.test.conftest import ray_executor  # noqa: F401


pytestmark = pytest.mark.server


def test_model_manager(ray_executor: RayExecutor):  # noqa: F811
    from nos import hub
    from nos.server.service import ModelHandle, ModelManager

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
    from PIL import Image
    from tqdm import tqdm

    from nos import hub
    from nos.common import TaskType
    from nos.server.service import ModelHandle, ModelManager
    from nos.test.utils import NOS_TEST_IMAGE

    manager = ModelManager()
    assert manager is not None

    # Load a model spec
    spec = hub.load_spec("openai/clip-vit-base-patch32", task=TaskType.IMAGE_EMBEDDING)

    # Add the model to the manager (or via `manager.add()`)
    handle: ModelHandle = manager.get(spec)
    assert handle is not None

    img = Image.open(NOS_TEST_IMAGE)
    for _ in tqdm(range(10), desc="Inference with len(pool)=1 (warmup)"):
        result = handle.remote(images=[img] * 8)
        assert result is not None

    # Run inference
    img = Image.open(NOS_TEST_IMAGE)
    for _ in tqdm(range(500), desc="Inference with len(pool)=1"):
        result = handle.remote(images=[img] * 8)
        assert result is not None


@pytest.mark.skip(reason="This test is not ready yet.")
def test_inference_service_impl(ray_executor: RayExecutor):  # noqa: F811
    service = InferenceServiceImpl()
    assert service is not None
