import pytest
from loguru import logger

from nos.constants import DEFAULT_GRPC_PORT
from nos.server._runtime import InferenceServiceRuntime
from nos.test.utils import NOS_TEST_IMAGE, PyTestGroup, skip_if_no_torch_cuda


# See `nos.server._runtime.InferenceServiceRuntime` for `mmdet-dev` runtime spec
RUNTIME_ENV = "mmdet-dev"
pytestmark = pytest.mark.skipif(pytest.importorskip("mmdet") is None, reason="mmdet is not installed")


@pytest.mark.skip(reason="TODO (spillai): Skip registration until new mmlab docker runtime is available")
@pytest.fixture(scope="module")
def openmmlab_runtime():
    runtime = InferenceServiceRuntime(runtime=RUNTIME_ENV, name=f"nos-{RUNTIME_ENV}-test")
    assert runtime is not None

    logger.debug(f"Starting inference service runtime: {RUNTIME_ENV}")
    runtime.start(ports={f"{DEFAULT_GRPC_PORT - 1}/tcp": DEFAULT_GRPC_PORT - 1})
    assert runtime.get_container() is not None
    assert runtime.get_container_id() is not None
    assert runtime.get_container_name() is not None
    assert runtime.get_container_status() is not None

    yield runtime

    logger.debug(f"Stopping inference service runtime: {RUNTIME_ENV}")
    runtime.stop()
    logger.debug(f"Stopped inference service runtime: {RUNTIME_ENV}")


@pytest.mark.parametrize(
    "model_name",
    [
        "open-mmlab/efficientdet-d3",
        "open-mmlab/faster-rcnn",
        "open-mmlab/yolox-small",
        # Note (spillai): For now, we can only test pre-trained models,
        # but we should be able to use custom models as listed below.
        # "open-mmlab/mmdetection/custom/yolox_s_8xb8-300e_coco_latest",
    ],
)
def test_mmdetection_predict(model_name):

    from PIL import Image

    from nos.models import MMDetection

    model = MMDetection(model_name)

    img = Image.open(NOS_TEST_IMAGE)
    predictions = model([img, img])
    assert predictions is not None

    assert predictions["scores"] is not None
    assert isinstance(predictions["scores"], list)
    assert len(predictions["scores"]) == 2

    assert predictions["labels"] is not None
    assert isinstance(predictions["labels"], list)
    assert len(predictions["labels"]) == 2

    assert predictions["bboxes"] is not None
    assert isinstance(predictions["bboxes"], list)
    assert len(predictions["bboxes"]) == 2


@pytest.mark.skip(reason="mmdetection benchmarking not yet implemented")
@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.MODEL_BENCHMARK)
@pytest.mark.parametrize(
    "model_name",
    [
        "open-mmlab/efficientdet-d3",
        "open-mmlab/faster-rcnn",
    ],
)
def test_mmdetection_benchmark(model_name):
    """Benchmark mmdetection models."""
    raise NotImplementedError()
