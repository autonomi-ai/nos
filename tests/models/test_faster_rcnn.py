import pytest
from PIL import Image

from nos.models import FasterRCNN
from nos.test.benchmark import run_benchmark
from nos.test.utils import NOS_TEST_IMAGE, PyTestGroup, skip_all_if_no_torch_cuda, skip_if_no_torch_cuda


pytestmark = skip_all_if_no_torch_cuda()


def _test_predict(_model):
    img = Image.open(NOS_TEST_IMAGE)
    predictions = _model.predict([img, img])
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


MODEL_NAME = "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn"


@pytest.fixture(scope="module")
def model():
    yield FasterRCNN(model_name=MODEL_NAME)


def test_fasterrcnn_predict(model):
    _test_predict(model)


@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.HUB)
def test_fasterrcnn_model_variants():
    for model_name in FasterRCNN.configs.keys():
        model = FasterRCNN(model_name=model_name)
        _test_predict(model)


@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.BENCHMARK_MODELS)
@pytest.mark.parametrize(
    "model_name",
    [
        "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",
    ],
)
def test_fasterrcnn_benchmark(model_name):
    """
    Benchmark results (NVIDIA GeForce RTX 2080 Ti):

    """

    img = Image.open(NOS_TEST_IMAGE)

    model = FasterRCNN(model_name=model_name)
    time_ms = run_benchmark(
        lambda: model.predict(img),
        num_iters=1000,
    )
    print(f"BENCHMARK [{model_name}]: {time_ms:.2f} ms / step")


@pytest.mark.skip(reason="Not implemented yet")
def test_fasterrcnn_predict_image(model):
    from PIL import Image

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
