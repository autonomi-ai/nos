"""Tests for object detection models.

Benchmark results (2080 Ti):

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
(640, 480)

[torchvision/fasterrcnn_mobilenet_v3_large_320_fpn]: 14.79 ms / step
[yolox/small]: 10.67 ms / step
[yolox/medium]: 12.49 ms / step
[yolox/large]: 17.63 ms / step
[yolox/xlarge]: 27.38 ms / step
[yolox/tiny]: 10.63 ms / step
[yolox/nano]: 13.08 ms / step

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
(1280, 960)

[torchvision/fasterrcnn_mobilenet_v3_large_320_fpn]: 24.65 ms / step
[yolox/small]: 25.15 ms / step
[yolox/medium]: 38.11 ms / step
[yolox/large]: 59.41 ms / step
[yolox/xlarge]: 93.09 ms / step
[yolox/tiny]: 22.82 ms / step
[yolox/nano]: 24.67 ms / step
"""

import os

import numpy as np
import pytest
from loguru import logger
from PIL import Image

from nos import hub
from nos.common import TaskType
from nos.models import YOLOX, FasterRCNN
from nos.test.benchmark import run_benchmark
from nos.test.utils import NOS_TEST_IMAGE, PyTestGroup, skip_if_no_torch_cuda


MODELS = list(FasterRCNN.configs.keys()) + list(YOLOX.configs.keys())
UNIQUE_MODELS = list(FasterRCNN.configs.keys())[:1] + list(YOLOX.configs.keys())[:1]


# Only enable YOLOX TRT models in trt-dev and trt-runtime environments
if os.getenv("NOS_ENV", "") in ("trt-dev", "trt-runtime"):
    MODELS += ["yolox/medium-trt"]


def _test_predict(_model):
    B = 1
    img = Image.open(NOS_TEST_IMAGE)
    img = img.resize((640, 480))
    predictions = _model([img for _ in range(B)])
    assert predictions is not None

    assert predictions["scores"] is not None
    assert isinstance(predictions["scores"], list)
    assert len(predictions["scores"]) == B
    for scores in predictions["scores"]:
        assert np.min(scores) >= 0.0 and np.max(scores) <= 1.0

    assert predictions["labels"] is not None
    assert isinstance(predictions["labels"], list)
    assert len(predictions["labels"]) == B
    for labels in predictions["labels"]:
        assert len(np.unique(labels)) >= 3, "At least 3 different classes should be detected"
        assert labels.dtype == np.int32

    assert predictions["bboxes"] is not None
    assert isinstance(predictions["bboxes"], list)
    assert len(predictions["bboxes"]) == B
    for bbox in predictions["bboxes"]:
        assert (bbox[:, 0] >= -5e-1).all() and (bbox[:, 0] <= W).all()
        assert (bbox[:, 1] >= -5e-1).all() and (bbox[:, 1] <= H).all()


@skip_if_no_torch_cuda
@pytest.mark.parametrize("model_name", UNIQUE_MODELS)
def test_object_detection_predict_one(model_name):
    logger.debug(f"Testing model: {model_name}")
    spec = hub.load_spec(model_name, task=TaskType.OBJECT_DETECTION_2D)
    model = hub.load(spec.name, task=spec.task)
    _test_predict(model)


@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.HUB)
@pytest.mark.parametrize("model_name", MODELS)
def test_object_detection_predict_variants(model_name):
    logger.debug(f"Testing model: {model_name}")
    spec = hub.load_spec(model_name, task=TaskType.OBJECT_DETECTION_2D)
    model = hub.load(spec.name, task=spec.task)
    _test_predict(model)


@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.BENCHMARK_MODELS)
@pytest.mark.parametrize("model_name", MODELS)
# @pytest.mark.parametrize("img_size", [(640, 480), (1280, 960)])
@pytest.mark.parametrize(
    "img_size",
    [
        (640, 480),
    ],
)
def test_object_detection_predict_benchmark(model_name, img_size):
    """object detection models."""

    img = Image.open(NOS_TEST_IMAGE)
    img = img.resize(img_size)

    logger.debug(f"Benchmarking model: {model_name}, img_size: {img_size}")
    spec = hub.load_spec(model_name, task=TaskType.OBJECT_DETECTION_2D)
    model = hub.load(spec.name, task=spec.task)
    time_ms = run_benchmark(
        lambda: model(img),
        num_iters=100,
    )
    logger.debug(f"[{model_name}]: {time_ms:.2f} ms / step")
