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

from typing import List

import numpy as np
import pytest
from loguru import logger
from PIL import Image

from nos import hub
from nos.models import YOLOX, FasterRCNN
from nos.test.utils import NOS_TEST_IMAGE, PyTestGroup, skip_if_no_torch_cuda


MODELS = list(FasterRCNN.configs.keys()) + list(YOLOX.configs.keys())
UNIQUE_MODELS = list(FasterRCNN.configs.keys())[:1] + list(YOLOX.configs.keys())[:1]


def package_installed(package_name: str) -> bool:
    """Check if the current environment has a package installed"""
    import importlib

    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


if package_installed("torch_tensorrt"):
    UNIQUE_MODELS += ["yolox/medium-trt"]

if package_installed("mmdet"):
    UNIQUE_MODELS += ["open-mmlab/yolox-small"]


def _test_predict(_model, img_size):
    B = 2
    W, H = img_size
    img1 = Image.open(NOS_TEST_IMAGE)
    img1 = img1.resize((W, H))
    # we add an empty image to test batch inference,
    # and to make sure that the model can handle empty images.
    img2 = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
    for im_type in (List[Image.Image], List[np.ndarray], np.ndarray):
        if im_type == List[Image.Image]:
            images = [img1, img2]
            logger.debug("Testing List[Image.Image] inference")
        elif im_type == List[np.ndarray]:
            images = [np.asarray(img) for img in images]
            logger.debug("Testing List[np.ndarray] inference")
        elif im_type == np.ndarray:
            images = np.stack([np.asarray(img) for img in images])
            logger.debug("Testing stacked np.ndarray inference")
        predictions = _model(images)
        assert predictions is not None

        assert predictions["scores"] is not None
        assert isinstance(predictions["scores"], list)
        assert len(predictions["scores"]) == B
        assert len(predictions["scores"][0]) > 0
        assert len(predictions["scores"][1]) == 0  # empty image should have no detections
        for scores in predictions["scores"]:
            assert scores.dtype == np.float32
            if not len(scores):
                continue
            assert np.min(scores) >= 0.0 and np.max(scores) <= 1.0

        assert predictions["labels"] is not None
        assert isinstance(predictions["labels"], list)
        assert len(predictions["labels"]) == B
        for labels in predictions["labels"]:
            assert labels.dtype in (np.int32, np.int64)
            if not len(labels):
                continue
            assert len(np.unique(labels)) >= 3, "At least 3 different classes should be detected"
            assert labels.dtype in (np.int32, np.int64)

        assert predictions["bboxes"] is not None
        assert isinstance(predictions["bboxes"], list)
        assert len(predictions["bboxes"]) == B
        for bbox in predictions["bboxes"]:
            assert bbox.dtype == np.float32
            if not len(bbox):
                continue
            # Check if predictions are within 1% of the image dimensions
            assert (bbox[:, 0] >= -W * 0.01).all() and (bbox[:, 0] <= W + W * 0.01).all()
            assert (bbox[:, 1] >= -H * 0.01).all() and (bbox[:, 1] <= H + H * 0.01).all()


@skip_if_no_torch_cuda
@pytest.mark.parametrize("model_name", UNIQUE_MODELS)
@pytest.mark.parametrize("img_size", [(640, 480), (1280, 720), (2880, 1620)])
def test_object_detection_predict_one(model_name, img_size):
    logger.debug(f"Testing model: {model_name}")
    spec = hub.load_spec(model_name)
    model = hub.load(spec.name)
    logger.info("Test prediction with model: {}".format(model))
    _test_predict(model, img_size)


@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.HUB)
@pytest.mark.parametrize("model_name", MODELS)
def test_object_detection_predict_all(model_name):
    """ "Benchmark load/infer all object detection models."""
    logger.debug(f"Testing model: {model_name}")
    spec = hub.load_spec(model_name)
    model = hub.load(spec.name)
    _test_predict(model)


@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.MODEL_BENCHMARK)
@pytest.mark.parametrize("model_name", UNIQUE_MODELS)
@pytest.mark.parametrize("img_size", [(640, 480), (1280, 960)])
def test_object_detection_predict_benchmark(model_name, img_size):
    """ "Benchmark inference for all object detection models."""

    from nos.common import tqdm

    img = Image.open(NOS_TEST_IMAGE)
    img = img.resize(img_size)

    logger.debug(f"Benchmarking model: {model_name}, img_size: {img_size}")
    spec = hub.load_spec(model_name)
    model = hub.load(spec.name)
    for _ in tqdm(duration=10, unit="images", desc=f"{model_name}"):
        model(img)
