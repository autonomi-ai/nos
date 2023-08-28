"""Tests for super-resolution models."""

import numpy as np
import pytest
from loguru import logger
from PIL import Image

from nos import hub
from nos.common import TaskType
from nos.models import SuperResolution
from nos.models.super_resolution import SuperResolutionLDM, SuperResolutionSwin2SR
from nos.test.benchmark import run_benchmark
from nos.test.utils import NOS_TEST_IMAGE, PyTestGroup, skip_if_no_torch_cuda


MODELS = list(SuperResolution.configs.keys())
UNIQUE_MODELS = list(SuperResolutionSwin2SR.configs.keys())[:1] + list(SuperResolutionLDM.configs.keys())[:1]


def _test_predict(_model):
    img = Image.open(NOS_TEST_IMAGE)
    img = img.resize((160, 120))
    W, H = img.size

    predictions = _model(img)
    assert predictions is not None
    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype == np.uint8
    UH, UW = predictions.shape[-2:]
    assert UW >= W * 2 and UH >= H * 2

    predictions = _model([img, img])
    assert predictions is not None
    assert len(predictions) == 2
    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype == np.uint8
    UH, UW = predictions.shape[-2:]
    assert UW >= W * 2 and UH >= H * 2


@skip_if_no_torch_cuda
@pytest.mark.parametrize("model_name", UNIQUE_MODELS)
def test_superres_predict_one(model_name):
    """ "Load/infer first depth estimation model."""
    logger.debug(f"Testing model: {model_name}")
    spec = hub.load_spec(model_name, task=TaskType.IMAGE_SUPER_RESOLUTION)
    model = hub.load(spec.name, task=spec.task)
    _test_predict(model)


@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.HUB)
@pytest.mark.parametrize("model_name", MODELS)
def test_superres_predict_all(model_name):
    """ "Benchmark load/infer all depth estimation models."""
    logger.debug(f"Testing model: {model_name}")
    spec = hub.load_spec(model_name, task=TaskType.IMAGE_SUPER_RESOLUTION)
    model = hub.load(spec.name, task=spec.task)
    _test_predict(model)


@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.MODEL_BENCHMARK)
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("img_size", [(640, 480), (1280, 960)])
def test_superres_predict_benchmark(model_name, img_size):
    """ "Benchmark inference for all depth estimation models."""

    img = Image.open(NOS_TEST_IMAGE)
    img = img.resize(img_size)

    logger.debug(f"Benchmarking model: {model_name}, img_size: {img_size}")
    spec = hub.load_spec(model_name, task=TaskType.IMAGE_SUPER_RESOLUTION)
    model = hub.load(spec.name, task=spec.task)
    time_ms = run_benchmark(
        lambda: model(img),
        num_iters=100,
    )
    logger.debug(f"[{model_name}]: {time_ms:.2f} ms / step")
