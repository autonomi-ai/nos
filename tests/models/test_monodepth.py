"""Tests for monocular depth estimation models.

Benchmark results (2080 Ti):

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
(640, 480)
[isl-org/MiDaS_small]: 11.23 ms / step
[isl-org/MiDaS]: 19.05 ms / step

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
(1280, 960)
[isl-org/MiDaS_small]: 16.41 ms / step
[isl-org/MiDaS]: 24.54 ms / step
"""

import numpy as np
import pytest
from loguru import logger
from PIL import Image

from nos import hub
from nos.common import TaskType
from nos.models import MonoDepth
from nos.test.benchmark import run_benchmark
from nos.test.utils import NOS_TEST_IMAGE, PyTestGroup, skip_if_no_torch_cuda


MODELS = list(MonoDepth.configs.keys())


def _test_predict(_model):
    img = Image.open(NOS_TEST_IMAGE)
    W, H = img.size
    predictions = _model([img, img])
    assert predictions is not None
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 2
    for pred in predictions:
        assert pred.shape[-2:] == (H, W)


@skip_if_no_torch_cuda
@pytest.mark.parametrize("model_name", MODELS[:1])
def test_monodepth_predict_one(model_name):
    """ "Load/infer first depth estimation model."""
    logger.debug(f"Testing model: {model_name}")
    spec = hub.load_spec(model_name, task=TaskType.DEPTH_ESTIMATION_2D)
    model = hub.load(spec.name, task=spec.task)
    _test_predict(model)


@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.HUB)
@pytest.mark.parametrize("model_name", MODELS)
def test_monodepth_predict_all(model_name):
    """ "Benchmark load/infer all depth estimation models."""
    logger.debug(f"Testing model: {model_name}")
    spec = hub.load_spec(model_name, task=TaskType.DEPTH_ESTIMATION_2D)
    model = hub.load(spec.name, task=spec.task)
    _test_predict(model)


@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.MODEL_BENCHMARK)
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("img_size", [(640, 480), (1280, 960)])
def test_monodepth_predict_benchmark(model_name, img_size):
    """ "Benchmark inference for all depth estimation models."""

    img = Image.open(NOS_TEST_IMAGE)
    img = img.resize(img_size)

    logger.debug(f"Benchmarking model: {model_name}, img_size: {img_size}")
    spec = hub.load_spec(model_name, task=TaskType.DEPTH_ESTIMATION_2D)
    model = hub.load(spec.name, task=spec.task)
    time_ms = run_benchmark(
        lambda: model(img),
        num_iters=100,
    )
    logger.debug(f"[{model_name}]: {time_ms:.2f} ms / step")
