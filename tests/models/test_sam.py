"""
SAM model tests and benchmarks.
"""

import numpy as np
import pytest
from PIL import Image
from typing import List

from nos import hub
from nos.common import TaskType
from nos.logging import logger
from nos.models import SAM
from nos.test.utils import NOS_TEST_IMAGE, skip_if_no_torch_cuda

import gc
import torch


def test_sam(_model, img_size):
    # Test segmentations with a variety of sizes.
    # Sample points will be fixed by grid.
    W, H = img_size
    img1 = Image.open(NOS_TEST_IMAGE)
    img1 = img1.resize((W, H))
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
        masks = _model(images)
        assert masks is not None
        gc.collect()
        torch.cuda.empty_cache()

@skip_if_no_torch_cuda
@pytest.mark.parametrize("model_name", SAM.configs.keys())
# @pytest.mark.parametrize("img_size", [(640, 480), (1280, 960)])
@pytest.mark.parametrize("img_size", [(640, 480)])
def test_object_detection_predict_one(model_name, img_size):
    logger.debug(f"Testing model: {model_name}")
    spec = hub.load_spec(model_name, task=TaskType.IMAGE_SEGMENTATION_2D)
    model = hub.load(spec.name, task=spec.task)
    logger.info("Test prediction with model: {}".format(model))
    test_sam(model, img_size)