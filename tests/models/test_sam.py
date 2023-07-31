"""
SAM model tests and benchmarks.
"""

from typing import List

import numpy as np
import pytest
from PIL import Image

from nos import hub
from nos.common import TaskType
from nos.logging import logger
from nos.models import SAM
from nos.test.utils import NOS_TEST_IMAGE, skip_if_no_torch_cuda


@skip_if_no_torch_cuda
@pytest.mark.parametrize("model_name", SAM.configs.keys())
@pytest.mark.parametrize("img_size", [(640, 480), (1280, 960)])
# @pytest.mark.parametrize("img_size", [(640, 480)])
def test_sam(model_name, img_size):
    # Test segmentations with a variety of sizes.
    # Sample points will be fixed by grid.
    # NOTE: This will OOM on 2080 if anything else is running on card.

    logger.debug(f"Testing model: {model_name}")
    spec = hub.load_spec(model_name, task=TaskType.IMAGE_SEGMENTATION_2D)
    model = hub.load(spec.name, task=spec.task)
    logger.info("Test prediction with model: {}".format(model))

    W, H = img_size
    img1 = Image.open(NOS_TEST_IMAGE)
    img1 = img1.resize((W, H))
    Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))

    for im_type in (List[Image.Image], List[np.ndarray], np.ndarray):
        if im_type == List[Image.Image]:
            images = [img1]
            logger.debug("Testing List[Image.Image] inference")
        elif im_type == List[np.ndarray]:
            images = [np.asarray(img) for img in images]
            logger.debug("Testing List[np.ndarray] inference")
        elif im_type == np.ndarray:
            images = np.stack([np.asarray(img) for img in images])
            logger.debug("Testing stacked np.ndarray inference")
        masks = model(images)
        assert masks is not None
