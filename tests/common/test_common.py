import numpy as np
import pytest
import torch
from PIL import Image

from nos.common import tqdm
from nos.common.io import prepare_images
from nos.test.utils import NOS_TEST_IMAGE


def test_common_tqdm():
    import time

    # Use tqdm as a regular progress bar
    for i1, i2 in zip(range(10), tqdm(range(10)), strict=False):
        assert i1 == i2

    # Use tqdm as a timer
    for _ in tqdm(duration=1):
        time.sleep(0.1)


def test_prepare_images():
    img = Image.open(NOS_TEST_IMAGE)

    # Image.Image, List[Image.Image]
    images = prepare_images(img)
    assert isinstance(images, list)
    assert isinstance(images[0], np.ndarray)
    assert len(images) == 1

    images = prepare_images([img for _ in range(2)])
    assert isinstance(images, list)
    assert isinstance(images[0], np.ndarray)
    assert len(images) == 2

    # np.ndarray, List[np.ndarray]
    images = prepare_images(np.asarray(img))
    assert isinstance(images, list)
    assert isinstance(images[0], np.ndarray)
    assert len(images) == 1

    images = prepare_images([np.asarray(img) for _ in range(2)])
    assert isinstance(images, list)
    assert isinstance(images[0], np.ndarray)
    assert len(images) == 2

    # Invalid types / shapes
    INVALID_TYPES = [torch.tensor([1, 2, 3]), [[[1, 2, 3]]]]
    INVALID_SHAPES = [(1, 1), (1, 1, 1, 1, 1)]
    for shape in INVALID_SHAPES:
        with pytest.raises(ValueError):
            prepare_images(np.zeros(shape=shape))

    for t in INVALID_TYPES:
        with pytest.raises(TypeError):
            prepare_images(t)
