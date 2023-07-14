from typing import List, Union

import numpy as np
from PIL import Image

from .video.opencv import VideoReader, VideoWriter  # noqa: F401


def prepare_images(images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> List[np.ndarray]:
    """Prepare images for inference.

    All images are converted to np.ndarray, either as a stacked batch or a list.
    Currently, this implementation does not stack images into a batch unless
    they are already stacked as a batch.

    Args:
        images: A single image or a list of images (PIL.Image or np.ndarray)

    Returns:
        images: A list of images (np.ndarray)
    """
    if isinstance(images, np.ndarray):
        # Only convert unary images as list, otherwise assume batched ndarray
        if images.ndim < 3 or images.ndim > 4:
            raise ValueError(f"Invalid number of dimensions for images: {images.ndim}")
        if images.ndim == 3:
            images = [images]
    elif isinstance(images, Image.Image):
        images = [np.asarray(images.convert("RGB"))]
    elif isinstance(images, list):
        if isinstance(images[0], Image.Image):
            images = [np.asarray(image.convert("RGB")) for image in images]
        elif isinstance(images[0], np.ndarray):
            pass
        else:
            raise TypeError(f"Invalid type for images: {type(images[0])}")
    else:
        raise TypeError(f"Invalid type for images: {type(images)}")
    # import pdb; pdb.set_trace()
    return images
