from typing import List, Union

import numpy as np
from PIL import Image

from .video.opencv import VideoReader, VideoWriter  # noqa: F401


def prepare_images(images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> List[np.ndarray]:
    """Prepare images for inference.

    Args:
        images: A single image or a list of images (PIL.Image or np.ndarray)

    Returns:
        images: A list of images (np.ndarray)
    """
    if isinstance(images, np.ndarray):
        images = [images]
    elif isinstance(images, Image.Image):
        images = [np.asarray(images.convert("RGB"))]
    elif isinstance(images, list):
        if isinstance(images[0], Image.Image):
            images = [np.asarray(image.convert("RGB")) for image in images]
        elif isinstance(images[0], np.ndarray):
            pass
        else:
            raise ValueError(f"Invalid type for images: {type(images[0])}")
    return images
