from typing import List, Union

import numpy as np
from PIL import Image

from nos import hub
from nos.common import ImageSpec, TaskType
from nos.common.types import Batch, ImageT


class NoOp:
    """No-op model."""

    def process_images(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> List[int]:
        if (isinstance(images, np.ndarray) and images.ndim == 3) or isinstance(images, Image.Image):
            images = [images]
        return list(range(len(images)))

    def process_texts(self, texts: List[str]) -> List[int]:
        return list(range(len(texts)))


# Register noop model separately for each method
hub.register(
    "noop/process-images",
    TaskType.CUSTOM,
    NoOp,
    inputs={
        "images": Union[
            Batch[ImageT[Image.Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 8],
            Batch[ImageT[Image.Image, ImageSpec(shape=(960, 1280, 3), dtype="uint8")], 1],
        ]
    },
    outputs={"result": List[int]},
    method="process_images",
)
hub.register(
    "noop/process-texts",
    TaskType.CUSTOM,
    NoOp,
    inputs={
        "texts": Batch[str, 1],
    },
    outputs={"result": List[int]},
    method="process_texts",
)

# Register model with two methods under the same name
hub.register(
    "noop/process",
    TaskType.CUSTOM,
    NoOp,
    inputs={
        "images": Union[
            Batch[ImageT[Image.Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 8],
            Batch[ImageT[Image.Image, ImageSpec(shape=(960, 1280, 3), dtype="uint8")], 1],
        ]
    },
    outputs={"result": List[int]},
    method="process_images",
)
hub.register(
    "noop/process",
    TaskType.CUSTOM,
    NoOp,
    inputs={
        "texts": Batch[str, 1],
    },
    outputs={"result": List[int]},
    method="process_texts",
)
