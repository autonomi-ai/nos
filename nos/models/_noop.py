from typing import Dict, List, Union

import numpy as np
from PIL import Image

from nos import hub
from nos.common import ImageSpec, TaskType
from nos.common.types import Batch, ImageT


class NoOp:
    """No-op model."""

    def process_images(
        self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if isinstance(images, (Image.Image, np.ndarray)):
            images = [images]
        return list(range(len(images)))


hub.register(
    "noop/process-images",
    TaskType.CUSTOM,
    NoOp,
    method_name="process_images",
    inputs={
        "images": Union[
            Batch[ImageT[Image.Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 8],
            Batch[ImageT[Image.Image, ImageSpec(shape=(960, 1280, 3), dtype="uint8")], 1],
        ]
    },
    outputs={"result": List[int]},
)
