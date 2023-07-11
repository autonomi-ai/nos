from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
from PIL import Image

from nos import hub
from nos.common import ImageSpec, TaskType
from nos.common.types import Batch, ImageT

from ._noop import NoOp  # noqa: F401
from .clip import CLIP  # noqa: F401
from .faster_rcnn import FasterRCNN  # noqa: F401
from .openmmlab.mmdetection.mmdetection import MMDetection  # noqa: F401
from .sam import SAM
from .stable_diffusion import StableDiffusion  # noqa: F401
from .yolox import YOLOX  # noqa: F401


class NoOp:
    """Noop model rep for benchmarking."""

    def process_images(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> bool:
        return True


hub.register(
    "noop/process_images",
    TaskType.CUSTOM,
    NoOp,
    method_name="process_images",
    inputs={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(None, None, 3), dtype="uint8")]]},
    outputs={"result": bool},
)
