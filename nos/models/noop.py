from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
from PIL import Image

from nos import hub
from nos.common import ImageSpec, TaskType
from nos.common.types import Batch, ImageT


@dataclass(frozen=True)
class NoopConfig:
    name: str


class Noop:
    """Noop model rep for benchmarking."""

    configs = {
        "noop/noop-image": NoopConfig(
            name="noop/noop-image",
        ),
    }

    def __init__(self):
        pass

    def __call__(
        self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        return {"success": True}


hub.register(
    "noop/noop-image",
    TaskType.BENCHMARK,
    Noop,
    method_name="__call__",
    inputs={
        "images": Union[
            Batch[ImageT[Image.Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 8],
            Batch[ImageT[Image.Image, ImageSpec(shape=(960, 1280, 3), dtype="uint8")], 1],
        ]
    },
    outputs={"success": bool},
)
