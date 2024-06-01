import time
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
from PIL import Image

from nos import hub
from nos.common import ImageSpec, ModelResources, TaskType
from nos.common.types import Batch, ImageT


class NoOp:
    """No-op model."""

    def process_images(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> List[int]:
        if (isinstance(images, np.ndarray) and images.ndim == 3) or isinstance(images, Image.Image):
            images = [images]
        return list(range(len(images)))

    def process_texts(self, texts: List[str]) -> List[int]:
        return list(range(len(texts)))

    def process_file(self, path: Path) -> bool:
        assert path.exists(), f"File not found: {path}"
        return True

    def process_sleep(self, seconds: float) -> bool:
        time.sleep(seconds)
        return True

    def stream_texts(self, texts: List[str]) -> Iterable[str]:
        for line in texts:
            yield line.rstrip()
        for line in Path(__file__).open("r").readlines():
            yield line.rstrip()


# Register noop model separately for each method
resources = ModelResources(cpu=1, memory="100Mi", device="cpu")
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
    outputs=List[int],
    method="process_images",
    resources=resources,
)
hub.register(
    "noop/process-texts",
    TaskType.CUSTOM,
    NoOp,
    inputs={
        "texts": Batch[str, 1],
    },
    outputs=List[int],
    method="process_texts",
    resources=resources,
)
hub.register(
    "noop/process-file",
    TaskType.CUSTOM,
    NoOp,
    method="process_file",
    resources=resources,
)
hub.register(
    "noop/process-sleep",
    TaskType.CUSTOM,
    NoOp,
    method="process_sleep",
    resources=resources,
)
hub.register(
    "noop/stream-texts",
    TaskType.CUSTOM,
    NoOp,
    outputs=Iterable[str],
    method="stream_texts",
    resources=resources,
)

# Register model with multiple methods under the same name
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
    method="process_images",
    resources=resources,
)
hub.register(
    "noop/process",
    TaskType.CUSTOM,
    NoOp,
    inputs={
        "texts": Batch[str, 1],
    },
    method="process_texts",
    resources=resources,
)
hub.register(
    "noop/process",
    TaskType.CUSTOM,
    NoOp,
    method="process_file",
    resources=resources,
)
hub.register(
    "noop/process",
    TaskType.CUSTOM,
    NoOp,
    method="process_sleep",
)
hub.register(
    "noop/process",
    TaskType.CUSTOM,
    NoOp,
    method="stream_texts",
    resources=resources,
)
