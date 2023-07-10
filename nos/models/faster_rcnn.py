from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from nos import hub
from nos.common import ImageSpec, TaskType, TensorSpec
from nos.common.types import Batch, ImageT, TensorT
from nos.hub import TorchHubConfig


@dataclass(frozen=True)
class FasterRCNNConfig(TorchHubConfig):
    pass


class FasterRCNN:
    """FasterRCNN model from torchvision"""

    configs = {
        "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn": FasterRCNNConfig(
            model_name="torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",
            repo="pytorch/vision",
        ),
    }

    def __init__(self, model_name: str = "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn"):
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

        try:
            self.cfg = FasterRCNN.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {FasterRCNN.configs.keys()}")
        model_name = self.cfg.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT").to(self.device)
        self.model.eval()

    def __call__(
        self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Predict bounding boxes for images."""
        with torch.inference_mode():
            if isinstance(images, np.ndarray):
                images = [images]
            elif isinstance(images, Image.Image):
                images = [np.asarray(images)]
            elif isinstance(images, list):
                if isinstance(images[0], Image.Image):
                    images = [np.asarray(image) for image in images]
                elif isinstance(images[0], np.ndarray):
                    pass
                else:
                    raise ValueError(f"Invalid type for images: {type(images[0])}")

            images = torch.stack([F.to_tensor(image) for image in images])
            images = images.to(self.device)
            predictions = self.model(images)
            return {
                "bboxes": [pred["boxes"].cpu().numpy() for pred in predictions],
                "scores": [pred["scores"].cpu().numpy() for pred in predictions],
                "labels": [pred["labels"].cpu().numpy().astype(np.int32) for pred in predictions],
            }
        

    def __benchmark__(
        self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        return {"result": True}


hub.register(
    "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",
    TaskType.OBJECT_DETECTION_2D,
    FasterRCNN,
    init_args=("torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",),
    method_name="__call__",
    inputs={
        "images": Union[
            Batch[ImageT[Image.Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 8],
            Batch[ImageT[Image.Image, ImageSpec(shape=(960, 1280, 3), dtype="uint8")], 1],
        ]
    },
    outputs={
        "bboxes": Batch[TensorT[np.ndarray, TensorSpec(shape=(None, 4), dtype="float32")]],
        "scores": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="float32")]],
        "labels": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="int32")]],
    },
)


""" Register a noop equivalent for benchmarking. Going forward we should have these for most models."""
hub.register(
    "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",
    TaskType.BENCHMARK,
    FasterRCNN,
    init_args=("torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",),
    method_name="__call__",
    inputs={
        "images": Union[
            Batch[ImageT[Image.Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 8],
            Batch[ImageT[Image.Image, ImageSpec(shape=(960, 1280, 3), dtype="uint8")], 1],
        ]
    },
    outputs={"result" : bool},
)
