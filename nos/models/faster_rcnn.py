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

        self.cfg = FasterRCNN.configs.get(model_name)
        model_name = self.cfg.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT").to(self.device)
        self.model.eval()

    def predict(
        self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        with torch.inference_mode():
            if isinstance(images, np.ndarray):
                images = [images]
            elif isinstance(images, Image.Image):
                images = [np.asarray(images)]
            elif isinstance(images, list):
                pass

            images = torch.stack([F.to_tensor(image) for image in images])
            images = images.to(self.device)
            predictions = self.model(images)
            return {
                "scores": [pred["boxes"].cpu().numpy() for pred in predictions],
                "labels": [pred["labels"].cpu().numpy() for pred in predictions],
                "bboxes": [pred["boxes"].cpu().numpy() for pred in predictions],
            }


hub.register(
    "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",
    TaskType.OBJECT_DETECTION_2D,
    FasterRCNN,
    init_args=("torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",),
    method_name="predict",
    inputs={
        "images": Union[
            Batch[ImageT[Image.Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 8],
            Batch[ImageT[Image.Image, ImageSpec(shape=(960, 1280, 3), dtype="uint8")], 4],
            Batch[ImageT[Image.Image, ImageSpec(shape=(1080, 1920, 3), dtype="uint8")], 1],
        ]
    },
    outputs={
        "scores": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="float32")]],
        "labels": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="float32")]],
        "bboxes": Batch[TensorT[np.ndarray, TensorSpec(shape=(None, 4), dtype="float32")]],
    },
)
