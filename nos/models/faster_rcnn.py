from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image

from nos import hub
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
        # initialize fasterrcnn with pretrained weights
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        self.model.eval()

    def predict(
        self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        with torch.inference_mode():
            if isinstance(images, np.ndarray):
                pass
            elif isinstance(images, Image.Image):
                images = np.asarray(images)
            elif isinstance(images, list):
                raise ValueError("Batching not yet supported")
            
            tensor = torch.as_tensor(images.astype("float32")).permute(2, 0, 1)
            predictions = self.model([tensor])
            return {
                "scores": predictions[0]['boxes'].cpu().numpy(),
                "labels": predictions[0]['labels'].cpu().numpy(),
                "bboxes": predictions[0]['boxes'].cpu().numpy(),
            }


hub.register(
    "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",
    "img2bbox",
    FasterRCNN,
    args=("torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",),
)