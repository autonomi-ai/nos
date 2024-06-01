from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from nos import hub
from nos.common import ImageSpec, TaskType
from nos.common.io import prepare_images
from nos.common.types import Batch, ImageT, TensorT
from nos.hub import TorchHubConfig


@dataclass(frozen=True)
class DepthConfig(TorchHubConfig):
    """Depth model configuration."""

    def get_transforms(self):
        """Get model-specific transforms for pre-processing."""
        if self.model_name.startswith("DPT_"):
            return torch.hub.load(self.repo, "transforms").dpt_transform
        return torch.hub.load(self.repo, "transforms").small_transform


class MonoDepth:
    """Monodepth models for depth understanding."""

    configs = {
        "isl-org/MiDaS-small": DepthConfig(
            repo="isl-org/MiDaS",
            model_name="MiDaS_small",
        ),
        "isl-org/MiDaS": DepthConfig(
            repo="isl-org/MiDaS",
            model_name="MiDaS",
        ),
    }

    def __init__(self, model_name: str = "isl-org/MiDaS-small"):
        """Initialize the model."""
        try:
            self.cfg = MonoDepth.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {MonoDepth.configs.keys()}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transforms = self.cfg.get_transforms()
        self.model = torch.hub.load(self.cfg.repo, self.cfg.model_name, pretrained=True).to(self.device)
        self.model.eval()

    def __call__(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> np.ndarray:
        """Predict depth from image."""
        images = prepare_images(images)
        H, W = images[0].shape[:2]
        with torch.inference_mode():
            input_batch = torch.cat([self.transforms(image) for image in images]).to(self.device)
            prediction = self.model(input_batch)  # (B, dH, dW)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            return prediction.cpu().numpy()


for model_name in MonoDepth.configs:
    hub.register(
        model_name,
        TaskType.DEPTH_ESTIMATION_2D,
        MonoDepth,
        init_args=(model_name,),
        method="__call__",
        inputs={
            "images": Union[
                Batch[ImageT[Image.Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 16],
                Batch[ImageT[Image.Image, ImageSpec(shape=(960, 1280, 3), dtype="uint8")], 4],
            ]
        },
        outputs={
            "depths": Batch[TensorT[np.ndarray, ImageSpec(shape=(None, None, 1), dtype="float32")]],
        },
    )
