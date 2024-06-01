from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from nos.common.io import prepare_images

from .config import SuperResolutionConfig


@dataclass(frozen=True)
class SuperResolutionLDMConfig(SuperResolutionConfig):
    """SuperResolution model configuration for LDM-based models."""

    num_inference_steps: int = 100
    """Number of inference steps for LDM-based models."""
    eta: float = 1.0
    """Eta parameter for LDM-based models."""


class SuperResolutionLDM:
    """SuperResolution model using LDM."""

    configs = {
        "CompVis/ldm-super-resolution-4x-openimages": SuperResolutionLDMConfig(
            model_name="CompVis/ldm-super-resolution-4x-openimages", method="ldm"
        ),
    }

    def __init__(self, model_name: str = "CompVis/ldm-super-resolution-4x-openimages"):
        from diffusers import LDMSuperResolutionPipeline

        try:
            self.cfg = SuperResolutionLDM.configs[model_name]
        except KeyError:
            raise ValueError(
                f"Invalid model_name: {model_name}, available models: {SuperResolutionLDM.configs.keys()}"
            )
        model_name = self.cfg.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = LDMSuperResolutionPipeline.from_pretrained(model_name)
        self.pipe.to(self.device)

    def __call__(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> np.ndarray:
        """Upsample image/video."""
        images = prepare_images(images)
        with torch.inference_mode():
            images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(self.device)
            H, W = images.shape[2:]
            PH = (16 - H % 16) % 16
            PW = (16 - W % 16) % 16
            images = torch.nn.functional.pad(images, (0, PW, 0, PH))
            return np.stack(
                [
                    np.asarray(im).transpose(2, 0, 1)
                    for im in self.pipe(
                        images, num_inference_steps=self.cfg.num_inference_steps, eta=self.cfg.eta
                    ).images
                ]
            )
