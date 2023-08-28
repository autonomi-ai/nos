from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from nos.common.io import prepare_images

from .config import SuperResolutionConfig


@dataclass(frozen=True)
class SuperResolutionSwin2SRConfig(SuperResolutionConfig):
    """SuperResolution model configuration for Swin2SR."""

    pass


class SuperResolutionSwin2SR:
    """SuperResolution model using Swin2SR."""

    configs = {
        "caidas/swin2SR-classical-sr-x2-64": SuperResolutionSwin2SRConfig(
            model_name="caidas/swin2SR-classical-sr-x2-64", method="swin2sr"
        ),
    }

    def __init__(self, model_name: str = "caidas/swin2SR-classical-sr-x2-64"):
        from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

        try:
            self.cfg = SuperResolutionSwin2SR.configs[model_name]
        except KeyError:
            raise ValueError(
                f"Invalid model_name: {model_name}, available models: {SuperResolutionSwin2SR.configs.keys()}"
            )
        model_name = self.cfg.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Swin2SRForImageSuperResolution.from_pretrained(model_name)
        self.model.to(self.device)

    def __call__(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> np.ndarray:
        """Upsample image/video."""
        images = prepare_images(images)
        inputs = self.processor(images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = (output * 255.0).round().astype(np.uint8)
        return output
