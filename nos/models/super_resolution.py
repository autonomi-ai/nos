from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from nos.hub import HuggingFaceHubConfig


@dataclass(frozen=True)
class SuperResolutionConfig(HuggingFaceHubConfig):
    pass


class SuperResolution:
    # Super resolution using https://huggingface.co/eugenesiow/drln
    configs = {
        "eugenesiow/drln": SuperResolutionConfig(
            model_name="eugenesiow/drln",
        ),
    }

    def __init__(self, model_name: str = "eugenesiow/drln", scale_factor: int = 2):
        from super_image import DrlnModel, ImageLoader

        self.cfg = SuperResolution.configs.get(model_name)
        model_name = self.cfg.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DrlnModel.from_pretrained(model_name, scale=scale_factor).to(self.device)
        self.model.eval()

    def predict(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> np.ndarray:
        with torch.inference_mode():
            # inputs = ImageLoader.load_image(image)
            preds = self.model(inputs)
            return preds.cpu().numpy()
