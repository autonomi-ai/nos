from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from nos import hub
from nos.hub import HuggingFaceHubConfig


@dataclass(frozen=True)
class CLIPConfig(HuggingFaceHubConfig):
    pass


class CLIP:
    """CLIP model for image and text encoding."""

    configs = {
        "openai/clip-vit-base-patch32": CLIPConfig(
            model_name="openai/clip-vit-base-patch32",
        ),
        "openai/clip-vit-large-patch14": CLIPConfig(
            model_name="openai/clip-vit-large-patch14",
        ),
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": CLIPConfig(
            model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        ),
        "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": CLIPConfig(
            model_name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        ),
    }

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

        self.cfg = CLIP.configs.get(model_name)
        model_name = self.cfg.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_image(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            return image_features.cpu().numpy()

    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        with torch.inference_mode():
            if isinstance(text, str):
                text = [
                    text,
                ]
            inputs = self.tokenizer(
                text,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            return text_features.cpu().numpy()


# Register all CLIP models
for model_name in CLIP.configs:
    hub.register(model_name, "txt2vec", CLIP, args=(model_name,))
