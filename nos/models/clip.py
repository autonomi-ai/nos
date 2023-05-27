from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from nos import hub
from nos.common import EmbeddingSpec, TaskType
from nos.common.types import Batch, ImageT, TensorT
from nos.hub import HuggingFaceHubConfig


@dataclass(frozen=True)
class CLIPConfig(HuggingFaceHubConfig):
    D: int = 512


class CLIP:
    """CLIP model for image and text encoding."""

    configs = {
        "openai/clip": CLIPConfig(
            model_name="openai/clip-vit-base-patch32",
            D=512,
        ),
        "openai/clip-vit-base-patch32": CLIPConfig(
            model_name="openai/clip-vit-base-patch32",
            D=512,
        ),
        "openai/clip-vit-large-patch14": CLIPConfig(
            model_name="openai/clip-vit-large-patch14",
            D=768,
        ),
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": CLIPConfig(
            model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            D=1024,
        ),
        "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": CLIPConfig(
            model_name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            D=768,
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
        """Encode image into an embedding."""
        with torch.inference_mode():
            if isinstance(images, (np.ndarray, Image.Image)):
                images = [images]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            return image_features.cpu().numpy()

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text into an embedding."""
        with torch.inference_mode():
            if isinstance(texts, str):
                texts = [texts]
            inputs = self.tokenizer(
                texts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            return text_features.cpu().numpy()


# Register all CLIP models (for both tasks img2vec and txt2vec)
for model_name in CLIP.configs:
    cfg = CLIP.configs[model_name]
    hub.register(
        model_name,
        TaskType.TEXT_EMBEDDING,
        CLIP,
        init_args=(model_name,),
        method_name="encode_text",
        inputs={"texts": Batch[str]},
        outputs={"embedding": Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(cfg.D,), dtype="float32")]]},
    )
    hub.register(
        model_name,
        TaskType.IMAGE_EMBEDDING,
        CLIP,
        init_args=(model_name,),
        method_name="encode_image",
        inputs={"images": Batch[ImageT[Image.Image]]},
        outputs={"embedding": Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(cfg.D,), dtype="float32")]]},
    )
