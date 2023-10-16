from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from nos import hub
from nos.common import EmbeddingSpec, ImageSpec, TaskType
from nos.common.io import prepare_images
from nos.common.types import Batch, ImageT, TensorT
from nos.hub import HuggingFaceHubConfig


@dataclass(frozen=True)
class CLIPConfig(HuggingFaceHubConfig):
    D: int = 512
    """Dimension of the embedding."""
    height: int = 224
    """Height of the input image."""
    width: int = 224
    """Width of the input image."""


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

        try:
            self.cfg = CLIP.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {CLIP.configs.keys()}")
        model_name = self.cfg.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # TOFIX (spillai): Regression with fp16
        # https://github.com/autonomi-ai/nos/issues/198
        # torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        torch_dtype = torch.float32
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=torch_dtype, torchscript=True).to(self.device)
        self.model.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_image(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> np.ndarray:
        """Encode image into an embedding."""
        images = prepare_images(images)
        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            if self.device != "cuda" and self.model.dtype == torch.float16:
                inputs = {k: v.half() for k, v in inputs.items()}
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
        TaskType.IMAGE_EMBEDDING,
        CLIP,
        init_args=(model_name,),
        method="encode_image",
        inputs={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(cfg.height, cfg.width, 3), dtype="uint8")], 16]},
        outputs={"embedding": Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(cfg.D,), dtype="float32")]]},
    )
    hub.register(
        model_name,
        TaskType.TEXT_EMBEDDING,
        CLIP,
        init_args=(model_name,),
        method="encode_text",
        inputs={"texts": Batch[str, 16]},
        outputs={"embedding": Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(cfg.D,), dtype="float32")]]},
    )
