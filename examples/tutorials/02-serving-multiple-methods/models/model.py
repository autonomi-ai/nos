from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image


class CustomCLIPModel:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

        self.device = "cpu"
        assert not torch.cuda.is_available(), "CUDA should not be available here"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.eval()

    @torch.inference_mode()
    def __call__(
        self, images: Union[List[Image.Image], List[np.ndarray]], texts: Union[str, List[str]]
    ) -> Dict[str, np.ndarray]:
        """Encode image and text into embeddings."""
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        output = self.model(**inputs.to(self.device))
        return {
            "image_embeds": output["image_embeds"].cpu().numpy(),
            "text_embeds": output["text_embeds"].cpu().numpy(),
        }

    @torch.inference_mode()
    def encode_image(self, images: Union[List[Image.Image], List[np.ndarray]]) -> np.ndarray:
        """Encode image into an embedding."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy()

    @torch.inference_mode()
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text into an embedding."""
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy()
