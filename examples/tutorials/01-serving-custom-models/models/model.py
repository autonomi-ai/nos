from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image


class CustomCLIPModel:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor

        assert torch.cuda.is_available(), "CUDA not available"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
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
