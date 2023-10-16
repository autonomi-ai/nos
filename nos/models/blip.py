from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from nos.common.io import prepare_images
from nos.hub import HuggingFaceHubConfig


@dataclass(frozen=True)
class BLIPConfig(HuggingFaceHubConfig):
    pass


class BLIP:
    """BLIP model for image and text encoding."""

    configs = {
        "Salesforce/blip2-opt-2.7b": BLIPConfig(
            model_name="Salesforce/blip2-opt-2.7b",
        ),
    }

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", dtype: torch.dtype = torch.float32):
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        try:
            self.cfg = BLIP.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {BLIP.configs.keys()}")

        if dtype not in (torch.float32, torch.float16, torch.int8):
            raise ValueError(
                f"Invalid dtype: {dtype}, available dtypes: {torch.float32}, {torch.float16}, {torch.int8}"
            )
        if dtype == torch.int8:
            self.dtype = torch.float16
            init_kwargs = {"load_in_8bit": True, "device_map": "auto"}
        elif dtype == torch.float16:
            self.dtype = torch.float16
            init_kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
        else:
            self.dtype = torch.float32
            init_kwargs = {"device_map": "auto"}

        model_name = self.cfg.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, **init_kwargs).to(
            self.device, self.dtype
        )

    def __call__(
        self,
        images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]],
        prompt: Union[str, List[str]] = None,
    ) -> np.ndarray:
        """Caption image."""
        images = prepare_images(images)
        with torch.inference_mode():
            if prompt:
                # conditional image captioning
                inputs = self.processor(images, prompt, return_tensors="pt").to(self.device, self.dtype)
            else:
                # unconditional image captioning
                inputs = self.processor(images, return_tensors="pt").to(self.device, self.dtype)

            out = self.model.generate(**inputs)
            return self.processor.decode(out[0], skip_special_tokens=True).strip()
