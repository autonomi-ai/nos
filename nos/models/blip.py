from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from nos.common.io import prepare_images
from nos.hub import HuggingFaceHubConfig


@dataclass(frozen=True)
class BLIPConfig(HuggingFaceHubConfig):
    model_type: str = "blip2"
    """Type of the model (blip / blip2)."""


class BLIP:
    """BLIP model for image and text encoding."""

    configs = {
        "Salesforce/blip2-opt-2.7b": BLIPConfig(
            model_name="Salesforce/blip2-opt-2.7b",
            model_type="blip2",
        ),
        "Salesforce/blip-image-captioning-large": BLIPConfig(
            model_name="Salesforce/blip-image-captioning-large",
            model_type="blip",
        ),
        "Salesforce/blip-image-captioning-base": BLIPConfig(
            model_name="Salesforce/blip-image-captioning-base",
            model_type="blip",
        ),
    }

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large", dtype: torch.dtype = torch.float32):
        try:
            self.cfg = BLIP.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {BLIP.configs.keys()}")

        if self.cfg.model_type == "blip2":
            from transformers import Blip2ForConditionalGeneration as Blip2ForConditionalGeneration
            from transformers import Blip2Processor as BlipProcessor
        elif self.cfg.model_type == "blip":
            from transformers import BlipForConditionalGeneration, BlipProcessor
        else:
            raise ValueError(f"Invalid model_type: {self.cfg.model_type}, available models: blip, blip2")

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

        # BLIP does not support device_map
        if self.cfg.model_type == "blip":
            init_kwargs.pop("device_map")
            if dtype == torch.int8:
                raise ValueError(
                    f"BLIP does not support dtype: {dtype}, available dtypes: {torch.float32}, {torch.float16}"
                )

        model_name = self.cfg.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name, **init_kwargs).to(
            self.device, self.dtype
        )

    def __call__(
        self,
        images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]],
        prompt: Union[str, List[str]] = None,
        max_length: int = 100,
    ) -> np.ndarray:
        """Caption image."""
        images = prepare_images(images)
        assert isinstance(images, list)
        with torch.inference_mode():
            if prompt:
                # conditional image captioning
                inputs = self.processor(images, prompt, return_tensors="pt").to(self.device, self.dtype)
            else:
                # unconditional image captioning
                inputs = self.processor(images, return_tensors="pt").to(self.device, self.dtype)

            out = self.model.generate(**inputs, max_length=max_length)
            return [self.processor.decode(out[idx], skip_special_tokens=True).strip() for idx in range(len(images))]
