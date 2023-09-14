from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch
from PIL import Image

from nos import hub
from nos.common import EmbeddingSpec, ImageSpec, TaskType
from nos.common.io import prepare_images
from nos.common.types import Batch, ImageT, TensorT
from nos.constants import NOS_MODELS_DIR
from nos.logging import logger

from .config import SuperResolutionConfig
from .ldm import SuperResolutionLDM
from .swin2sr import SuperResolutionSwin2SR


class SuperResolution:
    """SuperResolution model for image/video upsampling."""

    configs = {
        **SuperResolutionSwin2SR.configs,
        **SuperResolutionLDM.configs,
    }

    def __init__(self, model_name: str = "CompVis/ldm-super-resolution-4x-openimages"):
        try:
            self.cfg = SuperResolution.configs[model_name]
        except KeyError:
            raise ValueError(
                f"Invalid model_name: {model_name}, available models: {SuperResolutionConfig.configs.keys()}"
            )
        model_name = self.cfg.model_name

        if self.cfg.method == "ldm":
            self.model = SuperResolutionLDM(model_name)
        elif self.cfg.method == "swin2sr":
            self.model = SuperResolutionSwin2SR(model_name)
        else:
            raise ValueError(f"Invalid method: {self.cfg.method}, available methods: ['ldm', 'swin2sr']")

    def __call__(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> np.ndarray:
        """Upsample image/video."""
        images = prepare_images(images)
        with torch.inference_mode():
            return self.model(images)


for model_name in SuperResolution.configs:
    hub.register(
        model_name,
        TaskType.IMAGE_SUPER_RESOLUTION,
        SuperResolution,
        init_args=(model_name,),
        method_name="__call__",
        inputs={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(240, 320, 3), dtype="uint8")], 1]},
        outputs={
            "outputs": Batch[TensorT[np.ndarray, ImageSpec(shape=(None, None, 3), dtype="uint8")]],
        },
    )
