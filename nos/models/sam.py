from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from nos import hub
from nos.common import TaskType
from nos.common.types import Batch, ImageT
from nos.hub import HuggingFaceHubConfig

from nos.logging import logger


@dataclass(frozen=True)
class SAMConfig(HuggingFaceHubConfig):
    pass


class SAM:
    """SAM Segment Anything Model."""

    configs = {
        "facebook/sam-vit-large": SAMConfig(
            model_name="facebook/sam-vit-large",
        ), 
    }

    def __init__(self, model_name: str = "facebook/sam-vit-large"):
        from transformers import SamModel, SamProcessor

        self.cfg = SAM.configs.get(model_name)
        model_name = self.cfg.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = SamProcessor.from_pretrained(model_name)

    def __call__(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> np.ndarray:
        with torch.inference_mode():
            # 50 X 50 grid, evenly spaced across input image resolution
            # h, w = images.size 
            h, w = 640, 480
            grid_x, grid_y = torch.meshgrid(
                torch.linspace(0, w, 50, dtype=int), torch.linspace(0, h, 50, dtype=int)
            )
            # flatten grid to a list of (x, y) coordinates
            input_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            inputs = self.processor(images=images, input_points=[input_points.tolist()], return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )
            assert len(masks) > 0
            return [mask.cpu().numpy() for mask in masks]


for model_name in SAM.configs:
    hub.register(
        model_name,
        TaskType.IMAGE_SEGMENTATION_2D,
        SAM,
        init_args=(model_name,),
        method_name="__call__",
        inputs={"images": Batch[ImageT[Image.Image]]},
        outputs={"masks": Batch[ImageT[Image.Image]]},
    )
