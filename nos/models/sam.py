from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from nos import hub
from nos.hub import HuggingFaceHubConfig


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

    def segment_image(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> np.ndarray:
        with torch.inference_mode():
            # empty points for now
            inputs = self.processor(images=images, input_points = [], return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
            return masks.cpu().numpy()


# Register all SAM models
for model_name in SAM.configs:
    hub.register(model_name, "segmentation", SAM, args=(model_name,))
