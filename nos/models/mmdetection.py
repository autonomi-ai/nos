from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from nos import hub
from nos.hub import MMLabConfig

from pathlib import Path

from typing import Dict


@dataclass(frozen=True)
class MMDetectionConfig(MMLabConfig):
    score_threshold: float = 0.3

class MMDetection:

    configs = {
        "open-mmlab/efficientdet-d3": MMDetectionConfig(
            config="configs/open-mmlab/efficientdet_effb3_bifpn_8xb16-crop896-300e_coco.py",
            checkpoint="https://download.openmmlab.com/mmdetection/v3.0/efficientdet/efficientdet_effb3_bifpn_8xb16-crop896-300e_coco/efficientdet_effb3_bifpn_8xb16-crop896-300e_coco_20230223_122457-e6f7a833.pth",
        ),
    }

    def __init__(self, model_name: str = "open-mmlab/efficientdet-d3"):
        from mmdet.apis import inference_detector, init_detector

        self.cfg = MMDetection.configs.get(model_name)
        checkpoint = self.cfg.cached_checkpoint()
        config = str(Path(__file__).parent / self.cfg.config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = init_detector(config, checkpoint, device=self.device)
        self.inference_detector = inference_detector

    def predict(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> Dict[str, np.ndarray]:
        with torch.inference_mode():
            predictions = self.inference_detector(self.model, images)
            return {"scores" : predictions.pred_instances.scores.cpu().numpy(), "labels" : predictions.pred_instances.labels.cpu().numpy(), "bboxes" : predictions.pred_instances.bboxes.cpu().numpy()}


hub.register(
    "open-mmlab/efficientdet-d3",
    "img2bbox",
    MMDetection,
    args=("open-mmlab/efficientdet-d3",),
)