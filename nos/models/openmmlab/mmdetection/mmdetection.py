from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image

from nos.hub import MMLabConfig


@dataclass(frozen=True)
class MMDetectionConfig(MMLabConfig):
    score_threshold: float = 0.3
    """Score threshold for predictions."""


class MMDetection:
    """MMDetection models from open-mmlab."""

    configs = {
        "open-mmlab/efficientdet-d3": MMDetectionConfig(
            config="configs/efficientdet/efficientdet_effb3_bifpn_8xb16-crop896-300e_coco.py",
            checkpoint="https://download.openmmlab.com/mmdetection/v3.0/efficientdet/efficientdet_effb3_bifpn_8xb16-crop896-300e_coco/efficientdet_effb3_bifpn_8xb16-crop896-300e_coco_20230223_122457-e6f7a833.pth",
        ),
        "open-mmlab/faster-rcnn": MMDetectionConfig(
            config="configs/faster-rcnn/faster-rcnn_r50_fpn_1x_coco.py",
            checkpoint="https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        ),
    }

    def __init__(self, model_name: str = "open-mmlab/efficientdet-d3"):
        from mmdet.apis import inference_detector, init_detector

        try:
            self.cfg = MMDetection.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {MMDetection.configs.keys()}")
        checkpoint = self.cfg.cached_checkpoint()
        config = str(Path(__file__).parent / self.cfg.config)
        # TODO (spillai): Add config validation
        assert Path(config).exists(), f"Config {config} does not exist."
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = init_detector(config, checkpoint, device=self.device)
        self.inference_detector = inference_detector

    def predict(
        self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        with torch.inference_mode():
            if isinstance(images, np.ndarray):
                images = [images]
            elif isinstance(images, Image.Image):
                images = [np.asarray(images)]
            elif isinstance(images, list):
                images = [np.asarray(image) if isinstance(image, Image.Image) else image for image in images]
            predictions = self.inference_detector(self.model, images)
            return {
                "scores": [pred.pred_instances.scores.cpu().numpy() for pred in predictions],
                "labels": [pred.pred_instances.labels.cpu().numpy() for pred in predictions],
                "bboxes": [pred.pred_instances.bboxes.cpu().numpy() for pred in predictions],
            }


# TODO (spillai): Skip registration until new mmlab docker runtime is available
# for model_name in MMDetection.configs:
#     hub.register(
#         model_name,
#         TaskType.OBJECT_DETECTION_2D,
#         MMDetection,
#         init_args=(model_name,),
#         method_name="predict",
#     )
