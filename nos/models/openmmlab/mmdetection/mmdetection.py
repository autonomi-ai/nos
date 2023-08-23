from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image

from nos import hub
from nos.common import ImageSpec, TaskType, TensorSpec
from nos.common.io import prepare_images
from nos.common.types import Batch, ImageT, TensorT
from nos.hub import MMLabConfig, MMLabHub


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
        "open-mmlab/yolox_s": MMDetectionConfig(
            config="configs/yolox/yolox_s_8xb8-300e_coco.py",
            checkpoint="https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth",
        ),
        "open-mmlab/yolox_l": MMDetectionConfig(
            config="configs/yolox/yolox_l_8xb8-300e_coco.py",
            checkpoint="https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
        ),
        "open-mmlab/yolox_x": MMDetectionConfig(
            config="configs/yolox/yolox_x_8xb8-300e_coco.py",
            checkpoint="https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth",
        ),
        "open-mmlab/yolox_tiny": MMDetectionConfig(
            config="configs/yolox/yolox_tiny_8xb8-300e_coco.py",
            checkpoint="https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth",
        ),
        # Note: The following registers the configs for all models in the local hub.
        **MMLabHub().configs,
    }

    def __init__(self, model_name: str = "open-mmlab/efficientdet-d3"):
        from mmdet.apis import inference_detector, init_detector

        try:
            self.cfg = MMDetection.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {MMDetection.configs.keys()}")
        checkpoint = self.cfg.cached_checkpoint
        config = str(Path(__file__).parent / self.cfg.config)
        # TODO (spillai): Add config validation
        assert Path(config).exists(), f"Config {config} does not exist."
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = init_detector(config, checkpoint, device=self.device)
        self.inference_detector = inference_detector

    def __call__(
        self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Predict bounding boxes for a list of images."""
        images = prepare_images(images)
        with torch.inference_mode():
            predictions = self.inference_detector(self.model, images)
            return {
                "scores": [pred.pred_instances.scores.cpu().numpy() for pred in predictions],
                "labels": [pred.pred_instances.labels.cpu().numpy() for pred in predictions],
                "bboxes": [pred.pred_instances.bboxes.cpu().numpy() for pred in predictions],
            }


# TODO (spillai): Skip registration until new mmlab docker runtime is available
for model_name in MMDetection.configs:
    hub.register(
        model_name,
        TaskType.OBJECT_DETECTION_2D,
        MMDetection,
        init_args=(model_name,),
        method_name="__call__",
        inputs={
            "images": Union[
                Batch[ImageT[Image.Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 16],
                Batch[ImageT[Image.Image, ImageSpec(shape=(960, 1280, 3), dtype="uint8")], 4],
            ]
        },
        outputs={
            "bboxes": Batch[TensorT[np.ndarray, TensorSpec(shape=(None, 4), dtype="float32")]],
            "scores": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="float32")]],
            "labels": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="int32")]],
        },
    )
