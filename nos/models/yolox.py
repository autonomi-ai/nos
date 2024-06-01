from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import ops

from nos import hub
from nos.common import ImageSpec, TaskType, TensorSpec
from nos.common.io import prepare_images
from nos.common.types import Batch, ImageT, TensorT
from nos.hub import TorchHubConfig


@dataclass(frozen=True)
class YOLOXConfig(TorchHubConfig):
    confidence_threshold: float = 0.3
    """Confidence threshold for object detection."""
    nms_threshold: float = 0.3
    """Non-maximum suppression threshold for object detection."""
    class_agnostic: bool = False
    """Whether to perform class-agnostic object detection."""

    def __post_init__(self):
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError(f"Invalid confidence_threshold: {self.confidence_threshold}")
        if self.nms_threshold < 0 or self.nms_threshold > 1:
            raise ValueError(f"Invalid nms_threshold: {self.nms_threshold}")


def postprocess(
    prediction: torch.Tensor,
    num_classes: int = 80,
    conf_threshold: float = 0.7,
    nms_threshold: float = 0.45,
    class_agnostic: bool = False,
):
    """Postprocessing for YOLOX."""

    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [np.empty(shape=(0, 7), dtype=np.float32) for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_threshold).squeeze()

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        # Class-agnostic / Class-specific NMS
        if class_agnostic:
            nms_out_index = ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_threshold,
            )
        else:
            nms_out_index = ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_threshold,
            )

        # Filter detections based on NMS
        detections = detections[nms_out_index]
        output[i] = detections.cpu().numpy()
    return output


class YOLOX:
    """YOLOX Object Detection
    https://github.com/Megvii-BaseDetection/YOLOX/tree/main#benchmark
    """

    configs = {
        "yolox/small": YOLOXConfig(
            repo="Megvii-BaseDetection/YOLOX",
            model_name="yolox_s",
        ),
        "yolox/medium": YOLOXConfig(
            repo="Megvii-BaseDetection/YOLOX",
            model_name="yolox_m",
        ),
        "yolox/large": YOLOXConfig(
            repo="Megvii-BaseDetection/YOLOX",
            model_name="yolox_l",
        ),
        "yolox/xlarge": YOLOXConfig(
            repo="Megvii-BaseDetection/YOLOX",
            model_name="yolox_x",
        ),
        "yolox/tiny": YOLOXConfig(
            repo="Megvii-BaseDetection/YOLOX",
            model_name="yolox_tiny",
        ),
        "yolox/nano": YOLOXConfig(
            repo="Megvii-BaseDetection/YOLOX",
            model_name="yolox_nano",
        ),
    }

    def __init__(self, model_name: str = "yolox/small"):
        try:
            self.cfg = YOLOX.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {YOLOX.configs.keys()}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load(self.cfg.repo, self.cfg.model_name).to(self.device)
        self.model.eval()

    def __call__(
        self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Predict bounding boxes for images."""
        images = prepare_images(images)
        with torch.inference_mode():
            images = (
                torch.stack([F.to_tensor(image) for image in images]) * 255
            )  # yolox expects non-normalized 0-255 tensor
            # Pad image to be divisible by 32
            _, _, H, W = images.shape
            PH = int(np.ceil(H / 32) * 32)
            PW = int(np.ceil(W / 32) * 32)
            images = torch.nn.functional.pad(images, (0, PW - W, 0, PH - H))
            images = images.to(self.device)
            predictions = self.model(images)
            predictions = postprocess(
                predictions,
                conf_threshold=self.cfg.confidence_threshold,
                nms_threshold=self.cfg.nms_threshold,
                class_agnostic=self.cfg.class_agnostic,
            )
            return {
                "bboxes": [p[:, :4] for p in predictions],
                "scores": [(p[:, 4] * p[:, 5]) for p in predictions],  # obj_conf * class_conf
                "labels": [p[:, 6].astype(np.int32) for p in predictions],
            }


for model_name in YOLOX.configs:
    hub.register(
        model_name,
        TaskType.OBJECT_DETECTION_2D,
        YOLOX,
        init_args=(model_name,),
        method="__call__",
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
