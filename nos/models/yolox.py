from dataclasses import dataclass
from pathlib import Path
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
from nos.compilers import compile
from nos.constants import NOS_MODELS_DIR
from nos.hub import TorchHubConfig
from nos.logging import logger


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


class YOLOXTensorRT(YOLOX):
    """TensorRT accelerated for YOLOX with Torch TensorRT."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.verbose = kwargs.get("verbose", False)
        self._model_dir = Path(NOS_MODELS_DIR, f"cache/{self.cfg.model_name}")
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._patched = False
        self._patched_shape = None

    @staticmethod
    def _get_model_id(name: str, shape: torch.Size, dtype: torch.dtype) -> str:
        """Get model id from model name, shape and dtype."""
        replacements = {"/": "-", " ": "-"}
        for k, v in replacements.items():
            name = name.replace(k, v)
        shape = list(map(int, shape))
        shape_str = "x".join([str(s) for s in shape])
        precision_str = str(dtype).split(".")[-1]
        return f"{name}_{shape_str}_{precision_str}"

    def __compile__(self, inputs: List[torch.Tensor], precision: torch.dtype = torch.float32) -> torch.nn.Module:
        """Model compilation flow."""
        assert isinstance(inputs, list), f"inputs must be a list, got {type(inputs)}"
        assert len(inputs) == 1, f"inputs must be a list of length 1, got {len(inputs)}"
        keys = {"input"}
        args = dict(zip(keys, inputs))

        # Check if we have a cached model
        slug = "backbone"
        model_id = YOLOXTensorRT._get_model_id(f"{self.cfg.model_name}--{slug}", inputs[0].shape, precision)
        filename = f"{self._model_dir}/{model_id}.torchtrt.pt"
        if Path(filename).exists():
            logger.debug(f"Found cached {model_id}: {filename}")
            trt_model = torch.load(filename)
            self.model.backbone = trt_model
            return

        # Compile the model backbone
        try:
            trt_model = compile(self.model.backbone, args, concrete_args=None, precision=precision, slug=model_id)
            logger.debug(f"Saving compiled {model_id} model to {filename}")
            torch.save(trt_model, filename)
            self.model.backbone = trt_model
            logger.debug(f"Patched {model_id} model")
        except Exception as e:
            logger.error(f"Failed to compile {model_id} model: {e}")

    def __call__(
        self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Predict bounding boxes for images."""
        images = prepare_images(images)
        B = len(images)
        H, W = images[0].shape[:2]
        if not self._patched:
            assert H is not None and W is not None, "Must provide image size for first call to __call__"
            inputs = [torch.rand(B, 3, H, W).to(self.device)]
            self.__compile__(inputs, precision=torch.float32)
            self._patched = True  # we set this to patched even if the compilation fails
            self._patched_shape = (B, H, W)
        if (B, H, W) != self._patched_shape:
            raise ValueError(f"Image size changed from {self._patched_shape} to {(B, H, W)}")
        return super().__call__(images)


for model_name in YOLOX.configs:
    hub.register(
        model_name,
        TaskType.OBJECT_DETECTION_2D,
        YOLOX,
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

# for model_name in YOLOXTensorRT.configs:
#     hub.register(
#         model_name,
#         TaskType.OBJECT_DETECTION_2D,
#         YOLOXTensorRT,
#         init_args=(model_name,),
#         method_name="__call__",
#         inputs={
#             "images": Batch[ImageT[Image.Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 1],
#         },
#         outputs={
#             "bboxes": Batch[TensorT[np.ndarray, TensorSpec(shape=(None, 4), dtype="float32")]],
#             "scores": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="float32")]],
#             "labels": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="int32")]],
#         },
#     )
