from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import ops

from nos import hub
from nos.common import ImageSpec, TaskType, TensorSpec
from nos.common.types import Batch, ImageT, TensorT
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

    def __call__(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> np.ndarray:
        """Predict bounding boxes for images."""
        with torch.inference_mode():
            if isinstance(images, np.ndarray):
                images = [images]
            elif isinstance(images, Image.Image):
                images = [np.asarray(images)]
            elif isinstance(images, list):
                if isinstance(images[0], Image.Image):
                    images = [np.asarray(image) for image in images]
                elif isinstance(images[0], np.ndarray):
                    pass
                else:
                    raise ValueError(f"Invalid type for images: {type(images[0])}")

            images = (
                torch.stack([F.to_tensor(image) for image in images]) * 255
            )  # yolox expects non-normalized 0-255 tensor
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


class YOLOX_TRT(YOLOX):
    """Torch-TensorRT runtime for YOLOX."""

    configs = {f"{k}-trt": v for k, v in YOLOX.configs.items()}

    def __init__(self, model_name: str = "yolox/medium-trt"):
        _model_name = model_name.replace("-trt", "")
        self.model_name = _model_name
        super().__init__(_model_name)

        self.model_dir = Path(NOS_MODELS_DIR, f"cache/{_model_name}")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.patched = False

    def __compile__(self, inputs: List[torch.Tensor], filename) -> str:
        """Model compilation flow."""
        import torch_tensorrt.fx.tracer.acc_tracer.acc_tracer as acc_tracer
        from torch_tensorrt.fx import InputTensorSpec, TRTInterpreter, TRTModule
        from torch_tensorrt.fx.tools.trt_splitter import TRTSplitter
        from torch_tensorrt.fx.utils import LowerPrecision

        # Trace the model backbone (TODO: Catch failures here and log problem layer?)
        traced = acc_tracer.trace(self.model.backbone, [inputs])

        # Split out TRT eligible segments
        splitter = TRTSplitter(traced, [inputs])
        split_mod = splitter()

        _ = splitter.node_support_preview(dump_graph=False)

        logger.info("Graph: \n" + str(split_mod.graph))

        def get_submod_inputs(mod, submod, inputs):
            acc_inputs = None

            def get_input(self, inputs):
                nonlocal acc_inputs
                acc_inputs = inputs

            handle = submod.register_forward_pre_hook(get_input)
            mod(*inputs)
            handle.remove()
            return acc_inputs

        # We need to lower each TRT eligible segment.
        # TODO: If we know the model can be fully lowered, we can skip the splitter part.
        for name, _ in split_mod.named_children():
            logger.info(f"Splitting {name}")
            if "_run_on_acc" in name:
                submod = getattr(split_mod, name)
                # Get submodule inputs for fx2trt
                acc_inputs = get_submod_inputs(split_mod, submod, [inputs])

                # fx2trt replacement
                interp = TRTInterpreter(
                    submod,
                    InputTensorSpec.from_tensors(acc_inputs),
                    explicit_batch_dimension=True,
                )
                r = interp.run(lower_precision=LowerPrecision.FP32)
                trt_mod = TRTModule(*r)
                setattr(split_mod, name, trt_mod)

        # write out split mod to file
        torch.save(split_mod, filename)
        logger.info(f"Saved compiled model to {filename}")
        return True

    def __call__(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> np.ndarray:

        # TODO: A little annoying that we need to do this twice. Should be cleaned up in YOLOX refactor.
        W, H = None, None
        if isinstance(images, np.ndarray):
            H, W = images.shape[-2:]
        elif isinstance(images, Image.Image):
            W, H = images.size
        elif isinstance(images, list):
            if isinstance(images[0], Image.Image):
                W, H = images[0].size
            elif isinstance(images[0], np.ndarray):
                H, W = images[0].shape[-2:]
            else:
                raise ValueError(f"Invalid type for images: {type(images[0])}")

        model_id = self.model_name.replace("/", "-") + "_" + f"{W}x{H}" + "_fp32"
        self.filename = f"{self.model_dir}/{model_id}.torchtrt.pt"
        if not Path(self.filename).exists():
            compiled = self.__compile__(torch.rand(1, 3, H, W).to(self.device), self.filename)
            assert compiled, "Failed to compile model."

        if not self.patched:
            # Monkey patch backbone
            logger.info("Patching backbone...")
            trt_backbone = torch.load(self.filename)
            self.model.backbone = trt_backbone
            self.patched = True

        logger.info("Compiled with height, width: " + str(H) + ", " + str(W))

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
                Batch[ImageT[Image.Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 8],
                Batch[ImageT[Image.Image, ImageSpec(shape=(960, 1280, 3), dtype="uint8")], 1],
            ]
        },
        outputs={
            "bboxes": Batch[TensorT[np.ndarray, TensorSpec(shape=(None, 4), dtype="float32")]],
            "scores": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="float32")]],
            "labels": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="int32")]],
        },
    )

for model_name in YOLOX_TRT.configs:
    hub.register(
        model_name,
        TaskType.OBJECT_DETECTION_2D,
        YOLOX_TRT,
        init_args=(model_name,),
        method_name="__call__",
        inputs={
            "images": Batch[ImageT[Image.Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 1],
        },
        outputs={
            "bboxes": Batch[TensorT[np.ndarray, TensorSpec(shape=(None, 4), dtype="float32")]],
            "scores": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="float32")]],
            "labels": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="int32")]],
        },
    )
