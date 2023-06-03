from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from nos import hub
from nos.common import ImageSpec, TaskType, TensorSpec
from nos.common.types import Batch, ImageT, TensorT
from nos.hub import TorchHubConfig
from nos.hub import TRTConfig

from nos.constants import NOS_MODELS_DIR

import tensorrt as trt
import pycuda.driver as cuda


@dataclass(frozen=True)
class FasterRCNNConfig(TorchHubConfig, TRTConfig):
    pass


class FasterRCNN:
    """FasterRCNN model from torchvision"""

    configs = {
        "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn": FasterRCNNConfig(
            model_name="torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",
            repo="pytorch/vision",
            # Place the trt engine file in the .nosd/models dir for now
            engine_file=[NOS_MODELS_DIR/"trt/fasterrcnn_mobilenet_v3_large_320_fpn.engine"]
        ),
    }

    def __init__(self, model_name: str = "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn"):
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

        try:
            self.cfg = FasterRCNN.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {FasterRCNN.configs.keys()}")
        model_name = self.cfg.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT").to(self.device)
        self.model.eval()

        # Load the TRT engine file if availble
        if os.path.exists(self.cfg.engine_file):
            with open(self.cfg.engine_file, "rb") as engine_file, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(engine_file.read())
                self.context = self.engine.create_execution_context()


    def predict(
        self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        with torch.inference_mode():
            if isinstance(images, np.ndarray):
                images = [images]
            elif isinstance(images, Image.Image):
                images = [np.asarray(images)]
            elif isinstance(images, list):
                pass

            images = torch.stack([F.to_tensor(image) for image in images])
            images = images.to(self.device)
            predictions = self.model(images)
            return {
                "scores": [pred["boxes"].cpu().numpy() for pred in predictions],
                "labels": [pred["labels"].cpu().numpy() for pred in predictions],
                "bboxes": [pred["boxes"].cpu().numpy() for pred in predictions],
            }
        

    def predict_trt(
        self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        with torch.inference_mode():
            if isinstance(images, np.ndarray):
                images = [images]
            elif isinstance(images, Image.Image):
                images = [np.asarray(images)]
            elif isinstance(images, list):
                pass

            images = torch.stack([F.to_tensor(image) for image in images])
            images = images.to(self.device).contiguous()

            
            """
            inputs = []
            outputs = []
            allocations = []
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                dtype = self.engine.get_binding_dtype(i)
                shape = self.engine.get_binding_shape(i)
                if self.engine.binding_is_input(i):
                    is_input = True
                if is_input:
                    assert shape == images.shape,  "Mismatch between ModelSpec and TRT engine input shape"
                    assert dtype == images.dtype, "Mismatch between ModelSpec and TRT engine input dtype"
                    self.batch_size = shape[0]
                size = np.dtype(trt.nptype(dtype)).itemsize
                for s in shape:
                    size *= s
                allocation = cuda.mem_alloc(size)
                binding = {
                    'index': i,
                    'name': name,
                    'dtype': np.dtype(trt.nptype(dtype)),
                    'shape': list(shape),
                    'allocation': allocation,
                }
                self.allocations.append(allocation)
                if self.engine.binding_is_input(i):
                    self.inputs.append(binding)
                else:
                    self.outputs.append(binding)

            # Don't need to specify the cuda stream for async execution?
            self.context.execute_v2(self.allocations)
            """

            return {
                "scores": [],
                "labels": [],
                "bboxes": [],
            }

            """
            return {
                "scores": [output["boxes"].cpu().numpy() for output in outputs],
                "labels": [output["labels"].cpu().numpy() for output in outputs],
                "bboxes": [output["boxes"].cpu().numpy() for output in outputs],
            }
            """


hub.register(
    "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",
    TaskType.OBJECT_DETECTION_2D,
    FasterRCNN,
    init_args=("torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",),
    method_name="predict",
    inputs={
        "images": Union[
            Batch[ImageT[Image.Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 8],
            Batch[ImageT[Image.Image, ImageSpec(shape=(960, 1280, 3), dtype="uint8")], 4],
            Batch[ImageT[Image.Image, ImageSpec(shape=(1080, 1920, 3), dtype="uint8")], 1],
        ]
    },
    outputs={
        "scores": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="float32")]],
        "labels": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="float32")]],
        "bboxes": Batch[TensorT[np.ndarray, TensorSpec(shape=(None, 4), dtype="float32")]],
    },
)


hub.register(
    "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",
    TaskType.OBJECT_DETECTION_2D_TRT,
    FasterRCNN,
    init_args=("torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",),
    method_name="predict_trt",
    inputs={
        "images": Union[
            Batch[ImageT[Image.Image, ImageSpec(shape=(480, 640, 3), dtype="uint8")], 8],
            Batch[ImageT[Image.Image, ImageSpec(shape=(960, 1280, 3), dtype="uint8")], 4],
            Batch[ImageT[Image.Image, ImageSpec(shape=(1080, 1920, 3), dtype="uint8")], 1],
        ]
    },
    outputs={
        "scores": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="float32")]],
        "labels": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="float32")]],
        "bboxes": Batch[TensorT[np.ndarray, TensorSpec(shape=(None, 4), dtype="float32")]],
    },
)
