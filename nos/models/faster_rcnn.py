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

# TODO(Scott): Take another look at these dependencies
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from nos.logging import logger

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

        # Load the TRT engine file and runtime if availble
        if os.path.exists(self.cfg.engine_file):
            logger.info("Found TRT engine file at {self.cfg.engine_file}, initializing runtime and context...") 
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            self.trt_runtime = trt.Runtime(self.trt_logger)
            with open(self.cfg.engine_file, "rb") as engine_file:
                self.trt_engine = self.trt_runtime.deserialize_cuda_engine(engine_file.read())
                self.trt_context = self.trt_engine.create_execution_context()
                self.cuda_stream = cuda.Stream()

            logger.info("Initialized TRT runtime and context")

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
        logger.info("Run TRT Inference")
        
        # Check that we have the trt dependencies set up
        assert self.trt_logger is not None
        assert self.trt_runtime is not None
        assert self.trt_engine is not None
        assert self.trt_context is not None
        assert self.cuda_stream is not None

        with torch.inference_mode():
            if isinstance(images, np.ndarray):
                images = [images]
            elif isinstance(images, Image.Image):
                images = [np.asarray(images)]
            elif isinstance(images, list):
                pass

            images = torch.stack([F.to_tensor(image) for image in images])
            images = images.to(self.device).contiguous()

            output_shape = (1, 1000, 4)

            # Input/Output allocations
            logger.info("Setup IO allocations for TRT inference...")
            input_host = cuda.pagelocked_empty(trt.volume(images.shape), dtype=np.float32)
            output_host = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
            input_device = cuda.mem_alloc(input_host.nbytes)
            output_device = cuda.mem_alloc(output_host.nbytes)

            # TODO(Scott): do we want to use execute_async_v2?
            logger.info("Copy to device and execute...")
            cuda.memcpy_htod_async(input_device, input_host, self.cuda_stream)
            self.trt_context.execute_async_v2([int(input_device), int(output_device)], self.cuda_stream.handle, None)
            cuda.memcpy_dtoh_async(output_host, output_device, self.cuda_stream.handle)
            self.cuda_stream.synchronize()

            logger.info("Finished TRT execution")

            return {
                "scores": [],
                "labels": [],
                "bboxes": [],
            }


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
