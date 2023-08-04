from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch
from PIL import Image

from nos import hub
from nos.common import EmbeddingSpec, ImageSpec, TaskType
from nos.common.io import prepare_images
from nos.common.types import Batch, ImageT, TensorT
from nos.compilers import compile
from nos.constants import NOS_MODELS_DIR
from nos.hub import HuggingFaceHubConfig
from nos.logging import logger


@dataclass(frozen=True)
class CLIPConfig(HuggingFaceHubConfig):
    D: int = 512
    """Dimension of the embedding."""
    height: int = 224
    """Height of the input image."""
    width: int = 224
    """Width of the input image."""


class CLIP:
    """CLIP model for image and text encoding."""

    configs = {
        "openai/clip": CLIPConfig(
            model_name="openai/clip-vit-base-patch32",
            D=512,
        ),
        "openai/clip-vit-base-patch32": CLIPConfig(
            model_name="openai/clip-vit-base-patch32",
            D=512,
        ),
        "openai/clip-vit-large-patch14": CLIPConfig(
            model_name="openai/clip-vit-large-patch14",
            D=768,
        ),
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": CLIPConfig(
            model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            D=1024,
        ),
        "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": CLIPConfig(
            model_name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            D=768,
        ),
    }

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

        try:
            self.cfg = CLIP.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {CLIP.configs.keys()}")
        model_name = self.cfg.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # TOFIX (spillai): Regression with fp16
        # https://github.com/autonomi-ai/nos/issues/198
        # torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        torch_dtype = torch.float32
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=torch_dtype, torchscript=True).to(self.device)
        self.model.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_image(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> np.ndarray:
        """Encode image into an embedding."""
        images = prepare_images(images)
        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            if self.device != "cuda" and self.model.dtype == torch.float16:
                inputs = {k: v.half() for k, v in inputs.items()}
            image_features = self.model.get_image_features(**inputs)
            return image_features.cpu().numpy()

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text into an embedding."""
        with torch.inference_mode():
            if isinstance(texts, str):
                texts = [texts]
            inputs = self.tokenizer(
                texts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            return text_features.cpu().numpy()


class CLIPTensorRT(CLIP):
    """TensorRT accelerated for CLIP with Torch TensorRT."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._model_dir = Path(NOS_MODELS_DIR, f"cache/{self.cfg.model_name}")
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._patched = {}
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

    def _compile_vision_model(
        self, inputs: List[torch.Tensor], precision: torch.dtype = torch.float32
    ) -> torch.nn.Module:
        """Vision model compilation flow."""
        import torch.nn as nn
        import torch_tensorrt.fx.converter_registry as registry
        from torch_tensorrt.fx.tracer.acc_tracer import acc_ops
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        assert isinstance(inputs, list), f"inputs must be a list, got {type(inputs)}"
        assert len(inputs) == 1, f"inputs must be a list of length 1, got {len(inputs)}"
        logger.debug(f"Compiling {self.cfg.model_name} (vision_model) with precision: {precision}")

        class _CLIPVisionTransformer(nn.Module):
            """Wrapper for the vision model with patched outputs.

            We need this since the compilation removes the output types and
            requires us to patch the outputs manually with the correct types
            so that they can be used identically to the original model.
            """

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                return BaseModelOutputWithPooling(self.model(*args, **kwargs))

        # TODO (spillai): The compilation for CLIP requires acc_ops_expand_tensor to be disabled
        # to avoid the assertion thrown by the TRT backend. This is a temporary workaround
        # until the issue is fixed upstream.
        logger.debug("Disabling acc_ops.expand_tensor for CLIP compilation")
        expand_op = None
        if acc_ops.expand in registry.CONVERTERS.keys():
            expand_op = registry.CONVERTERS.pop(acc_ops.expand)
            logger.debug(f"Disabled {acc_ops.expand} from registry.CONVERTERS")

        # Check if we have a cached model
        model_id = CLIPTensorRT._get_model_id(f"{self.cfg.model_name}", inputs[0].shape, precision)
        filename = f"{self._model_dir}/{model_id}.torchtrt.pt"
        if Path(filename).exists():
            logger.debug(f"Found cached {model_id}: {filename}")
            trt_model = torch.load(filename)
            self.model.vision_model = _CLIPVisionTransformer(trt_model)
            return

        # Compile the model backbone
        # Note (spillai): Currently we hard-code the iamge size to 1x3x224x224
        B, H, W = 1, self.cfg.height, self.cfg.width
        args = {
            "pixel_values": torch.randn((B, 3, H, W), dtype=precision, device="cuda"),
            "output_attentions": self.model.config.output_attentions,
            "output_hidden_states": self.model.config.output_hidden_states,
            "return_dict": True,
        }
        try:
            trt_model = compile(self.model.vision_model, args, concrete_args=None, precision=precision, slug=model_id)
            logger.debug(f"Saving compiled {model_id} model to {filename}")
            torch.save(trt_model, filename)
            self.model.vision_model = _CLIPVisionTransformer(trt_model)
            logger.debug(f"Patched {model_id} model")
        except Exception as e:
            import traceback

            logger.error(f"Failed to compile {model_id} model\n{traceback.format_exc()}")
            raise e

        # Restore acc_ops.expand_tensor
        if expand_op is not None:
            logger.debug("Restoring acc_ops.expand_tensor for CLIP compilation")
            registry.CONVERTERS[acc_ops.expand] = expand_op
            logger.debug(f"Restored {acc_ops.expand} to registry.CONVERTERS")

    def encode_image(self, images: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> np.ndarray:
        """Encode image into an embedding."""
        images = prepare_images(images)

        # Resize images if necessary
        if self._patched_shape is None:
            H, W = self.cfg.height, self.cfg.width
        else:
            _, H, W = self._patched_shape
        images = [cv2.resize(image, (W, H)) if image.shape[-2:] != (H, W) else image for image in images]

        if not len(self._patched):
            # Note (spillai): Force compilation with resized images
            inputs = [torch.tensor(images)]
            self._compile_vision_model(inputs, precision=self.model.dtype)
            self._patched["vision_model"] = True
            self._patched_shape = (len(images), H, W)

        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            if self.device != "cuda" and self.model.dtype == torch.float16:
                inputs = {k: v.half() for k, v in inputs.items()}
            image_features = self.model.get_image_features(**inputs)
            return image_features.cpu().numpy()


# Register all CLIP models (for both tasks img2vec and txt2vec)
for model_name in CLIP.configs:
    cfg = CLIP.configs[model_name]
    hub.register(
        model_name,
        TaskType.TEXT_EMBEDDING,
        CLIP,
        init_args=(model_name,),
        method_name="encode_text",
        inputs={"texts": Batch[str, 16]},
        outputs={"embedding": Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(cfg.D,), dtype="float32")]]},
    )
    hub.register(
        model_name,
        TaskType.IMAGE_EMBEDDING,
        CLIP,
        init_args=(model_name,),
        method_name="encode_image",
        inputs={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(cfg.height, cfg.width, 3), dtype="uint8")], 16]},
        outputs={"embedding": Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(cfg.D,), dtype="float32")]]},
    )
