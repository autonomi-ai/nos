import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Union

import torch
from PIL import Image
from transformers.utils.fx import HFTracer, symbolic_trace

from nos import hub
from nos.common import TaskType
from nos.common.types import Batch, ImageSpec, ImageT
from nos.constants import NOS_MODELS_DIR
from nos.hub import HuggingFaceHubConfig
from nos.logging import logger


def get_model_id(name: str, shape: torch.Size, dtype: torch.dtype) -> str:
    """Get model id from model name, shape and dtype."""
    replacements = {"/": "-", " ": "-"}
    for k, v in replacements.items():
        name = name.replace(k, v)
    shape = list(map(int, shape))
    shape_str = "x".join([str(s) for s in shape])
    precision_str = str(dtype).split(".")[-1]
    return f"{name}_{shape_str}_{precision_str}"


@dataclass(frozen=True)
class StableDiffusionConfig(HuggingFaceHubConfig):
    """Configuration for StableDiffusion model."""

    pass


class StableDiffusion:
    """StableDiffusion model for text to image generation."""

    configs = {
        "CompVis/stable-diffusion-v1-4": StableDiffusionConfig(
            model_name="CompVis/stable-diffusion-v1-4",
        ),
        "runwayml/stable-diffusion-v1-5": StableDiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
        ),
        "stabilityai/stable-diffusion-2": StableDiffusionConfig(
            model_name="stabilityai/stable-diffusion-2",
        ),
        "stabilityai/stable-diffusion-2-1": StableDiffusionConfig(
            model_name="stabilityai/stable-diffusion-2-1",
        ),
    }

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-2",
        scheduler: str = "ddim",
        dtype: torch.dtype = torch.float16,
    ):
        from diffusers import (
            DDIMScheduler,
            DiffusionPipeline,
            EulerDiscreteScheduler,
        )

        try:
            self.cfg = StableDiffusion.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {StableDiffusion.configs.keys()}")

        # Only run on CUDA if available
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = dtype
            self.revision = "fp16"
            # TODO (spillai): Investigate if this has any effect on performance
            # tf32 vs fp32: https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/279
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            self.revision = None

        if scheduler == "ddim":
            self.scheduler = DDIMScheduler.from_pretrained(self.cfg.model_name, subfolder="scheduler")
        elif scheduler == "euler-discrete":
            self.scheduler = EulerDiscreteScheduler.from_pretrained(self.cfg.model_name, subfolder="scheduler")
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}, choose from: ['ddim', 'euler-discrete']")
        self.pipe = DiffusionPipeline.from_pretrained(
            self.cfg.model_name,
            scheduler=self.scheduler,
            torch_dtype=self.dtype,
            revision=self.revision,
        )
        self.pipe = self.pipe.to(self.device)

        # TODO (spillai): Pending xformers integration
        # self.pipe.enable_attention_slicing()
        # self.pipe.enable_xformers_memory_efficient_attention()

    def __call__(
        self,
        prompts: Union[str, List[str]],
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = None,
        width: int = None,
    ) -> List[Image.Image]:
        """Generate images from text prompt."""
        if isinstance(prompts, str):
            prompts = [prompts]
        return self.pipe(
            prompts * num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        ).images


@dataclass(frozen=True)
class StableDiffusionTensorRTCompilationConfig:
    """Compilation configuration for StableDiffusion model."""

    precision: torch.dtype = torch.float16
    """Precision to use for TensorRT."""
    width: int = 512
    """Width of the image."""
    height: int = 512
    """Height of the image."""


class StableDiffusionTensorRT(StableDiffusion):
    """TensorRT accelerated StableDiffusion with Torch TensorRT."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.verbose = kwargs.get("verbose", False)
        self.model_dir = Path(NOS_MODELS_DIR, f"cache/{self.cfg.model_name}")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._compilation_cfg = None
        self._patched = {}

    def get_inputs_vae(self, width: int = 512, height: int = 512, batch_size: int = 1):
        """Get inputs for VAE."""
        # x = torch.rand(batch_size, 3, height // 8, width // 8, dtype=self.pipe.vae.dtype, device=self.pipe.vae.device)
        z = torch.rand(
            2 * batch_size, 4, height // 8, width // 8, dtype=self.pipe.vae.dtype, device=self.pipe.vae.device
        )
        return [z]

    def get_inputs_text_encoder(self):
        """Get inputs for text encoder."""
        model = self.pipe.text_encoder
        args = model.dummy_inputs
        inputs = list(args.values())
        return [x.to(self.pipe.text_encoder.device) for x in inputs]

    def _trace_text_encoder(
        self, model: torch.nn.Module, dtype: torch.dtype = torch.float32
    ) -> torch.fx.graph_module.GraphModule:
        """Trace the text encoder of the model."""
        model = self.pipe.text_encoder
        args = model.dummy_inputs
        return symbolic_trace(model, input_names=list(args.keys()), disable_check=False, tracer_cls=HFTracer)

    def _trace_vae(
        self, model: torch.nn.Module, dtype: torch.dtype = torch.float32
    ) -> torch.fx.graph_module.GraphModule:
        """Trace the VAE of the model."""
        import torch.nn as nn

        # Note: For SDv2, we need to only trace `vae.decode` since that is method that is
        # called within SDv2's forward pass. If we try to trace the full VAE directly,
        # it returns a `GraphModule` that cannot access the `decode` method. We workaround
        # this by wrapping the decode method below in a `_AutoEncoderKLDecoder` class that
        # calls the `vae.decode` method and only trace the relevant codepaths for inference.
        class _AutoEncoderKLDecoder(nn.Module):
            """Wrapper for VAE to only trace the `decode` method."""

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, z: torch.FloatTensor, return_dict: bool = False):
                return self.model.decode(z, return_dict=return_dict)

        model = _AutoEncoderKLDecoder(self.pipe.vae)
        (z,) = self.get_inputs_vae()
        args = {"z": z}
        concrete_args = {
            "return_dict": False,
        }

        logger.debug("Tracing VAE with HFTracer")
        tracer = HFTracer()
        with torch.inference_mode():
            traced_graph = tracer.trace(model, concrete_args=concrete_args, dummy_inputs=args)
            traced_model = torch.fx.GraphModule(model, traced_graph)
        if self.verbose:
            traced_graph.print_tabular()

        traced_model.config = self.pipe.vae.config
        # The model class must be stored as an attribute to allow model deserialization, which uses trace, and thus
        # _generate_dummy_input, where the model class is needed.
        traced_model.class_for_deserialization = model.__class__
        traced_model.device = self.pipe.vae.device
        return traced_model

    def _trace_unet(self, dtype: torch.dtype = torch.float32) -> torch.fx.graph_module.GraphModule:
        raise NotImplementedError("TODO: Implement UNet tracing")

    def _compile(
        self,
        trace_fn: Callable,
        model: torch.nn.Module,
        inputs: List[torch.Tensor],
        precision: torch.dtype = torch.float32,
        slug: str = "model",
    ) -> torch.nn.Module:
        """Compile the text encoder / VAE

        Models are written to disk in the following format:
            f"{model_dir}/{model_id}.torchtrt.pt"
            ~/.nos/models/cache/openai-clip-vit-base-32--vae-HxW-fp32.torchtrt.pt
        """
        import torch_tensorrt
        from torch_tensorrt.fx.utils import LowerPrecision

        model_id = get_model_id(f"{self.cfg.model_name}--{slug}", inputs[0].shape, precision)
        filename = f"{self.model_dir}/{model_id}.torchtrt.pt"
        if Path(filename).exists():
            logger.debug(f"Found cached {slug}: {filename}")
            trt_model = torch.load(filename)
            return trt_model

        # TODO (spillai): Check if we have traced the right inputs here
        try:
            traced_model = trace_fn(model, precision)
        except Exception as e:
            logger.error(f"Failed to trace: {e}, skipping compilation")
            return None

        with torch.inference_mode():
            st = time.time()
            logger.debug(f"Compiling {slug} with TensorRT")
            try:
                GB_bytes = 1024 * 1024 * 1024
                trt_model = torch_tensorrt.fx.compile(
                    traced_model,
                    inputs,
                    min_acc_module_size=5,
                    max_batch_size=2048,
                    max_workspace_size=16 * GB_bytes,
                    explicit_batch_dimension=True,
                    lower_precision=LowerPrecision.FP32 if precision == torch.float32 else LowerPrecision.FP16,
                    verbose_log=True,
                    timing_cache_prefix="",
                    save_timing_cache=False,
                    cuda_graph_batch_size=-1,
                    dynamic_batch=False,
                    is_aten=False,
                    use_experimental_fx_rt=False,
                    # correctness_atol=0.1,
                    # correctness_rtol=0.1
                )
            except Exception as e:
                logger.error(f"Failed to compile {slug}: {e}, skipping compilation")
                return None
            logger.debug(f"Compiled {slug} in {time.time() - st:.2f}s")

        logger.debug(f"Saving compiled {slug} model to {filename}")
        torch.save(trt_model, filename)
        return trt_model

    def _compile_vae(self, inputs: List[torch.Tensor], precision: torch.dtype = torch.float32) -> bool:
        """Compile the VAE model."""
        return self._compile(self._trace_vae, self.pipe.vae, inputs, precision, slug="vae")

    def _compile_text_encoder(self, inputs: List[torch.Tensor], precision: torch.dtype = torch.float32) -> bool:
        """Compile the text encoder model."""
        return self._compile(self._trace_text_encoder, self.pipe.text_encoder, inputs, precision, slug="text-encoder")

    def _compile_unet(self, inputs: List[torch.Tensor], precision: torch.dtype = torch.float32) -> str:
        """Compile the UNet model."""
        try:
            logger.info("Compiling UNet model")
            torch._dynamo.config.verbose = False
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.reset()
            with torch.no_grad():
                torch.cuda.empty_cache()
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        except RuntimeError as e:
            logger.error(f"Failed to compile UNet: {e}")

    def __compile__(
        self, inputs: List[torch.Tensor], precision: torch.dtype = torch.float32, width: int = 512, height: int = 512
    ) -> str:
        """Model compilation flow."""
        st = time.time()
        logger.debug("Compiling and patching model")
        try:
            logger.info("Compiling VAE model")
            vae_inputs = self.get_inputs_vae(width=width, height=height)
            trt_model = self._compile_vae(vae_inputs, precision)
            if trt_model is None:
                raise RuntimeError("Failed to compile VAE")
            logger.debug("Patching VAE model")
            self.pipe.vae.decode = trt_model
            self._patched["vae"] = True
            logger.debug("Done patching VAE model")
        except Exception as e:
            logger.error(f"Failed to compile VAE: {e}")
            self._patched["vae"] = False

        try:
            raise NotImplementedError("TODO: Implement text-encoder compilation")
            logger.info("Compiling text encoder model")
            text_inputs = self.get_inputs_text_encoder()
            trt_model = self._compile_text_encoder(text_inputs, precision)
            if trt_model is None:
                raise RuntimeError("Failed to compile text encoder")
            logger.debug("Patching text encoder model")
            self.pipe.text_encoder = trt_model
            self._patched["text_encoder"] = True
            logger.debug("Done patching text encoder model")
        except Exception as e:
            logger.error(f"Failed to compile text encoder: {e}")
            self._patched["text_encoder"] = False

        try:
            raise NotImplementedError("TODO: Implement UNet compilation")
            logger.info("Compiling UNet model")
            opt_model = self._compile_unet(inputs, precision)
            if opt_model is None:
                raise RuntimeError("Failed to compile UNet")
            logger.debug("Patching UNet model")
            self.pipe.unet = opt_model
            self._patched["unet"] = True
            logger.debug("Done patching UNet model")
        except Exception as e:
            logger.error(f"Failed to compile UNet: {e}")
            self._patched["unet"] = False

        # TODO (spillai): Investigate why we need to do this here.
        try:
            self.pipe._execution_device = self.device
        except Exception:
            logger.warning("Failed to set execution device")
        logger.debug(f"Done compiling models in {time.time() - st:.2f}s")

    def __call__(
        self,
        prompts: Union[str, List[str]],
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
    ) -> List[Image.Image]:
        """Generate images from text prompt."""

        if isinstance(prompts, str):
            prompts = [prompts]

        # TODO (spillai): Capture all of this in StableDiffusionTensorRTConfig?
        if not len(self._patched):
            self._compilation_cfg = StableDiffusionTensorRTCompilationConfig(
                precision=self.pipe.vae.dtype,
                width=width,
                height=height,
            )
            self.__compile__(prompts, precision=self.pipe.unet.dtype, width=width, height=height)

        return self.pipe(
            prompts * num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        ).images


# Register the model with the hub
# TODO (spillai): Ideally, we should do this via a decorator
# @hub.register("stabilityai/stable-diffusion-2")
# def stable_diffusion_2_ddim_fp16():
#     return StableDiffusion2("stabilityai/stable-diffusion-2", scheduler="ddim", dtype=torch.float16)
#
for model_name in StableDiffusion.configs.keys():
    hub.register(
        model_name,
        TaskType.IMAGE_GENERATION,
        StableDiffusion,
        init_args=(model_name,),
        init_kwargs={"scheduler": "ddim", "dtype": torch.float16},
        method_name="__call__",
        inputs={"prompts": Batch[str, 1], "num_images": int, "height": int, "width": int},
        outputs={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(None, None, 3), dtype="uint8")]]},
    )
