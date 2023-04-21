import torch

from nos import hub

from .stable_diffusion import StableDiffusion2  # noqa: F401


@hub.register("stabilityai/stable-diffusion-2")
def stable_diffusion_2_ddim_fp16():
    return StableDiffusion2("stabilityai/stable-diffusion-2", scheduler="ddim", dtype=torch.float16)
