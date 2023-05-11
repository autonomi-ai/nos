import torch

from nos import hub

from .clip import CLIP  # noqa: F401
from .faster_rcnn import FasterRCNN
from .openmmlab.mmdetection.mmdetection import MMDetection  # noqa: F401
from .stable_diffusion import StableDiffusion2  # noqa: F401
