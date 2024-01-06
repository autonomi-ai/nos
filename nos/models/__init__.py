from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
from PIL import Image

from nos import hub, internal_libs_available
from nos.common import ImageSpec, TaskType
from nos.common.types import Batch, ImageT

from ._noop import NoOp  # noqa: F401
from .blip import BLIP  # noqa: F401
from .clip import CLIP  # noqa: F401
from .faster_rcnn import FasterRCNN  # noqa: F401
from .llm import LLM  # noqa: F401
from .monodepth import MonoDepth  # noqa: F401
from .owlvit import OwlViT  # noqa: F401
from .sam import SAM
from .stable_diffusion import StableDiffusion  # noqa: F401
from .super_resolution import SuperResolution  # noqa: F401
from .tts import TextToSpeech  # noqa: F401
from .whisper import Whisper  # noqa: F401
from .yolox import YOLOX  # noqa: F401


if internal_libs_available():
    # Register internal models with hub
    from autonomi.nos._internal import models  # noqa: F401, F403
