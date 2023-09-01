from loguru import logger
from PIL import Image

from nos.models.dreambooth.dreambooth import StableDiffusionLoRA
from nos.test.utils import skip_if_no_torch_cuda


def test_dreambooth_lora_configs():
    lora_configs = StableDiffusionLoRA.configs
    assert len(lora_configs) >= 0


@skip_if_no_torch_cuda
def test_dreambooth_lora():
    from nos.models.dreambooth.dreambooth import StableDiffusionDreamboothHub

    namespace = "custom"
    hub = StableDiffusionDreamboothHub(namespace=namespace)
    for model_name in hub:
        logger.debug(f"Loading StableDiffusionLoRA from custom hub [namespace={namespace}, model={model_name}]")
        model = StableDiffusionLoRA(model_name)
        (img,) = model(prompts="bench on the moon", num_images=1)
        assert img is not None
        assert isinstance(img, Image.Image)
