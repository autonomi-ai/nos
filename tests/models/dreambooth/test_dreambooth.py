import pytest
from loguru import logger
from PIL import Image

from nos.models.dreambooth.dreambooth import StableDiffusionLoRA
from nos.test.utils import PyTestGroup, skip_if_no_torch_cuda


def test_dreambooth_lora_configs():
    lora_configs = StableDiffusionLoRA.configs
    assert len(lora_configs) >= 0


@skip_if_no_torch_cuda
def test_dreambooth_lora():
    from nos.models.dreambooth.hub import StableDiffusionDreamboothHub

    namespace = "custom"
    hub = StableDiffusionDreamboothHub(namespace=namespace)
    for model_name in hub:
        logger.debug(f"Loading StableDiffusionLoRA from custom hub [namespace={namespace}, model={model_name}]")
        model = StableDiffusionLoRA(model_name)
        (img,) = model(prompts="bench on the moon", num_images=1)
        assert img is not None
        assert isinstance(img, Image.Image)


@pytest.mark.benchmark(group=PyTestGroup.HUB)
@skip_if_no_torch_cuda
def test_dreambooth_lora_civit():
    import tempfile
    from pathlib import Path

    # Dowload some weights from civit AI through their REST API:
    HOWLS_CASTLE_CIVIT_URL = "https://civitai.com/api/v1/models/14605"

    import requests

    response = requests.get(HOWLS_CASTLE_CIVIT_URL)
    assert response.status_code == 200
    response_json = response.json()
    assert len(response_json) > 0

    first_model_version = response_json["modelVersions"][0]
    assert first_model_version["baseModel"] == "SD 1.5"

    with tempfile.TemporaryDirectory() as tmpdir:
        weights_dir = Path(tmpdir) / "howls_castle" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "pytorch_lora_weights.safetensors"

        if not weights_path.exists():
            download_url = first_model_version["downloadUrl"]
            response = requests.get(download_url)
            assert response.status_code == 200

            with open(str(weights_path), "wb") as f:
                f.write(response.content)

        model = StableDiffusionLoRA(weights_dir=str(weights_path), model_name="runwayml/stable-diffusion-v1-5")
        (img,) = model(prompts="a castle on a hillside", num_images=1)
    assert img is not None
    assert isinstance(img, Image.Image)
