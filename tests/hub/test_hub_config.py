from pathlib import Path

from nos.hub import MMLabHub
from nos.logging import logger
from nos.models.dreambooth.dreambooth import StableDiffusionDreamboothHub


def test_hub_mmlab_hub_config():
    hub = MMLabHub(namespace="openmmlab/mmdetection")
    assert hub is not None
    assert len(hub) >= 0

    # Test singleton
    assert hub is MMLabHub(namespace="openmmlab/mmdetection")

    for key in hub:
        # Get item by key
        cfg = hub[key]
        assert hub.get(key) == cfg
        assert Path(cfg.config).exists(), f"Failed to load config={cfg.config}."
        assert Path(cfg.cached_checkpoint).exists(), f"Failed to load checkpoint={cfg.cached_checkpoint}."


def test_hub_stablediffusion_dreambooth_hub_config():
    from nos.models.dreambooth.dreambooth import StableDiffusionDreamboothConfig, StableDiffusionDreamboothLoRAConfig

    hub = StableDiffusionDreamboothHub(namespace="diffusers/dreambooth")
    assert hub is not None
    assert len(hub) >= 0

    tmp_hub = StableDiffusionDreamboothHub(namespace="custom")
    assert tmp_hub is not None
    assert len(tmp_hub) >= 0

    # Test singleton
    hub_ = StableDiffusionDreamboothHub(namespace="diffusers/dreambooth")
    assert hub is hub_

    # Test singleton with different namespace
    hub_ = StableDiffusionDreamboothHub(namespace="diffusers/dreambooth2")
    assert hub is not hub_

    for key in tmp_hub:
        logger.debug(f"key={key}")
        cfg = tmp_hub[key]
        assert isinstance(cfg, (StableDiffusionDreamboothConfig, StableDiffusionDreamboothLoRAConfig))
