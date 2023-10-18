from nos.logging import logger


def test_hub_stablediffusion_dreambooth_hub_config():
    from nos.models.dreambooth.hub import (
        StableDiffusionDreamboothConfig,
        StableDiffusionDreamboothHub,
        StableDiffusionDreamboothLoRAConfig,
    )

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
