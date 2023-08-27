from pathlib import Path

from nos.hub import MMLabHub


def test_hub_mmlab_hub_config():
    hub = MMLabHub()
    assert hub is not None
    assert len(hub) >= 0

    # Test singleton
    hub_ = MMLabHub()
    assert hub is hub_

    for key in hub:
        # Get item by key
        cfg = hub[key]
        assert hub.get(key) == cfg
        assert Path(cfg.config).exists(), f"Failed to load config={cfg.config}."
        assert Path(cfg.cached_checkpoint).exists(), f"Failed to load checkpoint={cfg.cached_checkpoint}."
