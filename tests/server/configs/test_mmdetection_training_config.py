import tempfile
from pathlib import Path

import pytest

from nos.server.train.openmmlab.mmdetection.config import MMDetectionTrainingJobConfig


@pytest.mark.skip(reason="This test requires autonomi/nos:latest-mmdet-gpu docker image.")
def test_mmdetection_training_job_config():
    from nos.server.train.openmmlab.mmdetection import config

    config_dir = Path(config.__file__).parent
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = MMDetectionTrainingJobConfig(
            working_directory=tmp_dir, config_filename=str(config_dir / "configs/yolox_s_8xb8-300e_coco_ft.py")
        )

        # Test the default values
        _ = cfg.entrypoint

        # Test working directory
        assert cfg.working_directory is not None
        assert Path(cfg.working_directory).exists()

        # Test weights directory
        assert cfg.weights_directory is not None
        assert Path(cfg.weights_directory).exists()

        # Test saving the configuration
        cfg.save()
        assert Path(cfg.output_directory).glob("*_config.json")

        # Test job configuration
        job_config = cfg.job_configuration()
        assert job_config is not None
        assert "submission_id" in job_config
        assert "runtime_env" in job_config
