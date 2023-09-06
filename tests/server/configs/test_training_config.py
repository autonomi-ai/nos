import tempfile
from pathlib import Path

import pytest

from nos.server.train.config import TrainingJobConfig


def test_training_job_config():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = TrainingJobConfig(working_directory=tmp_dir)

        # Test the default values
        with pytest.raises(NotImplementedError):
            _ = cfg.entrypoint

        # Test working directory
        assert cfg.working_directory is not None
        assert Path(cfg.working_directory).exists()

        # Test weights directory
        assert cfg.weights_directory is not None
        assert Path(cfg.weights_directory).exists()

        # Test saving the configuration
        cfg.save()
        assert Path(cfg.working_directory).exists()
        assert Path(cfg.working_directory).is_dir()
        assert Path(cfg.working_directory).glob("*_config.json")
        assert Path(cfg.working_directory).glob("weights")

        # Test job configuration
        job_config = cfg.job_configuration()
        assert job_config is not None
        assert "submission_id" in job_config
        assert "runtime_env" in job_config
        assert "entrypoint" in job_config
        assert "entrypoint_resources" in job_config

        # Check if entrypoint requires GPUs
        # TODO (spillai): gpu configuration Not yet supported
        # assert "gpu" in job_config["entrypoint_resources"]
