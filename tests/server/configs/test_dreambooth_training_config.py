import tempfile
from pathlib import Path

from nos.server.train.dreambooth.config import StableDiffusionDreamboothTrainingJobConfig


def test_dreambooth_training_job_config():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = StableDiffusionDreamboothTrainingJobConfig(
            working_directory=tmp_dir,
            instance_directory=str(Path(tmp_dir)),
            instance_prompt="A photo of sks dog in a bucket",
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
