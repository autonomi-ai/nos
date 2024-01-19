import contextlib
import os
from pathlib import Path
from typing import Union

import pytest
from typer.testing import CliRunner


pytestmark = pytest.mark.cli
runner = CliRunner()


@contextlib.contextmanager
def path_ctx(path: Union[Path, str]):
    prev_cwd = Path.cwd()
    os.chdir(str(path))
    yield
    os.chdir(str(prev_cwd))


def test_cli_serve_help():
    from nos.cli.cli import app_cli

    result = runner.invoke(app_cli, ["serve", "--help"])
    assert result.exit_code == 0


# @pytest.mark.skip
def test_cli_serve():

    from nos.cli.cli import app_cli
    from nos.test.utils import NOS_TEST_DATA_DIR, NOS_TEST_IMAGE

    config_path = NOS_TEST_DATA_DIR / "hub/custom_model/config.yaml"
    # Move to the test data directory
    with path_ctx(config_path.parent):
        serve_up_result = runner.invoke(app_cli, ["serve", "up", "-c", "config.yaml", "-d"])
        assert serve_up_result.exit_code == 0

    from nos.client import Client

    # Wait for the server to be ready
    client = Client()
    client.WaitForServer()
    assert client.IsHealthy()

    # Check if the model is registered
    model = client.Module("custom-model")
    assert model is not None

    with client.UploadFile(NOS_TEST_IMAGE) as remote_path:
        assert remote_path is not None

    # Tear down the model
    serve_down_result = runner.invoke(app_cli, ["serve", "down", "--target", "custom_model"])
    assert serve_down_result.exit_code == 0
