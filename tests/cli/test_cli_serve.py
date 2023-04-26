import pytest
from typer.testing import CliRunner

from nos.cli.cli import app_cli


runner = CliRunner()


@pytest.mark.skip(reason="Not yet implemented.")
def test_cli_serve_list():
    result = runner.invoke(app_cli, ["serve", "list"])
    assert result.exit_code == 0


@pytest.mark.skip(reason="Not yet implemented.")
@pytest.mark.parametrize("model_name", ["stabilityai/stable-diffusion-v2"])
def test_cli_serve_deploy(model_name: str):
    result = runner.invoke(app_cli, ["serve", "deploy", "-m", model_name, "-d"])
    assert result.exit_code == 0
