import pytest
from typer.testing import CliRunner

from nos.cli.cli import app_cli
from nos.test.utils import PyTestGroup, skip_if_no_torch_cuda


runner = CliRunner()


def test_cli_profile_list():
    result = runner.invoke(app_cli, ["profile", "list"])
    assert result.exit_code == 0


@skip_if_no_torch_cuda
def test_cli_profile_model():
    result = runner.invoke(app_cli, ["profile", "model", "-m", "openai/clip"])
    assert result.exit_code == 0


@pytest.mark.benchmark(group=PyTestGroup.MODEL_PROFILE)
def test_cli_profile_all():
    result = runner.invoke(app_cli, ["profile", "all"])
    assert result.exit_code == 0
