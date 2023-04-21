import pytest
from typer.testing import CliRunner

from nos.cli.cli import app_cli


runner = CliRunner()


def test_cli_hub_list():
    result = runner.invoke(app_cli, ["hub", "list"])
    assert result.exit_code == 0


@pytest.mark.skip(reason="Not yet implemented.")
def test_cli_hub_download():
    from nos import hub
    from nos.constants import NOS_MODELS_DIR

    models = hub.list()
    for model in models:
        result = runner.invoke(app_cli, ["hub", "download", "-m", model, "-d", NOS_MODELS_DIR])
        assert result.exit_code == 0
